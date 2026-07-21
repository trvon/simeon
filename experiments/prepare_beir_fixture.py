#!/usr/bin/env python3
"""Convert an unpacked BEIR dataset into a model-free Simeon fixture.

Usage: prepare_beir_fixture.py <beir_dataset_dir> <fixture_dir>

The source directory must contain corpus.jsonl, queries.jsonl, qrels/test.tsv,
and either qrels/dev.tsv or qrels/train.tsv. Train is used as the selection
split only when dev is absent (for example, SciFact).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def clean(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def read_jsonl(path: Path, kind: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
                identifier = str(row.get("_id", ""))
                if not identifier:
                    raise ValueError(f"{path}:{line_number}: missing _id")
                if identifier in rows:
                    raise ValueError(f"{path}:{line_number}: duplicate {kind} id {identifier!r}")
                rows[identifier] = row
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: no {kind} rows")
    return rows


def read_qrels(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            required = {"query-id", "corpus-id", "score"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path}: expected tab-separated header {sorted(required)}")
            for line_number, row in enumerate(reader, start=2):
                query_id = row["query-id"]
                document_id = row["corpus-id"]
                try:
                    relevance = int(row["score"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: score must be an integer") from error
                if relevance <= 0:
                    continue
                key = (query_id, document_id)
                if key in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate positive qrel {key!r}")
                seen.add(key)
                rows.append((query_id, document_id, relevance))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: no positive qrels")
    rows.sort()
    return rows


def validate_qrels(
    qrels: Iterable[tuple[str, str, int]], queries: dict[str, dict], corpus: dict[str, dict], path: Path
) -> None:
    for query_id, document_id, _ in qrels:
        if query_id not in queries:
            raise ValueError(f"{path}: qrel references unknown query {query_id!r}")
        if document_id not in corpus:
            raise ValueError(f"{path}: qrel references unknown document {document_id!r}")


def write_atomic(path: Path, lines: Iterable[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.writelines(lines)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert(source_dir: Path, output_dir: Path) -> str:
    corpus = read_jsonl(source_dir / "corpus.jsonl", "corpus")
    queries = read_jsonl(source_dir / "queries.jsonl", "query")
    qrels_dir = source_dir / "qrels"
    selection_name = "dev" if (qrels_dir / "dev.tsv").is_file() else "train"
    selection_path = qrels_dir / f"{selection_name}.tsv"
    holdout_path = qrels_dir / "test.tsv"
    selection_qrels = read_qrels(selection_path)
    holdout_qrels = read_qrels(holdout_path)
    validate_qrels(selection_qrels, queries, corpus, selection_path)
    validate_qrels(holdout_qrels, queries, corpus, holdout_path)

    selection_queries = sorted({query_id for query_id, _, _ in selection_qrels})
    holdout_queries = sorted({query_id for query_id, _, _ in holdout_qrels})
    output_dir.mkdir(parents=True, exist_ok=True)

    def corpus_lines() -> Iterable[str]:
        for document_id in sorted(corpus):
            row = corpus[document_id]
            text = clean(f"{row.get('title', '')} {row.get('text', '')}".strip())
            yield f"{document_id}\t{text}\n"

    def query_lines(query_ids: Iterable[str]) -> Iterable[str]:
        for query_id in query_ids:
            yield f"{query_id}\t{clean(str(queries[query_id].get('text', '')))}\n"

    def qrel_lines(qrels: Iterable[tuple[str, str, int]]) -> Iterable[str]:
        for query_id, document_id, relevance in qrels:
            yield f"{query_id}\t{document_id}\t{relevance}\n"

    write_atomic(output_dir / "corpus.tsv", corpus_lines())
    write_atomic(output_dir / "queries_dev.tsv", query_lines(selection_queries))
    write_atomic(output_dir / "qrels_dev.tsv", qrel_lines(selection_qrels))
    write_atomic(output_dir / "queries.tsv", query_lines(holdout_queries))
    write_atomic(output_dir / "qrels.tsv", qrel_lines(holdout_qrels))
    return (
        f"selection={selection_name} docs={len(corpus)} "
        f"dev_queries={len(selection_queries)} test_queries={len(holdout_queries)}"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        summary = convert(Path(argv[1]), Path(argv[2]))
    except ValueError as error:
        print(f"prepare_beir_fixture: {error}", file=sys.stderr)
        return 2
    print(f"[fixture] {argv[2]}: {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
