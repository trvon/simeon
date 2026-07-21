#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "experiments" / "prepare_beir_fixture.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PrepareBeirFixtureTest(unittest.TestCase):
    def test_converts_dev_and_test_without_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            target = root / "fixture"
            (source / "qrels").mkdir(parents=True)
            write_jsonl(
                source / "corpus.jsonl",
                [
                    {"_id": "d2", "title": "Second", "text": "beta\ntext"},
                    {"_id": "d1", "title": "First", "text": "alpha\ttext"},
                ],
            )
            write_jsonl(
                source / "queries.jsonl",
                [
                    {"_id": "q-test", "text": "test query"},
                    {"_id": "q-dev", "text": "dev query"},
                    {"_id": "unused", "text": "not judged"},
                ],
            )
            (source / "qrels" / "dev.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq-dev\td1\t2\n", encoding="utf-8"
            )
            (source / "qrels" / "test.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq-test\td2\t1\n", encoding="utf-8"
            )

            subprocess.run(
                [sys.executable, str(CONVERTER), str(source), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                (target / "corpus.tsv").read_text(encoding="utf-8"),
                "d1\tFirst alpha text\nd2\tSecond beta text\n",
            )
            self.assertEqual(
                (target / "queries_dev.tsv").read_text(encoding="utf-8"),
                "q-dev\tdev query\n",
            )
            self.assertEqual(
                (target / "qrels_dev.tsv").read_text(encoding="utf-8"),
                "q-dev\td1\t2\n",
            )
            self.assertEqual(
                (target / "queries.tsv").read_text(encoding="utf-8"),
                "q-test\ttest query\n",
            )

    def test_falls_back_to_train_when_dev_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            target = root / "fixture"
            (source / "qrels").mkdir(parents=True)
            write_jsonl(source / "corpus.jsonl", [{"_id": "d", "text": "document"}])
            write_jsonl(source / "queries.jsonl", [{"_id": "q", "text": "query"}])
            qrels = "query-id\tcorpus-id\tscore\nq\td\t1\n"
            (source / "qrels" / "train.tsv").write_text(qrels, encoding="utf-8")
            (source / "qrels" / "test.tsv").write_text(qrels, encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(CONVERTER), str(source), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("selection=train", completed.stderr)
            self.assertEqual((target / "qrels_dev.tsv").read_text(encoding="utf-8"), "q\td\t1\n")


if __name__ == "__main__":
    unittest.main()
