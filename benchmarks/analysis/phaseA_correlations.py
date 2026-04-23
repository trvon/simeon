#!/usr/bin/env python3
"""Phase A: correlate {nqc, wig_full, score_decay_rate} with per-query nDCG.

Reads oracle-router rows (config="router_oracle_*") from --router-per-query
JSONL on each corpus, computes Spearman rho for:
  1. predictor vs oracle-best nDCG@10  (overall difficulty QPP)
  2. predictor vs ndcg_atire           (Atire-specific QPP)
  3. predictor vs ndcg_sab             (SAB-specific QPP)
  4. predictor vs (ndcg_atire - ndcg_sab)  (routing-discriminator)

T1 / T2 gate criteria:
  - NQC validate: rho(nqc, oracle_best) > 0.4 on >=2/3 corpora
  - WIG validate: rho(wig_full) - rho(score_decay_rate) > 0.1 on >=2/3
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_oracle_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cfg = row.get("config", "")
            if cfg.startswith("router_oracle"):
                rows.append(row)
    return rows


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho via Pearson on ranks. Stdlib + numpy only."""
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def analyze(corpus: str, jsonl_path: Path) -> dict:
    rows = load_oracle_rows(jsonl_path)
    if not rows:
        return {"corpus": corpus, "n": 0}

    # Filter rows with a relevant doc — QPP correlations are only defined
    # when there is signal to predict.
    rows = [r for r in rows if r.get("has_relevant")]
    n = len(rows)

    nqc = [r["nqc"] for r in rows]
    wig_full = [r["wig_full"] for r in rows]
    sdr = [r["score_decay_rate"] for r in rows]

    ndcg_atire = [r["ndcg_atire"] for r in rows]
    ndcg_sab = [r["ndcg_sab"] for r in rows]
    ndcg_cascade = [r["ndcg_cascade"] for r in rows]
    ndcg_oracle = [max(a, s, c) for a, s, c in zip(ndcg_atire, ndcg_sab, ndcg_cascade)]
    ndcg_diff = [a - s for a, s in zip(ndcg_atire, ndcg_sab)]

    out = {"corpus": corpus, "n": n}
    for pname, p in [("nqc", nqc), ("wig_full", wig_full), ("score_decay_rate", sdr)]:
        out[pname] = {
            "vs_oracle_best": spearman(p, ndcg_oracle),
            "vs_atire": spearman(p, ndcg_atire),
            "vs_sab": spearman(p, ndcg_sab),
            "vs_atire_minus_sab": spearman(p, ndcg_diff),
        }
    return out


def fmt_row(name: str, r: dict, corpora: list[str]) -> str:
    cells = []
    for c in corpora:
        v = r[c].get(name)
        cells.append(f"{v:+.3f}" if isinstance(v, dict) is False and v == v else "  n/a ")
    return " | ".join(cells)


def main() -> int:
    res_dir = Path(__file__).resolve().parent.parent / "results" / "phaseA"
    corpora = ["scifact", "nfcorpus", "fiqa"]
    by_corpus: dict[str, dict] = {}
    for c in corpora:
        path = res_dir / f"{c}_per_query.jsonl"
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        by_corpus[c] = analyze(c, path)

    print("Phase A — Spearman rho (oracle-router rows only)")
    print("=" * 78)
    counts = " | ".join(f"{c}={by_corpus[c]['n']}" for c in corpora)
    print(f"queries (relevant): {counts}")
    print()

    targets = [
        ("vs_oracle_best", "vs oracle-best nDCG@10"),
        ("vs_atire", "vs Atire nDCG@10"),
        ("vs_sab", "vs SAB nDCG@10"),
        ("vs_atire_minus_sab", "vs (Atire - SAB) nDCG@10  [routing signal]"),
    ]
    predictors = ["nqc", "wig_full", "score_decay_rate"]

    for tkey, tlabel in targets:
        print(f"-- {tlabel}")
        header = f"  {'predictor':<20}" + " | ".join(f"{c:>8}" for c in corpora)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for p in predictors:
            cells = []
            for c in corpora:
                v = by_corpus[c][p][tkey]
                cells.append(f"{v:>+8.3f}" if v == v else f"{'  n/a ':>8}")
            print(f"  {p:<20}" + " | ".join(cells))
        print()

    # T1 gate
    print("=" * 78)
    print("T1 (NQC) gate: rho(nqc, oracle_best) > 0.4 on >=2/3 corpora")
    n_pass_t1 = sum(1 for c in corpora if by_corpus[c]["nqc"]["vs_oracle_best"] > 0.4)
    n_disprove_t1 = sum(1 for c in corpora if by_corpus[c]["nqc"]["vs_oracle_best"] < 0.2)
    print(f"  validate: {n_pass_t1}/3 corpora pass (need >=2)")
    print(f"  disprove: {n_disprove_t1}/3 corpora rho < 0.2 (>=2 = disprove)")
    if n_pass_t1 >= 2:
        print("  >>> T1 VALIDATED")
    elif n_disprove_t1 >= 2:
        print("  >>> T1 DISPROVED")
    else:
        print("  >>> T1 INCONCLUSIVE")

    # T2 gate
    print()
    print("T2 (WIG) gate: rho(wig_full) - rho(score_decay_rate) > 0.1 on >=2/3")
    deltas = []
    for c in corpora:
        delta = (by_corpus[c]["wig_full"]["vs_oracle_best"]
                 - by_corpus[c]["score_decay_rate"]["vs_oracle_best"])
        deltas.append(delta)
        print(f"  {c}: delta = {delta:+.3f}")
    n_pass_t2 = sum(1 for d in deltas if d > 0.1)
    n_disprove_t2 = sum(1 for d in deltas if abs(d) < 0.05)
    if n_pass_t2 >= 2:
        print("  >>> T2 VALIDATED")
    elif n_disprove_t2 >= 2:
        print("  >>> T2 DISPROVED (WIG-lite already captures the signal)")
    else:
        print("  >>> T2 INCONCLUSIVE")

    return 0


if __name__ == "__main__":
    sys.exit(main())
