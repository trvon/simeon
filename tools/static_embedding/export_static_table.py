#!/usr/bin/env python3
"""Export a model2vec static model to Simeon's SMSTAT01 table.

    python3 tools/static_embedding/export_static_table.py --model minishlab/potion-retrieval-32M \
        --dims 256 --out potion-retrieval-32m-256.smstat

Format (little-endian):
    magic        8 bytes  "SMSTAT01"
    version      u32      1
    dim          u32
    vocab_size   u32
    flags        u32      bit0 lowercase, bit1 strip_accents
    unk_id       u32      0xFFFFFFFF when the vocabulary has no [UNK]
    max_word     u32      WordPiece max_input_chars_per_word
    vocab        vocab_size x (u16 length, UTF-8 bytes); continuation pieces start with "##"
    scales       vocab_size x f32
    rows         vocab_size x dim x i8   (row value = int8 * scale)

Encoding is the mean of the token rows of a text, L2-normalized. PCA (centered, over the
rows) reduces the dimension before int8 quantization with one scale per row.
"""
import argparse
import json
import struct
import sys

import numpy as np


def quantize(E):
    """Per-row int8 with one f32 scale per row; returns (scales, int8 rows)."""
    E = np.asarray(E, dtype=np.float64)
    scales = np.abs(E).max(1) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.round(E / scales[:, None]), -127, 127).astype(np.int8)
    return scales.astype(np.float32), q


def write_smstat(path, tokens, E, lowercase, strip_accents, unk_id, max_word):
    scales, q = quantize(E)
    flags = (1 if lowercase else 0) | (2 if strip_accents else 0)
    with open(path, "wb") as f:
        f.write(b"SMSTAT01")
        f.write(struct.pack("<6I", 1, q.shape[1], len(tokens), flags, unk_id, max_word))
        for token in tokens:
            b = token.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        f.write(scales.astype("<f4").tobytes())
        f.write(q.tobytes())
    return scales, q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model2vec model name or local path")
    ap.add_argument("--dims", type=int, default=0, help="PCA target dimension (0 keeps the model's)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from model2vec import StaticModel

    m = StaticModel.from_pretrained(a.model)
    E = np.asarray(m.embedding, dtype=np.float64)
    tok = json.loads(m.tokenizer.to_str())
    model = tok["model"]
    if model.get("type") != "WordPiece":
        print(f"unsupported tokenizer model: {model.get('type')}", file=sys.stderr)
        return 2
    if model.get("continuing_subword_prefix", "##") != "##":
        print("unsupported continuing_subword_prefix", file=sys.stderr)
        return 2
    norm = tok.get("normalizer") or {}
    if norm.get("type") != "BertNormalizer":
        print(f"unsupported normalizer: {norm.get('type')}", file=sys.stderr)
        return 2
    lowercase = bool(norm.get("lowercase", True))
    strip_accents = norm.get("strip_accents")
    strip_accents = lowercase if strip_accents is None else bool(strip_accents)

    vocab = sorted(model["vocab"].items(), key=lambda kv: kv[1])
    if [i for _, i in vocab] != list(range(len(vocab))) or len(vocab) != E.shape[0]:
        print("vocabulary ids are not dense or do not match the table", file=sys.stderr)
        return 2
    unk_token = model.get("unk_token", "[UNK]")
    unk_id = model["vocab"].get(unk_token, 0xFFFFFFFF)

    if a.dims and a.dims < E.shape[1]:
        mu = E.mean(0)
        _, _, vt = np.linalg.svd(E - mu, full_matrices=False)
        E = (E - mu) @ vt[: a.dims].T
    dim = E.shape[1]

    write_smstat(a.out, [t for t, _ in vocab], E, lowercase, strip_accents, unk_id,
                 int(model.get("max_input_chars_per_word", 100)))
    print(f"wrote {a.out}: vocab={len(vocab)} dim={dim} lowercase={lowercase} strip_accents={strip_accents}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
