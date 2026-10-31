#!/usr/bin/env python3
"""Regenerate the StaticEmbedding parity fixture (tiny.smstat + cases.tsv).

    python3 tests/data/static_fixture/make_fixture.py

Needs `tokenizers` and `numpy`. The reference tokenizer is HuggingFace's WordPiece with the BERT
normalizer and pre-tokenizer, the same pipeline model2vec/potion models use. Each case line is:
    text <TAB> space-separated token ids <TAB> space-separated vector components (%.7g)
Text is written with \\t, \\n and \\\\ escaped.
"""
import os
import sys

import numpy as np
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "static_embedding"))
from export_static_table import write_smstat  # noqa: E402

WORDS = """the of and a to in is was for on with as by at from this that be are not it
vitamin d deficiency increases cardiovascular risk cancer cell cells protein gene genes
expression patients study results effect effects high low blood pressure heart disease
cafe naive angstrom resume uber zurich data model models deep learning search engine
run running ran fast faster fastest un able do does done 2024 10 100 covid 19 sars cov
""".split()
PIECES = """##s ##ing ##ed ##er ##est ##ly ##able ##tion ##ness ##al ##ic ##ize ##ization
##a ##b ##c ##d ##e ##f ##g ##h ##i ##k ##l ##m ##n ##o ##p ##r ##t ##u ##v ##x ##y ##z
##1 ##2 ##0 ##9 ##4 ##ll ##rd ##ps ##ee ##ou ##an ##en ##in ##on
""".split()
# Multi-character punctuation pieces as Tokenlearn adds them (normalized, lstrip/rstrip), plus a
# special token matched on the raw text.
ADDED = [").", "),", "...", ".,", "..", "”.", "!!", "²."]
SPECIAL = ["[MASK]"]
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789") + list(".,;:!?()[]{}'\"-/&%$#@*+=<>_~^|\\`") + ["中", "文", "の"]
CASES = [
    "Vitamin D deficiency increases cardiovascular risk",
    "the cells were running faster than the protein models",
    "COVID-19 and SARS-CoV-2: results (2024)!",
    "unable undone deepest learnings",
    "Café naïve Ångström résumé Über Zürich",
    "中文 search の engine",
    "tab\there\nnewline   spaces",
    "punctuation...!!! ??? -- // \\\\ @#$",
    "supercalifragilisticexpialidociousword",
    "xyzzyq plugh",
    "emoji 🙂 inside text",
    "control\x07char and nbsp\u00a0space",
    "",
    "   ",
    "a",
    "The THE the",
    "the end), and so on...",
    "word.,next and ...... dots",
    "e.g., (1990). “quoted”.",
    "x². wow!!! [MASK] and [mask]",
]
MAX_WORD = 20


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def main() -> int:
    tokens = ["[UNK]"]
    for t in WORDS + PIECES + CHARS + ADDED + SPECIAL:
        if t not in tokens:
            tokens.append(t)
    vocab = {t: i for i, t in enumerate(tokens)}
    tk = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=MAX_WORD))
    tk.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.add_tokens([AddedToken(t, normalized=True, lstrip=True, rstrip=True) for t in ADDED])
    tk.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in SPECIAL])
    added = [(vocab[t], True, False) for t in ADDED] + [(vocab[t], False, True) for t in SPECIAL]

    rng = np.random.default_rng(7)
    E = rng.standard_normal((len(tokens), 8)) * rng.uniform(0.2, 2.0, (len(tokens), 1))
    out = os.path.join(HERE, "tiny.smstat")
    scales, q = write_smstat(out, tokens, E, True, True, vocab["[UNK]"], MAX_WORD, added)
    deq = q.astype(np.float64) * scales.astype(np.float64)[:, None]

    with open(os.path.join(HERE, "cases.tsv"), "w", encoding="utf-8") as f:
        for text in CASES:
            ids = tk.encode(text, add_special_tokens=False).ids
            keep = [i for i in ids if i != vocab["[UNK]"]]
            v = deq[keep].mean(0) if keep else np.zeros(deq.shape[1])
            n = np.linalg.norm(v)
            v = v / n if n > 0 else v
            f.write(esc(text) + "\t" + " ".join(map(str, ids)) + "\t" + " ".join(f"{x:.7g}" for x in v) + "\n")
    print(f"wrote {out} ({len(tokens)} tokens) and cases.tsv ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
