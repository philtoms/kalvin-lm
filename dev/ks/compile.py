"""Compile a .ks script, printing each entry; optionally under a saved model's word_bits."""

import argparse
import json
from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="path to a .ks file")
    ap.add_argument("--model", help="path to a dialogue state JSON to compile under")
    args = ap.parse_args()

    word_bits = None
    known_words = None
    if args.model:
        word_bits = json.loads(Path(args.model).read_text()).get("word_bits")
        known_words = list(word_bits or {})

    tok = BPETokenizer()
    sigf = NLPSignifier()
    entries = compile_source(
        Path(args.script).read_text(),
        tokenizer=tok, signifier=sigf, dev=True,
        word_bits=word_bits, known_words=known_words,
    )
    for e in entries:
        print("   ", e.kline.signature.label, [n.label for n in e.kline.nodes],
              e.kline.dbg.op if e.kline.dbg else "?",
              "ann:", repr(e.kline.dbg.annotation if e.kline.dbg else None))


if __name__ == "__main__":
    main()
