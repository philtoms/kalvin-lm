"""Compile a .ks file into a data/dialogue state JSON (all entries in ltm)."""

import sys
from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.signifier import NLPSignifier
from dialogue.engine_state import EngineState
from ks.compiler import compile_source


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: build_state.py <source.ks | "kscript source"> [out.json]', file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    is_file = Path(arg).is_file() or arg.endswith(".ks")
    source = Path(arg).read_text() if is_file else arg

    tok = BPETokenizer()
    sigf = NLPSignifier()
    entries = compile_source(source, tokenizer=tok, signifier=sigf, dev=True)

    stem = Path(arg).stem if is_file else entries[0].kline.signature.label
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/dialogue") / (stem + ".json")

    state = EngineState(sigf)
    for e in entries:
        state.ground(e.kline)

    state.save(out_path)
    print(f"{arg} -> {out_path}: {len(entries)} entries, {len(state.ltm)} ltm signatures")


if __name__ == "__main__":
    main()
