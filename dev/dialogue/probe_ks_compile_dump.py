"""Compile the wdmh scripts and dump entries; then run the harness and trace."""
import sys
sys.path.insert(0, "src")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from kalvin.significance import BandLayout

layout = BandLayout()
tok = BPETokenizer()
sig = NLPSignifier()
bits: dict[str, int] = {}

for path in ("data/scripts/mhall.ks", "data/scripts/wdmh-underfit.ks"):
    print(f"== {path} ==")
    src = open(path).read()
    entries = compile_source(src, tokenizer=tok, signifier=sig, dev=True, word_bits=bits)
    for e in entries:
        d = e.kline.dbg
        name = getattr(e.kline.signature, "label", "") or hex(e.kline.signature)
        nodes = ", ".join(getattr(n, "label", "") or hex(n) for n in e.kline.nodes)
        ann = repr(d.annotation) if d else "?"
        print(
            f"  {name if d is None else d.label or name}:[{nodes}]"
            f"  op={d.op if d else '?'} scope={d.scope if d else '?'}"
            f" ann={ann} band={layout.classify(e.significance)}"
            f" sig_byte=0x{e.significance:02x}"
        )
    print()
