"""Dump the compiled wdmh.ks entries — where does the ask signature come from."""
import sys
sys.path.insert(0, "src")

from kalvin.bpe_tokenizer import BPETokenizer
from ks.compiler import compile_source
from dialogue.harness import load_engine

tok = BPETokenizer()
h = load_engine("data/dialogue/mhall.json", tok)

src = open("data/scripts/wdmh.ks").read()
entries = compile_source(src, tokenizer=tok, signifier=h.signifier, dev=True,
                         word_bits=h.word_bits, known_words=h.known_words)

ASK = 1 << 63


def words_of(v):
    upper = int(v) >> 32 & ~0x80000000
    return [w for w, b in h.word_bits.items() if upper & b]


for e in entries:
    k = e.kline
    sig = int(k.signature)
    mark = "?" if sig & ASK else " "
    print(f"{mark} {hex(sig)}  words={words_of(sig)}  tok={hex(sig & 0xFFFFFFFF)}  "
          f"op={k.dbg.op if k.dbg else '?'}  label={getattr(k.signature, 'label', '')!r}")
    for n in k.nodes:
        v = int(n)
        print(f"     node {hex(v)}  words={words_of(v)}  tok={v & 0xFFFFFFFF:#x} "
              f"tokstr={tok.decode([v & 0xFFFFFFFF]) if v & 0xFFFFFFFF else '-'}")
