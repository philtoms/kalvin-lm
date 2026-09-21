"""Decode the [a, Mod, lamb, a] proposal: values, labels, goal, why done."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from ks.compiler import compile_source
from dev.dialogue.harness import make_engine, load_engine
from kalvin.kline import KLine, is_ask, is_terminal
from kalvin import hop as hop_mod

tok = BPETokenizer()

def nm(v):
    return getattr(v, "label", "") or hex(int(v))

def word_bits_of(v):
    """The distinct word bits (upper 32) in a value."""
    upper = int(v) >> 32
    return [i for i in range(64) if upper >> i & 1]

h = load_engine("data/dialogue/mhall.json", tok)
sig = h.signifier
bits = h.word_bits

mhall_src = open("data/scripts/mhall.ks").read()
entries = compile_source(mhall_src, tokenizer=tok, signifier=sig, dev=True,
                         word_bits=bits, known_words=h.known_words)
h.engine.rationalise(entries)

print("word bits table (word -> bit, sig):")
for w, b in bits.items():
    ids = tok.encode(w)
    orval = 0
    for i in ids:
        orval |= int(i)
    print(f"  {w!r:12} bit {b}  tokens {[hex(int(i)) for i in ids]}  OR {hex(orval)}")

st = h.state
print("\nwork list:")
for k in st.work_list:
    print(f"  {nm(k.signature)}({hex(int(k.signature))}):[{', '.join(nm(n) for n in k.nodes)}]")

print("\nframe (non-terminal):")
for s, bucket in st.frame.items():
    for k in bucket:
        if is_terminal(k):
            continue
        print(f"  {nm(k.signature)}({hex(int(k.signature))}):[{', '.join(nm(n) for n in k.nodes)}]")
