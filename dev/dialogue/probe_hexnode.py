"""Decode the 0x1c00001efd proposal node: words, token, and who holds it."""
import sys
sys.path.insert(0, "src")

from kalvin.bpe_tokenizer import BPETokenizer
from dialogue.harness import load_engine

tok = BPETokenizer()

MYSTERY = [0x80000002F45, 0x1C00001EFD, 0xE000007FFE, 0x420000017E, 0x1F00001FFD]


def nm(v):
    return getattr(v, "label", "") or hex(int(v))


def words_for(h, v):
    upper = int(v) >> 32
    return [w for w, b in h.word_bits.items() if upper & b]


def decode(v):
    tok_id = int(v) & 0xFFFFFFFF
    if not tok_id:
        return "(no token half)"
    try:
        return repr(tok.decode([tok_id]))
    except Exception:
        return f"(undecodable token {tok_id})"


h = load_engine("data/dialogue/mhall.json", tok)

print("word bits (word -> bit):")
for w, b in sorted(h.word_bits.items(), key=lambda kv: kv[1]):
    print(f"  bit {b:2}  {w!r}")

print("\nmystery values:")
for v in MYSTERY:
    print(f"  {hex(v)}  words={words_for(h, v)}  token={decode(v)}")


def dump_stores(tag):
    print(f"\n{tag}:")
    st = h.state
    seen = set()
    for name, store in (("frame", st.frame), ("ltm", st.ltm)):
        for sig, bucket in store.items():
            for k in bucket:
                key = (name, int(k.signature), tuple(int(n) for n in k.nodes))
                if key in seen:
                    continue
                seen.add(key)
                vals = [int(k.signature)] + [int(n) for n in k.nodes]
                if any(m in vals for m in MYSTERY):
                    print(f"  [{name}] {nm(k.signature)}({hex(int(k.signature))}):"
                          f"[{', '.join(nm(n) + '(' + hex(int(n)) + ')' for n in k.nodes)}]")


dump_stores("prior state (mhall.json) — holders of mystery values")
