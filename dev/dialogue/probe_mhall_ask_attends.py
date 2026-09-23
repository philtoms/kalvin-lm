"""Counterfactual: feed the MHALL ask at S3 (not its graded-S4 byte) so it
attends instead of being fast-refused. Does the derivation then move?"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.kline import using_resolver, is_ask
from kalvin.kvalue import KValue
from ks.compiler import compile_source
from dev.dialogue.harness import make_rationaliser

tok = BPETokenizer()
h = make_rationaliser(tok)
st = h.state
sig = h.signifier
bits: dict[str, int] = {}


def turn(feeds):
    with using_resolver(st.find):
        h.rationaliser.rationalise(feeds)
        batch = []
        while True:
            size = len(st.work_list)
            batch.extend(cogitate(st))
            if len(st.work_list) == size:
                break
        return batch


def name(v):
    return getattr(v, "label", "") or hex(int(v))


def render(k):
    nodes = ", ".join(name(n) for n in k.nodes)
    return f"{name(k.signature)}:[{nodes}]"


entries = compile_source(open("data/scripts/mhall.ks").read(), tokenizer=tok,
                         signifier=sig, dev=True, word_bits=bits)
ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
scaffolding = [e for e in entries if e is not ask]

print("== entries ==")
for e in entries:
    print(f"  {render(e.kline)}  0x{e.significance:02x}  op={e.kline.dbg.op if e.kline.dbg else None}")

batch1 = turn(scaffolding)
print(f"\nscaffolding emissions: {len(batch1)}")
for v in batch1:
    print(f"  -> {render(v.kline)} 0x{v.significance:02x}")

# Counterfactual feed: the ask at S3 (attends) instead of its graded S4.
out = turn([KValue(ask.kline, 0x40)])
print(f"\nask-at-S3 emissions: {len(out)}")
for v in out:
    print(f"  -> {render(v.kline)} 0x{v.significance:02x}")
print(f"ask in work_list: {any(e.signature == ask.kline.signature for e in st.work_list)}")
print(f"ask refused: {st.is_refused(ask.kline)}")
print(f"ask grounded: {st.is_grounded(ask.kline)}")

print("\n== final work list ==")
for k in st.work_list:
    print(f"  {render(k)}")

# And with the harness's real graded byte for contrast.
h2 = make_rationaliser(tok)
st2 = h2.state
with using_resolver(st2.find):
    h2.rationaliser.rationalise(scaffolding)
    while True:
        size = len(st2.work_list)
        cogitate(st2)
        if len(st2.work_list) == size:
            break
    h2.rationaliser.rationalise([ask])
print(f"\nask-at-graded-byte: refused={st2.is_refused(ask.kline)} "
      f"work_list_len={len(st2.work_list)}")
