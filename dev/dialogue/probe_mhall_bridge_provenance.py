"""One clean derivation: the ask toward SVO on a fresh state. Dump every
bridge, walk arrival, and the re-entry chain with labels."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.derivation import Derivation
from kalvin.hop import run_hops
from kalvin.kline import KLine, using_resolver
from kalvin.kvalue import KValue
from ks.compiler import compile_source
from dev.dialogue.harness import make_rationaliser

tok = BPETokenizer()
h = make_rationaliser(tok)
st = h.state
sig = h.signifier
bits: dict[str, int] = {}

entries = compile_source(Path("data/scripts/mhall.ks").read_text(),
                         tokenizer=tok, signifier=sig, dev=True, word_bits=bits)
ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
goal = next(e for e in entries if e.kline.dbg and e.kline.dbg.label == "SVO"
            and e.kline.dbg.op == "CANONICALISES")
scaffolding = [e for e in entries if e is not ask]


def name(v):
    return getattr(v, "label", "") or hex(int(v))


def render(k):
    return f"{name(k.signature)}:[{', '.join(name(n) for n in k.nodes)}]"


with using_resolver(st.find):
    h.rationaliser.rationalise(scaffolding)
    while True:
        size = len(st.work_list)
        cogitate(st)
        if len(st.work_list) == size:
            break

print("== value inventory (label -> hex) ==")
seen: dict[int, str] = {}
for k in st.where(lambda k: True, True):
    for v in [k.signature, *k.nodes]:
        seen.setdefault(int(v), name(v) or hex(int(v)))
for v, lbl in seen.items():
    print(f"  {v:#x}  {lbl}")

print("\n== bridges written per derivation (instrumented) ==")
orig_ground = Derivation._ground_composed
def spy(self, composed):
    ok = orig_ground(self, composed)
    if ok:
        print(f"    bridge {render(composed)} acq={composed.acq_depth}")
    return ok
Derivation._ground_composed = spy

with using_resolver(st.find):
    h.rationaliser.rationalise([KValue(ask.kline, 0x00)])
    res = run_hops(st, ask.kline, sig)
print(f"\nchain ending={res.ending}")
for r in res.results:
    print(f"  {r.ending:9s} j1={r.j1:.3f} nodes={[name(n) or hex(int(n)) for n in r.trace[-1]]}")
