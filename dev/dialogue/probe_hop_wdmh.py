"""Bypass the routing seam: queue the WDMH MTS canon directly as A0 and hop.

The algebra's worked example: A0 = wdmh:[w,d,m,h] (the question at its own
resolution), goal B = mhall:[m,h,a,l,l] held as canon. Memory = the state a
fresh engine holds after running mhall.ks (frame) + the wdmh scaffolding.
"""
import sys
sys.path.insert(0, "src")

from pathlib import Path
from kalvin.bpe_tokenizer import BPETokenizer
from ks.compiler import compile_source
from dialogue.harness import make_engine
from kalvin.hop import Hop, run_hops
from kalvin.kline import KLine, is_terminal
from kalvin.kvalue import KValue

tok = BPETokenizer()
h = make_engine(tok)
sig = h.signifier
bits: dict[str, int] = {}

for path in ("data/scripts/mhall.ks", "data/scripts/wdmh-underfit.ks"):
    entries = compile_source(open(path).read(), tokenizer=tok, signifier=sig,
                             dev=True, word_bits=bits)
    h.engine.rationalise(entries)

st = h.state
print("frame entries:", sum(len(b) for b in st.frame.values()),
      " work_list:", len(st.work_list))

wdmh_canon = next(
    (k for b in st.frame.values() for k in b
     if getattr(k.signature, "label", "") == "WDMH" and k.nodes),
    None,
)
print("held WDMH canon nodes:", [str(n) for n in wdmh_canon.nodes])

memory = st.where(lambda k: not is_terminal(k))
a0 = KLine(wdmh_canon.signature, list(wdmh_canon.nodes))
# Exclude the trivial self-goal: klines isomorphic to A0 (same signature+nodes)
# complete at entry and say nothing about derivation toward MHALL.
memory = [
    k for k in memory
    if not (k.signature == a0.signature and k.nodes == a0.nodes)
]

res = run_hops(memory, a0, sig)
print(f"\nhop ending: {res.ending}")
print(f"writes ({len(res.writes)}):")
for w in res.writes:
    print(f"  {getattr(w.signature, 'label', hex(w.signature))}:[{[str(n) for n in w.nodes]}]")
print(f"results: {len(res.results)}")
for r in res.results:
    end_nodes = [str(n) for n in r.trace[-1]] if r.trace else []
    print(f"  ending={r.ending} gamma={r.gamma:.3f} final_nodes={end_nodes[:8]}")
    if r.ending == "done":
        prop = KLine(a0.signature, r.trace[-1])
        val = sig.signature_of(prop.nodes)
        print(f"  DONE proposal: WDMH:{[str(n) for n in prop.nodes]}")
        print(f"  content signature: {getattr(val, 'label', hex(val))}")
