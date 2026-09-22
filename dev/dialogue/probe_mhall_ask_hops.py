"""Why does the attending MHALL ask not derive? Per-hop: goals, scope,
results, bridges."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.hop import candidate_goals, trawl, Hop, run_hops
from kalvin.kline import is_terminal
from dev.dialogue.harness import make_rationaliser

source = Path("data/scripts/mhall.ks").read_text()
tok = BPETokenizer()
h = make_rationaliser(tok)
st = h.state
sig = h.signifier

h.run(source)

def name(v):
    return getattr(v, "label", "") or hex(int(v))

def render(k):
    nodes = ", ".join(name(n) for n in k.nodes)
    return f"{name(k.signature)}:[{nodes}]"

ask = next(k for k in st.work_list if getattr(k.signature, "label", "") == "MHALL")
print(f"ask: {render(ask)}  dbg.goal={ask.dbg.goal if ask.dbg else None}")

print("\n== candidate goals ==")
goals = candidate_goals(st, ask, sig)
for g in goals:
    print(f"  {render(g)}  label={g.dbg.label if g.dbg else None}")

print("\n== run_hops chain ==")
res = run_hops(st, ask, sig)
print(f"ending={res.ending} writes={len(res.writes)}")
for r in res.results:
    tail = ", ".join(name(n) for n in r.trace[-1])
    print(f"  {r.ending:10s} j1={r.j1:.3f} nodes=[{tail}] composed={len(r.composed)}")
for w in res.writes:
    print(f"  write: {render(w)} acq={w.acq_depth}")

print("\n== first hop detail ==")
hop = Hop(st, ask, sig).run()
for g in hop.goals:
    print(f"  goal {render(g)}")
    scope = trawl(st, ask.nodes, g.nodes, sig)
    print(f"    scope={len(scope)}: {[render(k) for k in scope]}")
