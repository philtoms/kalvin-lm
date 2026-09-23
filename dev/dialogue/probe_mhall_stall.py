"""Why does the mhall run propose nothing? Post-run, per work-list item:
Def 22 candidate goals, Def 23 trawl scope, and full hop results."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.hop import Hop, candidate_goals, trawl
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

print("== post-run work list ==")
for k in st.work_list:
    print(f"  {render(k)}")

memory = st.where(lambda k: not is_terminal(k), True)
print(f"\nnon-terminal memory klines: {len(memory)}")
for k in memory:
    print(f"  {render(k)}")

print("\n== per-item hop diagnostics ==")
for k in list(st.work_list):
    print(f"\n-- {render(k)}")
    goals = candidate_goals(st, k, sig)
    if not goals:
        print("   no candidate goals")
        continue
    for g in goals[:8]:
        scope = trawl(st, k.nodes, g.nodes, sig)
        print(f"   goal {render(g)}  scope={len(scope)}")
    hop = Hop(st, k, sig).run()
    print(f"   hop ending={hop.ending}  writes={len(hop.writes)}")
    for r in hop.results:
        trace = " > ".join(name(n) for n in r.trace[-1][:6])
        print(f"     {r.ending:10s} j1={r.j1:.3f} trace_tail=[{trace}]")
