"""Dump the goal list for the WDMH ask and the Def 23 scope per goal."""
import sys
sys.path.insert(0, "src")

from pathlib import Path
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import ASK_SIG, is_terminal, is_ask
from kalvin.hop import candidate_goals, trawl
from dialogue.harness import load_engine

h = load_engine(Path("data/dialogue/wdmh-underfit.json"), BPETokenizer())
st = h.state
sig = h.signifier

def name(v):
    return getattr(v, "label", "") or hex(int(v))

def render(k):
    m = "?" if is_ask(k.signature) else " "
    return f"{m}{name(k.signature)}:[{', '.join(name(n) for n in k.nodes)}]"

reservoir = st.where(lambda k: not is_terminal(k))
ask = next(k for k in st.work_list if is_ask(k.signature))
print(f"queued ask: {render(ask)}")
print(f"reservoir: {len(reservoir)} klines")
for k in reservoir:
    print(f"  {render(k)}")

goals = candidate_goals(reservoir, ask, sig)
print("\n== goal list (Def 22) ==")
for g in goals:
    print(f"  {render(g)}")

print("\n== scope per goal (Def 23 trawl, depth 4) ==")
for g in goals[:3]:
    scope = trawl(reservoir, ask.nodes, g.nodes)
    print(f"goal {render(g)} -> {len(scope)} scoped:")
    for k in scope:
        print(f"    {render(k)}")
    out = [k for k in reservoir if id(k) not in {id(s) for s in scope}]
    print(f"    (left out: {[render(k) for k in out]})")
