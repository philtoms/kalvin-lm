"""Which kline consumes 'what' in the WDMH vs MHALL expansion."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from pathlib import Path
from dev.dialogue.harness import load_rationaliser
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import Cogitator
from kalvin.kline import KLine, is_terminal, is_identity

tok = BPETokenizer()
h = load_rationaliser(Path("data/dialogue/mhall.json"), tok)
h.run(Path("data/scripts/wdmh-underfit.ks").read_text())
state, sig = h.state, h.state.signifier

orig = Cogitator.expand
def traced(self, underfit, overfit, fit):
    print(f"expand u={[n.label for n in underfit]} o={[n.label for n in overfit]} f={[n.label for n in fit]}")
    result = orig(self, underfit, overfit, fit)
    print(f"  -> {[p.label for p in result[0]]} d={result[1]}")
    return result
Cogitator.expand = traced
cog = Cogitator(state)

def by_label(label):
    return [k for k in state.where(lambda k: getattr(k.signature, "label", None) == label)]

w = [k for k in by_label("WDMH") if not is_terminal(k)]
m = [k for k in by_label("MHALL") if not is_terminal(k) and k.nodes and k.nodes != [k.signature]][0]
# pair: full WDMH vs MHALL word kline
q_set, c_set = set(w[0].nodes), set(m.nodes)
u, o, f = list(q_set - c_set), list(c_set - q_set), list(q_set & c_set)
print("pair WDMH:", [n.label for n in w[0].nodes], "vs MHALL:", [n.label for n in m.nodes])

what = [n for n in u if n.label == "what"][0]
print("== denotate('what') yields ==")
for kl, hops in cog.denotate(what):
    print(f"  hops={hops} sig={kl.signature.label!r} nodes={[n.label for n in kl.nodes]}")
    for name, m_nodes in (("overfit", o[:]), ("fit", f[:])):
        if kl.signature in m_nodes:
            print(f"    !! arm1: {kl.signature.label!r} in {name}")
        if all(x in m_nodes for x in kl.nodes):
            print(f"    !! arm2: all nodes {[n.label for n in kl.nodes]} in {name}")
