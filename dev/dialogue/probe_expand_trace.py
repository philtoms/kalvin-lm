"""Log per-node consumption inside expand for the WDMH query pairs."""
import sys
sys.path.insert(0, "src")
from pathlib import Path
from dialogue.harness import load_engine
from kalvin.bpe_tokenizer import BPETokenizer
from dialogue.cogitator import Cogitator
from kalvin.kline import is_terminal

tok = BPETokenizer()
h = load_engine(Path("data/dialogue/mhall.json"), tok)
h.run(Path("data/scripts/wdmh-underfit.ks").read_text())
state, sig = h.state, h.state.signifier

orig = Cogitator.expand

def traced_expand(self, underfit, overfit, fit):
    print(f"  expand: u={[n.label for n in underfit]} o={[n.label for n in overfit]} f={[n.label for n in fit]}")
    prop, dist = orig(self, underfit, overfit, fit)
    print(f"  -> proposal={[p.label for p in prop]} dist={dist} "
          f"(u drained={not underfit}, o drained={not overfit})")
    return prop, dist

Cogitator.expand = traced_expand
cog = Cogitator(state)

def by_label(label):
    return [k for k in state.where(lambda k: getattr(k.signature, "label", None) == label)]

w = [k for k in by_label("WDMH") if not is_terminal(k)][0]
for cand_label in ("Subject", "MHALL", "SVO", "ALL", "DH"):
    for m in by_label(cand_label):
        if is_terminal(m):
            continue
        q_set, c_set = set(w.nodes), set(m.nodes)
        u, o, f = list(q_set - c_set), list(c_set - q_set), list(q_set & c_set)
        print(f"pair WDMH vs {cand_label}:{[n.label for n in m.nodes]}")
        traced_expand(cog, u, o, f)
