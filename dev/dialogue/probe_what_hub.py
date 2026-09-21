"""Per-node consumption trace of expand for the WDMH vs MHALL pair."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from pathlib import Path
from dev.dialogue.harness import load_rationaliser
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import Cogitator
from kalvin.kline import is_terminal

tok = BPETokenizer()
h = load_rationaliser(Path("data/dialogue/mhall.json"), tok)
h.run(Path("data/scripts/wdmh-underfit.ks").read_text())
state, sig = h.state, h.state.signifier

def by_label(label):
    return [k for k in state.where(lambda k: getattr(k.signature, "label", None) == label)]

w = [k for k in by_label("WDMH") if not is_terminal(k)][0]
m = [k for k in by_label("MHALL") if not is_terminal(k) and k.nodes != [k.signature]][0]
q_set, c_set = set(w.nodes), set(m.nodes)
u, o, f = list(q_set - c_set), list(c_set - q_set), list(q_set & c_set)

cog = Cogitator(state)
for m1, m2 in [(u, o), (o, [])]:
    while m1:
        n = m1.pop(0)
        consumed = None
        for kl, hops in cog.denotate(n):
            for m_nodes in [m2, f]:
                if kl.signature in m_nodes:
                    consumed = (f"hops={hops} arm1 sig {kl.signature.label!r} from {[x.label for x in m_nodes]}")
                    m_nodes.remove(kl.signature)
                    break
                if all(x in m_nodes for x in kl.nodes):
                    consumed = (f"hops={hops} arm2 nodes {[x.label for x in kl.nodes]} of {kl.signature.label!r} from {[x.label for x in m_nodes]}")
                    for x in kl.nodes:
                        m_nodes.remove(x)
                    break
            if consumed:
                break
        print(f"node {n.label!r}: {consumed if consumed else 'UNMATCHED -> remainder'}")
