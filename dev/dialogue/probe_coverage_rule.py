"""Prototype: underfit node matched only if covered by the reached kline;
unmatched query node => empty proposal."""
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

def expand_cov(self, underfit, overfit, fit):
    proposal, remainder = [], []
    distance = 0
    for m1, m2 in [(underfit, overfit), (overfit, remainder)]:
        while len(m1) > 0:
            n = m1.pop(0)
            reserve = True if m1 is underfit else False
            for kl, hops in self.connotate(n):
                covers = n in kl.nodes or self.signifier.bit_in(n, kl.signature)
                for m_nodes in [m2, fit]:
                    if kl.signature in m_nodes:
                        if covers:
                            distance += hops
                            proposal.append(kl.signature)
                            m_nodes.remove(kl.signature)
                            reserve = False
                        break
                    if all(x in m_nodes for x in kl.nodes):
                        if covers:
                            proposal.append(kl.signature)
                            for x in kl.nodes:
                                distance += hops
                                m_nodes.remove(x)
                            reserve = False
                        break
                if not reserve:
                    break
            if reserve:
                remainder.append(n)
    if remainder:
        return [], 0
    return proposal, distance

Cogitator.expand = expand_cov
cog = Cogitator(state)

def by_label(label):
    return [k for k in state.where(lambda k: getattr(k.signature, "label", None) == label)]

w_full = [k for k in by_label("WDMH") if not is_terminal(k)]
queries = {"WDMH:[wdmh]": w_full[0], "WDMH:[DH]": w_full[1]}
for qn, w in queries.items():
    print(f"== query {qn}: {[n.label for n in w.nodes]}")
    for cand_label in ("MHALL", "SVO", "ALL", "DH", "Subject", "Verb", "Object"):
        for m in by_label(cand_label):
            if is_terminal(m):
                continue
            q_set, c_set = set(w.nodes), set(m.nodes)
            u, o, f = list(q_set - c_set), list(c_set - q_set), list(q_set & c_set)
            prop, dist = cog.expand(u, o, f)
            print(f"  vs {cand_label}:{[n.label for n in m.nodes]} -> {[p.label for p in prop]} d={dist}")
