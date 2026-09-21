"""Variant C: query node satisfied iff some reached kline covers it
(bit-containment / node membership). Consumption from the opposite list
only builds the proposal. Any unsatisfied node => empty proposal."""
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

def expand_c(self, underfit, overfit, fit):
    proposal, remainder = [], []
    distance = 0
    for m1, m2 in [(underfit, overfit), (overfit, remainder)]:
        while len(m1) > 0:
            n = m1.pop(0)
            reserve = True if m1 is underfit else False
            for kl, hops in self.denotate(n):
                covers = n in kl.nodes or self.signifier.bit_in(n, kl.signature)
                if covers:
                    reserve = False
                    for m_nodes in [m2, fit]:
                        if kl.signature in m_nodes:
                            distance += hops
                            proposal.append(kl.signature)
                            m_nodes.remove(kl.signature)
                            break
                        if all(x in m_nodes for x in kl.nodes):
                            proposal.append(kl.signature)
                            for x in kl.nodes:
                                distance += hops
                                m_nodes.remove(x)
                            break
                    break
            if reserve:
                remainder.append(n)
    if remainder:
        return [], 0
    return proposal, distance

Cogitator.expand = expand_c
cog = Cogitator(state)

def by_label(label):
    return [k for k in state.where(lambda k: getattr(k.signature, "label", None) == label)]

w_full = [k for k in by_label("WDMH") if not is_terminal(k)]
for qn, w in {"WDMH:[wdmh]": w_full[0], "WDMH:[DH]": w_full[1]}.items():
    print(f"== query {qn}: {[n.label for n in w.nodes]}")
    for cand_label in ("MHALL", "SVO", "ALL", "DH"):
        for m in by_label(cand_label):
            if is_terminal(m):
                continue
            q_set, c_set = set(w.nodes), set(m.nodes)
            u, o, f = list(q_set - c_set), list(c_set - q_set), list(q_set & c_set)
            prop, dist = cog.expand(u, o, f)
            print(f"  vs {cand_label}:{[n.label for n in m.nodes]} -> {[p.label for p in prop]} d={dist}")

# variant D: denotate yields the original kline (signature intact)
from kalvin.cogitator import Cogitator as _C
def denotate_orig(self, s, depth=100):
    state = self._state
    signifier = state.signifier
    frontier = [KLine(s, [])]
    visited = set()
    hop_count = 0
    while frontier and hop_count < depth:
        hop_count += 1
        nxt = []
        for cur in frontier:
            edges = list(state.find_sig(cur.signature))
            edges.extend(
                k for k in state.where(
                    lambda k: k.signature != cur.signature
                    and not is_terminal(k) and not is_identity(k)
                    and (signifier.signifies(cur.signature, k.signature)
                         or cur.signature in k.nodes))
            )
            for kline in edges:
                if kline is None or is_terminal(kline) or is_identity(kline) or kline in visited:
                    continue
                visited.add(kline)
                yield kline, hop_count
                nxt.append(kline)
        frontier = nxt
_C.denotate = denotate_orig
print("== variant D (original signatures preserved) ==")
cog2 = Cogitator(state)
w_full = [k for k in by_label("WDMH") if not is_terminal(k)]
for qn, w in {"WDMH:[wdmh]": w_full[0], "WDMH:[DH]": w_full[1]}.items():
    print(f"== query {qn}: {[n.label for n in w.nodes]}")
    for cand_label in ("MHALL", "SVO", "ALL", "DH"):
        for m in by_label(cand_label):
            if is_terminal(m):
                continue
            q_set, c_set = set(w.nodes), set(m.nodes)
            u, o, f = list(q_set - c_set), list(c_set - q_set), list(q_set & c_set)
            prop, dist = cog2.expand(u, o, f)
            print(f"  vs {cand_label}:{[n.label for n in m.nodes]} -> {[p.label for p in prop]} d={dist}")

# instrument: which node lands in remainder, per pair, variant D
def expand_dbg(self, underfit, overfit, fit):
    proposal, remainder = [], []
    distance = 0
    for m1, m2 in [(underfit, overfit), (overfit, remainder)]:
        while len(m1) > 0:
            n = m1.pop(0)
            reserve = True if m1 is underfit else False
            for kl, hops in self.denotate(n):
                covers = n in kl.nodes or self.signifier.bit_in(n, kl.signature)
                if covers:
                    reserve = False
                    for m_nodes in [m2, fit]:
                        if kl.signature in m_nodes:
                            distance += hops
                            proposal.append(kl.signature)
                            m_nodes.remove(kl.signature)
                            break
                        if all(x in m_nodes for x in kl.nodes):
                            proposal.append(kl.signature)
                            for x in kl.nodes:
                                distance += hops
                                m_nodes.remove(x)
                            break
                    break
            if reserve:
                remainder.append(n)
                print(f"    UNSATISFIED: {n.label}")
    if remainder:
        return [], 0
    return proposal, distance

_C.expand = expand_dbg
cog3 = Cogitator(state)
for cand_label in ("MHALL", "SVO"):
    for m in by_label(cand_label):
        if is_terminal(m) or m.nodes == [m.signature]:
            continue
        q_set, c_set = set(w_full[0].nodes), set(m.nodes)
        u, o, f = list(q_set - c_set), list(c_set - q_set), list(q_set & c_set)
        print(f"pair vs {cand_label}:{[n.label for n in m.nodes]}")
        prop, dist = cog3.expand(u, o, f)
        print(f"  -> {[p.label for p in prop]} d={dist}")
