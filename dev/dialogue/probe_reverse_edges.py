"""Prototype a reverse-edge denotate and test WDMH=>MHALL expansion."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from pathlib import Path
from dev.dialogue.harness import load_engine
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import Cogitator
from kalvin.kline import KLine, is_terminal, is_identity
from kalvin.signifier import _TYPE_MASK

tok = BPETokenizer()
h = load_engine(Path("data/dialogue/mhall.json"), tok)
h.run(Path("data/scripts/wdmh-underfit.ks").read_text())
state, sig = h.state, h.state.signifier


def denotateR(self, start, depth=6):
    """BFS from a node over forward sig edges AND reverse edges
    (bit-containment on signatures + node containment)."""
    frontier = [KLine(start, [])]
    visited = set()
    hops = 0
    while frontier and hops < depth:
        hops += 1
        nxt = []
        for cur in frontier:
            edges = list(state.find_sig(cur.signature))
            # reverse: klines whose signature carries a word bit of cur,
            # or whose nodes contain cur's signature
            for k in state.where(lambda k: k.signature != cur.signature
                                 and not is_terminal(k) and not is_identity(k)
                                 and ((cur.signature & k.signature & _TYPE_MASK) != 0
                                      or cur.signature in k.nodes)):
                edges.append(k)
            for kline in edges:
                if kline is None or is_terminal(kline) or is_identity(kline) or kline in visited:
                    continue
                visited.add(kline)
                reached = KLine(sig.signature_of(kline.nodes), kline.nodes)
                yield reached, hops
                nxt.append(reached)
        frontier = nxt


Cogitator.denotate = denotateR
cog = Cogitator(state)

def by_label(label):
    return [k for k in state.where(lambda k: getattr(k.signature, "label", None) == label)]

ws = [k for k in by_label("WDMH") if not is_terminal(k)]
for w in ws[:1]:
    print("WDMH:", [n.label for n in w.nodes])
    print("  denotateR from WDMH sig:")
    for kl, hp in denotateR(cog, w.signature):
        print(f"    hops={hp} sig={kl.signature.label} nodes={[n.label for n in kl.nodes]}")
    for m in by_label("MHALL"):
        q_set, c_set = set(w.nodes), set(m.nodes)
        u, o, f = list(q_set-c_set), list(c_set-q_set), list(q_set & c_set)
        print(f"  vs MHALL{[n.label for n in m.nodes]}:")
        print(f"    u={[x.label for x in u]} o={[x.label for x in o]} f={[x.label for x in f]}")
        prop, dist = cog.expand(u, o, f)
        print(f"    proposal={[p.label for p in prop]} dist={dist}")
