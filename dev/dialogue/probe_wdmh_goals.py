"""Instrument candidate_goals + Derivation to expose the little:[Mod] hop."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin import hop as hop_mod
from kalvin.derivation import Derivation

def nm(v):
    return getattr(v, "label", "") or hex(int(v))

def render(k):
    return f"{nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}]"

import os
TARGET = os.environ.get("TARGET")

orig_cg = hop_mod.candidate_goals
def cg(state, queued, signifier):
    goals = orig_cg(state, queued, signifier)
    q = render(queued)
    if TARGET and TARGET in q:
        print(f"\n## candidate_goals for {q}")
        for g in goals:
            gc = int(signifier.signature_of(g.nodes))
            print(f"   goal {render(g)}  content_atoms={hex(gc & ((1<<40)-1))} "
                  f"acq={getattr(g, 'acq_depth', None)}")
    return goals
hop_mod.candidate_goals = cg

orig_run = Derivation.run
def run(self):
    q = render(self.queued)
    if TARGET and TARGET in q:
        print(f"\n>> derivation queued={q} goal={render(self.goal)} "
              f"acq={getattr(self.goal, 'acq_depth', None)}")
        print(f"   scope ({len(self.memory)}):")
        for k in self.memory:
            print(f"     {render(k)}")
    r = orig_run(self)
    if TARGET and TARGET in q:
        print(f"   ending={r.ending} j0={r.j0:.3f} j1={r.j1:.3f} "
              f"composed={[render(k) for k in r.composed]}")
    return r
Derivation.run = run

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin import engine as eng_mod
eng_mod.run_hops.__globals__["Hop"] = hop_mod.Hop

sys.argv = ["harness", "data/scripts/wdmh.ks", "-p", "data/dialogue/mhall.json"]
from dev.dialogue.harness import main
main()
