"""Trace the walk that writes Det:[Object] — slot, end_mask, arrival, path."""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.derivation import Derivation

def nm(v):
    return getattr(v, "label", "") or hex(int(v))

orig_sw = Derivation.slot_walk
last_walks: list = []

def sw(self, slot, end_mask=None):
    r = orig_sw(self, slot, end_mask)
    last_walks.append((slot, end_mask, r, list(self.memory)))
    return r
Derivation.slot_walk = sw

orig_run = Derivation.run

def run(self):
    del last_walks[:]
    r = orig_run(self)
    for k in r.composed:
        head = nm(k.signature)
        if head in ("Det", "Mod", "little", "a") or True:
            print(f"\n## composed {nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}] "
                  f"acq={k.acq_depth}")
            print(f"   queued={nm(self.queued.signature)}:[{', '.join(nm(n) for n in self.queued.nodes)}] "
                  f"goal={nm(self.goal.signature)}:[{', '.join(nm(n) for n in self.goal.nodes)}]")
            print(f"   A content={nm(self.content())}  goal content={nm(self.goal_content())}  "
                  f"gap={nm(self.gap())}  excess={nm(self.excess())}")
            for slot, end_mask, res, mem in last_walks:
                print(f"   walk slot={nm(slot)} end_mask={nm(end_mask) if end_mask else None}"
                      f" -> {'None' if res is None else [nm(n) for n in res[0]]}")
                if res is not None:
                    for d, k in res[2]:
                        print(f"      {d:7s} {nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}]")
    return r
Derivation.run = run

sys.argv = ["harness", "data/scripts/wdmh.ks", "-p", "data/dialogue/mhall.json"]
from dev.dialogue.harness import main
main()
