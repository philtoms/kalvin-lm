"""Test the overfit-walk licence theory.

MODE=A — the overfit walk never runs (b_walks=False): each party walks only
         its own slots, writing slot:[arrival].
MODE=B — the overfit walk runs but writes under the underfit licence:
         KLine(slot, end) — departed slot headed, full arrival witnessed —
         instead of anchor:[filtered witness + slot].
"""
import sys, os
sys.path.insert(0, "src")
MODE = os.environ.get("MODE", "A")

from kalvin.derivation import Derivation, KLine

def nm(v):
    return getattr(v, "label", "") or hex(int(v))

orig_run = Derivation.run
writes_log: list[str] = []

def walk_b_same_licence(self) -> bool:
    for slot in self.goal.nodes:
        if not self.signifier.signifies(slot, self.excess()):
            continue
        walk = self.slot_walk(slot, end_mask=self.content())
        if walk is None:
            continue
        end, edges, path = walk
        end, edges, _ = self.refine(end, edges, path)
        if not any(n in self.nodes for n in end):
            continue
        composed = KLine(slot, end, acq_depth=edges)
        if self._ground_composed(composed):
            self.composed.append(composed)
            return True
    return False

if MODE == "B":
    Derivation._walk_b = walk_b_same_licence

def run(self):
    if MODE == "A":
        self.b_walks = False
    r = orig_run(self)
    for k in r.composed:
        writes_log.append(
            f"{nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}] acq={k.acq_depth}"
        )
    return r
Derivation.run = run

sys.argv = ["harness", "data/scripts/wdmh.ks", "-p", "data/dialogue/mhall.json"]
from dialogue.harness import main
main()

print(f"\n\n=== MODE {MODE}: composed writes ({len(writes_log)}) ===")
import collections
for w, c in collections.Counter(writes_log).most_common():
    print(f"  {c}x  {w}")
