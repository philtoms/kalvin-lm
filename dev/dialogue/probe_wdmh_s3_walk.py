"""MODE C — the S3-directed walk licence: node -> sig only.

slot_walk fires only reverse replaces (witness occurs -> contract to
signature). Forward expansion is removed from the walk entirely. Both
departures remain (A-side slot, B-side slot), both write the honest
slot:[arrival] shape. Refine (post-arrival canon expansion toward the
goal's witness resolution) is left as-is — a separate witnessed licence.
"""
import sys, os
sys.path.insert(0, "src")

from collections import Counter, deque
from kalvin.derivation import Derivation, KLine

def nm(v):
    return getattr(v, "label", "") or hex(int(v))

def slot_walk_s3(self, slot, end_mask=None):
    if end_mask is None:
        end_mask = self.excess()
    queue: deque = deque([([slot], frozenset(), 0, [])])
    seen = {(int(slot),)}
    expanded = 0
    while queue:
        if expanded >= self.max_walk_states:
            return None
        expanded += 1
        nodes, used, edges, path = queue.popleft()
        if edges and self.signifier.signifies(
            int(self.signifier.signature_of(nodes)), end_mask
        ):
            return nodes, edges, path
        if edges >= self.max_walk_edges:
            continue
        for k in self.memory:
            if not self.usable(k):
                continue
            key = (int(k.signature), tuple(int(n) for n in k.nodes))
            if key in used:
                continue
            if self.occurs_rev(k, nodes):  # node -> sig only
                new = self.replace_rev(k, nodes)
                t = tuple(sorted(int(x) for x in new))
                if t in seen:
                    continue
                seen.add(t)
                queue.append((new, used | {key}, edges + 1, path + [("reverse", k)]))
    return None

def walk_b_slot_headed(self) -> bool:
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

Derivation.slot_walk = slot_walk_s3
Derivation._walk_b = walk_b_slot_headed

orig_run = Derivation.run
writes_log: list[str] = []

def run(self):
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

print(f"\n\n=== MODE C: composed writes ({len(writes_log)}) ===")
for w, c in Counter(writes_log).most_common():
    print(f"  {c}x  {w}")
