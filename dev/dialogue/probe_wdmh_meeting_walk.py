"""MODE D — the meeting walk: both parties descend, the shared value is the bridge.

Per the trainer's algorithm:

    _walk:
        path_a = walk_a               # descend from A's gap slots (sig->witness)
        walk_b(path_a)                # descend from B's excess slots;
                                     # cur in path_a  =>  bridge

A descent edge: a held kline HEADED at the current value expands into its
witness members (you only walk through klines your value heads). The
meeting value must be delivered by DISTINCT klines on the two sides (the
ALL > O < W rule: two klines meet at O). Bridge: replace the A-side
departure value with the B-side departure value — written as the composed
correspondence  slot_a:[slot_b]  for the main line to consume.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.derivation import Derivation, KLine

MAX_EDGES = 8


def nm(v):
    return getattr(v, "label", "") or hex(int(v))

def _descend(self, starts):
    """BFS sig->witness: {value: (depth, delivering kline key, start)}."""
    reached: dict[int, tuple[int, tuple, int]] = {}
    frontier = [(int(v), 0, None, int(v)) for v in starts]
    while frontier:
        nxt = []
        for v, d, _k, start in frontier:
            if d >= MAX_EDGES:
                continue
            for k in self.memory:
                if not self.usable(k):
                    continue
                if int(k.signature) != v:
                    continue
                if int(k.signature) in [int(n) for n in k.nodes]:
                    continue  # self-containing
                key = (int(k.signature), tuple(int(n) for n in k.nodes))
                for n in k.nodes:
                    if int(n) not in reached:
                        reached[int(n)] = (d + 1, key, start)
                        nxt.append((int(n), d + 1, key, start))
        frontier = nxt
    return reached

def _walk_meeting(self) -> bool:
    from kalvin.significance import WORD_BITS
    a_starts = [n for n in self.nodes if self.signifier.signifies(n, self.gap())]
    # B departs the goal-witness value(s) CONTAINING the excess — the
    # overfit at the goal's witness resolution (Def 15), not any overlap.
    excess = self.excess() & WORD_BITS
    b_starts = [
        n for n in self.goal.nodes
        if excess & int(n) & WORD_BITS == excess
    ]
    if not a_starts or not b_starts:
        return False
    path_a = self._descend(a_starts)
    if not path_a:
        return False
    # walk_b: BFS from B, checking membership in path_a as values are reached
    frontier = [(int(v), 0, int(v)) for v in b_starts]
    seen = {int(v) for v in b_starts}
    while frontier:
        nxt = []
        for v, d, start in frontier:
            if d >= MAX_EDGES:
                continue
            for k in self.memory:
                if not self.usable(k):
                    continue
                if int(k.signature) != v:
                    continue
                if int(k.signature) in [int(n) for n in k.nodes]:
                    continue
                for n in k.nodes:
                    ni = int(n)
                    if ni in path_a:
                        ad, akey, astart = path_a[ni]
                        bkey = (int(k.signature), tuple(int(x) for x in k.nodes))
                        if akey != bkey and astart != start:
                            # bridge: replace slot_a with the B-side departure
                            # value, canonically expanded under a held
                            # well-founded canon before the construction
                            witness = [start]
                            from kalvin.kline import is_canon
                            kanon = next(
                                (kk for kk in self.memory
                                 if int(kk.signature) == start
                                 and is_canon(kk, self.signifier)
                                 and kk.signature not in kk.nodes),
                                None,
                            )
                            if kanon is not None:
                                witness = list(kanon.nodes)
                            comp = KLine(astart, witness, acq_depth=ad + d + 1)
                            if self._ground_composed(comp):
                                self.composed.append(comp)
                                return True
                        continue
                    if ni not in seen:
                        seen.add(ni)
                        nxt.append((ni, d + 1, start))
        frontier = nxt
    return False

Derivation._descend = _descend
Derivation._walk_meeting = _walk_meeting
Derivation._walk_a = lambda self: self._walk_meeting()
Derivation._walk_b = lambda self: False  # the meeting subsumes the b-walk

orig_run = Derivation.run
writes_log: list[str] = []

import os
if os.environ.get("TRACE_WDMH"):
    from kalvin import hop as hop_mod
    _orig_cg = hop_mod.candidate_goals
    def cg(state, queued, signifier):
        goals = _orig_cg(state, queued, signifier)
        if any(int(n) == 0x80000002f45 for n in queued.nodes):
            print(f"\n## goals for nodes={[nm(n) for n in queued.nodes]}:")
            for g in goals[:10]:
                print(f"   {nm(g.signature)}:[{', '.join(nm(n) for n in g.nodes)}]")
        return goals
    hop_mod.candidate_goals = cg

def run(self):
    r = orig_run(self)
    for k in r.composed:
        writes_log.append(
            f"{nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}] acq={k.acq_depth}"
        )
    if os.environ.get("TRACE_WDMH") and any(
        int(n) == 0x80000002f45 for n in self.queued.nodes
    ):
        print(f"   deriv goal={nm(self.goal.signature)}:[{','.join(nm(n) for n in self.goal.nodes)}]"
              f" -> {r.ending} j1={r.j1:.2f} trace={[','.join(nm(n) for n in s) for s in r.trace]}"
              f" composed={[f'{nm(k.signature)}:[{nm(k.nodes[0])}]' for k in r.composed]}")
    return r
Derivation.run = run

sys.argv = ["harness", "data/scripts/wdmh.ks", "-p", "data/dialogue/mhall.json"]
from dev.dialogue.harness import main
main()

print(f"\n\n=== MODE D: composed writes ({len(writes_log)}) ===")
import collections
for w, c in collections.Counter(writes_log).most_common():
    print(f"  {c}x  {w}")
