"""Verify Def 17 read from ν_B — the overfit slot walk — against the pure
algebra. No engine, compiler, or tokenizer: values are ints (one bit per
atom), klines are records. Scenario:

    A0 = mh:[m,h]          the fragment   ("Mary had")
    B  = mhall:[m,h,all]   the goal       ("Mary had a little lamb")

    C(A0,B) = mh:[m,h,all] — Overfit (S2): u = ∅, o = {a,l}
    No node of ν_A carries an underfit atom, so no A-side slot exists;
    no held correspondence adopts the overfit at any node of ν_A, so
    targeting alone is stuck and the misfit asks — falsely, because
    memory connects the parties (o:[m]).

    Def 17, walk from ν_B — the overfit slot is the goal's node `all`:

        all:[all] → all:[o]  (all:[o] Connotation, forward)
                  → all:[m]  (o:[m]  Connotation, forward)  — anchor m

    Arrival in σ(ν_A); the anchor is already a node of ν_A (no
    refinement edge) and the departed end already sits at the goal's
    witness resolution. The composed correspondence is written head-ward:

        m:[m,all]  (Overfit — adopt fwd)  at acquisition depth 2

    The main line consumes it:

        m ⇉ [m,all] → [m,all,h]  σ = mhall  Δ 2→0  done

    Measurement: J 0.5→1.0, Ĥ = 2/3 (all@2, m@0, h@0), γ = 2^-(2/3) ≈ 0.63.

The control run disables ν_B walks: stuck at entry — the ask the
one-party def would give.

Run from the repo root:  python3 dev/algebra/overfit-anchor-walk.py
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations

δ = 0.5
MAX_STEPS = 32
MAX_WALK_EDGES = 8

m, h, a, l, w, d, o = (1 << i for i in range(7))
mhall, allv, mh = m | h | a | l, a | l, m | h
NAMES = {
    m: "m", h: "h", a: "a", l: "l", w: "w", d: "d", o: "o",
    mhall: "mhall", allv: "all", mh: "mh",
}


@dataclass
class K:
    signature: int
    nodes: list
    acq_depth: int = 0
    tag: str = ""

    def __str__(self) -> str:
        nodes = ", ".join(name(n) for n in self.nodes)
        return f"{name(self.signature)}:[{nodes}]"


def name(v: int) -> str:
    if v in NAMES:
        return NAMES[v]
    bits = [NAMES.get(b, hex(b)) for b in atom_bits(v)]
    return "|".join(bits) if bits else "0"


def atom_bits(v: int) -> list[int]:
    return [1 << i for i in range(v.bit_length()) if v & (1 << i)]


def content(nodes: list) -> int:
    out = 0
    for n in nodes:
        out |= n
    return out


def popc(v: int) -> int:
    return bin(v).count("1")


def fit(signature: int, nodes: list) -> str:
    if not nodes or signature == 0:
        return "Unknown"
    if nodes == [signature]:
        return "Identity"
    sigma = content(nodes)
    if signature == sigma:
        return "Canon"
    covered = any(n & signature for n in nodes)
    if not covered:
        return "Connotation" if len(nodes) == 1 else "No-fit"
    underfit, overfit = signature & ~sigma, sigma & ~signature
    if underfit and not overfit:
        return "Denotation" if len(nodes) == 1 else "Underfit"
    if overfit and not underfit:
        return "Overfit"
    return "Under+over"


def band(shape: str) -> str:
    return {
        "Identity": "S1", "Canon": "S1",
        "Denotation": "S2", "Underfit": "S2", "Overfit": "S2", "Under+over": "S2",
        "Connotation": "S3", "No-fit": "S3", "Unknown": "S4",
    }[shape]


MEMORY = [
    K(mhall, [m, h, allv]),
    K(allv, [o]),
    K(allv, [a, l, l]),
    K(o, [m]),
    K(m, [m]),
]
A0 = K(mh, [m, h])
B = MEMORY[0]


def occurs_rev(k: K, nodes: list) -> bool:
    have = Counter(nodes)
    return all(have.get(n, 0) >= c for n, c in Counter(k.nodes).items())


def replace_fwd(k: K, nodes: list) -> list:
    i = nodes.index(k.signature)
    return nodes[:i] + list(k.nodes) + nodes[i + 1 :]


def replace_rev(k: K, nodes: list) -> list:
    want = Counter(k.nodes)
    out, placed = [], False
    for n in nodes:
        if want.get(n, 0) > 0:
            want[n] -= 1
            if not placed:
                out.append(k.signature)
                placed = True
        else:
            out.append(n)
    return out


def usable(k: K) -> bool:
    return fit(k.signature, k.nodes) not in ("Unknown", "Identity")


def wellfounded(k: K) -> bool:
    return k.signature not in k.nodes


class Derivation:
    """A ⊢_{M,B} … with Def 17's two-party slots."""

    def __init__(self, b_walks: bool = True) -> None:
        self.b_walks = b_walks
        self.M = list(MEMORY)
        self.composed: list[K] = []
        self.acq: dict[int, int] = {}
        self.stuck_at_entry = False

    def relationship(self, nodes: list) -> str:
        return fit(content(nodes), B.nodes)

    def find_contraction(self, nodes: list):
        n = len(nodes)
        for size in range(2, n):
            for idxs in combinations(range(n), size):
                group = [nodes[i] for i in idxs]
                for k in self.M:
                    if fit(k.signature, k.nodes) != "Canon":
                        continue
                    if Counter(k.nodes) != Counter(group):
                        continue
                    new = [x for i, x in enumerate(nodes) if i not in idxs]
                    new.insert(idxs[0], k.signature)
                    if not any(
                        usable(k2) and fit(k2.signature, k2.nodes) != "Canon"
                        and (k2.signature in new or occurs_rev(k2, new))
                        for k2 in self.M
                    ):
                        continue
                    return k, group, new
        return None

    def find_targeting(self, nodes: list):
        """Def 14, both misfit locations: a forward move departs the
        underfit or adopts the overfit; reverse consumes the underfit or
        arrives into the overfit. Δ must strictly fall either way."""
        cur = content(nodes)
        goal_c = content(B.nodes)
        d0 = popc(cur ^ goal_c)
        underfit, overfit = cur & ~goal_c, goal_c & ~cur
        region = self.relationship(nodes)
        for k in self.M + self.composed:
            if not usable(k):
                continue
            if k.signature in nodes:
                if region == "S2" and not (
                    k.signature & underfit or content(k.nodes) & overfit
                ):
                    continue
                new = replace_fwd(k, nodes)
                if popc(content(new) ^ goal_c) < d0:
                    return k, "forward", new, d0, popc(content(new) ^ goal_c)
            if occurs_rev(k, nodes):
                if region == "S2" and not (
                    all(n & underfit for n in k.nodes) or k.signature & overfit
                ):
                    continue
                new = replace_rev(k, nodes)
                if popc(content(new) ^ goal_c) < d0:
                    return k, "reverse", new, d0, popc(content(new) ^ goal_c)
        return None

    def walk(self, seed: int, end_mask: int):
        """Def 17 slot walk from either party: goal-less, occurrence
        licensed on either side, no-revisit keyed on correspondence
        identity, ending at arrival in end_mask."""
        queue = deque([([seed], frozenset(), 0, [])])
        seen = {(seed,)}
        while queue:
            nodes, used, edges, path = queue.popleft()
            if edges and content(nodes) & end_mask:
                return nodes, edges, path
            if edges >= MAX_WALK_EDGES:
                continue
            for k in self.M:
                if not usable(k):
                    continue
                key = (k.signature, tuple(k.nodes))
                if key in used:
                    continue
                moves = []
                if k.signature in nodes and (
                    fit(k.signature, k.nodes) != "Canon" or wellfounded(k)
                ):
                    moves.append(("forward", replace_fwd(k, nodes)))
                if occurs_rev(k, nodes):
                    moves.append(("reverse", replace_rev(k, nodes)))
                for direction, new in moves:
                    t = tuple(sorted(new))
                    if t in seen:
                        continue
                    seen.add(t)
                    queue.append((new, used | {key}, edges + 1, path + [(direction, k)]))
        return None

    def run(self) -> list:
        nodes = list(A0.nodes)
        goal_c = content(B.nodes)
        a_c = content(nodes)
        self.j0 = popc(a_c & goal_c) / popc(a_c | goal_c)
        print(f"A0 = {A0}")
        print(f"B  = {B}  (goal — held Canon)")
        underfit0 = a_c & ~goal_c
        overfit0 = goal_c & ~a_c
        rel0 = self.relationship(nodes)
        u_str = ', '.join(name(b) for b in atom_bits(underfit0))
        u_disp = f"{{{u_str}}}" if u_str else '∅'
        o_str = ', '.join(name(b) for b in atom_bits(overfit0))
        print(
            f"C(A0,B) = {rel0} ({band(rel0)})  "
            f"u={u_disp}  o={{{o_str}}}  "
            f"Δ0={popc(a_c ^ goal_c)}"
        )
        print(f"A-side slots: {[name(n) for n in nodes if n & underfit0] or '∅'}")
        print(f"B-side slots: {[name(n) for n in B.nodes if n & overfit0] or '∅'}")
        trace: list[list] = [list(nodes)]
        for step in range(1, MAX_STEPS + 1):
            cur = content(nodes)
            if cur == goal_c:
                return self.report_done(nodes, trace)
            underfit = cur & ~goal_c
            overfit = goal_c & ~cur
            d0 = popc(cur ^ goal_c)

            c = self.find_contraction(nodes)
            if c is not None:
                k, group, new = c
                print(
                    f"{step:2d}  witnessed  {{{', '.join(name(n) for n in group)}}} "
                    f"⇉ [{name(k.signature)}]"
                )
                nodes = new
                print(f"      → {fmt_state(nodes)}   Δ={d0} (content unchanged)")
                trace.append(list(nodes))
                continue

            t = self.find_targeting(nodes)
            if t is not None:
                k, direction, new, d0_, d1 = t
                lhs = f"[{', '.join(name(n) for n in k.nodes)}] ⇉ [{name(k.signature)}]" \
                    if direction == "reverse" else \
                    f"{name(k.signature)} ⇉ [{', '.join(name(n) for n in k.nodes)}]"
                tag = " (composed)" if k in self.composed else ""
                print(f"{step:2d}  targeting  {lhs}   licence {k}{tag}")
                arriving = content(new) & ~cur
                cost = k.acq_depth if k in self.composed else 1
                for atom in atom_bits(arriving):
                    self.acq[atom] = cost
                nodes = new
                j = popc(content(nodes) & goal_c) / popc(content(nodes) | goal_c)
                print(f"      → {fmt_state(nodes)}   Δ={d0_}→{d1}   J={j:.3f}")
                trace.append(list(nodes))
                continue

            for slot in [n for n in nodes if n & underfit]:
                w = self.walk(slot, overfit)
                if w is not None:
                    end, edges, path = w
                    self.compose_a(slot, end, edges, path, step)
                    break
            else:
                if self.b_walks:
                    for slot in [n for n in B.nodes if n & overfit]:
                        w = self.walk(slot, content(nodes))
                        if w is not None:
                            end, edges, path = w
                            self.compose_b(slot, end, edges, path, step, nodes)
                            break
                    else:
                        print(f"{step:2d}  STUCK  no bridge from either party — ask")
                        if step == 1:
                            self.stuck_at_entry = True
                        return trace
                else:
                    print(f"{step:2d}  STUCK  A-side slots ∅, ν_B walks off — ask")
                    if step == 1:
                        self.stuck_at_entry = True
                    return trace
                continue
            continue
        return trace

    def compose_a(self, slot: int, end: list, edges: int, path: list, step: int):
        hops = " → ".join(f"{d}({k})" for d, k in path)
        print(f"{step:2d}  ν_A walk from slot {name(slot)} ({edges} edges):  {hops}")
        composed = K(slot, end, acq_depth=edges, tag="composed")
        self.composed.append(composed)
        print(f"      write composed {composed}  acq_depth={edges}")

    def compose_b(self, slot: int, end: list, edges: int, path: list, step: int, nodes: list):
        goal_c = content(B.nodes)
        anchors = [n for n in end if n & content(nodes)]
        hops = " → ".join(f"{d}({k})" for d, k in path)
        print(f"{step:2d}  ν_B walk from overfit slot {name(slot)} ({edges} edges):  {hops}")
        print(f"      anchor {name(anchors[0])} ∈ ν_A (no refinement edge); "
              f"departed end {name(slot)} already at the goal's witness resolution")
        witness = [n for n in end if n & goal_c] + [slot]
        composed = K(anchors[0], witness, acq_depth=edges, tag="composed ν_B")
        self.composed.append(composed)
        print(
            f"      write composed {composed}  "
            f"({fit(composed.signature, composed.nodes)} — adopt fwd)  acq_depth={edges}"
        )

    def report_done(self, nodes: list, trace: list) -> list:
        rel = self.relationship(nodes)
        node_depths = [
            max((self.acq.get(b, 0) for b in atom_bits(n)), default=0) for n in nodes
        ]
        hbar = sum(node_depths) / len(node_depths)
        gamma = δ ** hbar
        depth_str = ", ".join(f"{name(n)}@{dep}" for n, dep in zip(nodes, node_depths))
        print(f" DONE  σ(ν_A) = {name(content(nodes))}")
        print(f"      witness {fmt_state(nodes)}   C(A,B) = {rel} ({band(rel)})")
        print(f"      measurement  J {self.j0:.3f}→1.000  Ĥ={hbar:.2f} ({depth_str})  "
              f"γ=2^-{hbar:.2f}≈{gamma:.3f}")
        doc = {
            "Entry relationship is Overfit (S2) with u = ∅ — no A-side slot":
                self.relationship(trace[0]) == "Overfit"
                and not (content(trace[0]) & ~content(B.nodes)),
            "ν_B walk: all → o (all:[o] fwd) → m (o:[m] fwd), anchor m, 2 edges":
                any(k.tag == "composed ν_B" and k.acq_depth == 2 for k in self.composed),
            "Composed correspondence is m:[m,all] — Overfit, adopt fwd, depth 2":
                any(k.signature == m and k.nodes == [m, allv] and k.acq_depth == 2
                    for k in self.composed),
            "Main line consumes m ⇉ [m,all] → σ = mhall, Δ 2→0, done":
                content(trace[-1]) == mhall and len(trace) == 2,
            "Done by value equality with C(A,B) = Canon":
                rel == "Canon",
            "The goal is never rewritten: B's witness unchanged":
                B.nodes == [m, h, allv],
            "Measurement Ĥ = 2/3 (all@2, m@0, h@0), γ = 2^-(2/3) ≈ 0.630":
                abs(hbar - 2 / 3) < 1e-9 and abs(gamma - 2 ** (-2 / 3)) < 1e-9,
        }
        print("\n── verdict ──")
        for expectation, ok in doc.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {expectation}")
        self.checks = list(doc.items())
        all_ok = all(ok for _, ok in doc.items())
        print(
            f"\n  {'MATCH — the algebra reproduces the ν_B walk' if all_ok else 'MISMATCH'}"
        )
        return trace


def fmt_state(nodes: list) -> str:
    return f"{name(A0.signature)}:[{', '.join(name(n) for n in nodes)}]"


def main() -> int:
    print("── ν_B walks on (Def 17 as written) ──\n")
    enabled = Derivation(b_walks=True)
    enabled.run()
    print("\n── control: ν_B walks off (the one-party def) ──\n")
    control = Derivation(b_walks=False)
    control.run()
    print(
        "\n── comparison ──\n"
        "  The control is stuck at entry: with slots defined only on ν_A, an\n"
        "  overfit relationship decomposes into nothing and the misfit asks.\n"
        "  The ask is false in this memory — o:[m] connects the parties; the\n"
        "  ν_B walk departs from the goal's overfit node, arrives at the\n"
        "  anchor, and writes the bridge head-ward for the main line to\n"
        "  adopt. The overfit content stays sealed in [all] (Ĥ = 2/3), as in\n"
        "  the lazy write of the worked-example probe: nothing in this\n"
        "  memory consumes bare a or l."
    )
    ok = (
        control.stuck_at_entry
        and all(ok for _, ok in enabled.checks)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
