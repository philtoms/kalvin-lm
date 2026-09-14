"""Verify the §9 worked example of docs/kalvin-algebra.md against the
pure algebra. No engine, compiler, or tokenizer: values are ints (one bit
per atom), klines are records. Memory, A0, and B are injected exactly as
the document lists them, and the verdict compares each step against the
documented trace and measurement:

    Step 1   {d,h} ⇉ [dh]   under Canon dh:[d,h]        → [w,dh,m],  Δ=4
    Step 2   dh ⇉ [h]       under Denotation dh:[h]     → [w,h,m],   Δ 4→3
    Step 3   walk w:[w] → w:[o] → w:[all] → w:[a,l,l];  write w:[a,l,l] at
             acquisition depth 3; consume w ⇉ [a,l,l]  → σ = mhall, done
    §11      Ĥ = (0+0+3+3+3)/5 = 9/5,  γ = 2^(-9/5) ≈ 0.29

Run from the repo root:  python3 dev/algebra/worked-example-wdmh.py
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations

δ = 0.5
MAX_STEPS = 32
MAX_WALK_EDGES = 8

# §1–2: atoms and values. One bit per atom; a value is any OR of them.
m, h, a, l, w, d, o = (1 << i for i in range(7))
mhall, dh, allv, wdmh = m | h | a | l, d | h, a | l, w | d | m | h
NAMES = {
    m: "m", h: "h", a: "a", l: "l", w: "w", d: "d", o: "o",
    mhall: "mhall", dh: "dh", allv: "all", wdmh: "wdmh",
}


@dataclass
class K:
    """A kline: signature, node sequence, acquisition depth (Def 5, §11)."""

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
    """σ — Def 4: OR-reduce; order and multiplicity discarded."""
    out = 0
    for n in nodes:
        out |= n
    return out


def popc(v: int) -> int:
    return bin(v).count("1")


def fit(signature: int, nodes: list) -> str:
    """Def 10 — the nine shapes, in case order."""
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
    gap, excess = signature & ~sigma, sigma & ~signature
    if gap and not excess:
        return "Denotation" if len(nodes) == 1 else "Underfit"
    if excess and not gap:
        return "Overfit"
    return "Under+over"


def band(shape: str) -> str:
    return {
        "Identity": "S1", "Canon": "S1",
        "Denotation": "S2", "Underfit": "S2", "Overfit": "S2", "Under+over": "S2",
        "Connotation": "S3", "No-fit": "S3", "Unknown": "S4",
    }[shape]


# §9 — the documented memory, injected verbatim.
MEMORY = [
    K(mhall, [m, h, a, l, l]),
    K(dh, [d, h]),
    K(allv, [a, l, l]),
    K(dh, [h]),
    K(w, [o]),
    K(allv, [o]),
    K(m, [m]),
]
A0 = K(wdmh, [w, d, m, h])
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
    """A ⊢_{M,B} … — Defs 12–17 over the injected memory."""

    def __init__(self, expand_on_write: bool = True) -> None:
        self.expand_on_write = expand_on_write
        self.M = list(MEMORY)
        self.composed: list[K] = []
        self.acq: dict[int, int] = {}  # atom -> acquisition depth
        self.checks: list[tuple[str, bool]] = []

    # ── licensing ────────────────────────────────────────────────────────

    def relationship(self, nodes: list) -> str:
        return fit(content(nodes), B.nodes)

    def find_contraction(self, nodes: list):
        """Canonicalisation (§6): smallest exactly-witnessed proper group
        whose compound exposes an applicable misfit correspondence."""
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
        """Def 14: C(A,B) scopes the region; licensed iff Δ strictly falls."""
        cur = content(nodes)
        d0 = popc(cur ^ content(B.nodes))
        gap = cur & ~content(B.nodes)
        region = self.relationship(nodes)
        for k in self.M + self.composed:
            if not usable(k):
                continue
            if k.signature in nodes:
                if region == "S2" and not k.signature & gap:
                    continue
                new = replace_fwd(k, nodes)
                if popc(content(new) ^ content(B.nodes)) < d0:
                    return k, "forward", new, d0, popc(content(new) ^ content(B.nodes))
            if occurs_rev(k, nodes):
                if region == "S2" and not all(n & gap for n in k.nodes):
                    continue
                new = replace_rev(k, nodes)
                if popc(content(new) ^ content(B.nodes)) < d0:
                    return k, "reverse", new, d0, popc(content(new) ^ content(B.nodes))
        return None

    def slot_walk(self, slot: int, excess: int):
        """Def 17: goal-less walk from the slot identity; occurrence on
        either side licenses; no-revisit keys on correspondence identity
        (the documented walk consumes signature all twice: all:[o], then
        all:[a,l,l])."""
        queue = deque([([slot], frozenset(), 0, [])])
        seen = {(slot,)}
        while queue:
            nodes, used, edges, path = queue.popleft()
            if edges and content(nodes) & excess:
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

    def refine(self, nodes: list, edges: int, path: list):
        """Write granularity (Def 17): expand the terminal's nodes
        covering the excess to the goal's own witness resolution of that
        content, under held well-founded canons. Policy A reproduces the
        documented write w:[a,l,l]; the lazy policy writes the walk's
        end-state w:[all] unchanged."""
        while True:
            for i, n in enumerate(nodes):
                kanon = next(
                    (
                        k
                        for k in self.M
                        if k.signature == n
                        and fit(k.signature, k.nodes) == "Canon"
                        and wellfounded(k)
                    ),
                    None,
                )
                if kanon is None:
                    continue
                nodes = nodes[:i] + list(kanon.nodes) + nodes[i + 1 :]
                path.append(("expand", kanon))
                edges += 1
                break
            else:
                return nodes, edges, path

    # ── the run ──────────────────────────────────────────────────────────

    def run(self) -> list:
        nodes = list(A0.nodes)
        goal_c = content(B.nodes)
        self.j0 = popc(content(nodes) & goal_c) / popc(content(nodes) | goal_c)
        print(f"A0 = {A0}")
        print(f"B  = {B}  (goal — held Canon)")
        gap0 = content(nodes) & ~goal_c
        print(
            f"C(A0,B) = {self.relationship(nodes)} ({band(self.relationship(nodes))})  "
            f"gap={{{', '.join(name(b) for b in atom_bits(gap0))}}}  "
            f"excess={{{', '.join(name(b) for b in atom_bits(goal_c & ~content(nodes)))}}}  "
            f"Δ0={popc(content(nodes) ^ goal_c)}"
        )
        trace: list[list] = [list(nodes)]
        for step in range(1, MAX_STEPS + 1):
            cur = content(nodes)
            if cur == goal_c:
                return self.report_done(nodes, trace, verdict=self.expand_on_write)
            gap = cur & ~goal_c
            excess = goal_c & ~cur
            d0 = popc(cur ^ goal_c)

            c = self.find_contraction(nodes)
            if c is not None:
                k, group, new = c
                print(
                    f"{step:2d}  witnessed  {{{', '.join(name(n) for n in group)}}} "
                    f"⇉ [{name(k.signature)}]   licence {k} ({fit(k.signature, k.nodes)}, reverse contract)"
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
                print(
                    f"{step:2d}  targeting  {lhs}   licence {k}{tag} "
                    f"({fit(k.signature, k.nodes)}, {direction})"
                )
                arriving = content(new) & ~cur
                cost = (k.acq_depth if k in self.composed
                        else 0 if band(fit(k.signature, k.nodes)) == "S1" else 1)
                for atom in atom_bits(arriving):
                    self.acq[atom] = cost
                nodes = new
                j = popc(content(nodes) & goal_c) / popc(content(nodes) | goal_c)
                print(f"      → {fmt_state(nodes)}   Δ={d0_}→{d1}   J={j:.3f}")
                trace.append(list(nodes))
                continue

            slots = [n for n in nodes if n & gap]
            walked = False
            for slot in slots:
                walk = self.slot_walk(slot, excess)
                if walk is None:
                    continue
                end, edges, path = walk
                if self.expand_on_write:
                    end, edges, path = self.refine(end, edges, path)
                hops = " → ".join(f"{d}({k})" for d, k in path)
                print(f"{step:2d}  slot {name(slot)} walks ({edges} edges):  {hops}")
                composed = K(slot, end, acq_depth=edges)
                self.composed.append(composed)
                print(f"      write composed {composed}  acq_depth={edges}")
                walked = True
                break
            if walked:
                continue

            print(f"{step:2d}  STUCK  slot asks: {[name(s) for s in slots]}")
            return trace
        return trace

    def report_done(self, nodes: list, trace: list, verdict: bool = True) -> list:
        rel = self.relationship(nodes)
        j1 = 1.0
        node_depths = [
            max((self.acq.get(b, 0) for b in atom_bits(n)), default=0) for n in nodes
        ]
        hbar = sum(node_depths) / len(node_depths)
        dbar = 0.0  # no canon expansion in the main line
        gamma = j1 * δ ** (dbar + hbar)
        depth_str = ", ".join(
            f"{name(n)}@{dep}" for n, dep in zip(nodes, node_depths)
        )
        print(f" DONE  σ(ν_A) = {name(content(nodes))}")
        print(f"      witness {fmt_state(nodes)}   C(A,B) = {rel} ({band(rel)})")
        print(f"      measurement  J {self.j0:.3f}→{j1:.3f}  Ĥ={hbar:.2f} ({depth_str})  "
              f"D̄={dbar:.1f}  γ=2^-{hbar:.2f}≈{gamma:.3f}")
        if not verdict:
            return trace

        doc = {
            "Step 1 contracts {d,h}→[dh] under dh:[d,h], reaching [w,dh,m]":
                trace[1] == [w, dh, m],
            "Step 2 sheds dh→[h] under dh:[h] (Denotation), reaching [w,h,m], Δ 4→3":
                trace[2] == [w, h, m],
            "Step 3 walks w:[w]→w:[o]→w:[all]→w:[a,l,l] and writes w:[a,l,l] at depth 3":
                any(k.signature == w and k.nodes == [a, l, l] and k.acq_depth == 3
                    for k in self.composed),
            "Step 3 consumes w⇉[a,l,l]; final witness ≡ [h,m,a,l,l] (multiset)":
                Counter(trace[-1]) == Counter([h, m, a, l, l]),
            "Done at value equality with C(A,B) = Canon":
                content(trace[-1]) == mhall and rel == "Canon",
            "The subject m is never replaced (m:[m] Identity, inert)":
                m in trace[-1] and all(m in state for state in trace),
            "§11: Ĥ = 9/5, γ = 2^(-9/5) ≈ 0.29":
                abs(hbar - 9 / 5) < 1e-9 and abs(gamma - 2 ** (-9 / 5)) < 1e-9,
        }
        print("\n── verdict ──")
        for expectation, ok in doc.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {expectation}")
        self.checks = list(doc.items())
        all_ok = all(ok for _, ok in doc.items())
        note = (
            "\n  Note: the walk's end-condition fires at w:[all] (content {a,l}\n"
            "  already overlaps the excess); the third edge is the write-time\n"
            "  expansion toward the goal's witness resolution of the excess,\n"
            "  per Def 17. No-revisit keys on correspondence identity, not\n"
            "  bare signature — the walk consumes `all` twice (all:[o], then\n"
            "  all:[a,l,l]), as T2 requires."
        )
        print(f"\n  {'MATCH — the algebra reproduces the documented example' if all_ok else 'MISMATCH'}")
        if all_ok:
            print(note)
        return trace


def fmt_state(nodes: list) -> str:
    return f"wdmh:[{', '.join(name(n) for n in nodes)}]"


def main() -> int:
    print("── policy A: expand at write (the documented example) ──\n")
    Derivation(expand_on_write=True).run()
    print("\n── policy B: lazy write (no expansion) ──\n")
    Derivation(expand_on_write=False).run()
    print(
        "\n── comparison ──\n"
        "  Both reach done by the same value equality. A delivers the excess\n"
        "  at the goal's witness resolution ([a,l,l], Ĥ=9/5, γ≈0.287); B\n"
        "  leaves it sealed in the compound ([all], Ĥ=2/3, γ≈0.63). The edge\n"
        "  costs 2^-1 whenever crossed — δ discounts D̄ and Ĥ alike — so B plus\n"
        "  a later in-derivation expansion totals the same 2^-9/5 as A; B is\n"
        "  strictly cheaper when nothing ever needs the finer granularity."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
