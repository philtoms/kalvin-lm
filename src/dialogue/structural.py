"""A structural supervisor: grades escalations from compiler-emitted evidence.

The script compiles to structure plus an *intended* significance — what the
script's klines prove when every declared kline is held. This module derives
that intent with the same structural predicates the engine applies to itself
(:func:`kalvin.kline.is_canon`, :meth:`signifier.residual`, ``signature_of``)
and grades a proposal against it:

- intended — the fixed point over the script's own entries (all declarations
  assumed ratified);
- attained — the same rule restricted to klines K actually holds (LTM);
- missing — the entries between the two, in ratification units.

Installed at the harness escalation seam; the harness itself never judges.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from kalvin.kline import KLine, KNode, is_canon, is_relationship
from kalvin.kvalue import KValue
from kalvin.significance import SIG8_MAX, SIG_S1, SIG_S2, SIG_S4, BandLayout

_LAYOUT = BandLayout()

Key = tuple[int, tuple[int, ...]]


def _key(kline: KLine) -> Key:
    return (kline.signature, tuple(kline.nodes))


@dataclass
class Grade:
    """A supervisor judgement: the reply band and how far K is from S1."""

    significance: int
    matched: KLine | None = None
    missing: list[KLine] = field(default_factory=list)

    def band(self) -> str:
        return _LAYOUT.classify(self.significance)


class SemanticEvidence:
    """The derived, cross-kline structure of a script: canon index, countersign
    pairs, denotation/connotation edges — the undeclared backbone the entries
    hold collectively. Grows as entries join the harness answering pools
    (no look-ahead)."""

    def __init__(self, signifier) -> None:
        self._signifier = signifier
        self._entries: list[KValue] = []
        self._by_key: dict[Key, KValue] = {}
        self._by_signature: dict[int, list[KValue]] = {}
        self._intended: set[Key] = set()
        self._dirty = True

    def add(self, value: KValue) -> None:
        if _key(value.kline) in self._by_key:
            return
        self._entries.append(value)
        self._by_key[_key(value.kline)] = value
        self._by_signature.setdefault(value.kline.signature, []).append(value)
        self._dirty = True

    @property
    def signifier(self):
        return self._signifier

    def under(self, signature: int) -> list[KValue]:
        return self._by_signature.get(signature, [])

    def intended_keys(self) -> set[Key]:
        """The fixed-point groundable set: every script kline whose structure
        completes, assuming all declarations hold.

        Identity and canon by the universal grounding rule (canon needs its
        nodes groundable — self-denoting); a declared relationship
        (COUNTERSIGNS/DENOTES/CONNOTES) is proven by its own declaration.
        Underfit/overfit questions never ground — they are the asks."""
        if self._dirty:
            seeds = {n for v in self._entries for n in v.kline.nodes}
            self._intended = self._fixpoint(
                {key for key in self._by_key}, seeds
            )
            self._dirty = False
        return self._intended

    def _fixpoint(self, candidates: set[Key], seeds: set[KNode] = set()) -> set[Key]:
        grounded: set[Key] = set()
        # Seed: every node value is a potential terminal word identity — the
        # same tokenizer-guaranteed identity the harness's _answer forges.
        grounded_sigs: set[int] = set(seeds)
        changed = True
        while changed:
            changed = False
            for key in candidates:
                if key in grounded:
                    continue
                kline = self._by_key[key].kline
                if self._proves(kline, grounded_sigs):
                    grounded.add(key)
                    grounded_sigs.add(kline.signature)
                    changed = True
        return grounded

    def _proves(self, kline: KLine, grounded_sigs: set[int]) -> bool:
        sig = self._signifier
        nodes = kline.nodes
        if not nodes:
            return False  # an unknown is an ask, never proof
        if nodes == [kline.signature]:
            return True  # identity
        if is_relationship(kline) and kline.dbg and kline.dbg.op in (
            "COUNTERSIGNS", "DENOTES", "CONNOTES"
        ):
            return True  # declared proof
        if is_canon(kline, sig):
            return all(n in grounded_sigs for n in nodes)
        return False

    def grade(self, proposal: KLine, held: set[Key]) -> Grade:
        """Grade ``proposal``: intended band from the script's proof, missing
        = intended-minus-held entries on the path to S1.

        ``held`` is K's current grounding in the same key space (LTM).
        """
        sig = self._signifier
        intended = self.intended_keys()
        nodes_sig = sig.signature_of(proposal.nodes)
        gap = sig.residual(proposal.signature, nodes_sig)
        excess = sig.residual(nodes_sig, proposal.signature)

        if nodes_sig == proposal.signature:
            # The proposal is itself a canon (an alternative decomposition) —
            # exact full-value equality, the same rule as ``is_canon``.
            if all(_sig_held(n, held) for n in proposal.nodes):
                return Grade(SIG_S1)
            missing = self._missing_for_nodes(proposal.nodes, intended, held)
            return Grade(_s2_byte(missing), missing=missing)
        if excess == 0 and gap != 0:
            # Underfit on the type dimension: the script's full decompositions
            # of this signature are the work left.
            missing = [
                v.kline for v in self.under(proposal.signature)
                if _key(v.kline) in intended and _key(v.kline) not in held
            ]
            if not missing and any(
                _key(v.kline) in intended for v in self.under(proposal.signature)
            ):
                return Grade(SIG_S1)
            return Grade(_s2_byte(missing), missing=missing)
        # The proposal's content is a canonical form the script knows:
        # signature_of(nodes) names a script kline (e.g. WDMH:[m h a l l]
        # whose content is MHALL). Distance = the ratifications still
        # missing on the content canon and on the proposal's own signature.
        if nodes_sig != proposal.signature:
            content = [
                v for v in self.under(nodes_sig) if _key(v.kline) in intended
            ]
            if content:
                missing = [
                    v.kline for v in content if _key(v.kline) not in held
                ]
                missing += [
                    v.kline
                    for v in self.under(proposal.signature)
                    if _key(v.kline) in intended and _key(v.kline) not in held
                ]
                return Grade(_s2_byte(missing), matched=content[0].kline, missing=missing)
        # Divergent fit (no-fit): claims S2; S4 only when nothing is shared.
        if sig.signifies(proposal.signature, nodes_sig):
            return Grade(SIG_S2)
        return Grade(SIG_S4)

    def _missing_for_nodes(
        self, nodes: Sequence[KNode], intended: set[Key], held: set[Key]
    ) -> list[KLine]:
        out: list[KLine] = []
        for n in nodes:
            if _sig_held(n, held):
                continue
            for v in self.under(n):
                if _key(v.kline) in intended and _key(v.kline) not in held:
                    out.append(v.kline)
                    break
        return out


def _sig_held(signature: int, held: set[Key]) -> bool:
    return any(k[0] == signature for k in held)


def _s2_byte(missing: list[KLine]) -> int:
    """A graded S2: one notch below S1 per missing ratification, floored at
    the bottom of S2."""
    return max(0x80, SIG8_MAX - len(missing))


class StructuralSupervisor:
    """Escalation callable: grade off-script proposals from evidence + K's
    held set, ratifying at S1 only when the script's proof completes."""

    def __init__(
        self,
        evidence: SemanticEvidence,
        state,
        render=None,
    ) -> None:
        self._evidence = evidence
        self._state = state
        self._render = render

    def observe(self, value: KValue) -> None:
        """A script entry joined the answering pools (harness, no look-ahead)."""
        self._evidence.add(value)

    def __call__(self, ask: KValue) -> KValue:
        held = {
            (kl.signature, tuple(kl.nodes))
            for bucket in self._state.ltm.values()
            for kl in bucket
        }
        grade = self._evidence.grade(ask.kline, held)
        if self._render is not None:
            for line in self._describe(ask, grade):
                print(line)
        return KValue(ask.kline, grade.significance)

    def _describe(self, ask: KValue, grade: Grade) -> list[str]:
        show = self._render or (lambda v: f"0x{v.kline.signature:x}")
        head = f"  ⚠ structural: {show(ask)} -> {grade.band()} ({grade.significance})"
        if grade.matched is not None:
            head += f"  content={show(KValue(grade.matched, grade.significance))}"
        lines = [head]
        for kl in grade.missing:
            lines.append(
                f"      missing {show(KValue(kl, grade.significance))}"
            )
        return lines
