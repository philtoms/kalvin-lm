"""Dialogue-driven runner — resolves a symbolic dialogue into a StubKAgent
ResponseTable + a paced trainer submission sequence, then drives an end-to-end
session through the real MessageBus + KAgentAdapter + StubKAgent.

This is bootstrap/dev tooling (Option C in the design): it proves the stub, the
dialogue artifact, the symbolic resolver, and the harness plumbing all integrate
into a completing session. The "brain" (which kline to submit next) is isolated
here, replaying the canonical trainer turns, so the future paced Trainer
(``@specs/trainer-satisfaction.md`` TS-1..TS-17) can replace it without touching
the stub or the resolver.

The dialogue references klines by *symbolic role* (e.g. ``canon_MHALL``,
``sw_Mary_1``), not raw signatures, so the artifact survives tokenizer
retraining. The resolver compiles the dialogue's ``source`` with ``dev=True``
and matches each role to a compiled entry by ``(op, signature)`` — see
``_resolution_notes`` in ``scripts/dialogue-mhall.json`` and the design notes in
``plans/impl/stub-kagent.md``.

Usage::

    python scripts/dialogue_runner.py scripts/dialogue-mhall.json
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SYS_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

from kalvin.events import RationaliseEvent  # noqa: E402
from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4  # noqa: E402
from kalvin.kline import KLine  # noqa: E402
from kalvin.kvalue import KValue  # noqa: E402
from kalvin.nlp_tokenizer import NLPTokenizer  # noqa: E402
from kalvin.signifier import NLPSignifier  # noqa: E402
from kalvin.stub_kagent import ResponseRow, StubKAgent  # noqa: E402
from ks.compiler import compile_source  # noqa: E402
from training.harness.adapter import KAgentAdapter  # noqa: E402
from training.harness.bus import MessageBus  # noqa: E402
from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE  # noqa: E402
from training.harness.message import Message  # noqa: E402

# Band label → band-representative significance.
BAND_TO_SIG = {"S1": SIG_S1, "S2": SIG_S2, "S3": SIG_S3, "S4": SIG_S4}
# Inverse, for readable verification diagnostics.
_SIG_TO_BAND = {v: k for k, v in BAND_TO_SIG.items()}

# The legend's op names map onto the compiler's dbg.op names; only CANON differs.
OP_MAP = {"CANON": "CANONIZED"}


class ResolveError(Exception):
    """A symbolic role could not be resolved to a concrete kline."""


class PacingError(Exception):
    """The kagent's reply to a paced trainer turn diverged from the table."""


# ── Role resolution ──────────────────────────────────────────────────────


class RoleResolver:
    """Resolve symbolic legend roles to concrete :class:`KLine` objects.

    Signatures compute bottom-up: an IDENTITY word's signature is
    ``make_signature(encode(word))`` (which for a multi-token word equals its
    subword-canon signature); a DERIVED_SIG is the OR of its operands; everything
    else inherits its ``signature_role``'s signature. Compiled entries are then
    matched by ``(op, signature)``.
    """

    def __init__(self, legend: dict, source: str, tok, sigf) -> None:
        self.legend = legend
        self.tok = tok
        self.sigf = sigf
        entries = compile_source(source, tokenizer=tok, signifier=sigf, dev=True)
        self.by_op_sig: dict[tuple[str, int], list[KValue]] = {}
        for e in entries:
            self.by_op_sig.setdefault((e.kline.dbg.op, e.kline.signature), []).append(e)
        self._sig: dict[str, int] = {}
        self._kline: dict[str, KLine] = {}
        self._resolve_constituents()

    def _first(self, op: str, sig: int) -> KLine | None:
        """The compiled kline for ``(op, sig); None if absent.

        Multiple matches collapse to the first: shared BPE constituents (e.g.
        little/lamb sharing 'l') are emitted once per word but are *equal*
        klines, so the first is representative.
        """
        ms = self.by_op_sig.get((op, sig))
        return ms[0].kline if ms else None

    def sig_of(self, role: str) -> int:
        if role in self._sig:
            return self._sig[role]
        le = self.legend[role]
        op = le["op"]
        if op == "IDENTITY":
            s = self.sigf.make_signature(self.tok.encode(le.get("word", role)))
        elif op == "DERIVED_SIG":
            s = 0
            for o in le["operands"]:
                s |= self.sig_of(o)
        else:
            s = self.sig_of(le["signature_role"])
        self._sig[role] = s
        return s

    def kline_of(self, role: str) -> KLine:
        if role in self._kline:
            return self._kline[role]
        if role not in self.legend:
            raise ResolveError(f"{role}: not in legend")
        le = self.legend[role]
        op = le["op"]
        sig = self.sig_of(role)
        if op == "DERIVED_SIG":
            # A derived signature has no own kline; its structural form is the
            # canon (or countersigned) entry carrying that signature.
            kl = self._first("CANONIZED", sig) or self._first("COUNTERSIGNED", sig)
            if kl is None:
                raise ResolveError(f"{role}: no canon/countersigned entry for sig 0x{sig:x}")
        elif op == "IDENTITY":
            kl = self._first("IDENTITY", sig)
            if kl is None:
                # Multi-token word → its subword canon shares this signature.
                kl = self._first("CANONIZED", sig)
            if kl is None:
                # Single-token atom the compiler never emitted standalone (it
                # only appears as a relation operand). Synthesise the identity.
                kl = KLine(sig, [])
        else:
            cop = OP_MAP.get(op, op)
            kl = self._first(cop, sig)
            if kl is None:
                raise ResolveError(f"{role}: no {cop} entry for sig 0x{sig:x}")
        self._kline[role] = kl
        return kl

    def _resolve_constituents(self) -> None:
        """Resolve sw_* constituents from their parent canon's nodes (positional).

        Per ``_resolution_notes``: a subword atom is the BPE constituent of its
        parent canon, matched positionally to the canon's node list (each node
        IS a constituent's signature).
        """
        for role, le in self.legend.items():
            if not (role.startswith("sw_") and le["op"] == "IDENTITY"):
                continue
            parent = next(
                (
                    r
                    for r, lp in self.legend.items()
                    if lp["op"] == "CANON" and role in lp.get("operands", [])
                ),
                None,
            )
            if parent is None:
                raise ResolveError(f"{role}: no parent subword canon")
            canon_kl = self.kline_of(parent)
            idx = self.legend[parent]["operands"].index(role)
            node_sig = canon_kl.nodes[idx]
            kl = self._first("IDENTITY", node_sig)
            self._kline[role] = kl if kl is not None else KLine(node_sig, [])


# ── Dialogue → submissions + ResponseTable ───────────────────────────────


@dataclass
class ResolvedDialogue:
    """A symbolic dialogue resolved into paced submissions + a stub table."""

    trainer_turns: list[list[KValue]] = field(default_factory=list)  # per turn, in order
    table: list[ResponseRow] = field(default_factory=list)  # stub rows, keyed by trigger
    turn_expected: list[list[KValue]] = field(default_factory=list)
    expected: list[KValue] = field(default_factory=list)
    primary: KLine | None = None  # the primary kline (lesson completes on its S1)

    @property
    def submissions(self) -> list[KValue]:
        """Trainer turns flattened to one KValue per submission, in order."""
        return [kv for turn in self.trainer_turns for kv in turn]


def _emission_sig(event: str, reported: str | None) -> int:
    if event == "request":
        return SIG_S4  # "I'm missing this operand"
    if event == "countersign":
        return SIG_S1  # a primary's completion
    return BAND_TO_SIG[reported]  # ground at its structural band


def resolve_dialogue(dial: dict, tok=None, sigf=None) -> ResolvedDialogue:
    """Resolve a dialogue dict into submissions + a StubKAgent ResponseTable."""
    tok = tok if tok is not None else NLPTokenizer()
    sigf = sigf if sigf is not None else NLPSignifier()
    resolver = RoleResolver(dial["legend"], dial["source"], tok, sigf)

    turns = dial["turns"]
    out = ResolvedDialogue(table=[], primary=None)
    if "primary" in dial["legend"]:
        out.primary = resolver.kline_of("primary")

    # The stub keys rows by kline (KV-2) and fires each once, so a kline
    # submitted twice (e.g. the shared 'l' BPE piece of little/lamb) collides:
    # the first submission fires its row, the second is a silent no-match. We
    # mirror that here — build one row per unique trigger kline, and suppress
    # grounds of klines already grounded (an atom grounded once stays grounded).
    seen_triggers: set[KLine] = set()
    grounded: set[KLine] = set()

    i = 0
    n = len(turns)
    while i < n:
        t = turns[i]
        if t["actor"] != "trainer":
            # A stub turn with no preceding trainer turn — not expected in the
            # bootstrap single-cascade dialogue. Skip defensively.
            i += 1
            continue
        # The trainer turn: roles + declared bands, in listed order.
        trainer = [(k, t.get("declared")) for k in t["klines"]]
        i += 1
        # The stub's responses: following consecutive stub turns.
        stub_emis: list[tuple[str, str, str | None]] = []  # (event, role, reported)
        while i < n and turns[i]["actor"] == "stub":
            st = turns[i]
            for k in st["klines"]:
                stub_emis.append((st["event"], k, st.get("reported")))
            i += 1

        # Resolve submitted KValues; classify each as a fresh trigger or a
        # re-submission (its kline already drove a row earlier in the cascade).
        role_kv: dict[str, KValue] = {}
        is_new: dict[str, bool] = {}
        for role, declared in trainer:
            kl = resolver.kline_of(role)
            role_kv[role] = KValue(kl, BAND_TO_SIG[declared])
            fresh = kl not in seen_triggers
            is_new[role] = fresh
            if fresh:
                seen_triggers.add(kl)
        # The trainer submits every listed kline in this turn (re-submissions
        # are silent no-ops at the stub, but the trainer does send them). One
        # trainer turn is a single paced submission unit — see run_session,
        # which drains the bus (delivering the kagent's reply to the
        # supervisor) before the next turn is ever submitted.
        out.trainer_turns.append([role_kv[r] for r, _ in trainer])

        new_roles = [r for r, _ in trainer if is_new[r]]
        last_new_role = new_roles[-1] if new_roles else None
        buckets: dict[str, dict[str, list[KValue]]] = {
            r: {"requests": [], "grounds": [], "countersigns": []} for r in new_roles
        }

        # Assign each stub emission to the fresh submission whose row fires it.
        # A ground of an already-grounded kline is suppressed (the stub grounds
        # each kline once); everything else attaches to the turn's last fresh
        # submission, realising the table-prescribed cascade.
        for event, role, reported in stub_emis:
            kl = resolver.kline_of(role)
            if event == "ground" and kl in grounded:
                continue  # already grounded — idempotent, stub emits nothing
            if last_new_role is None:
                # No fresh trigger this turn (all re-submissions): the emission
                # has nowhere to attach. This does not arise in the bootstrap
                # single-cascade dialogue; surface it loudly if it ever does.
                raise ResolveError(
                    f"stub emission for {role!r} with no fresh trigger in turn"
                )
            kv = KValue(kl, _emission_sig(event, reported))
            buckets[last_new_role][
                "requests" if event == "request"
                else "grounds" if event == "ground"
                else "countersigns"
            ].append(kv)
            if event == "ground":
                grounded.add(kl)

        # Build one row per fresh submitted role, in submission order. The
        # ordered proposals each row fires (requests → grounds → countersigns)
        # are also captured as this turn's expected reply — the oracle
        # run_session verifies the kagent against, turn by turn.
        turn_exp: list[KValue] = []
        for role, _ in trainer:
            if not is_new[role]:
                continue
            b = buckets[role]
            out.table.append(
                ResponseRow(
                    trigger=role_kv[role],
                    requests=tuple(b["requests"]),
                    grounds=tuple(b["grounds"]),
                    countersigns=tuple(b["countersigns"]),
                )
            )
            turn_exp.extend([*b["requests"], *b["grounds"], *b["countersigns"]])
        out.turn_expected.append(turn_exp)

    # The stub emits each row's requests, then grounds, then countersigns
    # (ST-7). ``turn_expected`` records that exact sequence per trainer turn;
    # ``expected`` is its flatten, matching the order the stub actually produces
    # — not the dialogue's narrative turn order, which occasionally interleaves
    # a ground and a later request within one row.
    out.expected = [kv for turn in out.turn_expected for kv in turn]
    return out


# ── Session execution ────────────────────────────────────────────────────


@dataclass
class SessionReport:
    """Outcome of driving a resolved dialogue through the harness."""

    captured: list[KValue]  # stub proposals observed by the supervisor, in order
    expected: list[KValue]
    fired: int  # rows fired
    grounded: int  # signatures grounded
    primary: KLine | None
    primary_countersigned: bool

    @property
    def n_match(self) -> int:
        n = 0
        for got, exp in zip(self.captured, self.expected):
            if got.kline == exp.kline and got.significance == exp.significance:
                n += 1
        return n

    @property
    def ok(self) -> bool:
        if len(self.captured) != len(self.expected):
            return False
        if self.n_match != len(self.expected):
            return False
        return self.primary_countersigned


def _drain(bus: MessageBus) -> None:
    """Process every queued message synchronously (deterministic test drive)."""
    while True:
        try:
            msg = bus._queue.get_nowait()  # noqa: SLF001
        except queue.Empty:
            break
        bus._dispatch(msg)  # noqa: SLF001


def _describe(kv: KValue) -> str:
    """One-line, robust description of a KValue for verification diagnostics."""
    kl = kv.kline
    op = getattr(getattr(kl, "dbg", None), "op", None) or "kline"
    band = _SIG_TO_BAND.get(kv.significance, f"0x{kv.significance:x}")
    return f"{op} sig=0x{kl.signature:x} nodes={list(kl.nodes)} band={band}"


def _kv_eq(a: KValue, b: KValue) -> bool:
    """KValue equality by kline AND significance.

    :class:`KValue` hashes/compares by kline only (KV-2) so it can key the
    stub's row index; significance is a separate axis the table authors, so a
    real equality check must compare it explicitly.
    """
    return a.kline == b.kline and a.significance == b.significance


def _verify_turn(
    captured: list[KValue], before: int, expected: list[KValue], turn_idx: int
) -> None:
    """Assert the kagent's reply to a paced turn equals the table's prescription.

    ``captured[before:]`` is exactly what the kagent emitted in response to
    trainer turn ``turn_idx`` (the proposals delivered to the supervisor's bus
    subscription during that turn's drain). It must equal ``expected`` — the
    table's prescribed proposals for that turn — in order, by kline and
    significance. The first divergence raises :class:`PacingError`, naming the
    turn and the offending proposal.
    """
    actual = captured[before:]
    for j, (got, exp) in enumerate(zip(actual, expected)):
        if not _kv_eq(got, exp):
            raise PacingError(
                f"table divergence at trainer turn {turn_idx}, proposal {j}: "
                f"expected {_describe(exp)}, got {_describe(got)}"
            )
    if len(actual) != len(expected):
        raise PacingError(
            f"table divergence at trainer turn {turn_idx}: "
            f"expected {len(expected)} proposal(s), got {len(actual)}"
        )


def run_session(resolved: ResolvedDialogue) -> SessionReport:
    """Wire the real harness and drive the resolved submissions against the stub."""
    bus = MessageBus()
    adapter = KAgentAdapter(bus, role=TRAINEE_ROLE)
    stub = StubKAgent(adapter, resolved.table)
    adapter.bind(stub)

    captured: list[KValue] = []

    def on_supervisor(msg: Message) -> None:
        if isinstance(msg.message, RationaliseEvent):
            captured.append(msg.message.proposal)

    bus.subscribe(SUPERVISOR_ROLE, on_supervisor)

    # Paced request-response loop (trainer-satisfaction §Prompt the Primary,
    # §Drive the Cascade). The trainer submits one turn, then yields to the
    # bus so the kagent's reply — the stub's requests, grounds, and
    # countersigns — is delivered back to the supervisor (its bus-subscription
    # callback) before the next trainer turn. The primary turn is the only
    # proactive prompt; every later turn is submitted only after the kagent
    # has emitted its events for the preceding turn — never as a batch, never
    # in succession (TS-5, TS-8).
    #
    # After each turn the captured reply is verified against the table's
    # prescribed proposals for that turn (turn_expected), in order — a live
    # check that the kagent emits exactly the interleaved klines the table
    # author wrote, not a silent drain. A mismatch is a table/kagent
    # divergence and fails loudly via _verify_turn (PacingError).
    if len(resolved.turn_expected) != len(resolved.trainer_turns):
        raise PacingError(
            "turn_expected is not aligned with trainer_turns "
            f"({len(resolved.turn_expected)} vs {len(resolved.trainer_turns)})"
        )
    for idx, turn in enumerate(resolved.trainer_turns):
        before = len(captured)
        for kv in turn:
            bus.send(
                Message(role=TRAINEE_ROLE, action="rationalise", message=kv, sender=SUPERVISOR_ROLE)
            )
        _drain(bus)
        _verify_turn(captured, before, resolved.turn_expected[idx], idx)

    primary_countersigned = any(
        kv.kline == resolved.primary and kv.significance == SIG_S1 for kv in captured
    ) if resolved.primary is not None else False

    return SessionReport(
        captured=captured,
        expected=resolved.expected,
        fired=len(stub.fired),
        grounded=len(stub.grounded),
        primary=resolved.primary,
        primary_countersigned=primary_countersigned,
    )


def _format_summary(report: SessionReport, resolved: ResolvedDialogue) -> str:
    total = len(resolved.submissions)
    fired = report.fired
    return (
        f"MHALL dialogue session\n"
        f"  trainer submissions : {total}\n"
        f"  stub rows fired     : {fired}\n"
        f"  stub proposals      : {len(report.captured)} captured / "
        f"{len(report.expected)} expected\n"
        f"  proposals matched   : {report.n_match}\n"
        f"  signatures grounded : {report.grounded}\n"
        f"  primary countersigned (S1): {report.primary_countersigned}\n"
        f"  result              : {'PASS — lesson complete' if report.ok else 'FAIL'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a dialogue end-to-end against the StubKAgent."
    )
    parser.add_argument(
        "--dialogue", default="scripts/dialogue-mhall.json", help="Path to a dialogue JSON (e.g. scripts/dialogue-mhall.json)"
    )
    args = parser.parse_args(argv)

    dial = json.loads(Path(args.dialogue).read_text())
    resolved = resolve_dialogue(dial)
    report = run_session(resolved)
    print(_format_summary(report, resolved))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
