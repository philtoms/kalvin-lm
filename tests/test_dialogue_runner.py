"""The bus-agnostic dialogue runner tests.

Spec: ``@specs/dialogue-driven-training.md`` DDT-6..16. The runner owns the table
cursor: at each step it reads whose row is next and asks that actor for its next
row. Greediness is automatic — consecutive same-actor rows are served to the same
actor (e.g. ``T,T,K`` asks the trainer twice, then the trainee). Each actor yields
its own rows one at a time. Every response is validated against the table.

The decisive acceptance test is the canonical end-to-end run (the "Mary had
a little lamb" reference dialogue, frozen in :mod:`tests._fixtures`):
runs to exhaustion, every row validated.
"""

from __future__ import annotations

import pytest

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from tests._fixtures import mhall_table
from training.dialogue import (
    ActorDivergence,
    TableTrainee,
    TableTrainer,
    decode,
    load_table,
    run,
)
from training.dialogue.runner import _TableActor, run_table


@pytest.fixture(scope="module")
def _decoded_mhall():
    tok, sigf = NLPTokenizer(), NLPSignifier()
    table = load_table(mhall_table())
    return decode(table, tokenizer=tok, signifier=sigf)


@pytest.fixture(scope="module")
def _compiled_mhall():
    """The MHALL compiled script + the signifier it was built with."""
    from ks.compiler import compile_source

    tok, sigf = NLPTokenizer(), NLPSignifier()
    table = load_table(mhall_table())
    compiled = compile_source(table.script, tokenizer=tok, signifier=sigf, dev=True)
    return compiled, sigf


def _ev(sig: int = 0, role: str | None = None) -> RationaliseEvent:
    return RationaliseEvent(
        kind="frame",
        query=KValue(KLine(sig, ()), 0),
        proposal=KValue(KLine(sig, ()), 0),
        role=role,
    )


def _row(role: str, sig: int):
    from training.dialogue.decoder import DecodedTurn

    return DecodedTurn(role=role, op="IDENTITY", value=KValue(KLine(sig, ()), 0))


# ── DDT-9/10/16: an actor yields its rows one at a time, never inspecting incoming ─


def test_actor_yields_rows_one_at_a_time(_decoded_mhall):
    """DDT-9/16: each respond() returns the next row's event; None when exhausted.
    The actor returns no cursor — the runner owns the validation index."""
    actor = TableTrainer(_decoded_mhall)
    t_rows = [t for t in _decoded_mhall if t.role == "T"]
    for i, row in enumerate(t_rows):
        event = actor.respond(None)
        assert event is not None
        assert event.proposal.kline == row.value.kline
    assert actor.respond(None) is None


def test_actor_does_not_inspect_incoming(_decoded_mhall):
    """DDT-10: the emitted row is independent of the incoming event."""
    first_t = next(t for t in _decoded_mhall if t.role == "T")
    a = TableTrainer(_decoded_mhall)
    b = TableTrainer(_decoded_mhall)
    ea = a.respond(_ev())
    eb = b.respond(None)
    assert ea.proposal.kline == eb.proposal.kline == first_t.value.kline


def test_actor_query_is_incoming_proposal(_decoded_mhall):
    """DDT-9: the event's query is the incoming proposal (own turn on opening)."""
    trainer = TableTrainer(_decoded_mhall)
    opening = trainer.respond(None)
    assert opening.query.kline == opening.proposal.kline  # no incoming → own turn


def test_actor_yields_first_row_on_first_respond(_decoded_mhall):
    """DDT-16: the actor yields its first row on the first respond (no cursor is
    returned to the runner; the runner owns the validation index)."""
    actor = TableTrainee(_decoded_mhall)
    first_k = next(t for t in _decoded_mhall if t.role == "K")
    event = actor.respond(None)
    assert event is not None
    assert event.proposal.kline == first_k.value.kline


# ── DDT-7: greedy — the runner serves a run of same-actor rows to one actor ─


def test_greedy_runner_serves_run_to_same_actor():
    """DDT-7: for a T,T,K table the trainer is asked twice, then the trainee once.
    Greediness is the runner's behaviour, not the actor's."""
    decoded = [_row("T", 0xAA), _row("T", 0xBB), _row("K", 0xCC)]
    seen: list[str] = []

    class _RecordingTrainer(_TableActor):
        def respond(self, incoming):
            r = super().respond(incoming)
            if r is not None:
                seen.append("T")
            return r

    class _RecordingTrainee(_TableActor):
        def respond(self, incoming):
            r = super().respond(incoming)
            if r is not None:
                seen.append("K")
            return r

    trainer = _RecordingTrainer(decoded, role="T", kind="frame")
    trainee = _RecordingTrainee(decoded, role="K", kind="frame")
    result = run(decoded, trainer=trainer, trainee=trainee)
    assert result.complete
    assert seen == ["T", "T", "K"]
    assert [e.proposal.kline.signature for e in result.events] == [0xAA, 0xBB, 0xCC]


# ── DDT-8: trainer and trainee are symmetric ──────────────────────────────


def test_trainer_and_trainee_are_symmetric_readers(_decoded_mhall):
    """DDT-8: both yield their own rows in order; together they cover the table."""
    assert isinstance(TableTrainer(_decoded_mhall), _TableActor)
    assert isinstance(TableTrainee(_decoded_mhall), _TableActor)
    result = run(
        _decoded_mhall, trainer=TableTrainer(_decoded_mhall), trainee=TableTrainee(_decoded_mhall)
    )
    assert [e.proposal.kline for e in result.events] == [t.value.kline for t in _decoded_mhall]


# ── DDT-6/13: the run is driven by the table cursor, opens with the trainer ─


def test_run_opens_with_trainer_and_ends_on_exhaustion(_decoded_mhall):
    """DDT-6/13: the first row is the trainer's; the run ends at the table end."""
    result = run(
        _decoded_mhall, trainer=TableTrainer(_decoded_mhall), trainee=TableTrainee(_decoded_mhall)
    )
    assert result.complete
    t_rows = [t for t in _decoded_mhall if t.role == "T"]
    assert result.events[0].proposal.kline == t_rows[0].value.kline
    assert len(result.events) == len(_decoded_mhall)


def test_run_greedy_multi_row_table_completes():
    """DDT-7/13: a T,T,K,K,T table runs to exhaustion with greedy multi-row runs."""
    decoded = [
        _row("T", 0x1),
        _row("T", 0x2),
        _row("K", 0x3),
        _row("K", 0x4),
        _row("T", 0x5),
    ]
    result = run(decoded, trainer=TableTrainer(decoded), trainee=TableTrainee(decoded))
    assert result.complete
    assert [e.proposal.kline.signature for e in result.events] == [0x1, 0x2, 0x3, 0x4, 0x5]


# ── DDT-11: both actors are validated against the table ─────────────────


def test_trainee_divergence_raises(_decoded_mhall):
    """DDT-11: a trainee response that does not match the table raises ActorDivergence."""

    class _BadTrainee:
        def respond(self, incoming):
            # Declares role K, emits a kline that matches nothing in the table.
            return _ev(0xDEAD, role="K")

    with pytest.raises(ActorDivergence) as exc_info:
        run(_decoded_mhall, trainer=TableTrainer(_decoded_mhall), trainee=_BadTrainee())
    assert exc_info.value.role == "K"
    # The first K-row is at full-table cursor 1 (after the T-row at cursor 0);
    # the runner validates against decoded[cursor], so the divergence is at 1.
    assert exc_info.value.cursor == 1
    assert exc_info.value.emitted.kline.signature == 0xDEAD


def test_trainer_divergence_raises(_decoded_mhall):
    """DDT-11: a synthesising trainer whose response does not match the table raises
    ActorDivergence. A real trainer is expected to synthesise its responses and
    is validated against the table like the trainee."""

    class _SynthesisingTrainer:
        def respond(self, incoming):
            # Declares role T, emits a synthesised non-table kline — rejected.
            return _ev(0x99, role="T")

    with pytest.raises(ActorDivergence) as exc_info:
        run(_decoded_mhall, trainer=_SynthesisingTrainer(), trainee=TableTrainee(_decoded_mhall))
    assert exc_info.value.role == "T"
    assert exc_info.value.emitted.kline.signature == 0x99


def test_validation_keys_off_event_role_not_table_key(_decoded_mhall):
    """DDT-11: the runner validates by the role the actor declared on its event,
    not by the table's view of who responded. An actor that declares a role with
    no matching table rows is caught by the role-keyed lookup."""

    class _WrongRoleTrainee:
        def respond(self, incoming):
            # The table expects a K row here, but the actor declares role "X" —
            # a role the table has no rows for. Validation keys off the declared
            # role and raises.
            return _ev(0xDEAD, role="X")

    with pytest.raises(ActorDivergence) as exc_info:
        run(_decoded_mhall, trainer=TableTrainer(_decoded_mhall), trainee=_WrongRoleTrainee())
    assert exc_info.value.role == "X"  # the declared role, not the table key "K"


def test_event_carries_emitting_role(_decoded_mhall):
    """The default actors self-declare their role on every emitted event."""
    result = run(
        _decoded_mhall, trainer=TableTrainer(_decoded_mhall), trainee=TableTrainee(_decoded_mhall)
    )
    roles = [e.role for e in result.events]
    # The MHALL table opens with a trainer turn (role "T").
    assert roles[0] == "T"
    assert set(roles) == {"T", "K"}


# ── DDT-14: bus-agnostic ──────────────────────────────────────────────────


def test_runner_has_no_bus_dependency():
    """DDT-14: the runner module does not import the harness bus/adapter."""
    import training.dialogue.runner as runner_mod

    import_lines = [
        line
        for line in open(runner_mod.__file__).read().splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    joined = "\n".join(import_lines)
    assert "training.harness" not in joined
    assert "MessageBus" not in joined
    assert "KAgentAdapter" not in joined


# ── DDT-15: no learned/grounding notion ───────────────────────────────────


def test_runner_has_no_learning_notion():
    """DDT-15: the runner carries no learned/grounding notion or signal."""
    import training.dialogue.runner as runner_mod

    src = open(runner_mod.__file__).read()
    assert "def _learned" not in src
    assert "grounding_event" not in src
    assert "Learned" not in src


# ── Canonical end-to-end (decisive acceptance) ────────────────────────────


def test_canonical_run_completes(_decoded_mhall):
    """The full MHALL dialogue runs to exhaustion through the default actors,
    every row validated, closing on the trainee's S1 countersign of the primary."""
    result = run(
        _decoded_mhall, trainer=TableTrainer(_decoded_mhall), trainee=TableTrainee(_decoded_mhall)
    )
    assert result.complete
    assert result.events[-1].proposal.significance == SIG_S1
    assert result.events[-1].proposal.kline == result.events[0].proposal.kline


def test_run_table_convenience():
    """run_table decodes and runs with default actors in one call."""
    table = load_table(mhall_table())
    result = run_table(table, tokenizer=NLPTokenizer(), signifier=NLPSignifier())
    assert result.complete


# ── Rationaliser integration (plan §Phase 2.2) ────────────────────────────


def test_rationaliser_runs_mhall_to_exhaustion(_decoded_mhall):
    """The RationalisingTrainee (a real, stateful trainee) runs MHALL to exhaustion
    with zero divergence against the golden master, driven through the runner's
    ``run()`` like any Actor. The trainer stays a ``TableTrainer`` (the
    deterministic oracle). This is the canonical end-to-end proof that a
    rationalising trainee is a drop-in replacement for ``TableTrainee``."""
    from training.dialogue.runner import RationalisingTrainee

    result = run(
        _decoded_mhall,
        trainer=TableTrainer(_decoded_mhall),
        trainee=RationalisingTrainee(NLPSignifier()),
    )
    assert result.complete
    # Every emitted event validated against decoded[cursor] (run would have
    # raised ActorDivergence otherwise). Confirm the closing S1 countersign.
    assert result.events[-1].proposal.significance == SIG_S1
    assert result.events[-1].proposal.kline == result.events[0].proposal.kline
    # The rationaliser produced exactly the table's rows (no more, no less).
    assert len(result.events) == len(_decoded_mhall)


# ── SynthesizingTrainer integration (plan §Phase 4.2) ────────────────────


def test_synthesizing_trainer_runs_mhall_to_exhaustion(_decoded_mhall, _compiled_mhall):
    """The SynthesizingTrainer (a real, script-deriving trainer) runs MHALL to
    exhaustion with zero divergence against the golden master, driven through
    the runner's ``run()`` like any Actor. The trainee stays a ``TableTrainee``
    (the deterministic oracle). This is the canonical end-to-end proof that a
    synthesizing trainer is a drop-in replacement for ``TableTrainer`` — the
    symmetric counterpart to the Rationaliser test above."""
    from training.dialogue.runner import SynthesizingTrainer

    compiled, sigf = _compiled_mhall
    result = run(
        _decoded_mhall,
        trainer=SynthesizingTrainer(compiled, sigf),
        trainee=TableTrainee(_decoded_mhall),
    )
    assert result.complete
    # Every emitted event validated against decoded[cursor] (run would have
    # raised ActorDivergence otherwise). Confirm the closing S1 countersign.
    assert result.events[-1].proposal.significance == SIG_S1
    assert result.events[-1].proposal.kline == result.events[0].proposal.kline
    # The synthesizer produced exactly the table's rows (no more, no less).
    assert len(result.events) == len(_decoded_mhall)


# ── R4 — open-dialog-close (plan @plans/implement-synthesizing-trainer.md) ─


def test_synthesizing_trainer_replies_to_any_incoming(_compiled_mhall):
    """The trainer does not detect script closes and never withholds on its
    own — it synthesises a reply for whatever ``incoming`` it is handed.
    Close-detection is the runner's job (it owns the ``close`` markers and
    routes accordingly). This replaces the former R4 trainer-side withhold."""
    from training.dialogue.runner import SynthesizingTrainer

    compiled, sigf = _compiled_mhall
    trainer = SynthesizingTrainer(compiled, sigf)
    # A primary-shaped incoming (formerly the close): the trainer still replies
    # (R3 echoes the matching compiled kline) — it does not withhold.
    primary_event = RationaliseEvent(
        kind="frame",
        query=KValue(KLine(compiled[0].kline.signature, ()), 0),
        proposal=KValue(compiled[0].kline, SIG_S1),
        role="K",
    )
    assert trainer.respond(primary_event) is not None


def test_run_routes_next_turn_as_open_after_close():
    """Script boundaries are the runner's: after a turn carrying a ``close``
    marker, the next actor is handed ``incoming=None`` (an open) rather than
    the close event (a reply). A close ends a script; the next turn opens a
    fresh one. The trainer never detects the close — the runner does."""
    from training.dialogue.decoder import DecodedTurn

    # Decoded rows with distinct values so the recording actors can reproduce
    # them in order: T(11), K(22), K(33, close:1), T(44).
    decoded = [
        DecodedTurn(role="T", op="IDENTITY", value=KValue(KLine(11, ()), 0)),
        DecodedTurn(role="K", op="IDENTITY", value=KValue(KLine(22, ()), 0)),
        DecodedTurn(role="K", op="IDENTITY", value=KValue(KLine(33, ()), 0), close=1),
        DecodedTurn(role="T", op="IDENTITY", value=KValue(KLine(44, ()), 0)),
    ]
    seen: list = []  # the incoming each respond() got

    def _recorder(role_label):
        rows = [t for t in decoded if t.role == role_label]
        i = 0

        class _A:
            role = role_label

            def respond(self, incoming):
                nonlocal i
                seen.append(incoming)
                turn = rows[i]
                i += 1
                return RationaliseEvent(
                    kind="frame",
                    query=turn.value,
                    proposal=turn.value,
                    role=role_label,
                )

        return _A()

    run(decoded, _recorder("T"), _recorder("K"))
    # 4 turns taken in role order T,K,K,T → 4 recorded incomings.
    # [0] None (opening); [1] T event; [2] K event;
    # [3] None (close at index 2 routes the next turn as an open).
    assert len(seen) == 4
    assert seen[0] is None
    assert seen[1] is not None and seen[1].role == "T"
    assert seen[2] is not None and seen[2].role == "K"
    assert seen[3] is None
