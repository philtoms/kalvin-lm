"""Tests for NLP tokenizer compatibility with existing curricula.

Verifies that all curricula compile and rationalize correctly with
the NLPTokenizer without requiring parenthetical comment annotations.

Spec ref: specs/tokenizer.md §NLP Tokenizer › Curriculum Compatibility (TOK-NLP-9..12)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kalvin.agent import KAgent
from kalvin.kline import sig_level
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signature import make_signature
from ks import compile_source
from tests.conftest import requires_nlp_data

# The entire module requires the NLP tokenizer data assets; skip it cleanly
# when those assets are absent on a fresh clone.
pytestmark = requires_nlp_data

# ── Fixtures ──────────────────────────────────────────────────────────

CURRICULA_DIR = Path(__file__).resolve().parents[1] / "curricula"


def _kscript_from_curriculum(path: Path) -> str:
    """Extract KScript source from a curriculum markdown file."""
    text = path.read_text(encoding="utf-8")
    in_block = False
    blocks: list[str] = []
    for line in text.splitlines():
        if line.strip() == "```":
            in_block = not in_block
            continue
        if in_block:
            blocks.append(line)
    return "\n".join(blocks)


class _CountingAdapter:
    """Adapter that counts frame and ground events."""

    def __init__(self) -> None:
        self.frame = 0
        self.ground = 0

    def on_event(self, event: object) -> None:
        kind = getattr(event, "kind", None)
        if kind == "frame":
            self.frame += 1
        elif kind == "ground":
            self.ground += 1


# ── TOK-NLP-9: Bare single-char signatures ────────────────────────────────


class TestBareSingleChar:
    """TOK-NLP-9: Bare single-char sigs produce consistent NLP-BPE tokens."""

    def test_m_compiles_to_nlp_bpe(self, nlp_tokenizer: NLPTokenizer) -> None:
        entries = compile_source("M", tokenizer=nlp_tokenizer, dev=True)
        assert len(entries) == 1
        sig = entries[0].signature
        # Upper 32 bits should have NLP type info (not zero)
        assert (sig >> 32) != 0
        # Lower 32 bits should be the BPE token ID for 'M'
        assert (sig & 0xFFFFFFFF) != 0

    def test_consistent_encoding(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Same character always produces the same node value."""
        e1 = compile_source("M", tokenizer=nlp_tokenizer)
        e2 = compile_source("M", tokenizer=nlp_tokenizer)
        assert e1[0].signature == e2[0].signature

    def test_different_chars_different_tokens(self, nlp_tokenizer: NLPTokenizer) -> None:
        e_m = compile_source("M", tokenizer=nlp_tokenizer)
        e_h = compile_source("H", tokenizer=nlp_tokenizer)
        assert e_m[0].signature != e_h[0].signature


# ── TOK-NLP-10: Multi-token signature decomposition ────────────────────


class TestMultiTokenDecomposition:
    """TOK-NLP-10: Multi-token sigs decompose correctly with NLP tokenizer."""

    def test_mhall_decomposition(self, nlp_tokenizer: NLPTokenizer) -> None:
        entries = compile_source("MHALL", tokenizer=nlp_tokenizer, dev=True)
        # §8 MTS: four distinct component identities (M, H, A, L — second L
        # intra-expansion deduped) + one CANONIZED. Compounds are exempt
        # from §11.3 (canonical encoding), so the literal string is not
        # BPE-re-decomposed.
        assert len(entries) == 5
        s4 = [e for e in entries if sig_level(e) == "S4"]
        s2 = [e for e in entries if sig_level(e) == "S2"]
        assert len(s4) == 4
        assert len(s2) == 1

    def test_countersign_with_multi_token(self, nlp_tokenizer: NLPTokenizer) -> None:
        entries = compile_source("MHALL == SVO", tokenizer=nlp_tokenizer, dev=True)
        # Should have countersign entries for MHALL:SVO and SVO:MHALL
        countersigns = [e for e in entries if sig_level(e) == "S1"]
        assert len(countersigns) == 2


# ── TOK-NLP-11: Curriculum compilation ─────────────────────────────────────


class TestCurriculumCompilation:
    """TOK-NLP-11: All curricula compile and rationalize correctly."""

    @pytest.mark.parametrize(
        "curriculum_file",
        [
            "first-steps.md",
            "first-steps-s2.md",
            "mhall-svo-single.md",
            "mhall-svo-equivalence.md",
            "cascade-pressure.md",
            "conflict-drill.md",
            "s3-auto-countersign.md",
        ],
    )
    def test_curriculum_compiles(self, curriculum_file: str, nlp_tokenizer: NLPTokenizer) -> None:
        path = CURRICULA_DIR / curriculum_file
        if not path.exists():
            pytest.skip(f"Curriculum not found: {path}")

        kscript = _kscript_from_curriculum(path)
        entries = compile_source(kscript, tokenizer=nlp_tokenizer, dev=True)
        assert len(entries) > 0

    @pytest.mark.parametrize(
        "curriculum_file",
        [
            "first-steps.md",
            "first-steps-s2.md",
            "mhall-svo-single.md",
            "mhall-svo-equivalence.md",
            "cascade-pressure.md",
            "conflict-drill.md",
            "s3-auto-countersign.md",
        ],
    )
    def test_curriculum_rationalizes(
        self, curriculum_file: str, nlp_tokenizer: NLPTokenizer
    ) -> None:
        path = CURRICULA_DIR / curriculum_file
        if not path.exists():
            pytest.skip(f"Curriculum not found: {path}")

        kscript = _kscript_from_curriculum(path)
        entries = compile_source(kscript, tokenizer=nlp_tokenizer, dev=True)
        assert len(entries) > 0

        adapter = _CountingAdapter()
        agent = KAgent(tokenizer=nlp_tokenizer, adapter=adapter)

        # Rationalize all entries without errors
        for entry in entries:
            agent.rationalise(entry)

        # Should have at least some events
        assert adapter.frame + adapter.ground > 0


# ── TOK-NLP-12: Comments are optional ──────────────────────────────────────


class TestCommentsOptional:
    """TOK-NLP-12: Bare sigs work; comments add semantic resolution."""

    def test_bare_and_annotated_both_work(self, nlp_tokenizer: NLPTokenizer) -> None:
        bare = compile_source("M", tokenizer=nlp_tokenizer, dev=True)
        annotated = compile_source("M(ary)", tokenizer=nlp_tokenizer, dev=True)

        # Both produce entries
        assert len(bare) == 1
        # M(ary) → "Mary" → [M, ary]: two component identities + one
        # canonize. The packed-sig IDENTITY is suppressed (CONTEXT.md
        # "Identity"), so 3 klines total.
        assert len(annotated) == 3

        # Both signatures are NLP-BPE encoded (high bits set)
        assert (bare[0].signature >> 32) != 0
        assert (annotated[0].signature >> 32) != 0

        # The annotation expands "M" into more entries than the bare sig
        assert len(annotated) > len(bare)

    def test_block_comment_binding(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Block comment before multi-token sig enables positional binding."""
        from ks.compiler import compile_source

        source = "(Mary Had A Little Lamb)\nMHALL"
        entries = compile_source(source, tokenizer=nlp_tokenizer, dev=True)

        # Block comment should bind M→Mary, H→Had, A→A, L→Little, L→Lamb
        # producing NLP-encoded signature entries.  With dev=True the sig field
        # is human-readable.
        assert len(entries) > 0

        # At least the first entry should resolve via binding
        # (signature should be an NLP-BPE token with high bits set)
        assert (entries[0].signature >> 32) != 0


# ── KB-320: first-steps-s2.md dual-misfit S2 routing ────────────────────


class _RoutingAdapter:
    """Adapter that records every RationaliseEvent (kind, significance, candidate).

    ``on_event`` is called from two threads — the caller's thread (fast-path
    ground/S1/S4 events published synchronously by ``KAgent.rationalise``) and
    the Cogitator's background daemon thread (S2/S3 expansion events via
    ``on_expansion``). ``list.append`` is atomic under the GIL, so the events
    list is safe to append from either thread; it must only be *read* after a
    ``cogitate_drain`` has let all background work finish.
    """

    def __init__(self) -> None:
        self.events: list = []

    def on_event(self, event: object) -> None:
        self.events.append(event)


def _replay_first_steps_s2(path: Path, tok: NLPTokenizer) -> tuple[_RoutingAdapter, list[list]]:
    """Replay ``first-steps-s2.md`` per-lesson through a fresh KAgent.

    Mirrors ``KAgentAdapter._handle_submit``: for each lesson, compile its
    joined KScript, pre-register every compiled entry in STM, then rationalise
    each entry against a single persistent agent. The Cogitator is drained
    after every lesson (and once more at the end) so background S2/S3 expansion
    events are collected before the caller reads them.

    Returns the recording adapter and the per-lesson compiled-entry lists.
    """
    from training.trainer.curriculum_document import CurriculumDocument

    doc = CurriculumDocument.from_file(path)
    assert doc.all_labels() == ["1", "2", "3", "4", "5"], doc.all_labels()

    adapter = _RoutingAdapter()
    agent = KAgent(tokenizer=tok, adapter=adapter)
    lesson_entries: list[list] = []
    for lesson in doc.lessons:
        src = "\n".join(lesson.kscript)
        entries = compile_source(src, tokenizer=tok, dev=True)
        for entry in entries:
            agent.model.add_to_stm(entry)
        for entry in entries:
            agent.rationalise(entry)
        agent.cogitate_drain(5.0)
        lesson_entries.append(entries)
    agent.cogitate_drain(5.0)
    return adapter, lesson_entries


def _event_summary(events: list) -> list:
    """Compact (kind, significance, has-candidate) summary for assertion messages."""
    return [(ev.kind, hex(ev.significance), ev.candidate is not None) for ev in events]


class TestFirstStepsS2Routing:
    """KB-320: the first-steps-s2.md dual-misfit lesson must reach the S2 band.

    Regression for KB-309's finding that the curriculum produced byte-identical
    S1-only runs. Root causes were (A) lessons 2-4 accidentally deleted by
    commit a1c40f1, and (B) the `=>` CANONIZED operator making the lesson-5
    kline canonical by construction, so AGT-14's self-grounded short-circuit
    fired before candidate retrieval. The fix restores lessons 2-4 and rewrites
    lesson 5 with the `>` CONNOTED operator so the kline is a genuine misfit.
    """

    def test_dual_misfit_routes_s2_not_short_circuit(self, nlp_tokenizer: NLPTokenizer) -> None:
        """The lesson-5 misfit klines reach candidate retrieval (S2/S3), not
        the AGT-14 self-grounded fast path (S1)."""
        from kalvin.expand import D_MAX

        path = CURRICULA_DIR / "first-steps-s2.md"
        adapter, lesson_entries = _replay_first_steps_s2(path, nlp_tokenizer)

        # Lesson 5 is the dual-misfit lesson. Its misfit klines are the
        # non-canonical (sig != make_signature(nodes)) entries with nodes.
        lesson5 = lesson_entries[-1]
        misfits = [e for e in lesson5 if e.nodes and e.signature != make_signature(e.nodes)]
        assert misfits, "lesson 5 should compile genuine misfit klines"
        misfit_keys = {(e.signature, tuple(e.nodes)) for e in misfits}

        # (1) At least one S2/S3 expansion event: candidate-bearing frame event
        # whose significance is strictly between 0 (S4) and D_MAX (S1). This is
        # the KB-309 significance-level observable that was previously absent.
        expansions = [
            ev
            for ev in adapter.events
            if ev.kind == "frame"
            and ev.candidate is not None
            and 0 < ev.significance < D_MAX
            and (ev.query.signature, tuple(ev.query.nodes)) in misfit_keys
        ]
        assert expansions, (
            "expected at least one S2/S3 expansion event (candidate-bearing, "
            "0 < significance < D_MAX) for a lesson-5 misfit kline; the "
            "curriculum is not reaching the S2 band. "
            f"events={_event_summary(adapter.events)}"
        )

        # (2) No lesson-5 misfit took the AGT-14 FAST-PATH self-grounded
        # short-circuit. The fast path publishes `_publish("frame", kline,
        # kline, D_MAX)` -> query == proposal. A cogitation-discovered S1
        # (`on_s1`) also has candidate is None + significance == D_MAX but has
        # query != proposal, so the query==proposal test isolates the fast path
        # that indicates AGT-14 fired before candidate retrieval.
        fast_path_s1 = [
            ev
            for ev in adapter.events
            if (ev.query.signature, tuple(ev.query.nodes)) in misfit_keys
            and ev.query == ev.proposal
            and ev.candidate is None
            and ev.significance == D_MAX
        ]
        assert not fast_path_s1, (
            "a lesson-5 misfit kline took the AGT-14 self-grounded short-circuit "
            "(fast-path S1, query == proposal) instead of reaching candidate "
            f"retrieval: {[(hex(ev.query.signature), ev.query.nodes) for ev in fast_path_s1]}"
        )

    def test_emits_ratify_request_full_stack(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Full-stack replay emits a ratify_request for the dual-misfit lesson.

        Uses a deterministic single-threaded FIFO bus (the real ``MessageBus``
        dispatches on a ``bus.run()`` thread, which combined with the
        Cogitator daemon makes a real-bus replay ordering-sensitive). Lesson 5's
        ``{MH: [Alpha]}`` dual misfit escapes the auto-countersign backstop, so
        it legitimately stalls without a supervisor participant — this asserts
        only that a ratify_request fires, not that the session reaches complete.
        """
        from collections import deque

        from training.harness.adapter import KAgentAdapter
        from training.harness.bus import MessageBus
        from training.harness.constants import SUPERVISOR_ROLE
        from training.trainer.curriculum import Curriculum
        from training.trainer.curriculum_document import CurriculumDocument
        from training.trainer.trainer import Trainer

        class _StepBus(MessageBus):
            """FIFO bus: send() enqueues; pump() dispatches one message on the
            caller's thread (no bus.run() thread, no recursive dispatch)."""

            def __init__(self) -> None:
                super().__init__()
                self._q: deque = deque()

            def send(self, msg: object) -> None:  # type: ignore[override]
                self._q.append(msg)

            def pump(self) -> bool:
                if not self._q:
                    return False
                self._dispatch(self._q.popleft())
                return True

        path = CURRICULA_DIR / "first-steps-s2.md"
        bus = _StepBus()
        captured: list = []
        bus.subscribe("*", lambda m: captured.append(m))
        bus.subscribe(SUPERVISOR_ROLE, lambda m: None)  # sink supervisor relays

        adapter = KAgentAdapter(bus, tokenizer=nlp_tokenizer)
        agent = KAgent(tokenizer=nlp_tokenizer, adapter=adapter)
        adapter.bind(agent)

        doc = CurriculumDocument.from_file(path)
        trainer = Trainer(bus, Curriculum(doc), curriculum_file=path)
        trainer.start_session()

        # Drive the FIFO queue, draining the Cogitator between pumps so async
        # S2/S3 events are enqueued. Stop once a ratify_request is observed.
        try:
            for _ in range(4000):
                agent.cogitate_drain(2.0)
                if not bus.pump():
                    break
                if any(m.action == "ratify_request" for m in captured):
                    agent.cogitate_drain(2.0)
                    for _ in range(50):
                        if not bus.pump():
                            break
                    break
        finally:
            agent.cogitate_join(2.0)

        ratify = [m for m in captured if m.action == "ratify_request"]
        assert ratify, (
            "expected at least one ratify_request for the lesson-5 dual misfit "
            "(proving the auto-countersign backstop did not swallow it)"
        )
