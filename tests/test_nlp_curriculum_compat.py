"""Tests for NLP tokenizer compatibility with existing curricula.

Verifies that all curricula compile and rationalize correctly with
the NLPTokenizer without requiring parenthetical comment annotations.

Spec ref: specs/tokenizer.md §NLP Tokenizer › Curriculum Compatibility (TOK-NLP-9..12)
"""

from __future__ import annotations

import difflib
import re
import subprocess
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


def _replay_curriculum(path: Path, tok: NLPTokenizer) -> tuple[_RoutingAdapter, list[list]]:
    """Replay any curriculum per-lesson through a single persistent ``KAgent``.

    Generalises the KB-320 ``first-steps-s2`` replay to an arbitrary curriculum
    path. Mirrors ``KAgentAdapter._handle_submit``: for each lesson, compile its
    joined KScript, pre-register every compiled entry in STM, then rationalise
    each entry against one persistent agent (so earlier lessons' countersigns
    are available as candidates for later lessons' misfits). The Cogitator is
    drained after every lesson (and once more at the end) so background S2/S3
    expansion events are collected before the caller reads them.

    Returns the recording adapter and the per-lesson compiled-entry lists.
    """
    from training.trainer.curriculum_document import CurriculumDocument

    doc = CurriculumDocument.from_file(path)

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
        adapter, lesson_entries = _replay_curriculum(path, nlp_tokenizer)

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

    def test_ratify_request_round_trip_completes(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Full-stack regression (KB-337): a supervisor countersign round-trips.

        Closes the KB-320 gap: ``test_emits_ratify_request_full_stack`` only
        asserts the ``ratify_request`` *fires*. This test continues past it —
        it extracts the buffered proposal, converts it to its wire-dict form
        ``{"signature", "nodes"}`` (faithfully simulating the WebSocket JSON
        round-trip — a real supervisor re-emits the canonical shape, not a
        live KLine), and sends the resulting ``countersign`` onto the bus.

        Before KB-337 the wire dict reached ``KAgent.countersign`` and raised
        ``AttributeError: 'dict' object has no attribute 'nodes'`` on the bus
        dispatch thread, killing the run. After the fix the adapter
        materialises the dict to a KLine (Step 1) and the reciprocal kline is
        rationalised.
        """
        from collections import deque

        from kalvin.signature import make_signature
        from training.harness.adapter import KAgentAdapter
        from training.harness.bus import MessageBus
        from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
        from training.harness.message import Message
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

        try:
            # Drive the FIFO queue until a ratify_request fires (the dual-misfit
            # that escapes the auto-countersign backstop), draining the
            # Cogitator between pumps so async S2/S3 events are enqueued.
            ratify = None
            for _ in range(4000):
                agent.cogitate_drain(2.0)
                if not bus.pump():
                    break
                ratify_msgs = [m for m in captured if m.action == "ratify_request"]
                if ratify_msgs:
                    ratify = ratify_msgs[0]
                    agent.cogitate_drain(2.0)
                    for _ in range(50):
                        if not bus.pump():
                            break
                    break
            assert ratify is not None, (
                "expected a ratify_request for the lesson-5 dual misfit "
                "(precondition for the countersign round-trip)"
            )

            # The on-bus ratify_request carries a LIVE KLine proposal (no
            # WebSocket round-trip yet). Convert it to the canonical wire dict
            # a real supervisor re-emits — do NOT pass the live object, or the
            # dict crash is not reproduced.
            proposal = ratify.message["proposal"]
            proposal_wire = {
                "signature": proposal.signature,
                "nodes": list(proposal.nodes),
            }
            # The reciprocal kline countersign will create + rationalise:
            # {make_signature(proposal.nodes): [proposal.signature]}.
            reciprocal_sig = make_signature(list(proposal.nodes))

            # Supervisor answers the ratify_request with a countersign carrying
            # the wire-dict proposal. Before KB-337 this raised AttributeError
            # inside _handle_countersign on the bus dispatch thread.
            raised: BaseException | None = None
            bus.send(
                Message(
                    role=TRAINEE_ROLE,
                    action="countersign",
                    message=proposal_wire,
                    sender="supervisor",
                )
            )

            # Pump the countersign and any resulting cogitation events, draining
            # the Cogitator between pumps. Bounded so the test cannot hang.
            for _ in range(4000):
                agent.cogitate_drain(2.0)
                try:
                    if not bus.pump():
                        break
                except BaseException as exc:  # noqa: BLE001 — surface the crash
                    raised = exc
                    break

            assert raised is None, (
                f"countersign round-trip raised {type(raised).__name__}: {raised}"
            )
            # The reciprocal kline was rationalised — rationalise always stages
            # the kline in the model (at minimum STM), so it is now findable.
            assert agent.model.find(reciprocal_sig) is not None, (
                "expected the countersign reciprocal kline to be rationalised "
                "(present in the model) after the wire-dict countersign"
            )
        finally:
            agent.cogitate_join(2.0)


# ── KB-333: curriculum lesson-count + structure snapshot guard ──────────

#: All seven curriculum files this guard watches.
ALL_CURRICULUM_FILES = [
    "first-steps.md",
    "first-steps-s2.md",
    "mhall-svo-single.md",
    "mhall-svo-equivalence.md",
    "cascade-pressure.md",
    "conflict-drill.md",
    "s3-auto-countersign.md",
]

# Remove any parenthetical (...) group. Real curriculum annotations are
# mixed-case inline forms (M(ary), H(ad), L(ittle), D(et)) and block word
# lists ((Mary Had A Little Lamb), (A L L), (Alpha Beta)), so any-paren
# stripping is the robust normalisation. Stripping is what makes the snapshot
# stable across cosmetic annotation-word edits (M(ary) and M(ark) both reduce
# to M) while remaining sensitive to actual sig/node/structure drift.
_ANNOTATION = re.compile(r"\([^)]*\)")


def _lesson_count(path: Path) -> int:
    """Number of ``### <n>`` lesson headers in a curriculum markdown file.

    Counts ``^### `` lines directly from the raw text (no fenced-block
    extraction, since ``### `` headers live outside code fences in these
    files). Unambiguous: no ``### `` line appears inside any code fence.
    """
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("### ")
    )


def _normalize_structure(text: str) -> list[str]:
    """Structural skeleton of curriculum markdown: ``### N`` headers plus the
    annotation-stripped lines of every fenced code block, in document order.

    Prose lines are dropped (annotations only ever appear inside fences per
    rule 45, so prose carries nothing the guard should pin). Blanked lines
    (e.g. a block word-list like ``(Mary Had A Little Lamb)`` that becomes
    empty after stripping) are removed. The result is a faithful, stable
    representation of the curriculum's kscript content and lesson roster.
    """
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        if line.startswith("### "):
            out.append(line.rstrip())
        elif line.strip() == "```":
            in_block = not in_block
        elif in_block:
            stripped = _ANNOTATION.sub("", line).rstrip()
            if stripped.strip():
                out.append(stripped)
    return out


def _normalized_structure(path: Path) -> list[str]:
    return _normalize_structure(path.read_text(encoding="utf-8"))


def _git_blob(rev: str, path: str) -> str | None:
    """Read ``git show <rev>:<path>`` or return ``None`` if unavailable.

    Returns ``None`` (rather than raising) when the revision is absent from a
    shallow/archive checkout, so the historical sanity checks can skip cleanly.
    """
    try:
        res = subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return res.stdout


# Audited baselines (KB-333). ``_EXPECTED_LESSON_COUNTS`` are the ``### ``
# header counts; ``_EXPECTED_STRUCTURE`` is the post-audit, post-restore
# (KB-320 for first-steps-s2.md, KB-333 for mhall-svo-single.md) structural
# skeleton with annotations stripped. These snapshots MUST be updated whenever
# a lesson is deliberately added/removed or a kline's sig/nodes/structure is
# intentionally changed -- the guard then forces awareness instead of letting
# an edit pass silently delete or truncate curriculum content (the a1c40f1
# accident this guard was written to prevent).
_EXPECTED_LESSON_COUNTS: dict[str, int] = {
    "first-steps.md": 3,
    "first-steps-s2.md": 5,
    "mhall-svo-single.md": 1,
    "mhall-svo-equivalence.md": 5,
    "cascade-pressure.md": 5,
    "conflict-drill.md": 4,
    "s3-auto-countersign.md": 3,
}

_EXPECTED_STRUCTURE: dict[str, list[str]] = {
    "first-steps.md": [
        "### 1",
        "M",
        "### 2",
        "H",
        "### 3",
        "M == H",
    ],
    "first-steps-s2.md": [
        "### 1",
        "M",
        "### 2",
        "H",
        "### 3",
        "M == H",
        "### 4",
        "A",
        "### 5",
        "MH > H A",
    ],
    "mhall-svo-single.md": [
        "### 1",
        "MHALL == SVO =>",
        "   S = M",
        "   V = H",
        "   O = ALL =>",
        "     A > D",
        "     L > M",
        "     L > O",
    ],
    "mhall-svo-equivalence.md": [
        "### 1",
        "M",
        "H",
        "A",
        "L",
        "L",
        "S",
        "V",
        "O",
        "### 2",
        "M == H",
        "### 3",
        "ALL => A L L",
        "### 4",
        "M = S",
        "H = V",
        "ALL = O",
        "### 5",
        "MHALL == SVO =>",
        "   S = M",
        "   V = H",
        "   O = ALL =>",
        "     A > D",
        "     L > M",
        "     L > O",
    ],
    "cascade-pressure.md": [
        "### 1",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "### 2",
        "A == B",
        "C == D",
        "E == F",
        "G == H",
        "I == J",
        "### 3",
        "AB > C",
        "CD > E",
        "EF > G",
        "GH > I",
        "### 4",
        "ABCD > E F",
        "EFGH > I J",
        "ACEGI > B D F H J",
        "### 5",
        "ABCDEFGHIJ > A C E G",
    ],
    "conflict-drill.md": [
        "### 1",
        "A",
        "B",
        "C",
        "D",
        "### 2",
        "A == B",
        "### 3",
        "AB > D C",
        "### 4",
        "E",
        "ACE > B D",
    ],
    "s3-auto-countersign.md": [
        "### 1",
        "M",
        "H",
        "A",
        "P",
        "X",
        "### 2",
        "M == H",
        "### 3",
        "H > A",
        "A == A",
        "HPA => P X",
    ],
}


class TestCurriculumLessonCountGuard:
    """KB-333: durable snapshot guard over all 7 curriculum files.

    Two complementary assertions, each parametrised over every curriculum file:

    1. **Lesson count** (``test_lesson_count_matches_baseline``): the ``### <n>``
       header count must equal its audited baseline. Catches a whole-lesson
       add/drop -- the KB-320 class, where commit a1c40f1 collapsed
       first-steps-s2.md from 5 lessons to 2.
    2. **Normalised structure** (``test_normalized_structure_unchanged``): the
       annotation-stripped structural skeleton (headers + fenced-block klines)
       must equal its audited baseline. Catches an *in-lesson* kline truncation
       -- the KB-333 class, where a1c40f1 truncated mhall-svo-single.md's
       compound object node ``O = ALL`` down to ``O = A(ll)`` while the ``### 1``
       header (and thus the count) survived. Lesson count alone is blind to
       this; the structure snapshot is the only signal that catches it.

    Note on placement: this class lives in the NLP-gated module (per KB-333's
    spec) and so is skipped wherever NLPTokenizer data is absent -- the guard
    therefore runs in CI (which has the data). The assertions themselves are
    pure markdown inspection and need no tokenizer.
    """

    @pytest.mark.parametrize("curriculum_file", ALL_CURRICULUM_FILES)
    def test_lesson_count_matches_baseline(self, curriculum_file: str) -> None:
        path = CURRICULA_DIR / curriculum_file
        if not path.exists():
            pytest.skip(f"Curriculum not found: {path}")
        actual = _lesson_count(path)
        expected = _EXPECTED_LESSON_COUNTS[curriculum_file]
        assert actual == expected, (
            f"{curriculum_file}: lesson count drifted from {expected} to {actual}. "
            "If a lesson was deliberately added/removed, update "
            "_EXPECTED_LESSON_COUNTS (and _EXPECTED_STRUCTURE)."
        )

    @pytest.mark.parametrize("curriculum_file", ALL_CURRICULUM_FILES)
    def test_normalized_structure_unchanged(self, curriculum_file: str) -> None:
        path = CURRICULA_DIR / curriculum_file
        if not path.exists():
            pytest.skip(f"Curriculum not found: {path}")
        actual = _normalized_structure(path)
        expected = _EXPECTED_STRUCTURE[curriculum_file]
        if actual != expected:
            diff = "\n".join(
                difflib.unified_diff(
                    expected,
                    actual,
                    fromfile=f"expected ({curriculum_file})",
                    tofile=f"actual ({curriculum_file})",
                    lineterm="",
                )
            )
            pytest.fail(
                f"{curriculum_file}: normalised kscript structure drifted.\n{diff}\n\n"
                "If a kline/sig/node/lesson was deliberately changed, update "
                "_EXPECTED_STRUCTURE (and _EXPECTED_LESSON_COUNTS). Otherwise this "
                "flags a non-additive edit like the a1c40f1 truncation (KB-333)."
            )

    def test_guard_would_have_detected_kb320_truncation(self) -> None:
        """Sanity: the guard catches the KB-320 accident (commit a1c40f1
        collapsed first-steps-s2.md from 5 lessons to 2).

        The known-bad a1c40f1 blob has 2 headers (vs the baseline 5) and a
        different structure, so both guard assertions would fail on it.
        """
        bad = _git_blob("a1c40f1", "curricula/first-steps-s2.md")
        if bad is None:
            pytest.skip("commit a1c40f1 not available in this checkout")
        bad_count = sum(1 for line in bad.splitlines() if line.startswith("### "))
        assert bad_count == 2, (
            f"expected the known-bad a1c40f1 first-steps-s2.md blob to have 2 "
            f"headers (the KB-320 accident); got {bad_count}"
        )
        assert _EXPECTED_LESSON_COUNTS["first-steps-s2.md"] == 5
        assert _normalize_structure(bad) != _EXPECTED_STRUCTURE["first-steps-s2.md"]

    def test_guard_would_have_detected_kb333_in_lesson_truncation(self) -> None:
        """Sanity: the *structure* guard catches the KB-333 in-lesson
        truncation that the *count* guard misses.

        Commit a1c40f1 truncated mhall-svo-single.md's compound object node
        ``O = ALL`` down to ``O = A(ll)`` while the ``### 1`` header (and so
        the lesson count) stayed at 1. The count guard is therefore blind to
        this class; the normalised-structure snapshot is what flags it.
        """
        bad = _git_blob("a1c40f1", "curricula/mhall-svo-single.md")
        if bad is None:
            pytest.skip("commit a1c40f1 not available in this checkout")
        # The count was unchanged by this truncation -- proving count alone is blind.
        bad_count = sum(1 for line in bad.splitlines() if line.startswith("### "))
        assert bad_count == 1, (
            f"expected the a1c40f1 mhall-svo-single.md blob to still have 1 "
            f"lesson (the truncation was in-lesson); got {bad_count}"
        )
        bad_struct = _normalize_structure(bad)
        assert "   O = A =>" in bad_struct, (
            f"expected the a1c40f1 blob to carry the truncated `O = A` line; got {bad_struct}"
        )
        assert bad_struct != _EXPECTED_STRUCTURE["mhall-svo-single.md"], (
            "the normalised-structure guard must flag the KB-333 in-lesson "
            "truncation (O=ALL -> O=A)"
        )


# ── KB-334: parametrised misfit-routing guard across audited curricula ────

#: Curricula whose prose INTENDS S2/S3 misfit routing (the audited set). Each
#: must emit at least one candidate-bearing ``0 < significance < D_MAX`` event
#: when replayed through ``KAgent`` — the rule-47 enforcement (@curriculum rule
#: 47): a misfit-intending lesson must NOT express its misfit with the ``=>``
#: CANONIZED operator (canonical by construction → AGT-14 S1 short-circuit →
#: never reaches candidate retrieval). ``first-steps-s2.md`` is the KB-320
#: known-good reference. ``cascade-pressure.md`` and ``conflict-drill.md`` were
#: fixed by KB-334 (``=>`` → ``>`` CONNOTED). ``mhall-svo-*`` were already
#: correct (their leaf misfits use ``>``). Excluded: ``first-steps.md`` (no
#: S2/S3 intent — covered by the compile smoke) and ``s3-auto-countersign.md``
#: (its design is auto-countersign: the ``H > A`` misfit resolves to a
#: structural S1 by intent, so no mid-band event fires by design — covered by
#: ``tests/test_s3_auto_countersign.py``).
_S2_S3_CURRICULA = [
    "first-steps-s2.md",
    "mhall-svo-single.md",
    "mhall-svo-equivalence.md",
    "cascade-pressure.md",
    "conflict-drill.md",
]


class TestCurriculumMisfitRouting:
    """KB-334: every S2/S3-intending curriculum reaches the S2/S3 band.

    Generalises the KB-320 ``TestFirstStepsS2Routing`` mid-band assertion across
    the audited set of curricula whose prose intends misfit routing. This guards
    against the canonical-implies-not-a-misfit gap (rule 47): if a misfit lesson
    regresses to the ``=>`` CANONIZED operator, the compiled kline is canonical
    by construction and resolves S1 via the AGT-14 self-grounded short-circuit
    before any candidate is retrieved — emitting only S1/S4 events, never a
    candidate-bearing ``0 < significance < D_MAX`` expansion.
    """

    @pytest.mark.parametrize("curriculum_file", _S2_S3_CURRICULA)
    def test_curriculum_reaches_s2_s3_band(
        self, curriculum_file: str, nlp_tokenizer: NLPTokenizer
    ) -> None:
        """At least one candidate-bearing mid-band (S2/S3) event fires for a
        curriculum whose prose intends misfit routing."""
        from kalvin.expand import D_MAX

        path = CURRICULA_DIR / curriculum_file
        adapter, _lesson_entries = _replay_curriculum(path, nlp_tokenizer)

        expansions = [
            ev
            for ev in adapter.events
            if ev.kind == "frame" and ev.candidate is not None and 0 < ev.significance < D_MAX
        ]
        assert expansions, (
            f"expected at least one S2/S3 expansion event (candidate-bearing, "
            f"0 < significance < D_MAX) for {curriculum_file}; the curriculum is "
            f"not reaching the S2 band — a misfit lesson may be using the `=>` "
            f"CANONIZED operator (rule 47). "
            f"events={_event_summary(adapter.events)}"
        )
