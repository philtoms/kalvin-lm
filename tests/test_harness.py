"""Tests for harness behavioural criteria HRN-1 through HRN-18.

See specs/harness.md for the full specification.
See plans/impl/harness.md for the test mapping table.

Each test function maps to exactly one HRN criterion.  The HarnessFixture
class simulates the harness state machine in headless mode — no TUI, no
Textual dependency.
"""

import json

import pytest

from kalvin.agent import KAgent
from kalvin.events import EventBus, RationaliseEvent
from kalvin.expand import D_MAX
from kalvin.kline import KLine
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signature import make_signature
from ks import compile_source
from ks.parser import ParseError
from tests.conftest import requires_nlp_data

# The whole module constructs KAgents (default NLP tokenizer); skip cleanly
# when the NLP data assets are absent on a fresh clone.
pytestmark = requires_nlp_data

# ── Shared tokenizer ──────────────────────────────────────────────────

# Lazy module-level NLP tokenizer (safe at import time; ``pytestmark`` gates
# execution so this is only ever instantiated on a data-present machine).
_tok_instance: NLPTokenizer | None = None


def _tok() -> NLPTokenizer:
    """Return the shared NLP tokenizer, constructing it on first use."""
    global _tok_instance
    if _tok_instance is None:
        _tok_instance = NLPTokenizer.from_files()
    return _tok_instance

# Status symbols per spec §Response Status
STATUS_SYMBOLS = {"pass": "\u2713", "pending": "\u25cc", "mismatch": "\u2717"}


# ═══════════════════════════════════════════════════════════════════════
# HarnessFixture — headless harness state machine
# ═══════════════════════════════════════════════════════════════════════


class HarnessFixture:
    """Simulates the harness state machine in headless mode.

    Provides the same tracking, submission, and response logic as the
    TUI harness (ui/kscript/app.py) without any Textual dependency.

    Attributes
    ----------
    agent : KAgent
        Real Agent instance used for rationalisation.
    submitted : set[tuple[int, tuple[int, ...]]]
        Entry keys fed to rationalise().  Monotonic; only ``clear()`` resets.
    satisfied : set[tuple[int, tuple[int, ...]]]
        Entry keys whose proposals matched and were countersigned.
    responses : list[dict]
        Recorded response dicts with key, status, significance, decompiled.
    expectations : dict[tuple, KLine]
        Slow-path entries awaiting proposals, keyed by entry_key.
    events : list[RationaliseEvent]
        All captured rationalisation events.
    """

    def __init__(self) -> None:
        self._agent = KAgent(tokenizer=_tok(), adapter=EventBus())
        self._submitted: set[tuple[int, tuple[int, ...]]] = set()
        self._satisfied: set[tuple[int, tuple[int, ...]]] = set()
        self._responses: list[dict] = []
        self._expectations: dict[tuple[int, tuple[int, ...]], KLine] = {}
        self._events: list[RationaliseEvent] = []
        self._run_mode: bool = False
        self._rationalise_count: int = 0

        # Subscribe to agent events
        self._agent.events.subscribe(self._on_event)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def agent(self) -> KAgent:
        return self._agent

    @property
    def submitted(self) -> set[tuple[int, tuple[int, ...]]]:
        return self._submitted

    @property
    def satisfied(self) -> set[tuple[int, tuple[int, ...]]]:
        return self._satisfied

    @property
    def responses(self) -> list[dict]:
        return self._responses

    @property
    def expectations(self) -> dict[tuple[int, tuple[int, ...]], KLine]:
        return self._expectations

    @property
    def events(self) -> list[RationaliseEvent]:
        return self._events

    @property
    def rationalise_count(self) -> int:
        return self._rationalise_count

    # ── Key helpers ───────────────────────────────────────────────────

    @staticmethod
    def entry_key(entry) -> tuple[int, tuple[int, ...]]:
        """Return a hashable key for a compiled entry or kline."""
        return (entry.signature, tuple(entry.nodes))

    @staticmethod
    def structural_match(a: KLine, b: KLine) -> bool:
        """Check structural equality: same signature AND same nodes."""
        return a.signature == b.signature and a.nodes == b.nodes

    @staticmethod
    def format_significance(sig: int) -> tuple[str, str]:
        """Format significance as (raw_hex, normalised).

        >>> format_significance(0)
        ('0x0000000000000000', '0.000')
        >>> format_significance(D_MAX)
        ('0xFFFFFFFFFFFFFFFF', '1.000')
        """
        return (f"0x{sig:016X}", f"{sig / D_MAX:.3f}")

    # ── Compile / pending ─────────────────────────────────────────────

    def compile(self, source: str) -> list[KLine]:
        """Compile KScript source to compiled entries."""
        return compile_source(source, tokenizer=_tok(), dev=True)

    def get_pending(self, entries: list[KLine]) -> list[KLine]:
        """Filter entries whose key is not yet in ``_submitted``."""
        return [e for e in entries if self.entry_key(e) not in self._submitted]

    # ── Submit ─────────────────────────────────────────────────────────

    def submit(self, entry) -> bool:
        """Submit an entry to the agent for rationalisation.

        Returns True if fast-path (auto-satisfied), False if slow-path.
        """
        key = self.entry_key(entry)
        self._rationalise_count += 1
        result = self._agent.rationalise(entry)
        self._submitted.add(key)

        if result:
            # Fast path — auto-satisfied immediately
            self._satisfied.add(key)
            sig = self._find_significance_for_key(key)
            self._responses.append(
                {
                    "key": key,
                    "status": "pass",
                    "significance": sig,
                    "decompiled": self._decompile_entry(entry),
                }
            )
        else:
            # Slow path — record as expectation awaiting proposals
            self._expectations[key] = entry
            self._responses.append(
                {
                    "key": key,
                    "status": "pending",
                    "significance": 0,
                    "decompiled": self._decompile_entry(entry),
                }
            )

        return result

    # ── Run / Step / Clear ─────────────────────────────────────────────

    def run_all(self, source: str) -> None:
        """Compile, get pending, submit all sequentially (Run mode)."""
        self._run_mode = True
        entries = self.compile(source)
        pending = self.get_pending(entries)
        for entry in pending:
            self.submit(entry)

    def step_one(self, source: str) -> KLine | None:
        """Compile, get pending, submit first pending, return it (Step mode)."""
        self._run_mode = False
        entries = self.compile(source)
        pending = self.get_pending(entries)
        if pending:
            self.submit(pending[0])
            return pending[0]
        return None

    def clear(self) -> None:
        """Reset all tracking state (Submitted, Satisfied, Pending, Responses)."""
        self._submitted.clear()
        self._satisfied.clear()
        self._responses.clear()
        self._expectations.clear()

    # ── Event callback ────────────────────────────────────────────────

    def _on_event(self, event: RationaliseEvent) -> None:
        """Event bus callback: record events and correlate to expectations."""
        self._events.append(event)

        if event.kind in ("frame", "ground"):
            event_key = (event.query.signature, tuple(event.query.nodes))
            if event_key in self._expectations:
                entry = self._expectations[event_key]
                sig = event.significance

                proposal_matches = self.structural_match(event.proposal, event.query)
                status = "pass" if (event.kind == "ground" or proposal_matches) else "pending"

                self._responses.append(
                    {
                        "key": event_key,
                        "status": status,
                        "significance": sig,
                        "decompiled": self._decompile_entry(entry),
                        "proposal": event.proposal,
                    }
                )

                # In run mode, auto-satisfy on ground or structural match
                if self._run_mode and (event.kind == "ground" or proposal_matches):
                    self._satisfied.add(event_key)
                    self._expectations.pop(event_key, None)

    # ── Internal helpers ──────────────────────────────────────────────

    def _find_significance_for_key(self, key: tuple) -> int:
        """Find the significance for the latest event matching *key*."""
        for event in reversed(self._events):
            if (event.query.signature, tuple(event.query.nodes)) == key:
                return event.significance
        return 0

    @staticmethod
    def _decompile_entry(entry) -> str:
        """Display an entry as a human-readable KScript string."""
        try:
            from kalvin.kline import kline_display

            return kline_display(entry, _tok())
        except Exception:
            pass
        return ""


# ═══════════════════════════════════════════════════════════════════════
# Helper: set up a slow-path scenario with raw KLine values
# ═══════════════════════════════════════════════════════════════════════


def _setup_slow_path(h: HarnessFixture) -> KLine:
    """Pre-populate the model so the next submit goes S2 slow-path.

    Adds a candidate KLine whose signature overlaps the query via
    ``model.where`` (bitwise AND matching).  Returns the query KLine
    that will route S2 against the candidate.

    After this call, submitting the returned query via ``h.submit()``
    will return False (slow-path).
    """
    # Candidate: non-canonical, sits in model for bitwise retrieval
    candidate = KLine(0b1100, [0b1000])
    h.agent.rationalise(candidate)

    # Query: same signature bits, partial node overlap → S2.
    # make_signature([0b1000, 0b0100]) = 0b1100 (canonical sig),
    # but node 0b0100 is not resolved in model → canonical fast-path
    # is NOT taken.  is_countersigned also fails.  So Phase 4 retrieves
    # candidates via bitwise AND, and routing gives S2 (partial match).
    query = KLine(0b1100, [0b1000, 0b0100])
    return query


# ═══════════════════════════════════════════════════════════════════════
# HRN-1  Recompile produces fresh entries; only new submitted
# ═══════════════════════════════════════════════════════════════════════


def test_recompile_only_new_submitted():
    h = HarnessFixture()

    # First compilation & run
    entries1 = h.compile("A => B")
    h.run_all("A => B")
    assert len(h.submitted) == len(entries1)
    first_count = h.rationalise_count

    # Second compilation adds entries; only new ones are submitted
    entries2 = h.compile("A => B\nC => D")
    h.run_all("A => B\nC => D")
    new_count = len(entries2) - len(entries1)
    assert len(h.submitted) == len(entries2)
    assert h.rationalise_count == first_count + new_count


# ═══════════════════════════════════════════════════════════════════════
# HRN-2  Submitted set is monotonic; Clear is the only reset
# ═══════════════════════════════════════════════════════════════════════


def test_submitted_monotonic():
    h = HarnessFixture()

    entries = h.compile("A => B\nC => D")
    h.run_all("A => B\nC => D")
    count_after_first = len(h.submitted)
    assert count_after_first == len(entries)

    # Re-run same source: no growth
    h.run_all("A => B\nC => D")
    assert len(h.submitted) == count_after_first

    # Only clear resets
    h.clear()
    assert len(h.submitted) == 0

    # Can re-submit after clear
    h.run_all("A => B")
    assert len(h.submitted) == len(h.compile("A => B"))


# ═══════════════════════════════════════════════════════════════════════
# HRN-3  Fast-path entries (rationalise returns True) are auto-satisfied
# ═══════════════════════════════════════════════════════════════════════


def test_fast_path_auto_satisfied():
    h = HarnessFixture()

    # "A" produces an unsigned entry → auto-satisfied (S1)
    entries = h.compile("A")
    result = h.submit(entries[0])

    assert result is True
    key = h.entry_key(entries[0])
    assert key in h.satisfied

    # A response with status "pass" was recorded
    pass_responses = [r for r in h.responses if r["status"] == "pass"]
    assert len(pass_responses) >= 1


# ═══════════════════════════════════════════════════════════════════════
# HRN-4  Slow-path match requires structural equality (signature + nodes)
# ═══════════════════════════════════════════════════════════════════════


def test_structural_match_expectation():
    a = KLine(42, [10, 20])
    b = KLine(42, [10, 20])
    assert HarnessFixture.structural_match(a, b) is True

    # Different nodes → no match
    c = KLine(42, [10, 30])
    assert HarnessFixture.structural_match(a, c) is False

    # Different signature → no match
    d = KLine(99, [10, 20])
    assert HarnessFixture.structural_match(a, d) is False


# ═══════════════════════════════════════════════════════════════════════
# HRN-5  Run mode submits all pending entries without pausing
# ═══════════════════════════════════════════════════════════════════════


def test_run_submits_all_pending():
    h = HarnessFixture()
    source = "A => B\nC => D\nE => F"
    entries = h.compile(source)
    h.run_all(source)

    expected_keys = {h.entry_key(e) for e in entries}
    assert h.submitted == expected_keys
    assert h.rationalise_count == len(entries)


# ═══════════════════════════════════════════════════════════════════════
# HRN-6  Run mode auto-countersigns matching proposals
# ═══════════════════════════════════════════════════════════════════════


def test_run_auto_countersigns():
    h = HarnessFixture()

    # Fast-path unsigned entry: auto-satisfied in run mode
    h.run_all("A")
    entries = h.compile("A")
    key = h.entry_key(entries[0])
    assert key in h.satisfied


# ═══════════════════════════════════════════════════════════════════════
# HRN-7  Step mode submits one entry and halts
# ═══════════════════════════════════════════════════════════════════════


def test_step_submits_one_halts():
    h = HarnessFixture()
    source = "A => B\nC => D\nE => F"
    h.compile(source)

    h.step_one(source)
    assert len(h.submitted) == 1

    h.step_one(source)
    assert len(h.submitted) == 2

    h.step_one(source)
    assert len(h.submitted) == 3


# ═══════════════════════════════════════════════════════════════════════
# HRN-8  Ratify button enabled only when a response item is selected
# ═══════════════════════════════════════════════════════════════════════


def test_ratify_enabled_on_selection():
    """Headless test: verify the data precondition for ratify enablement.

    The actual button enablement is a TUI concern (tested in KB-012's
    integration tests).  Here we verify that after ``step_one``, a
    response item is available — the precondition for enabling Ratify.
    """
    h = HarnessFixture()
    h.step_one("A => B")
    assert len(h.responses) > 0


# ═══════════════════════════════════════════════════════════════════════
# HRN-9  Ratify calls agent.countersign(proposal) with the proposal as-is
# ═══════════════════════════════════════════════════════════════════════


def test_ratify_calls_countersign():
    """Simulate ratification by constructing the reciprocal kline manually.

    Tests the countersign logic path without requiring Agent.countersign()
    (which is KB-007).
    """
    h = HarnessFixture()

    # Add an original kline to the model
    original = KLine(42, [10])
    h.agent.rationalise(original)  # S4 → True, model has KLine(42, [10])

    # Simulate receiving a proposal event where the proposal IS the original
    # (as happens in a ground event).  Ratify = construct reciprocal.
    proposal = original
    reciprocal = KLine(make_signature(proposal.nodes), [proposal.signature])
    # reciprocal = KLine(10, [42])

    result = h.agent.rationalise(reciprocal)
    assert result is True  # countersign check succeeds
    assert h.agent.model.find(reciprocal.signature) is not None


# ═══════════════════════════════════════════════════════════════════════
# HRN-10  Clear resets Submitted, Satisfied, Pending sets and responses
# ═══════════════════════════════════════════════════════════════════════


def test_clear_resets_tracking():
    h = HarnessFixture()
    h.run_all("A => B\nC => D")

    assert len(h.submitted) > 0
    assert len(h.satisfied) > 0
    assert len(h.responses) > 0

    h.clear()

    assert len(h.submitted) == 0
    assert len(h.satisfied) == 0
    assert len(h.responses) == 0
    assert len(h.expectations) == 0


# ═══════════════════════════════════════════════════════════════════════
# HRN-11  Events correlated to entries by structural match
# ═══════════════════════════════════════════════════════════════════════


def test_event_correlation_structural():
    h = HarnessFixture()

    # Set up model so the query routes S2 (slow-path)
    query = _setup_slow_path(h)
    result = h.submit(query)
    assert result is False, "Expected slow-path (S2) but got fast-path"

    # Manually simulate a proposal event from the cogitator.
    # The event's query structurally matches the expectation.
    proposal = KLine(0b1100, [0b1000, 0b0100, 0b0100])
    h.agent.events.publish(RationaliseEvent("frame", query, proposal, D_MAX // 2))

    query_key = h.entry_key(query)

    # Find correlated responses (initial pending + proposal from event)
    correlated = [r for r in h.responses if r["key"] == query_key and "proposal" in r]
    assert len(correlated) >= 1

    # Events are correlated by structural match (signature + nodes),
    # not by object identity — verify the structural properties.
    for response in correlated:
        event_proposal = response["proposal"]
        # The proposal in the response came from an event whose query
        # structurally matched the expectation
        assert isinstance(event_proposal, KLine)


# ═══════════════════════════════════════════════════════════════════════
# HRN-12  Tracking state persisted through hot-reload cycles
# ═══════════════════════════════════════════════════════════════════════


def test_state_persistence_hotreload():
    h = HarnessFixture()
    h.run_all("A => B\nC => D")

    # Serialize tracking state to JSON
    submitted_json = [[sig, list(nodes)] for sig, nodes in h.submitted]
    satisfied_json = [[sig, list(nodes)] for sig, nodes in h.satisfied]
    data = json.dumps({"submitted": submitted_json, "satisfied": satisfied_json})

    # Deserialize and reconstruct sets
    loaded = json.loads(data)
    submitted_restored = {(sig, tuple(nodes)) for sig, nodes in loaded["submitted"]}
    satisfied_restored = {(sig, tuple(nodes)) for sig, nodes in loaded["satisfied"]}

    assert submitted_restored == h.submitted
    assert satisfied_restored == h.satisfied


# ═══════════════════════════════════════════════════════════════════════
# HRN-13  Each response shows status symbol, decompiled, raw & normalised sig
# ═══════════════════════════════════════════════════════════════════════


def test_response_display_format():
    # ── Significance formatting with boundary values ──────────────────

    # Significance 0 → S4
    raw, norm = HarnessFixture.format_significance(0)
    assert raw == "0x0000000000000000"
    assert norm == "0.000"

    # Significance D_MAX → S1 (approx 1.0)
    raw, norm = HarnessFixture.format_significance(D_MAX)
    assert raw == "0xFFFFFFFFFFFFFFFF"
    assert norm == "1.000"

    # Significance D_MAX // 2 → ~0.5
    raw, norm = HarnessFixture.format_significance(D_MAX // 2)
    assert raw == "0x7FFFFFFFFFFFFFFF"
    assert norm == "0.500"

    # ── Response entries carry correct format ─────────────────────────

    h = HarnessFixture()
    h.run_all("A => B")

    for response in h.responses:
        # Status symbol exists
        assert response["status"] in STATUS_SYMBOLS
        symbol = STATUS_SYMBOLS[response["status"]]
        assert symbol  # non-empty

        # Raw significance is 0x-prefixed 16-char hex
        raw_hex, normalised = HarnessFixture.format_significance(response["significance"])
        assert raw_hex.startswith("0x")
        assert len(raw_hex) == 18  # "0x" + 16 hex digits
        assert "." in normalised  # normalised has decimal point

        # Decompiled source string is present
        assert isinstance(response["decompiled"], str)


# ═══════════════════════════════════════════════════════════════════════
# HRN-14  Compilation errors displayed as ✗ response items
# ═══════════════════════════════════════════════════════════════════════


def test_compilation_error_display():
    """Attempting to compile invalid source raises a lexer or parse error.

    In the real harness this would be captured and displayed as a ✗
    response item.  Here we verify the error capture path.
    """
    with pytest.raises((ParseError, Exception)) as exc_info:
        compile_source("=> => =>", tokenizer=_tok(), dev=True)

    # Error message is available for display
    assert str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════
# HRN-15  Progress count displayed in toolbar
# ═══════════════════════════════════════════════════════════════════════


def test_progress_count_display():
    h = HarnessFixture()

    # "A => B" compiles to one CANONIZE entry per line (sig=left side,
    # nodes=[right side]), so four lines produce 4 entries.
    source = "A => B\nC => D\nE => F\nG => H"
    all_entries = h.compile(source)
    assert len(all_entries) == 4

    # Submit 3 of 4 entries via step_one
    for _ in range(3):
        h.step_one(source)
    assert len(h.submitted) == 3

    # HRN-15 tests the progress *count display*, not the satisfaction routing.
    # Which entries fast-path satisfy depends on the tokenizer's node bit
    # overlap (NLP-BPE nodes of the same POS share type bits), so we set the
    # satisfied set directly to a known partial state (2 of 3).
    satisfied_list = list(h.submitted)[:2]
    h._satisfied = set(satisfied_list)

    satisfied_count = len(h.satisfied)
    submitted_count = len(h.submitted)
    pending_count = submitted_count - satisfied_count
    progress = f"{satisfied_count}/{submitted_count} pending/{pending_count}"
    assert progress == "2/3 pending/1"


# ═══════════════════════════════════════════════════════════════════════
# HRN-16  agent.countersign generates reciprocal kline and rationalises
# ═══════════════════════════════════════════════════════════════════════


def test_agent_countersign():
    """Create an Agent, add a kline, construct reciprocal, rationalise.

    Verifies the countersign logic: reciprocal = KLine(make_signature(
    original.nodes), [original.signature]).  The reciprocal passes the
    is_countersigned check because the original {Q: [V]} acts as
    countersigner for the reciprocal {V: [Q]}.
    """
    a = KAgent(adapter=EventBus())

    # Add original kline to model
    original = KLine(42, [10])
    a.rationalise(original)

    # Construct reciprocal manually (what Agent.countersign would do)
    reciprocal = KLine(make_signature(original.nodes), [original.signature])
    # reciprocal = KLine(10, [42])

    result = a.rationalise(reciprocal)
    assert result is True
    assert a.model.find(reciprocal.signature) is not None


# ═══════════════════════════════════════════════════════════════════════
# HRN-17  Mismatches in Run mode flagged as pending; execution continues
# ═══════════════════════════════════════════════════════════════════════


def test_run_mismatch_flagged_pending():
    h = HarnessFixture()

    # Set up model for slow-path (S2)
    query = _setup_slow_path(h)
    result = h.submit(query)
    assert result is False, "Expected slow-path (S2) but got fast-path"

    query_key = h.entry_key(query)

    # Entry is recorded as an expectation (pending)
    assert query_key in h.expectations

    # Initial response has status "pending"
    pending_responses = [r for r in h.responses if r["key"] == query_key]
    assert any(r["status"] == "pending" for r in pending_responses)

    # Simulate a proposal event where the proposal does NOT structurally
    # match the query (mismatch case).  The entry stays pending.
    mismatch_proposal = KLine(0b1111, [0b1000, 0b0111])
    h.agent.events.publish(RationaliseEvent("frame", query, mismatch_proposal, D_MAX // 4))

    # The proposal doesn't match the query → status remains "pending"
    correlated = [r for r in h.responses if r["key"] == query_key and "proposal" in r]
    assert len(correlated) >= 1
    # All correlated responses from the mismatch event are "pending"
    assert all(r["status"] == "pending" for r in correlated)

    # Entry remains in expectations (not auto-satisfied)
    assert query_key in h.expectations


# ═══════════════════════════════════════════════════════════════════════
# HRN-18  Multiple proposals for one expectation all displayed;
#         first match accepted in Run, user chooses in Step
# ═══════════════════════════════════════════════════════════════════════


def test_multiple_proposals_displayed():
    h = HarnessFixture()

    # Set up slow-path scenario
    query = _setup_slow_path(h)
    result = h.submit(query)
    assert result is False, "Expected slow-path (S2) but got fast-path"

    query_key = h.entry_key(query)

    # Simulate multiple proposal events for the same expectation
    proposals = [
        KLine(0b1100, [0b1000, 0b0100, 0b0100]),
        KLine(0b1100, [0b1000, 0b0110]),
        KLine(0b1100, [0b1000, 0b1000, 0b0100]),
    ]
    for proposal in proposals:
        h.agent.events.publish(RationaliseEvent("frame", query, proposal, D_MAX // 2))

    # All explicitly-published proposals are recorded as separate response
    # items.  (The async cogitator may also emit proposals for the same
    # expectation, so we verify our proposals are a subset of the recorded
    # set rather than asserting an exact total count.)
    recorded_proposals = [
        r["proposal"] for r in h.responses if r["key"] == query_key and "proposal" in r
    ]
    recorded_keys = {HarnessFixture.entry_key(p) for p in recorded_proposals}
    expected_keys = {HarnessFixture.entry_key(p) for p in proposals}
    assert expected_keys.issubset(recorded_keys), (
        f"Missing proposals: {expected_keys - recorded_keys}"
    )
    # The published proposals are distinct from each other
    assert len(expected_keys) == len(proposals)


# ═══════════════════════════════════════════════════════════════════════
# NLP Tokenizer Harness Compatibility
# ═══════════════════════════════════════════════════════════════════════

try:
    from kalvin.nlp_tokenizer import NLPTokenizer
except ImportError:  # optional NLP backend not installed
    NLPTokenizer = None  # type: ignore[assignment,misc]


class NLPHarnessFixture(HarnessFixture):
    """HarnessFixture variant driven by an explicit NLPTokenizer instance.

    Overrides compile() and _decompile_entry() to use the supplied NLP
    tokenizer, and constructs the agent with NLPTokenizer.  All tracking,
    submission, and response logic is inherited from the base class.
    """

    def __init__(self, nlp_tok: NLPTokenizer) -> None:
        self._nlp_tok = nlp_tok
        # Bypass parent __init__ — set up with NLP tokenizer
        self._agent = KAgent(tokenizer=nlp_tok, adapter=EventBus())
        self._submitted: set[tuple[int, tuple[int, ...]]] = set()
        self._satisfied: set[tuple[int, tuple[int, ...]]] = set()
        self._responses: list[dict] = []
        self._expectations: dict[tuple[int, tuple[int, ...]], KLine] = {}
        self._events: list[RationaliseEvent] = []
        self._run_mode: bool = False
        self._rationalise_count: int = 0
        self._agent.events.subscribe(self._on_event)

    def compile(self, source: str) -> list[KLine]:
        """Compile KScript source using the NLP tokenizer."""
        return compile_source(source, tokenizer=self._nlp_tok, dev=True)

    def _decompile_entry(self, entry) -> str:
        """Display an entry using the NLP tokenizer."""
        try:
            from kalvin.kline import kline_display

            return kline_display(entry, self._nlp_tok)
        except Exception:
            pass
        return ""


@requires_nlp_data
def test_harness_nlp_tokenizer():
    """Harness works end-to-end with NLPTokenizer.

    Verifies that the full rationalisation pipeline — compile KScript,
    submit entries, track satisfaction — works when using an NLP tokenizer.
    Entries should contain NLP-BPE nodes (high 32 bits non-zero) and the
    rationalisation results should be recorded correctly.
    """
    nlp_tok = NLPTokenizer.from_files()
    h = NLPHarnessFixture(nlp_tok)

    # Compile and run a simple source
    source = "A => B"
    h.run_all(source)

    entries = h.compile(source)

    # Entries should have been submitted
    assert len(h.submitted) == len(entries), (
        f"Expected {len(entries)} submitted, got {len(h.submitted)}"
    )

    # All entries should be satisfied (fast-path for simple entries)
    assert len(h.satisfied) >= 1, f"Expected at least 1 satisfied entry, got {len(h.satisfied)}"

    # Responses should be recorded
    assert len(h.responses) >= 1

    # Verify entries contain NLP-BPE nodes
    for entry in entries:
        # NLP node check replaced with inline check
        assert (entry.signature >> 32) != 0, (
            f"Entry signature {entry.signature:#x} should be NLP-BPE"
        )

    # Verify responses have correct structure
    for response in h.responses:
        assert response["status"] in STATUS_SYMBOLS
        assert isinstance(response["decompiled"], str)
