"""Tests for harness event correlation and satisfaction logic (KB-010).

Tests cover HRN-3, HRN-4, HRN-11, HRN-14, HRN-17, HRN-18:
- Structural matching of KLines by signature + nodes
- Event correlation to compiled entries
- Fast-path auto-satisfaction
- Slow-path proposal display
- Compilation error display
- Mismatch flagging as pending
- Multiple proposals per expectation
- Done event handling
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from kalvin.abstract import KLine
from kalvin.events import RationaliseEvent
from kscript import CompiledEntry


# ── Structural Match Tests ──────────────────────────────────────────────


class TestStructuralMatch:
    """Test KScriptApp._structural_match static method."""

    def test_identical_klines_match(self):
        """Two KLines with same signature and same nodes → True."""
        from ui.kscript.app import KScriptApp

        a = KLine(0x42, [1, 2, 3])
        b = KLine(0x42, [1, 2, 3])
        assert KScriptApp._structural_match(a, b) is True

    def test_different_signature_no_match(self):
        """Same nodes, different signature → False."""
        from ui.kscript.app import KScriptApp

        a = KLine(0x42, [1, 2, 3])
        b = KLine(0x99, [1, 2, 3])
        assert KScriptApp._structural_match(a, b) is False

    def test_different_nodes_no_match(self):
        """Same signature, different nodes → False."""
        from ui.kscript.app import KScriptApp

        a = KLine(0x42, [1, 2, 3])
        b = KLine(0x42, [4, 5, 6])
        assert KScriptApp._structural_match(a, b) is False

    def test_empty_nodes_match(self):
        """Two empty-node KLines with same signature → True."""
        from ui.kscript.app import KScriptApp

        a = KLine(0x42, [])
        b = KLine(0x42, [])
        assert KScriptApp._structural_match(a, b) is True

    def test_different_node_count_no_match(self):
        """Same signature, different node counts → False."""
        from ui.kscript.app import KScriptApp

        a = KLine(0x42, [1, 2])
        b = KLine(0x42, [1, 2, 3])
        assert KScriptApp._structural_match(a, b) is False


# ── Test Helper ─────────────────────────────────────────────────────────


def _make_app():
    """Create a KScriptApp with mocked Textual UI for testing.

    Returns (app, mock_responses) where mock_responses is the
    MagicMock that stands in for the ResponsesRegion.
    """
    with patch("ui.kscript.app.Agent"):
        from ui.kscript.app import KScriptApp
        app = KScriptApp.__new__(KScriptApp)
        # Initialize only the fields we need
        app._dev_mode = True
        app._agent = MagicMock()
        app._decompiler = MagicMock()
        app._execution_state = None
        app._pending_entries = []
        app._current_entry_index = 0
        app._cancelled = False
        app._last_script_dir = None
        app._last_state_dir = None
        app._auto_compile_interval = 1.0
        app._submitted = set()
        app._satisfied = set()
        app._compiled_entries = []
        app._expectations = {}
        app._fast_path_results = {}

        # Mock query_one to return a mock ResponsesRegion
        mock_responses = MagicMock()
        app.query_one = MagicMock(return_value=mock_responses)
        # Mock logger — App.log is a property returning self._logger
        app._logger = MagicMock()

    return app, mock_responses


def _make_entry(sig: int, nodes: list[int]) -> CompiledEntry:
    """Create a CompiledEntry with given signature and nodes."""
    return CompiledEntry(signature=sig, nodes=nodes)


# ── Event Correlation Tests ────────────────────────────────────────────


class TestEventCorrelation:
    """Test event-to-entry correlation (HRN-11)."""

    def test_event_correlation_matches_entry(self):
        """HRN-11: Event with query matching a compiled entry is correlated."""
        app, mock_responses = _make_app()

        entry = _make_entry(0x42, [1, 2, 3])
        app._compiled_entries = [entry]

        # Mock decompiler to return a valid entry
        from kscript.decompiler import DecompiledEntry
        app._decompiler.decompile.return_value = [
            DecompiledEntry(level="S1", sig="SIG", nodes=["A", "B"])
        ]

        # Publish a ground event whose query matches the entry
        event = RationaliseEvent(
            kind="ground",
            query=KLine(0x42, [1, 2, 3]),
            proposal=KLine(0x42, [1, 2, 3]),  # fast-path: query == proposal
            significance=0xFFFF_FFFF_FFFF_FFFF,
        )

        # Subscribe the event handler
        callback = app._agent.events.subscribe.call_args[0][0] \
            if app._agent.events.subscribe.called else None

        # Manually call _setup_events to get the callback
        app._setup_events()
        callback = app._agent.events.subscribe.call_args[0][0]

        callback(event)

        # Entry should be auto-satisfied (fast-path)
        key = app._entry_key(entry)
        assert key in app._satisfied

    def test_done_event_ignored(self):
        """Done events should not trigger any response or state change."""
        app, mock_responses = _make_app()
        app._setup_events()

        callback = app._agent.events.subscribe.call_args[0][0]

        initial_satisfied = set(app._satisfied)
        initial_submitted = set(app._submitted)

        event = RationaliseEvent(
            kind="done",
            query=KLine(0, []),
            proposal=KLine(0, []),
            significance=0,
        )
        callback(event)

        # No state changes
        assert app._satisfied == initial_satisfied
        assert app._submitted == initial_submitted
        # No response added
        mock_responses.add_response.assert_not_called()

    def test_unknown_event_kind_ignored(self):
        """Unknown event kinds should be logged and ignored."""
        app, mock_responses = _make_app()
        app._setup_events()
        callback = app._agent.events.subscribe.call_args[0][0]

        event = RationaliseEvent(
            kind="unknown_kind",
            query=KLine(0x10, [1]),
            proposal=KLine(0x10, [1]),
            significance=100,
        )
        callback(event)

        # No response added
        mock_responses.add_response.assert_not_called()
        # Was logged (App.log is a property that returns self._logger)
        app._logger.assert_called()


# ── Fast Path Tests ───────────────────────────────────────────────────


class TestFastPath:
    """Test fast-path auto-satisfaction (HRN-3)."""

    def test_fast_path_auto_satisfied(self):
        """HRN-3: Fast-path entry (query==proposal) auto-satisfied with status 'pass'."""
        app, mock_responses = _make_app()

        entry = _make_entry(0x50, [10, 20])
        app._compiled_entries = [entry]

        # Mock decompiler to return a valid entry
        from kscript.decompiler import DecompiledEntry
        app._decompiler.decompile.return_value = [
            DecompiledEntry(level="S1", sig="SIG", nodes=["A", "B"])
        ]

        app._setup_events()
        callback = app._agent.events.subscribe.call_args[0][0]

        # Fast-path event: query == proposal
        event = RationaliseEvent(
            kind="ground",
            query=KLine(0x50, [10, 20]),
            proposal=KLine(0x50, [10, 20]),
            significance=0xFFFF_FFFF_FFFF_FFFF,
        )
        callback(event)

        key = app._entry_key(entry)
        assert key in app._satisfied

        # Response should be added with status "pass"
        mock_responses.add_response.assert_called()
        call_kwargs = mock_responses.add_response.call_args
        assert call_kwargs.kwargs["status"] == "pass"


# ── Slow Path Tests ───────────────────────────────────────────────────


class TestSlowPath:
    """Test slow-path proposal handling (HRN-4, HRN-17)."""

    def test_slow_path_proposal_displayed(self):
        """HRN-4: Slow-path proposal displayed with status 'pending'."""
        app, mock_responses = _make_app()

        entry = _make_entry(0x60, [100, 200])
        app._compiled_entries = [entry]

        # Mock decompiler to return a valid entry
        from kscript.decompiler import DecompiledEntry
        app._decompiler.decompile.return_value = [
            DecompiledEntry(level="S2", sig="SIG", nodes=["A", "B"])
        ]

        app._setup_events()
        callback = app._agent.events.subscribe.call_args[0][0]

        # Slow-path event: query != proposal
        event = RationaliseEvent(
            kind="frame",
            query=KLine(0x60, [100, 200]),
            proposal=KLine(0x60, [100, 999]),  # different proposal
            significance=0x8000_0000_0000_0000,
        )
        callback(event)

        key = app._entry_key(entry)
        assert key in app._expectations
        assert len(app._expectations[key]) == 1

        # Response should be added with status "pending"
        mock_responses.add_response.assert_called()
        call_kwargs = mock_responses.add_response.call_args
        assert call_kwargs.kwargs["status"] == "pending"

    def test_slow_path_mismatch_flagged_pending(self):
        """HRN-17: Slow-path mismatch flagged as pending, not hard failure."""
        app, mock_responses = _make_app()

        # No compiled entries match — simulates mismatch scenario
        app._compiled_entries = []

        # Mock decompiler
        from kscript.decompiler import DecompiledEntry
        app._decompiler.decompile.return_value = [
            DecompiledEntry(level="S3", sig="UNK", nodes=["X"])
        ]

        app._setup_events()
        callback = app._agent.events.subscribe.call_args[0][0]

        # Event with no matching expectation
        event = RationaliseEvent(
            kind="frame",
            query=KLine(0x99, [500]),
            proposal=KLine(0x99, [600]),  # slow-path: query != proposal
            significance=0x1000_0000_0000_0000,
        )
        callback(event)

        # Response should be "pending" (NOT "pass" or "mismatch")
        mock_responses.add_response.assert_called()
        call_kwargs = mock_responses.add_response.call_args
        assert call_kwargs.kwargs["status"] == "pending"


# ── Compilation Error Tests ───────────────────────────────────────────


class TestCompilationError:
    """Test compilation error display (HRN-14)."""

    def test_compilation_error_displayed_as_mismatch(self):
        """HRN-14: Compilation error displayed as ✗ response (status='mismatch')."""
        app, mock_responses = _make_app()

        app._show_compilation_error("test error: bad syntax")

        mock_responses.add_response.assert_called_once_with(
            level="S4",
            decompiled_source="test error: bad syntax",
            status="mismatch",
            significance=0,
            kline=None,
            entry_key=None,
        )


# ── Multiple Proposals Tests ──────────────────────────────────────────


class TestMultipleProposals:
    """Test multiple proposals per expectation (HRN-18)."""

    def test_multiple_proposals_all_displayed(self):
        """HRN-18: Multiple proposals for one expectation are all displayed."""
        app, mock_responses = _make_app()

        entry = _make_entry(0x70, [1, 2])
        app._compiled_entries = [entry]

        # Mock decompiler to return entries for each proposal
        from kscript.decompiler import DecompiledEntry
        app._decompiler.decompile.side_effect = [
            [DecompiledEntry(level="S2", sig="A", nodes=["X"])],
            [DecompiledEntry(level="S3", sig="A", nodes=["Y"])],
        ]

        app._setup_events()
        callback = app._agent.events.subscribe.call_args[0][0]

        # First proposal event
        event1 = RationaliseEvent(
            kind="frame",
            query=KLine(0x70, [1, 2]),
            proposal=KLine(0x70, [1, 99]),
            significance=0x8000_0000_0000_0000,
        )
        callback(event1)

        # Second proposal event for the same expectation
        event2 = RationaliseEvent(
            kind="frame",
            query=KLine(0x70, [1, 2]),
            proposal=KLine(0x70, [1, 88]),
            significance=0x7000_0000_0000_0000,
        )
        callback(event2)

        key = app._entry_key(entry)
        assert key in app._expectations
        assert len(app._expectations[key]) == 2

        # Two response items added
        assert mock_responses.add_response.call_count == 2
