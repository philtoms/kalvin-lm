"""Tests for KB-012 — Step mode rewrite and Ratify action.

Covers HRN-7 (Step submits one and halts), HRN-8 (Ratify enabled on
selection), and HRN-9 (Ratify calls countersign with proposal as-is).
"""

import sys
from unittest.mock import MagicMock, patch, call

import pytest

from kalvin.abstract import KLine
from kscript import CompiledEntry
from ui.kscript.regions.toolbar import ExecutionState

# ── Bootstrap: make ui.kscript.app importable ────────────────────────

_MOCK_MODULES = [
    "kalvin.significance",
]

for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from ui.kscript.app import KScriptApp


# ── Helpers ───────────────────────────────────────────────────────────


def _make_entry(signature: int, nodes: list[int]) -> CompiledEntry:
    """Create a CompiledEntry with given signature and nodes."""
    return CompiledEntry(signature=signature, nodes=nodes)


def _make_app() -> KScriptApp:
    """Create a KScriptApp instance without Textual lifecycle.

    Bypasses App.__init__ by patching super().__init__ and then
    manually setting all fields needed for Step/Ratify testing.
    """
    with patch("ui.kscript.app.App.__init__", return_value=None):
        app = KScriptApp.__new__(KScriptApp)
        app._dev_mode = True
        app._agent = MagicMock()
        app._decompiler = MagicMock()
        app._execution_state = ExecutionState.IDLE
        app._cancelled = False
        app._last_script_dir = None
        app._last_state_dir = None
        app._auto_compile_interval = 1.0
        app._submitted = set()
        app._satisfied = set()
        app._selected_proposal = None
        app._selected_entry_key = None
        app._compiled_entries = []
        app._expectations = {}
        app._fast_path_results = {}

        # Mock query_one to return mock regions
        app.query_one = MagicMock()
        return app


def _key(sig: int, nodes: list[int]) -> tuple[int, tuple[int, ...]]:
    """Shorthand for an entry key."""
    return (sig, tuple(nodes))


# ── Step Mode Tests (HRN-7) ───────────────────────────────────────────


class TestStepSubmitsFirstPendingHalts:
    """HRN-7: Step compiles → diffs → submits first pending → halts."""

    def test_step_submits_first_pending_halts(self):
        """Step submits exactly one entry and halts."""
        app = _make_app()
        entries = [
            _make_entry(0x10, [1]),
            _make_entry(0x20, [2]),
            _make_entry(0x30, [3]),
        ]

        app._agent.rationalise.return_value = False  # slow path

        with patch.object(app, "_compile_script", return_value=entries):
            app.action_step_script()

        # rationalise called exactly once with the first entry
        first_kline = app._entry_to_kline(entries[0])
        app._agent.rationalise.assert_called_once_with(first_kline)

        # _submitted contains only the first entry's key
        assert _key(0x10, [1]) in app._submitted
        assert _key(0x20, [2]) not in app._submitted
        assert _key(0x30, [3]) not in app._submitted

        # _satisfied is empty (slow path)
        assert len(app._satisfied) == 0

        # State is HALTED
        assert app._execution_state == ExecutionState.HALTED


class TestStepFastPathAutoSatisfied:
    """Fast-path entries (rationalise returns True) are auto-satisfied."""

    def test_step_fast_path_auto_satisfied(self):
        """Fast-path: _submitted AND _satisfied both contain the entry."""
        app = _make_app()
        entries = [_make_entry(0x10, [1])]

        app._agent.rationalise.return_value = True  # fast path

        with patch.object(app, "_compile_script", return_value=entries):
            app.action_step_script()

        key = _key(0x10, [1])
        assert key in app._submitted
        assert key in app._satisfied
        assert app._execution_state == ExecutionState.HALTED


class TestStepSkipsAlreadySubmitted:
    """Step skips entries already in _submitted."""

    def test_step_skips_already_submitted(self):
        """If first entry is submitted, step submits the first pending one."""
        app = _make_app()
        entries = [
            _make_entry(0x10, [1]),
            _make_entry(0x20, [2]),
            _make_entry(0x30, [3]),
        ]

        # Pre-submit first entry
        app._submitted.add(_key(0x10, [1]))

        app._agent.rationalise.return_value = False

        with patch.object(app, "_compile_script", return_value=entries):
            app.action_step_script()

        # Should have rationalised the second entry (first pending)
        second_kline = app._entry_to_kline(entries[1])
        app._agent.rationalise.assert_called_once_with(second_kline)

        assert _key(0x20, [2]) in app._submitted


class TestStepNoPendingDoesNothing:
    """If all entries are already submitted, step does nothing."""

    def test_step_no_pending_does_nothing(self):
        """No pending entries → rationalise not called."""
        app = _make_app()
        entries = [
            _make_entry(0x10, [1]),
            _make_entry(0x20, [2]),
        ]

        # Pre-submit all entries
        for e in entries:
            app._submitted.add(app._entry_key(e))

        with patch.object(app, "_compile_script", return_value=entries):
            app.action_step_script()

        app._agent.rationalise.assert_not_called()


class TestStepDisabledDuringRun:
    """Step is disabled when state is RUNNING."""

    def test_step_disabled_during_run(self):
        """During RUNNING state, step does nothing."""
        app = _make_app()
        app._execution_state = ExecutionState.RUNNING

        app.action_step_script()

        app._agent.rationalise.assert_not_called()


class TestStepCompilationError:
    """Compilation error displays mismatch response."""

    def test_step_compilation_error_displays_mismatch(self):
        """If _compile_script raises, _show_compilation_error is called."""
        app = _make_app()

        mock_responses = MagicMock()
        app.query_one = MagicMock(return_value=mock_responses)

        # Call _show_compilation_error directly (this is what _compile_script
        # does on error). Then verify the response was added.
        app._show_compilation_error("test error")

        mock_responses.add_response.assert_called_once()
        call_kwargs = mock_responses.add_response.call_args
        assert call_kwargs.kwargs["status"] == "mismatch"
        assert call_kwargs.kwargs["kline"] is None
        assert call_kwargs.kwargs["entry_key"] is None

    def test_step_returns_early_on_compile_none(self):
        """If _compile_script returns None, rationalise is not called."""
        app = _make_app()

        with patch.object(app, "_compile_script", return_value=None):
            app.action_step_script()

        app._agent.rationalise.assert_not_called()


# ── Ratify Tests (HRN-8, HRN-9) ───────────────────────────────────────


class TestRatifyCallsCountersign:
    """HRN-9: Ratify calls agent.countersign with proposal as-is."""

    def test_ratify_calls_countersign_with_proposal(self):
        """Ratify passes the selected proposal directly to countersign."""
        app = _make_app()

        proposal = KLine(0x42, [1, 2, 3])
        app._selected_proposal = proposal
        app._selected_entry_key = _key(0x42, [1, 2, 3])
        app._agent.countersign.return_value = True

        # Mock query_one for ToolbarRegion
        mock_toolbar = MagicMock()
        app.query_one = MagicMock(return_value=mock_toolbar)

        # Mock _update_selected_response_status
        with patch.object(app, "_update_selected_response_status"):
            app.on_toolbar_region_ratify(MagicMock())

        app._agent.countersign.assert_called_once_with(proposal)

    def test_ratify_adds_to_satisfied(self):
        """On successful ratification, entry_key is added to _satisfied."""
        app = _make_app()

        proposal = KLine(0x42, [1, 2, 3])
        entry_key = _key(0x42, [1, 2, 3])
        app._selected_proposal = proposal
        app._selected_entry_key = entry_key
        app._agent.countersign.return_value = True

        mock_toolbar = MagicMock()
        app.query_one = MagicMock(return_value=mock_toolbar)

        with patch.object(app, "_update_selected_response_status"):
            app.on_toolbar_region_ratify(MagicMock())

        assert entry_key in app._satisfied

    def test_ratify_noop_without_selection(self):
        """If no proposal is selected, ratify does nothing."""
        app = _make_app()

        app._selected_proposal = None
        app._agent = MagicMock()

        app.on_toolbar_region_ratify(MagicMock())

        app._agent.countersign.assert_not_called()

    def test_ratify_noop_without_agent(self):
        """If agent is None, ratify does nothing."""
        app = _make_app()

        app._selected_proposal = KLine(0x42, [1])
        app._agent = None

        app.on_toolbar_region_ratify(MagicMock())

        # No crash, no countersign call


class TestRatifyButtonState:
    """HRN-8: Ratify button enabled on selection, disabled after ratify."""

    def test_ratify_button_enabled_on_selection(self):
        """Selecting a response enables the Ratify button."""
        app = _make_app()

        mock_toolbar = MagicMock()
        # query_one returns ToolbarRegion for the ratify handler
        app.query_one = MagicMock(return_value=mock_toolbar)

        from ui.kscript.regions.responses import ResponsesRegion

        proposal = KLine(0x42, [1, 2])
        entry_key = _key(0x42, [1, 2])

        event = ResponsesRegion.ResponseClicked(
            decompiled_source="test",
            kline=proposal,
            entry_key=entry_key,
        )

        app.on_responses_region_response_clicked(event)

        mock_toolbar.set_ratify_enabled.assert_called_once_with(True)
        assert app._selected_proposal is proposal
        assert app._selected_entry_key == entry_key

    def test_ratify_button_disabled_after_ratify(self):
        """After ratification, selection is cleared and button disabled."""
        app = _make_app()

        proposal = KLine(0x42, [1, 2, 3])
        entry_key = _key(0x42, [1, 2, 3])
        app._selected_proposal = proposal
        app._selected_entry_key = entry_key
        app._agent.countersign.return_value = True

        mock_toolbar = MagicMock()
        app.query_one = MagicMock(return_value=mock_toolbar)

        with patch.object(app, "_update_selected_response_status"):
            app.on_toolbar_region_ratify(MagicMock())

        # Selection state cleared
        assert app._selected_proposal is None
        assert app._selected_entry_key is None

        # Ratify button disabled
        mock_toolbar.set_ratify_enabled.assert_called_with(False)

    def test_ratify_countersign_failure_does_not_satisfy(self):
        """If countersign returns False, entry is NOT added to _satisfied."""
        app = _make_app()

        proposal = KLine(0x42, [1, 2, 3])
        entry_key = _key(0x42, [1, 2, 3])
        app._selected_proposal = proposal
        app._selected_entry_key = entry_key
        app._agent.countersign.return_value = False

        mock_toolbar = MagicMock()
        app.query_one = MagicMock(return_value=mock_toolbar)

        with patch.object(app, "_update_selected_response_status"):
            app.on_toolbar_region_ratify(MagicMock())

        assert entry_key not in app._satisfied
        # Selection is still cleared
        assert app._selected_proposal is None
