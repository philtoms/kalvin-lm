"""Tests for KB-011 — Run mode rewrite in KScriptApp.

Covers: HRN-2, HRN-3, HRN-5, HRN-6, HRN-17:
- action_run_script compiles → diffs → submits all pending sequentially
- _submit_all_pending iterates entries, calls rationalise, tracks submission
- _auto_countersign checks structural match and calls agent.countersign
- Fast-path entries auto-satisfied; slow-path entries await events
- Mismatches flagged as pending; execution continues
- Submitted set is monotonic — no re-submission
- Run mode stop cancels in-flight submission
- No pending entries → idle immediately
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kalvin.abstract import KLine
from kalvin.events import RationaliseEvent
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
    """Create a KScriptApp instance without initializing the Textual app.

    Sets up all fields needed for Run mode testing.
    """
    with patch("ui.kscript.app.App.__init__", return_value=None):
        app = KScriptApp.__new__(KScriptApp)
        app._dev_mode = True
        app._agent = MagicMock()
        app._decompiler = MagicMock()
        app._execution_state = ExecutionState.IDLE
        app._pending_entries = []
        app._current_entry_index = 0
        app._cancelled = False
        app._last_script_dir = Path("data/scripts")
        app._last_state_dir = Path("data")
        app._auto_compile_interval = 1.0
        app._submitted = set()
        app._satisfied = set()
        app._compiled_entries = []
        app._expectations = {}
        app._fast_path_results = {}
        app._run_mode_active = False
        app._logger = MagicMock()
        return app


def _entry_key(entry: CompiledEntry):
    """Shortcut for KScriptApp._entry_key."""
    return KScriptApp._entry_key(entry)


def _run_submit_all(app, entries):
    """Run _submit_all_pending with _set_state mocked (avoids Textual)."""
    with patch.object(app, "_set_state"):
        asyncio.run(app._submit_all_pending(entries))


# ── HRN-5: Run compiles and submits pending entries ──────────────────


class TestRunCompilesAndSubmits:
    """action_run_script compiles → diffs → submits all pending."""

    def test_run_compiles_and_submits_pending(self):
        """HRN-5: Only non-submitted entries are rationalised sequentially."""
        app = _make_app()

        e1 = _make_entry(0x10, [1, 2])
        e2 = _make_entry(0x20, [3, 4])
        e3 = _make_entry(0x30, [5, 6])

        # e1 already submitted
        app._submitted.add(_entry_key(e1))
        app._agent.rationalise.return_value = True

        # _get_pending filters out already-submitted entries
        pending = app._get_pending([e1, e2, e3])
        assert len(pending) == 2

        # Submit only pending entries
        _run_submit_all(app, pending)

        # Only e2 and e3 should be rationalised (e1 filtered out)
        assert app._agent.rationalise.call_count == 2
        rationalised_sigs = [
            c[0][0].signature for c in app._agent.rationalise.call_args_list
        ]
        assert 0x20 in rationalised_sigs
        assert 0x30 in rationalised_sigs
        assert 0x10 not in rationalised_sigs

    def test_run_submitted_monotonic(self):
        """HRN-2: After Run, calling Run again with same entries submits nothing."""
        app = _make_app()

        e1 = _make_entry(0x10, [1, 2])
        e2 = _make_entry(0x20, [3, 4])

        # First run: submit both
        app._agent.rationalise.return_value = True
        _run_submit_all(app, [e1, e2])
        assert len(app._submitted) == 2

        # Second run: same entries, should submit nothing
        app._agent.rationalise.reset_mock()
        pending = app._get_pending([e1, e2])
        assert pending == []
        assert app._agent.rationalise.call_count == 0

    def test_run_no_pending_goes_idle(self):
        """All entries already in _submitted → Run stays IDLE."""
        app = _make_app()

        e1 = _make_entry(0x10, [1, 2])
        app._submitted.add(_entry_key(e1))

        with patch.object(app, "_compile_script", return_value=[e1]):
            with patch.object(app, "_set_state") as mock_set_state:
                with patch.object(app, "run_worker") as mock_run_worker:
                    app.action_run_script()

        # Should NOT call run_worker — no pending entries
        assert mock_run_worker.called is False
        # _run_mode_active should be False
        assert app._run_mode_active is False

    def test_run_fast_path_auto_satisfied(self):
        """HRN-3: Fast-path entry (rationalise returns True) auto-satisfied."""
        app = _make_app()

        e1 = _make_entry(0x10, [1, 2])
        app._agent.rationalise.return_value = True

        _run_submit_all(app, [e1])

        key = _entry_key(e1)
        assert key in app._submitted
        assert key in app._satisfied
        assert app._fast_path_results[key] is True

    def test_run_slow_path_awaits_events(self):
        """Slow-path entry (rationalise returns False) in _submitted but NOT _satisfied."""
        app = _make_app()

        e1 = _make_entry(0x10, [1, 2])
        app._agent.rationalise.return_value = False

        _run_submit_all(app, [e1])

        key = _entry_key(e1)
        assert key in app._submitted
        assert key not in app._satisfied
        assert app._fast_path_results[key] is False

    def test_run_stop_cancels(self):
        """Start Run mode, set _cancelled = True → loop breaks, state IDLE."""
        app = _make_app()

        e1 = _make_entry(0x10, [1, 2])
        e2 = _make_entry(0x20, [3, 4])
        e3 = _make_entry(0x30, [5, 6])

        # Cancel after first entry is submitted
        def rationalise_side_effect(kline):
            app._cancelled = True
            return True

        app._agent.rationalise.side_effect = rationalise_side_effect

        _run_submit_all(app, [e1, e2, e3])

        # Only e1 should be submitted (cancelled after first)
        assert len(app._submitted) == 1
        assert _entry_key(e1) in app._submitted
        assert _entry_key(e2) not in app._submitted

        # _run_mode_active should be False
        assert app._run_mode_active is False


# ── HRN-6: Auto-countersign matching proposals ───────────────────────


class TestAutoCountersign:
    """_auto_countersign checks structural match and calls countersign."""

    def test_auto_countersign_matching_proposal(self):
        """HRN-6: Matching proposal → countersign called, entry satisfied."""
        app = _make_app()

        entry = _make_entry(0x42, [1, 2, 3])
        proposal = KLine(0x42, [1, 2, 3])  # structurally matches entry

        result = app._auto_countersign(entry, proposal)

        assert result is True
        app._agent.countersign.assert_called_once_with(proposal)
        assert _entry_key(entry) in app._satisfied

    def test_auto_countersign_non_matching_proposal(self):
        """HRN-6 negative: Non-matching proposal → no countersign, not satisfied."""
        app = _make_app()

        entry = _make_entry(0x42, [1, 2, 3])
        proposal = KLine(0x42, [1, 2, 99])  # different nodes

        result = app._auto_countersign(entry, proposal)

        assert result is False
        app._agent.countersign.assert_not_called()
        assert _entry_key(entry) not in app._satisfied

    def test_auto_countersign_different_signature(self):
        """Non-matching signature → no countersign."""
        app = _make_app()

        entry = _make_entry(0x42, [1, 2, 3])
        proposal = KLine(0x99, [1, 2, 3])  # different signature

        result = app._auto_countersign(entry, proposal)

        assert result is False
        app._agent.countersign.assert_not_called()


# ── HRN-17: Mismatches flagged as pending ─────────────────────────────


class TestMismatchPending:
    """Mismatches are flagged as pending for human review; execution continues."""

    def test_run_mismatch_flagged_pending(self):
        """HRN-17: Slow-path with non-matching proposal → status 'pending'."""
        app = _make_app()

        entry = _make_entry(0x60, [100, 200])
        app._compiled_entries = [entry]
        app._run_mode_active = True

        # Mock decompiler to return a valid entry
        from kscript.decompiler import DecompiledEntry
        app._decompiler.decompile.return_value = [
            DecompiledEntry(level="S2", sig="SIG", nodes=["A", "B"])
        ]

        # Mock query_one for ResponsesRegion
        mock_responses = MagicMock()
        app.query_one = MagicMock(return_value=mock_responses)

        # Set up event callback
        app._setup_events()
        callback = app._agent.events.subscribe.call_args[0][0]

        # Slow-path event with non-matching proposal
        event = RationaliseEvent(
            kind="frame",
            query=KLine(0x60, [100, 200]),
            proposal=KLine(0x60, [100, 999]),  # mismatch
            significance=0x8000_0000_0000_0000,
        )
        callback(event)

        # Should be flagged as pending (not "pass" or "mismatch")
        mock_responses.add_response.assert_called()
        call_kwargs = mock_responses.add_response.call_args
        assert call_kwargs.kwargs["status"] == "pending"

        # Entry should NOT be in satisfied
        key = _entry_key(entry)
        assert key not in app._satisfied

        # Execution continues — run_mode_active still True
        assert app._run_mode_active is True


# ── Run mode flag tests ──────────────────────────────────────────────


class TestRunModeFlag:
    """_run_mode_active flag distinguishes Run from Step."""

    def test_run_sets_flag_true(self):
        """Starting Run mode sets _run_mode_active = True."""
        app = _make_app()
        e1 = _make_entry(0x10, [1, 2])

        with patch.object(app, "_compile_script", return_value=[e1]):
            with patch.object(app, "run_worker"):
                with patch.object(app, "_set_state"):
                    app.action_run_script()

        assert app._run_mode_active is True

    def test_run_toggle_off_sets_flag_false(self):
        """Toggling Run off during execution sets _run_mode_active = False."""
        app = _make_app()
        app._execution_state = ExecutionState.RUNNING

        with patch.object(app, "_set_state"):
            app.action_run_script()

        assert app._run_mode_active is False
        assert app._cancelled is True

    def test_submit_all_pending_clears_flag_on_completion(self):
        """_submit_all_pending clears flag when loop completes."""
        app = _make_app()
        app._run_mode_active = True

        e1 = _make_entry(0x10, [1, 2])
        app._agent.rationalise.return_value = True

        with patch.object(app, "_set_state") as mock_set_state:
            asyncio.run(app._submit_all_pending([e1]))

        assert app._run_mode_active is False
        mock_set_state.assert_called_with(ExecutionState.IDLE)

    def test_run_empty_compilation_goes_idle(self):
        """Compilation returns None → Run mode immediately goes idle."""
        app = _make_app()

        with patch.object(app, "_compile_script", return_value=None):
            with patch.object(app, "_set_state"):
                with patch.object(app, "run_worker") as mock_run_worker:
                    app.action_run_script()

        assert app._run_mode_active is False
        assert mock_run_worker.called is False

    def test_run_empty_compilation_returns_empty(self):
        """Compilation returns empty list → Run mode immediately goes idle."""
        app = _make_app()

        with patch.object(app, "_compile_script", return_value=[]):
            with patch.object(app, "_set_state"):
                with patch.object(app, "run_worker") as mock_run_worker:
                    app.action_run_script()

        assert app._run_mode_active is False
        assert mock_run_worker.called is False


# ── Sequential submission tests ──────────────────────────────────────


class TestSequentialSubmission:
    """Entries are submitted sequentially in order (HRN-5)."""

    def test_entries_submitted_in_order(self):
        """Entries submitted in the order they appear in the list."""
        app = _make_app()

        e1 = _make_entry(0x10, [1])
        e2 = _make_entry(0x20, [2])
        e3 = _make_entry(0x30, [3])

        call_order = []

        def rationalise_side_effect(kline):
            call_order.append(kline.signature)
            return True

        app._agent.rationalise.side_effect = rationalise_side_effect

        _run_submit_all(app, [e1, e2, e3])

        assert call_order == [0x10, 0x20, 0x30]

    def test_all_pending_submitted(self):
        """All pending entries are submitted."""
        app = _make_app()

        entries = [_make_entry(i, [i]) for i in range(5)]
        app._agent.rationalise.return_value = True

        _run_submit_all(app, entries)

        assert len(app._submitted) == 5
        for e in entries:
            assert _entry_key(e) in app._submitted
