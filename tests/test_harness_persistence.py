"""Tests for KB-014 — HRN-12: Tracking set persistence through hot-reload cycles.

Covers: serialisation, deserialisation, roundtrip fidelity, missing-key
defaults, preservation of other state fields, empty sets, and multiple entries.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kscript import CompiledEntry

# ── Bootstrap: make ui.kscript.app importable ────────────────────────
# The UI module has dependencies that may not be available in the test
# environment.  Patch sys.modules before the first import.

# Ensure kalvin.Agent is importable (kalvin.__init__ doesn't export it)
import kalvin as _kalvin_pkg
from kalvin.agent import KAgent as _RealAgent
if not hasattr(_kalvin_pkg, "Agent"):
    _kalvin_pkg.Agent = _RealAgent

from ui.kscript.app import KScriptApp, UI_STATE_FILE, AGENT_STATE_FILE
from ui.kscript.regions.toolbar import ExecutionState


# ── Helpers ───────────────────────────────────────────────────────────

def _make_app() -> KScriptApp:
    """Create a KScriptApp instance without initializing the Textual app.

    Bypasses App.__init__ by patching super().__init__, then manually
    sets all fields that _save_state / _restore_state depend on.
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
        app._rationalise_buffer = []
        app._submitted = set()
        app._satisfied = set()
        # Textual App.log accesses _logger internally
        app._logger = MagicMock()
        return app


def _cleanup_state_files():
    """Remove temp state files if they exist."""
    UI_STATE_FILE.unlink(missing_ok=True)
    AGENT_STATE_FILE.unlink(missing_ok=True)


def _mock_editor(get_script_return: str = "") -> MagicMock:
    """Create a mock EditorRegion for patching query_one."""
    mock = MagicMock()
    mock.get_script.return_value = get_script_return
    return mock


# ── Tests ─────────────────────────────────────────────────────────────


class TestSaveStateIncludesSubmittedAndSatisfied:
    """_save_state must serialise both tracking sets as [sig, list(nodes)]."""

    def test_save_state_includes_submitted_and_satisfied(self):
        app = _make_app()
        app._submitted = {(42, (1, 2, 3)), (99, (7,))}
        app._satisfied = {(42, (1, 2, 3))}

        try:
            with patch.object(app, "query_one", return_value=_mock_editor("test script")):
                app._save_state()

            assert UI_STATE_FILE.exists()
            with open(UI_STATE_FILE) as f:
                data = json.load(f)

            # Check submitted — each entry is [sig, [nodes]]
            assert "submitted" in data
            submitted = sorted(data["submitted"], key=lambda p: p[0])
            assert submitted == [[42, [1, 2, 3]], [99, [7]]]

            # Check satisfied
            assert "satisfied" in data
            assert data["satisfied"] == [[42, [1, 2, 3]]]
        finally:
            _cleanup_state_files()


class TestRestoreStateReconstructsSubmittedAndSatisfied:
    """_restore_state must reconstruct both tracking sets from JSON."""

    def test_restore_state_reconstructs_submitted_and_satisfied(self):
        ui_state = {
            "editor_content": "restored script",
            "execution_state": "IDLE",
            "submitted": [[42, [1, 2, 3]], [99, [7]]],
            "satisfied": [[42, [1, 2, 3]]],
        }

        app = _make_app()
        app._submitted = set()
        app._satisfied = set()
        app._setup_events = MagicMock()

        try:
            with open(UI_STATE_FILE, "w") as f:
                json.dump(ui_state, f)

            with patch.object(app, "query_one", return_value=_mock_editor()):
                app._restore_state()

            assert app._submitted == {(42, (1, 2, 3)), (99, (7,))}
            assert app._satisfied == {(42, (1, 2, 3))}
        finally:
            _cleanup_state_files()


class TestSaveRestoreRoundtrip:
    """Full serialise → deserialise cycle preserves both sets exactly."""

    def test_save_restore_roundtrip(self):
        app = _make_app()
        app._submitted = {(42, (1, 2, 3)), (99, (7,)), (7, (5,))}
        app._satisfied = {(42, (1, 2, 3)), (7, (5,))}

        editor = _mock_editor("roundtrip script")

        try:
            # Save
            with patch.object(app, "query_one", return_value=editor):
                app._save_state()

            # Snapshot originals, then reset
            original_submitted = app._submitted.copy()
            original_satisfied = app._satisfied.copy()
            app._submitted = set()
            app._satisfied = set()
            app._setup_events = MagicMock()

            # Restore
            with patch.object(app, "query_one", return_value=editor):
                app._restore_state()

            assert app._submitted == original_submitted
            assert app._satisfied == original_satisfied
        finally:
            _cleanup_state_files()


class TestRestoreMissingKeysLeavesEmptySets:
    """Old state files without tracking keys should leave sets at default empty."""

    def test_restore_missing_keys_leaves_empty_sets(self):
        ui_state = {
            "editor_content": "old script",
            "execution_state": "IDLE",
        }

        app = _make_app()
        app._submitted = set()
        app._satisfied = set()
        app._setup_events = MagicMock()

        try:
            with open(UI_STATE_FILE, "w") as f:
                json.dump(ui_state, f)

            with patch.object(app, "query_one", return_value=_mock_editor()):
                app._restore_state()

            assert app._submitted == set()
            assert app._satisfied == set()
        finally:
            _cleanup_state_files()


class TestRestorePreservesOtherState:
    """Restoring tracking sets must not break other persisted fields."""

    def test_restore_preserves_other_state(self):
        ui_state = {
            "editor_content": "full state script",
            "last_script_dir": "/some/path",
            "last_state_dir": "/other/path",
            "execution_state": "HALTED",
            "submitted": [[42, [1, 2, 3]]],
            "satisfied": [[99, [7]]],
        }

        app = _make_app()
        app._setup_events = MagicMock()

        try:
            with open(UI_STATE_FILE, "w") as f:
                json.dump(ui_state, f)

            editor = _mock_editor()
            with patch.object(app, "query_one", return_value=editor):
                app._restore_state()

            # All fields should be restored correctly
            assert app._execution_state == ExecutionState.HALTED
            assert app._last_script_dir == Path("/some/path")
            assert app._last_state_dir == Path("/other/path")
            editor.set_script.assert_called_once_with("full state script")
            assert app._submitted == {(42, (1, 2, 3))}
            assert app._satisfied == {(99, (7,))}
        finally:
            _cleanup_state_files()


class TestEmptySetsSerialiseToEmptyArrays:
    """Empty tracking sets should produce empty JSON arrays."""

    def test_empty_sets_serialise_to_empty_arrays(self):
        app = _make_app()
        app._submitted = set()
        app._satisfied = set()

        try:
            with patch.object(app, "query_one", return_value=_mock_editor("")):
                app._save_state()

            with open(UI_STATE_FILE) as f:
                data = json.load(f)

            assert data["submitted"] == []
            assert data["satisfied"] == []
        finally:
            _cleanup_state_files()


class TestMultipleEntriesRoundtrip:
    """3+ entries per set with varying signatures and nodes survive roundtrip."""

    def test_multiple_entries_roundtrip(self):
        app = _make_app()
        app._submitted = {
            (10, (1, 2)),
            (20, (3, 4, 5)),
            (30, ()),
        }
        app._satisfied = {
            (10, (1, 2)),
            (40, (99,)),
            (50, (7, 8, 9)),
        }

        editor = _mock_editor("")

        try:
            # Save
            with patch.object(app, "query_one", return_value=editor):
                app._save_state()

            # Snapshot originals
            original_submitted = app._submitted.copy()
            original_satisfied = app._satisfied.copy()

            # Reset
            app._submitted = set()
            app._satisfied = set()
            app._setup_events = MagicMock()

            # Restore
            with patch.object(app, "query_one", return_value=editor):
                app._restore_state()

            assert app._submitted == original_submitted
            assert app._satisfied == original_satisfied
            assert len(app._submitted) == 3
            assert len(app._satisfied) == 3
        finally:
            _cleanup_state_files()
