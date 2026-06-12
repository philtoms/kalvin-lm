"""Tests for KB-008 — Tracking state model in KScriptApp.

Covers: _entry_key, _get_pending, clear resets, save/restore roundtrip,
and monotonicity of _submitted.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ks import KLine

# ── Bootstrap: make ui.kscript.app importable ────────────────────────
# The UI module has dependencies that may not be available in the test
# environment.  Patch sys.modules before the first import of ui.kscript.app.

# Ensure kalvin.Agent is importable (kalvin.__init__ doesn't export it)
import kalvin as _kalvin_pkg
from kalvin.agent import KAgent as _RealAgent
if not hasattr(_kalvin_pkg, "Agent"):
    _kalvin_pkg.Agent = _RealAgent

from ui.kscript.app import KScriptApp, UI_STATE_FILE


# ── Helpers ───────────────────────────────────────────────────────────

def _make_entry(signature: int, nodes: list[int]) -> KLine:
    """Create a KLine with given signature and nodes."""
    return KLine(signature=signature, nodes=nodes)


def _make_app() -> KScriptApp:
    """Create a KScriptApp instance without initializing the Textual app.

    We bypass App.__init__ machinery by patching super().__init__ and
    then manually setting the fields we need.
    """
    with patch("ui.kscript.app.App.__init__", return_value=None):
        app = KScriptApp.__new__(KScriptApp)
        # Manually set __init__ fields (skip super().__init__ entirely)
        app._dev_mode = True
        app._agent = MagicMock()
        app._decompiler = MagicMock()
        app._execution_state = MagicMock()
        app._execution_state.name = "IDLE"
        app._pending_entries = []
        app._current_entry_index = 0
        app._cancelled = False
        app._last_script_dir = Path("data/scripts")
        app._last_state_dir = Path("data")
        app._auto_compile_interval = 1.0
        app._rationalise_buffer = []
        app._submitted = set()
        app._satisfied = set()
        return app


# ── _entry_key tests ──────────────────────────────────────────────────

class TestEntryKey:
    """KScriptApp._entry_key: identity key for KLine."""

    def test_entry_key_basic(self):
        entry = _make_entry(42, [1, 2, 3])
        assert KScriptApp._entry_key(entry) == (42, (1, 2, 3))

    def test_entry_key_empty_nodes(self):
        entry = _make_entry(99, [])
        assert KScriptApp._entry_key(entry) == (99, ())

    def test_entry_key_single_node(self):
        entry = _make_entry(7, [5])
        assert KScriptApp._entry_key(entry) == (7, (5,))

    def test_entry_key_deterministic(self):
        e1 = _make_entry(42, [1, 2, 3])
        e2 = _make_entry(42, [1, 2, 3])
        assert KScriptApp._entry_key(e1) == KScriptApp._entry_key(e2)


# ── _get_pending tests ────────────────────────────────────────────────

class TestGetPending:
    """KScriptApp._get_pending: filter entries not in _submitted."""

    def test_get_pending_filters_submitted(self):
        app = _make_app()
        e1 = _make_entry(10, [1, 2])
        e2 = _make_entry(20, [3, 4])
        app._submitted.add(KScriptApp._entry_key(e1))

        pending = app._get_pending([e1, e2])
        assert pending == [e2]

    def test_get_pending_returns_all_when_empty(self):
        app = _make_app()
        e1 = _make_entry(10, [1, 2])
        e2 = _make_entry(20, [3, 4])

        pending = app._get_pending([e1, e2])
        assert pending == [e1, e2]

    def test_get_pending_returns_none_when_all_submitted(self):
        app = _make_app()
        e1 = _make_entry(10, [1, 2])
        e2 = _make_entry(20, [3, 4])
        app._submitted.add(KScriptApp._entry_key(e1))
        app._submitted.add(KScriptApp._entry_key(e2))

        pending = app._get_pending([e1, e2])
        assert pending == []


# ── action_clear_responses tests ──────────────────────────────────────

class TestClearResetsTracking:
    """action_clear_responses resets _submitted and _satisfied."""

    def test_clear_resets_submitted_and_satisfied(self):
        app = _make_app()
        app._submitted = {(42, (1, 2, 3)), (99, (7,))}
        app._satisfied = {(42, (1, 2, 3))}

        # Mock query_one to avoid Textual widget lookups
        mock_responses = MagicMock()
        with patch.object(app, "query_one", return_value=mock_responses):
            app.action_clear_responses()

        assert app._submitted == set()
        assert app._satisfied == set()
        assert app._pending_entries == []
        assert app._current_entry_index == 0


# ── save/restore roundtrip ────────────────────────────────────────────

class TestSaveRestoreState:
    """_save_state / _restore_state roundtrip for tracking sets."""

    def test_save_restore_state_roundtrip(self):
        app = _make_app()
        app._submitted = {(42, (1, 2, 3)), (99, (7,))}
        app._satisfied = {(42, (1, 2, 3))}

        # We need to mock query_one for the editor in _save_state
        mock_editor = MagicMock()
        mock_editor.get_script.return_value = "test script"

        # Save state
        with patch.object(app, "query_one", return_value=mock_editor):
            app._save_state()

        # Verify JSON content
        assert UI_STATE_FILE.exists()
        with open(UI_STATE_FILE) as f:
            data = json.load(f)
        assert "submitted" in data
        assert "satisfied" in data
        assert len(data["submitted"]) == 2
        assert len(data["satisfied"]) == 1

        # Reset sets to empty to test restore
        app._submitted = set()
        app._satisfied = set()

        # Mock internals to avoid Textual reactive system
        app._logger = MagicMock()
        app._setup_events = MagicMock()

        # Restore state
        mock_editor2 = MagicMock()
        with patch.object(app, "query_one", return_value=mock_editor2):
            app._restore_state()

        assert app._submitted == {(42, (1, 2, 3)), (99, (7,))}
        assert app._satisfied == {(42, (1, 2, 3))}

        # Cleanup
        UI_STATE_FILE.unlink(missing_ok=True)


# ── Monotonicity ──────────────────────────────────────────────────────

class TestSubmittedMonotonic:
    """_submitted is monotonic — entries once submitted stay submitted."""

    def test_submitted_monotonic(self):
        app = _make_app()
        e1 = _make_entry(10, [1, 2])
        e2 = _make_entry(20, [3, 4])

        # Initially both are pending
        assert app._get_pending([e1, e2]) == [e1, e2]

        # Submit e1
        app._submitted.add(KScriptApp._entry_key(e1))

        # Now only e2 is pending
        assert app._get_pending([e1, e2]) == [e2]

        # Submit e2
        app._submitted.add(KScriptApp._entry_key(e2))

        # No pending entries — re-submitting same entries yields nothing
        assert app._get_pending([e1, e2]) == []
