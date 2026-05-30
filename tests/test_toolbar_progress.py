"""Tests for toolbar satisfaction progress display (KB-013)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Bootstrap: make ui.kscript.app importable ────────────────────────
# The UI module has dependencies (kalvin.significance, etc.) that may not
# be available in the test environment.  Patch sys.modules before the
# first import of ui.kscript.app.

_MOCK_MODULES = [
    "kalvin.significance",
]

for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

import kalvin as _kalvin_pkg
from kalvin.agent import KAgent as _RealAgent
if not hasattr(_kalvin_pkg, "Agent"):
    _kalvin_pkg.Agent = _RealAgent

from textual.css.query import NoMatches
from ui.kscript.regions.toolbar import ToolbarRegion, ExecutionState


# ---------------------------------------------------------------------------
# Helper: create a bare ToolbarRegion without mounting it in a Textual app.
# ---------------------------------------------------------------------------

def _make_toolbar() -> ToolbarRegion:
    """Create a ToolbarRegion instance with patched DOM init for testing.

    We patch Horizontal.__init__ to avoid the full DOM setup, then add
    just enough attributes for the reactive system to work without a
    running Textual app.
    """
    with patch("textual.containers.Horizontal.__init__", return_value=None):
        tb = ToolbarRegion.__new__(ToolbarRegion)
        # Minimal DOM node attributes needed by reactive system
        tb._id = "test-toolbar"
        tb._is_mounted = False
        # Mock refresh to avoid DOM tree access
        tb.refresh = MagicMock(return_value=tb)
        # Mock query_one to raise NoMatches (same as real unmounted state)
        tb.query_one = MagicMock(side_effect=NoMatches())
        # Set reactive defaults via __dict__ to avoid triggering watchers
        tb.__dict__["execution_state"] = ExecutionState.IDLE
        tb.__dict__["satisfied_count"] = 0
        tb.__dict__["total_count"] = 0
        tb.__dict__["pending_count"] = 0
    return tb


# ===========================================================================
# Test cases
# ===========================================================================


class TestStatusTextFormat:
    """Tests for _get_status_text() with and without progress."""

    def test_status_text_idle_no_progress(self):
        """When total_count is 0, show only the base state text."""
        tb = _make_toolbar()
        assert tb._get_status_text() == "○ IDLE"

    def test_status_text_halted_with_progress(self):
        """HALTED state with progress shows counts."""
        tb = _make_toolbar()
        tb.execution_state = ExecutionState.HALTED
        tb.satisfied_count = 5
        tb.total_count = 12
        tb.pending_count = 3
        assert tb._get_status_text() == "◐ HALTED  5/12 | 3 pending"

    def test_status_text_running_with_progress(self):
        """RUNNING state with progress shows counts."""
        tb = _make_toolbar()
        tb.execution_state = ExecutionState.RUNNING
        tb.satisfied_count = 3
        tb.total_count = 7
        tb.pending_count = 4
        assert tb._get_status_text() == "● RUNNING  3/7 | 4 pending"

    def test_status_text_no_progress_when_zero_total(self):
        """Even with non-zero satisfied/pending, if total is 0 show no progress."""
        tb = _make_toolbar()
        tb.execution_state = ExecutionState.RUNNING
        tb.satisfied_count = 0
        tb.total_count = 0
        tb.pending_count = 0
        assert tb._get_status_text() == "● RUNNING"


class TestSetProgress:
    """Tests for the set_progress() method."""

    def test_set_progress_updates_all_reactives(self):
        """set_progress sets all three reactive fields at once."""
        tb = _make_toolbar()
        tb.set_progress(2, 8, 6)
        assert tb.satisfied_count == 2
        assert tb.total_count == 8
        assert tb.pending_count == 6

    def test_progress_text_changes_on_update(self):
        """_get_status_text reflects updated progress after set_progress."""
        tb = _make_toolbar()
        tb.execution_state = ExecutionState.HALTED

        # Initially no progress
        assert tb._get_status_text() == "◐ HALTED"

        # After setting progress
        tb.set_progress(1, 4, 3)
        assert tb._get_status_text() == "◐ HALTED  1/4 | 3 pending"

        # After clearing progress
        tb.set_progress(0, 0, 0)
        assert tb._get_status_text() == "◐ HALTED"


class TestUpdateToolbarProgress:
    """Tests for KScriptApp._update_toolbar_progress()."""

    @patch("ui.kscript.app.KScriptApp.__init__", return_value=None)
    def test_update_toolbar_progress_method(self, mock_init):
        """_update_toolbar_progress reads _submitted/_satisfied and pushes counts."""
        from ui.kscript.app import KScriptApp

        app = KScriptApp.__new__(KScriptApp)
        # Set up tracking sets directly
        app._submitted = {(1, (2, 3)), (4, (5, 6)), (7, (8, 9))}
        app._satisfied = {(1, (2, 3))}

        # Mock toolbar
        mock_toolbar = MagicMock()
        app.query_one = MagicMock(return_value=mock_toolbar)

        app._update_toolbar_progress()

        mock_toolbar.set_progress.assert_called_once_with(1, 3, 2)

    @patch("ui.kscript.app.KScriptApp.__init__", return_value=None)
    def test_update_toolbar_progress_with_no_kb008(self, mock_init):
        """_update_toolbar_progress works even without _submitted/_satisfied (KB-008 not landed)."""
        from ui.kscript.app import KScriptApp

        app = KScriptApp.__new__(KScriptApp)
        # Deliberately do NOT set _submitted or _satisfied

        mock_toolbar = MagicMock()
        app.query_one = MagicMock(return_value=mock_toolbar)

        app._update_toolbar_progress()

        mock_toolbar.set_progress.assert_called_once_with(0, 0, 0)
