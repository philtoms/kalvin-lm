"""Tests for ResponseItem display formatting and ResponsesRegion.add_response."""

import importlib
import sys
import types
from unittest.mock import MagicMock

from kalvin.expand import D_MAX

# -------------------------------------------------------------------
# The production import chain requires modules that don't exist yet:
#   kalvin.Agent  (pulled in by ui.kscript.__init__ → app.py)
#   kscript.*     (pulled in by ui.kscript.app)
#   ui.kscript.dialogs (pulled in by ui.kscript.app)
# We stub all of these so we can load responses.py in isolation.
# -------------------------------------------------------------------

# Stub the kscript package and its submodules
_kscript_mod = types.ModuleType("kscript")
_kscript_mod.KScript = MagicMock()
_kscript_mod.CompiledEntry = MagicMock()
sys.modules.setdefault("kscript", _kscript_mod)
sys.modules.setdefault("kscript.decompiler", MagicMock())
sys.modules.setdefault("kscript.compiler", MagicMock())

# Stub ui.kscript.dialogs
sys.modules.setdefault("ui.kscript.dialogs", MagicMock())

# Stub kalvin.Agent
_kalvin_mod = sys.modules.get("kalvin")
if _kalvin_mod is None:
    import kalvin as _kalvin_mod  # type: ignore[no-redef]
if not hasattr(_kalvin_mod, "Agent"):
    _kalvin_mod.Agent = MagicMock  # type: ignore[attr-defined]

# Now safe to import
from ui.kscript.regions.responses import ResponseItem, ResponsesRegion, STATUS_SYMBOLS  # noqa: E402


class TestStatusSymbols:
    """Verify the status symbol mapping."""

    def test_symbols_defined(self) -> None:
        assert STATUS_SYMBOLS == {"pass": "✓", "pending": "◌", "mismatch": "✗"}


class TestResponseItemFormat:
    """Test _format_display() output for each status."""

    def test_response_item_pass_format(self) -> None:
        item = ResponseItem(
            level="S2",
            decompiled_source="MHALL => SVO",
            status="pass",
            significance=0xFFFF_FFFF_FFFF_FFFF,
        )
        text = item._format_display()
        assert "✓ MHALL => SVO" in text
        assert "0xFFFFFFFFFFFFFFFF" in text
        assert "1.000" in text

    def test_response_item_pending_format(self) -> None:
        item = ResponseItem(
            level="S1",
            decompiled_source="QUERY => DB",
            status="pending",
            significance=0x7FFF_FFFF_FFFF_FFFF,
        )
        text = item._format_display()
        assert text.startswith("◌")
        assert "0x7FFFFFFFFFFFFFFF" in text
        # ≈ 0.500 (0x7FFF.../0xFFFF... = 0.5 exactly)
        assert "0.500" in text

    def test_response_item_mismatch_format(self) -> None:
        item = ResponseItem(
            level="S3",
            decompiled_source="NOOP",
            status="mismatch",
            significance=0,
        )
        text = item._format_display()
        assert text.startswith("✗")
        assert "0x0000000000000000" in text
        assert "0.000" in text


class TestResponseItemBackwardCompat:
    """Default parameters preserve old behaviour."""

    def test_response_item_backward_compat(self) -> None:
        item = ResponseItem(level="S1", decompiled_source="HELLO")
        text = item._format_display()
        # Defaults: status="pending" → ◌, significance=0
        assert text.startswith("◌")
        assert "0x0000000000000000" in text
        assert "0.000" in text
        assert "HELLO" in text


class TestNormalisedRange:
    """Boundary and mid-point normalised significance values."""

    def test_zero(self) -> None:
        item = ResponseItem(level="S1", decompiled_source="A", significance=0)
        assert "0.000" in item._format_display()

    def test_max(self) -> None:
        item = ResponseItem(level="S1", decompiled_source="A", significance=D_MAX)
        assert "1.000" in item._format_display()

    def test_intermediate(self) -> None:
        half = D_MAX // 2
        item = ResponseItem(level="S1", decompiled_source="A", significance=half)
        normalised_str = f"{half / D_MAX:.3f}"
        assert normalised_str in item._format_display()


class TestResponseItemAttributes:
    """Verify constructor stores all fields correctly."""

    def test_stores_all_fields(self) -> None:
        item = ResponseItem(
            level="S2",
            decompiled_source="MHALL => SVO",
            status="pass",
            significance=D_MAX,
        )
        assert item.level == "S2"
        assert item.decompiled_source == "MHALL => SVO"
        assert item.status == "pass"
        assert item.significance == D_MAX

    def test_default_fields(self) -> None:
        item = ResponseItem(level="S3", decompiled_source="X")
        assert item.status == "pending"
        assert item.significance == 0


class TestAddResponseSignature:
    """Verify add_response accepts new keyword parameters (unit-level)."""

    def test_add_response_accepts_new_params(self) -> None:
        """Smoke test: add_response signature accepts status and significance."""
        import inspect

        sig = inspect.signature(ResponsesRegion.add_response)
        param_names = list(sig.parameters.keys())
        assert "status" in param_names
        assert "significance" in param_names
        # New params should have defaults for backward compatibility
        assert sig.parameters["status"].default == "pending"
        assert sig.parameters["significance"].default == 0
