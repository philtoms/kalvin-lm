"""Tests for ResponseItem display formatting and ResponsesRegion.add_response."""

import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalvin.expand import D_MAX

# Keys that the fixture will stub in sys.modules
_STUB_KEYS = ("kscript", "kscript.decompiler", "kscript.compiler", "ui.kscript.dialogs")


@pytest.fixture(scope="module")
def responses():
    """Install sys.modules stubs, import responses classes, then clean up."""
    # --- Save pre-existing state ---
    saved_modules: dict[str, object | None] = {}
    newly_installed: set[str] = set()
    for key in _STUB_KEYS:
        saved_modules[key] = sys.modules.get(key)
        newly_installed.add(key)

    # Save kalvin.Agent state
    import kalvin as _kalvin_mod  # noqa: F811

    had_agent = hasattr(_kalvin_mod, "Agent")
    saved_agent = getattr(_kalvin_mod, "Agent", None)

    # --- Install stubs ---
    _kscript_mod = types.ModuleType("kscript")
    _kscript_mod.KScript = MagicMock()  # type: ignore[attr-defined]
    _kscript_mod.CompiledEntry = MagicMock()  # type: ignore[attr-defined]
    sys.modules["kscript"] = _kscript_mod
    sys.modules["kscript.decompiler"] = MagicMock()
    sys.modules["kscript.compiler"] = MagicMock()
    sys.modules["ui.kscript.dialogs"] = MagicMock()

    # Stub kalvin.Agent
    _kalvin_mod.Agent = MagicMock  # type: ignore[attr-defined]

    # --- Import under test ---
    from ui.kscript.regions.responses import (  # noqa: E402
        STATUS_SYMBOLS,
        ResponseItem,
        ResponsesRegion,
    )

    try:
        yield SimpleNamespace(
            ResponseItem=ResponseItem,
            ResponsesRegion=ResponsesRegion,
            STATUS_SYMBOLS=STATUS_SYMBOLS,
        )
    finally:
        # --- Restore sys.modules ---
        for key in _STUB_KEYS:
            if saved_modules[key] is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = saved_modules[key]

        # Remove the responses module itself to avoid stale cached imports
        sys.modules.pop("ui.kscript.regions.responses", None)

        # --- Restore kalvin.Agent ---
        if had_agent:
            _kalvin_mod.Agent = saved_agent  # type: ignore[attr-defined]
        else:
            if hasattr(_kalvin_mod, "Agent"):
                delattr(_kalvin_mod, "Agent")


class TestStatusSymbols:
    """Verify the status symbol mapping."""

    def test_symbols_defined(self, responses) -> None:
        assert responses.STATUS_SYMBOLS == {"pass": "✓", "pending": "◌", "mismatch": "✗"}


class TestResponseItemFormat:
    """Test _format_display() output for each status."""

    def test_response_item_pass_format(self, responses) -> None:
        item = responses.ResponseItem(
            level="S2",
            decompiled_source="MHALL => SVO",
            status="pass",
            significance=0xFFFF_FFFF_FFFF_FFFF,
        )
        text = item._format_display()
        assert "✓ MHALL => SVO" in text
        assert "0xFFFFFFFFFFFFFFFF" in text
        assert "1.000" in text

    def test_response_item_pending_format(self, responses) -> None:
        item = responses.ResponseItem(
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

    def test_response_item_mismatch_format(self, responses) -> None:
        item = responses.ResponseItem(
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

    def test_response_item_backward_compat(self, responses) -> None:
        item = responses.ResponseItem(level="S1", decompiled_source="HELLO")
        text = item._format_display()
        # Defaults: status="pending" → ◌, significance=0
        assert text.startswith("◌")
        assert "0x0000000000000000" in text
        assert "0.000" in text
        assert "HELLO" in text


class TestNormalisedRange:
    """Boundary and mid-point normalised significance values."""

    def test_zero(self, responses) -> None:
        item = responses.ResponseItem(level="S1", decompiled_source="A", significance=0)
        assert "0.000" in item._format_display()

    def test_max(self, responses) -> None:
        item = responses.ResponseItem(level="S1", decompiled_source="A", significance=D_MAX)
        assert "1.000" in item._format_display()

    def test_intermediate(self, responses) -> None:
        half = D_MAX // 2
        item = responses.ResponseItem(level="S1", decompiled_source="A", significance=half)
        normalised_str = f"{half / D_MAX:.3f}"
        assert normalised_str in item._format_display()


class TestResponseItemAttributes:
    """Verify constructor stores all fields correctly."""

    def test_stores_all_fields(self, responses) -> None:
        item = responses.ResponseItem(
            level="S2",
            decompiled_source="MHALL => SVO",
            status="pass",
            significance=D_MAX,
        )
        assert item.level == "S2"
        assert item.decompiled_source == "MHALL => SVO"
        assert item.status == "pass"
        assert item.significance == D_MAX

    def test_default_fields(self, responses) -> None:
        item = responses.ResponseItem(level="S3", decompiled_source="X")
        assert item.status == "pending"
        assert item.significance == 0


class TestAddResponseSignature:
    """Verify add_response accepts new keyword parameters (unit-level)."""

    def test_add_response_accepts_new_params(self, responses) -> None:
        """Smoke test: add_response signature accepts status and significance."""
        sig = inspect.signature(responses.ResponsesRegion.add_response)
        param_names = list(sig.parameters.keys())
        assert "status" in param_names
        assert "significance" in param_names
        # New params should have defaults for backward compatibility
        assert sig.parameters["status"].default == "pending"
        assert sig.parameters["significance"].default == 0
