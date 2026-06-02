"""Regression test: verify that test_responses_region's module-scoped fixture
cleans up its sys.modules stubs after teardown.

This file must be collected AFTER test_responses_region.py for the test to
be meaningful.  The name starts with ``test_z_`` to encourage alphabetical
ordering, but the real guarantee is Step 3's cross-file pytest command:
    python -m pytest tests/test_responses_region.py tests/test_responses_region_isolation.py -x
"""

import sys

from unittest.mock import MagicMock


def test_kscript_stub_removed_after_responses_region_tests():
    """The kscript module should not be a MagicMock after test_responses_region
    has torn down its fixture."""
    kscript_mod = sys.modules.get("kscript")
    # If the stub leaked, kscript_mod would be a MagicMock (or a fake
    # types.ModuleType with MagicMock attributes).  After proper teardown
    # the key is either removed or points to the real package.
    if kscript_mod is not None:
        # If it exists, it must be the real package — not a bare ModuleType
        # with mock attributes.
        assert not isinstance(getattr(kscript_mod, "KScript", None), MagicMock), (
            "kscript.KScript is still a MagicMock — fixture teardown did not clean up"
        )
        assert not isinstance(getattr(kscript_mod, "CompiledEntry", None), MagicMock), (
            "kscript.CompiledEntry is still a MagicMock — fixture teardown did not clean up"
        )


def test_kscript_decompiler_stub_removed():
    """kscript.decompiler should not be a MagicMock after teardown."""
    mod = sys.modules.get("kscript.decompiler")
    if mod is not None:
        assert not isinstance(mod, MagicMock), (
            "kscript.decompiler is still a MagicMock — fixture teardown did not clean up"
        )


def test_kscript_compiler_stub_removed():
    """kscript.compiler should not be a MagicMock after teardown."""
    mod = sys.modules.get("kscript.compiler")
    if mod is not None:
        assert not isinstance(mod, MagicMock), (
            "kscript.compiler is still a MagicMock — fixture teardown did not clean up"
        )


def test_ui_kscript_dialogs_stub_removed():
    """ui.kscript.dialogs should not be a MagicMock after teardown."""
    mod = sys.modules.get("ui.kscript.dialogs")
    if mod is not None:
        assert not isinstance(mod, MagicMock), (
            "ui.kscript.dialogs is still a MagicMock — fixture teardown did not clean up"
        )


def test_responses_region_module_not_cached():
    """If ui.kscript.regions.responses is in sys.modules, it must be the real
    module (legitimately imported by another test), not a stale cached copy
    left behind by the fixture."""
    import types as _types

    mod = sys.modules.get("ui.kscript.regions.responses")
    if mod is not None:
        # If it exists, it must be a real module with real attributes,
        # not a MagicMock or a stub module with mock attributes.
        assert isinstance(mod, _types.ModuleType), (
            "ui.kscript.regions.responses in sys.modules is not a real module"
        )
        # The real module should have a ResponseItem class, not a MagicMock
        assert not isinstance(getattr(mod, "ResponseItem", None), MagicMock), (
            "ui.kscript.regions.responses.ResponseItem is a MagicMock — "
            "stale cached import from fixture"
        )
