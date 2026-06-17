"""Regression tests for the atomic removal of the Mod tokenizer (KB-277).

Locks in the removal so the Mod tokenizer (``ModTokenizer`` /
``Mod32Tokenizer`` / ``Mod64Tokenizer``) cannot silently return and so that
``NLPTokenizer`` is the sole production tokenizer.

Structural / source assertions run unconditionally on every machine.
Instance-based assertions are gated by :data:`requires_nlp_data` so they are
cleanly skipped on fresh clones without the ~35 MB NLP data assets.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

from kalvin.abstract import KTokenizer
from tests.conftest import requires_nlp_data


def test_mod_module_deleted():
    """The ``kalvin.mod_tokenizer`` module must no longer be importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("kalvin.mod_tokenizer")


def test_supports_mts_removed():
    """``KTokenizer`` (and its adapters) must not carry the ``supports_mts`` property.

    The codebase used the name ``supports_mts`` (the MCS→MTS rename was
    already applied to this property).  It had zero call sites — the encoder
    decides multi-token structurally via ``len(tokens)`` — and its only
    override lived in the now-deleted ``mod_tokenizer.py``.
    """
    assert not hasattr(KTokenizer, "supports_mts")
    assert not hasattr(KTokenizer, "supports_mcs")

    from kalvin.nlp_tokenizer import NLPTokenizer
    from kalvin.tokenizer import Tokenizer

    assert not hasattr(Tokenizer, "supports_mts")
    assert not hasattr(NLPTokenizer, "supports_mts")


def test_nlp_decode_has_no_literal_branch():
    """``NLPTokenizer.decode()`` must contain no legacy literal-node branch."""
    import kalvin.nlp_tokenizer as nlp_module

    source = inspect.getsource(nlp_module)
    assert "0xFFFFFFFF) == 0xFFFFFFFF" not in source
    assert "chr(node >> 32)" not in source


@requires_nlp_data
def test_nlp_decode_pure_bpe_for_literal_shaped_node(nlp_tokenizer):
    """A legacy literal-shaped node no longer decodes to its character.

    The legacy literal branch returned ``chr(node >> 32)`` (i.e. ``"A"``).
    Pure-BPE decode of the out-of-vocab low-32-bit token id (``0xFFFFFFFF``)
    now either raises (invalid BPE token) or yields something other than
    ``"A"``.  In either case the old ``chr()`` path is confirmed dead.
    """
    node = (ord("A") << 32) | 0xFFFFFFFF
    try:
        result = nlp_tokenizer.decode([node])
    except (KeyError, ValueError):
        # Invalid token raises — the literal path is definitively gone.
        return
    assert result != "A"


@requires_nlp_data
def test_migrated_imports_smoke():
    """All migrated importers import cleanly and KScript compiles."""
    from kalvin.agent import KAgent  # noqa: F401
    from ks import KScript
    from training.participants.auto_tune.events import enrich_event  # noqa: F401
    from training.trainer.trainer import Trainer  # noqa: F401

    KScript("A == B")  # compiles with the default NLPTokenizer


def test_no_mod_references_in_src_or_ui():
    """No source file under ``src/`` or ``ui/`` may reference the removed Mod tokenizer.

    ``tests/`` and ``scripts/`` are intentionally NOT scanned here — the
    eleven Mod-importing test files and ``scripts/ks_verify.py`` are owned by
    KB-279.
    """
    pattern = re.compile(
        r"mod_tokenizer|Mod32Tokenizer|Mod64Tokenizer|ModTokenizer"
        r"|supports_mcs|supports_mts"
    )
    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for base in (root / "src", root / "ui"):
        for path in base.rglob("*.py"):
            if pattern.search(path.read_text()):
                offenders.append(str(path.relative_to(root)))
    assert not offenders, f"Mod tokenizer references found in src/ui: {offenders}"
