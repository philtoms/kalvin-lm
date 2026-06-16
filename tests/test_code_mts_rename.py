"""Code-layer regression guard for KB-293: MCS -> MTS rename in src/.

Locks in the terminological rename so production source and tests cannot
silently regress to the obsolete 'MCS' (Multi-Character Signature) term.
The five spec/plan regression tests (test_spec_*.py, test_plan_mod_prune_*)
guard their respective layers; this test guards the implementation code.

The detection pattern matches *both* standalone 'MCS' (comments/docstrings)
and embedded 'MCS' inside CamelCase identifiers (e.g. a regressed
'TestKS19MCS' class name), plus lowercase identifier-style 'mcs' tokens.
A simple ``\\bMCS\\b`` word-boundary check would miss embedded CamelCase
regressions, so the pattern here is deliberately broader.

Run: python -m pytest tests/test_code_mts_rename.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Matches any uppercase 'MCS' (standalone or embedded in CamelCase) or a
# lowercase 'mcs' token (identifier-style, not surrounded by lowercase
# letters). Does NOT match the renamed 'MTS'/'mts'.
_MCS_RE = re.compile(r"MCS|(?<![a-z])mcs(?![a-z])")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_no_mcs(path: Path) -> None:
    """Assert no MCS/mcs abbreviation remains in the given file."""
    if not path.exists():
        return
    text = _read(path)
    match = _MCS_RE.search(text)
    assert match is None, f"residual MCS in {path}: {match.group(0)!r}"


class TestNoMcsInSource:
    """No 'MCS' token remains in renamed production source files."""

    def test_ast_emitter_no_mcs(self):
        _assert_no_mcs(ROOT / "src" / "ks" / "ast_emitter.py")

    def test_token_encoder_no_mcs(self):
        _assert_no_mcs(ROOT / "src" / "ks" / "token_encoder.py")

    def test_compiler_no_mcs(self):
        _assert_no_mcs(ROOT / "src" / "ks" / "compiler.py")

    def test_ks_verify_no_mcs(self):
        _assert_no_mcs(ROOT / "scripts" / "ks_verify.py")

    def test_abstract_no_mcs(self):
        # Conditional: KB-277 may have deleted supports_mts; only check if present.
        _assert_no_mcs(ROOT / "src" / "kalvin" / "abstract.py")

    def test_mod_tokenizer_no_mcs(self):
        # Conditional: KB-277 may have deleted this file; only check if present.
        _assert_no_mcs(ROOT / "src" / "kalvin" / "mod_tokenizer.py")


class TestMtsIdentifiersPresent:
    """The renamed identifiers exist and the old names are gone."""

    def test_emit_mts_method_exists(self):
        from ks.ast_emitter import ASTEmitter

        assert hasattr(ASTEmitter, "_emit_mts")
        assert not hasattr(ASTEmitter, "_emit_mcs")

    def test_mts_identity_seen_attr(self):
        from ks.ast_emitter import ASTEmitter

        emitter = ASTEmitter()
        assert hasattr(emitter, "_mts_identity_seen")
        assert not hasattr(emitter, "_mcs_identity_seen")

    def test_mts_canonize_seen_attr(self):
        from ks.ast_emitter import ASTEmitter

        emitter = ASTEmitter()
        assert hasattr(emitter, "_mts_canonize_seen")
        assert not hasattr(emitter, "_mcs_canonize_seen")

    def test_emit_mts_for_tokens_exists(self):
        from ks.token_encoder import TokenEncoder

        assert hasattr(TokenEncoder, "_emit_mts_for_tokens")
        assert not hasattr(TokenEncoder, "_emit_mcs_for_tokens")

    def test_supports_mts_property(self):
        # Conditional: KB-277 may have removed supports_mts. If the property
        # still exists, it must be the renamed form.
        from kalvin.abstract import KTokenizer

        if hasattr(KTokenizer, "supports_mts"):
            assert not hasattr(KTokenizer, "supports_mcs")


class TestNoMcsInTests:
    """Renamed test files are free of the MCS abbreviation."""

    @staticmethod
    def _check(name: str):
        _assert_no_mcs(ROOT / "tests" / name)

    def test_ks_no_mcs(self):
        self._check("test_ks.py")

    def test_ks_ast_emitter_no_mcs(self):
        self._check("test_ks_ast_emitter.py")

    def test_ks_compiler_no_mcs(self):
        self._check("test_ks_compiler.py")

    def test_ks_token_encoder_no_mcs(self):
        self._check("test_ks_token_encoder.py")

    def test_expand_no_mcs(self):
        self._check("test_expand.py")

    def test_nlp_curriculum_compat_no_mcs(self):
        self._check("test_nlp_curriculum_compat.py")
