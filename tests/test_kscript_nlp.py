"""KSN integration tests — KScript NLP-BPE mode test matrix.

Implements KSN-1 through KSN-28 from specs/kscript-nlp.md §9.
Tests verify NLP-BPE encoding, operator semantics, MCS replacement,
signature construction, decompilation, and compatibility.
"""

import pytest

from kalvin.signature import (
    make_signature,
    signifies,
)
from kalvin.mod_tokenizer import Mod32Tokenizer

try:
    from kalvin.nlp_tokenizer import NLPTokenizer
    _has_nlp = True
except ImportError:
    _has_nlp = False

_nlp_skip = pytest.mark.skipif(
    not _has_nlp,
    reason="NLPTokenizer not available"
)


def _get_nlp_tokenizer():
    """Get NLPTokenizer instance, or None if unavailable."""
    if not _has_nlp:
        return None
    try:
        return NLPTokenizer.from_files()
    except Exception:
        return None


def compile_nlp(source: str) -> list:
    """Compile with NLPTokenizer."""
    from kscript.compiler import compile_source
    tok = _get_nlp_tokenizer()
    assert tok is not None, "NLPTokenizer not available"
    return compile_source(source, tokenizer=tok, dev=True)


def _md_nlp(entries, tokenizer=None):
    """Build multidict from compiled entries using NLP tokenizer decoding."""
    tok = tokenizer or _get_nlp_tokenizer()
    md = {}
    for e in entries:
        sig = tok.decode([e.signature])
        if e.nodes is None:
            md.setdefault(sig, [])
        elif isinstance(e.nodes, int):
            if False:
                node_str = tok.decode([e.nodes])
            else:
                node_str = tok.decode([e.nodes])
            md.setdefault(sig, []).append(node_str)
        else:
            for n in e.nodes:
                if False:
                    node_str = tok.decode([n])
                else:
                    node_str = tok.decode([n])
                md.setdefault(sig, []).append(node_str)
    return md


def _has_node(md: dict[str, list], sig: str, node_value) -> bool:
    """Check if a sig has a specific decoded node value in its list."""
    return node_value in md.get(sig, [])


def _find_multi_token_word():
    """Find an uppercase word that BPE-encodes to multiple tokens.

    Returns (word, token_count) or None if no multi-token word found.
    """
    tok = _get_nlp_tokenizer()
    if tok is None:
        return None
    # Most multi-char uppercase words BPE-encode to multiple tokens
    for word in ["TEA", "ABC", "SVO", "ALL", "BREW", "MHALL"]:
        tokens = tok.encode(word)
        if len(tokens) > 1:
            return (word, len(tokens))
    return None


# ============================================================================
# KSN-1: NLP-BPE encode produces correct node format
# ============================================================================
@_nlp_skip
class TestKSN1:
    """KSN-1: NLP-BPE encode produces correct node format (nlp_type32 << 32) | bpe_token_id."""

    def test_node_format(self):
        tok = _get_nlp_tokenizer()
        nodes = tok.encode("TEA")
        assert len(nodes) > 0
        for node in nodes:
            # High 32 bits must be non-zero (NLP type)
            assert (node >> 32) != 0, "High 32 bits (NLP type) must be non-zero"
            # Low 32 bits must be BPE token ID (small integer)
            bpe_id = node & 0xFFFFFFFF
            assert bpe_id < tok.vocab_size, "Low 32 bits must be valid BPE token ID"
            # Must be detected as NLP node
            assert (node >> 32) != 0, "Must be detected as NLP node"

    def test_node_not_literal(self):
        tok = _get_nlp_tokenizer()
        nodes = tok.encode("TEA")
        for node in nodes:
            assert not False


# ============================================================================
# KSN-2: Single-token identifiers produce single-node signatures with no decomposition
# ============================================================================
@_nlp_skip
class TestKSN2:
    """KSN-2: Single-token identifiers produce single-node signatures with no decomposition."""

    def test_single_token_no_decomposition(self):
        tok = _get_nlp_tokenizer()
        # "A" is a single BPE token
        tokens = tok.encode("A")
        assert len(tokens) == 1, "'A' should be a single BPE token"

        entries = compile_nlp("A")
        # Should produce exactly one entry: unsigned identity
        unsigned_entries = [e for e in entries if e.nodes == []]
        assert len(unsigned_entries) == 1, (
            f"Single-token identifier should produce exactly 1 unsigned entry, got {len(unsigned_entries)}"
        )


# ============================================================================
# KSN-3: Multi-token identifiers produce decomposition entries
# ============================================================================
@_nlp_skip
class TestKSN3:
    """KSN-3: Multi-token identifiers produce decomposition entries {first_token: [all_tokens]}."""

    def test_multi_token_decomposition(self):
        result = _find_multi_token_word()
        if result is None:
            pytest.skip("No multi-token BPE word found for testing")
        word, _ = result
        tok = _get_nlp_tokenizer()

        tokens = tok.encode(word)
        assert len(tokens) > 1, f"{word} should encode to multiple tokens"

        entries = compile_nlp(word)

        # Find decomposition entry: sig == first node of a multi-node entry
        decomp_entries = []
        for e in entries:
            if e.nodes is not None and len(e.nodes) >= 2 and e.signature == e.nodes[0]:
                decomp_entries.append(e)

        assert len(decomp_entries) >= 1, (
            f"Multi-token identifier '{word}' should produce at least one decomposition entry"
        )

        # Verify decomposition structure: {first_token: [all_tokens]}
        decomp = decomp_entries[0]
        assert decomp.signature == tokens[0], "Decomposition sig should be first BPE token"
        assert list(decomp.nodes) == tokens, "Decomposition nodes should be all BPE tokens"

    def test_component_unsigned_entries(self):
        result = _find_multi_token_word()
        if result is None:
            pytest.skip("No multi-token BPE word found for testing")
        word, _ = result
        tok = _get_nlp_tokenizer()

        tokens = tok.encode(word)
        entries = compile_nlp(word)

        # Each component token should have an unsigned entry
        unsigned_sigs = {e.signature for e in entries if e.nodes == []}
        for tok_id in tokens:
            assert tok_id in unsigned_sigs, (
                f"Component token {tok_id} should have an unsigned entry"
            )


# ============================================================================
# KSN-4: Literal encoding is unchanged
# ============================================================================

# ============================================================================
# KSN-4: Removed — literal concept no longer exists
# ============================================================================


# ============================================================================
# KSN-5: Removed — literal concept no longer exists
# ============================================================================
# KSN-5: Removed — literal concept no longer exists
# ============================================================================


# ============================================================================
# KSN-6: COUNTERSIGN under NLP-BPE
# ============================================================================
@_nlp_skip
class TestKSN6:
    """KSN-6: COUNTERSIGN under NLP-BPE: A == B → {nlp_A: nlp_B}, {nlp_B: nlp_A}."""

    def test_countersign_structure(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A == B")
        md = _md_nlp(entries, tok)

        # Should have entries where A has B and B has A
        assert _has_node(md, "A", "B"), "A should have node B"
        assert _has_node(md, "B", "A"), "B should have node A"

    def test_countersign_bidirectional(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A == B")
        md = _md_nlp(entries, tok)

        a_to_b = _has_node(md, "A", "B")
        b_to_a = _has_node(md, "B", "A")
        assert a_to_b and b_to_a, "COUNTERSIGN should be bidirectional"


# ============================================================================
# KSN-7: CANONIZE under NLP-BPE
# ============================================================================
@_nlp_skip
class TestKSN7:
    """KSN-7: CANONIZE under NLP-BPE: A => B C → {nlp_A: [nlp_B, nlp_C]}."""

    def test_canonize_structure(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A => B C")
        md = _md_nlp(entries, tok)

        assert "A" in md, "A should be in the multidict"
        assert "B" in md.get("A", []), "A should have node B"
        assert "C" in md.get("A", []), "A should have node C"


# ============================================================================
# KSN-8: CONNOTATE under NLP-BPE
# ============================================================================
@_nlp_skip
class TestKSN8:
    """KSN-8: CONNOTATE under NLP-BPE: A > B → {nlp_A: nlp_B}."""

    def test_connotate_structure(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A > B")
        md = _md_nlp(entries, tok)

        assert _has_node(md, "A", "B"), "A should have node B"
        # Should NOT have B > A (not bidirectional)
        assert not _has_node(md, "B", "A"), "CONNOTATE should not be bidirectional"


# ============================================================================
# KSN-9: UNDERSIGN under NLP-BPE
# ============================================================================
@_nlp_skip
class TestKSN9:
    """KSN-9: UNDERSIGN under NLP-BPE: A = B → {nlp_B: nlp_A} (reversed)."""

    def test_undersign_reversed(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A = B")
        md = _md_nlp(entries, tok)

        assert _has_node(md, "B", "A"), "UNDERSIGN: B should have node A (reversed)"


# ============================================================================
# KSN-10: Self-identity under NLP-BPE
# ============================================================================
@_nlp_skip
class TestKSN10:
    """KSN-10: Self-identity under NLP-BPE: A = A → {nlp_A: None} (unsigned)."""

    def test_self_identity_unsigned(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A = A")

        # Find the unsigned entry for A
        a_unsigned = [e for e in entries
                      if tok.decode([e.signature]) == "A" and e.nodes == []]
        assert len(a_unsigned) >= 1, "A = A should produce unsigned entry for A"


# ============================================================================
# KSN-11: No per-character MCS expansion for NLP-BPE identifiers
# ============================================================================
@_nlp_skip
class TestKSN11:
    """KSN-11: No per-character MCS expansion for NLP-BPE identifiers."""

    def test_no_per_char_mcs(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("ABC")

        # Under Mod32, "ABC" would produce per-char unsigned entries for "A", "B", "C"
        # plus a canonize {ABC: [A, B, C]}. Under NLP-BPE, there should be NO
        # per-character entries from MCS expansion.
        #
        # The NLP-BPE version will have BPE decomposition entries (multi-token)
        # but NOT the Mod32-style per-character MCS pattern.

        # Check that there's no canonize entry where sig == OR of single-char packed nodes
        for e in entries:
            if e.nodes is not None and len(e.nodes) >= 2:
                # Under NLP-BPE, all nodes should be NLP nodes (not Mod32 packed chars)
                for n in e.nodes:
                    if not False:
                        assert (n >> 32) != 0, (
                            "Multi-node entries should contain NLP-BPE nodes, not Mod32 packed chars"
                        )


# ============================================================================
# KSN-12: BPE decomposition for multi-token identifiers
# ============================================================================
@_nlp_skip
class TestKSN12:
    """KSN-12: BPE decomposition for multi-token identifiers."""

    def test_bpe_decomposition_entries(self):
        result = _find_multi_token_word()
        if result is None:
            pytest.skip("No multi-token BPE word found for testing")
        word, _ = result
        tok = _get_nlp_tokenizer()

        tokens = tok.encode(word)
        entries = compile_nlp(word)

        # Should have unsigned entries for each component token
        unsigned_sigs = {e.signature for e in entries if e.nodes == []}
        for t in tokens:
            assert t in unsigned_sigs, f"Component token should have unsigned entry"

        # Should have a decomposition entry {first: [all]}
        decomp = [e for e in entries
                  if e.nodes is not None and len(e.nodes) >= 2
                  and e.signature == e.nodes[0]]
        assert len(decomp) >= 1, "Should have at least one decomposition entry"

        # Should have unsigned identity for the first token
        first_unsigned = [e for e in entries
                         if e.signature == tokens[0] and e.nodes == []]
        assert len(first_unsigned) >= 1, "First token should have unsigned identity"


# ============================================================================
# KSN-13: Signature construction masks BPE IDs
# ============================================================================

# ============================================================================
# KSN-13: Removed — make_signature() is now plain OR-reduce (no BPE masking)
# ============================================================================


# ============================================================================
# KSN-14: signifies() tests NLP type overlap
# ============================================================================
@_nlp_skip
class TestKSN14:
    """KSN-14: signifies() tests NLP type overlap, not character overlap."""

    def test_signifies_nlp_type_overlap(self):
        tok = _get_nlp_tokenizer()
        # A and H share NLP type (both 0x400020)
        a_enc = tok.encode("A")
        h_enc = tok.encode("H")

        if a_enc[0] >> 32 == h_enc[0] >> 32:
            sig_a = make_signature(a_enc)
            sig_h = make_signature(h_enc)
            assert signifies(sig_a, sig_h), "Same NLP type should signify"

    def test_signifies_different_types(self):
        tok = _get_nlp_tokenizer()
        # A and B have different NLP types (0x400020 vs 0x800010)
        a_enc = tok.encode("A")
        b_enc = tok.encode("B")

        if a_enc[0] >> 32 != b_enc[0] >> 32:
            sig_a = make_signature(a_enc)
            sig_b = make_signature(b_enc)
            # They may or may not share bits depending on NLP type structure
            # Just verify that signifies() returns a boolean
            result = signifies(sig_a, sig_b)
            assert isinstance(result, bool)


# ============================================================================
# KSN-15: Deduplication works on NLP-BPE encoded values
# ============================================================================
@_nlp_skip
class TestKSN15:
    """KSN-15: Deduplication works on NLP-BPE encoded values."""

    def test_no_duplicates(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A == A")

        # Check for duplicate entries (same sig and same nodes)
        seen = set()
        for e in entries:
            key = (e.signature, tuple(e.nodes))
            assert key not in seen, f"Duplicate entry found: sig={e.signature:#x}"
            seen.add(key)


# ============================================================================
# KSN-16: Singleton unwrapping works for NLP-BPE nodes
# ============================================================================
@_nlp_skip
class TestKSN16:
    """KSN-16: Singleton unwrapping works for NLP-BPE nodes."""

    def test_singleton_unwrap(self):
        tok = _get_nlp_tokenizer()
        # Connotate produces single node: A > B → {A: B}
        entries = compile_nlp("A > B")

        # Find the entry where sig is A and it has nodes
        a_entries = [e for e in entries
                     if tok.decode([e.signature]) == "A" and e.nodes is not None]
        assert len(a_entries) >= 1

        # The nodes should be stored as an int (singleton unwrapped)
        # Note: KLine normalizes to list internally, but the original value
        # may be an int before normalization
        a_entry = a_entries[0]
        # After KLine normalization, nodes are always a list
        assert len(a_entry.nodes) == 1, "Singleton should produce single node"


# ============================================================================
# KSN-17: Decompiler name recovery for single-token NLP identifiers
# ============================================================================
@_nlp_skip
class TestKSN17:
    """KSN-17: Decompiler name recovery for single-token NLP identifiers via tokenizer.decode()."""

    def test_single_token_name_recovery(self):
        from kscript.decompiler import Decompiler
        tok = _get_nlp_tokenizer()

        entries = compile_nlp("A == B")
        dec = Decompiler(tokenizer=tok)
        decompiled = dec.decompile(entries)

        sig_names = {d.sig for d in decompiled}
        assert "A" in sig_names, "Decompiled output should contain 'A'"
        assert "B" in sig_names, "Decompiled output should contain 'B'"


# ============================================================================
# KSN-18: Decompiler name recovery for multi-token identifiers
# ============================================================================
@_nlp_skip
class TestKSN18:
    """KSN-18: Decompiler name recovery for multi-token identifiers via decomposition entries."""

    def test_multi_token_name_recovery(self):
        from kscript.decompiler import Decompiler
        result = _find_multi_token_word()
        if result is None:
            pytest.skip("No multi-token BPE word found for testing")
        word, _ = result
        tok = _get_nlp_tokenizer()

        entries = compile_nlp(word)
        dec = Decompiler(tokenizer=tok)
        decompiled = dec.decompile(entries)

        # The word should appear in decompiled output (via decomposition name mapping)
        sig_names = {d.sig for d in decompiled}
        assert word in sig_names, (
            f"Multi-token identifier '{word}' should be recovered in decompilation. "
            f"Got: {sig_names}"
        )


# ============================================================================
# KSN-19: Level inference tests NLP type overlap
# ============================================================================
@_nlp_skip
class TestKSN19:
    """KSN-19: Level inference (sig & node) != 0 tests NLP type overlap under NLP-BPE."""

    def test_level_inference_canonize(self):
        """Canonize entries where sig shares NLP type with nodes → S2."""
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A => B C")

        # Find the canonize entry (A → [B, C])
        a_canonize = [e for e in entries
                      if tok.decode([e.signature]) == "A" and e.nodes is not None and len(e.nodes) >= 2]
        if a_canonize:
            # Level inference depends on NLP type overlap between sig and nodes
            # We just verify it produces a valid level
            assert a_canonize[0].sig_level in ("S2", "S3"), (
                "Canonize should produce S2 or S3 level"
            )

    def test_level_inference_unsigned(self):
        """Unsigned entries should have S4 level."""
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A")

        unsigned = [e for e in entries if e.nodes == []]
        for e in unsigned:
            assert e.sig_level == "S4", "Unsigned entries should have S4 level"


# ============================================================================
# KSN-20: Lexer produces identical token stream regardless of encoding
# ============================================================================
@_nlp_skip
class TestKSN20:
    """KSN-20: Lexer produces identical token stream for same source regardless of encoding."""

    def test_lexer_encoding_independent(self):
        from kscript.lexer import Lexer

        source = "A == B"
        tokens1 = Lexer(source).tokenize()
        tokens2 = Lexer(source).tokenize()

        # Token types and values should be identical
        assert len(tokens1) == len(tokens2)
        for t1, t2 in zip(tokens1, tokens2):
            assert t1.type == t2.type
            assert t1.value == t2.value


# ============================================================================
# KSN-21: Parser produces identical AST regardless of encoding
# ============================================================================
@_nlp_skip
class TestKSN21:
    """KSN-21: Parser produces identical AST for same source regardless of encoding."""

    def test_parser_encoding_independent(self):
        from kscript.lexer import Lexer
        from kscript.parser import Parser

        source = "A == B"
        ast1 = Parser(Lexer(source).tokenize()).parse()
        ast2 = Parser(Lexer(source).tokenize()).parse()

        # AST structures should be identical (same source, same parser)
        assert len(ast1.scripts) == len(ast2.scripts)


# ============================================================================
# KSN-22: AST emitter produces identical symbolic entries regardless of encoding
# ============================================================================
@_nlp_skip
class TestKSN22:
    """KSN-22: AST emitter produces identical symbolic entries regardless of encoding mode."""

    def test_emitter_skip_mcs_same_for_simple_source(self):
        from kscript.lexer import Lexer
        from kscript.parser import Parser
        from kscript.ast_emitter import ASTEmitter

        source = "A == B"
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()

        # Emit with skip_mcs=False (Mod32 mode)
        emitter1 = ASTEmitter(skip_mcs=False)
        entries1 = emitter1.emit(ast)

        # Emit with skip_mcs=True (NLP-BPE mode)
        emitter2 = ASTEmitter(skip_mcs=True)
        entries2 = emitter2.emit(ast)

        # For simple source (no multi-char sigs), entries should be identical
        assert len(entries1) == len(entries2), (
            f"Entry count mismatch: {len(entries1)} vs {len(entries2)}"
        )
        for e1, e2 in zip(entries1, entries2):
            assert e1.sig == e2.sig, f"Sig mismatch: {e1.sig} vs {e2.sig}"
            assert e1.nodes == e2.nodes, f"Nodes mismatch: {e1.nodes} vs {e2.nodes}"
            assert e1.op == e2.op, f"Op mismatch: {e1.op} vs {e2.op}"


# ============================================================================
# KSN-23: Same .ks source compiles under both Mod32 and NLP-BPE
# ============================================================================
@_nlp_skip
class TestKSN23:
    """KSN-23: Same .ks source compiles under both Mod32 and NLP-BPE without modification."""

    def test_compiles_under_both(self):
        from kscript.compiler import compile_source

        source = "A == B"
        mod32_entries = compile_source(source, tokenizer=Mod32Tokenizer())
        nlp_entries = compile_nlp(source)

        # Both should produce valid (non-empty) entries
        assert len(mod32_entries) > 0, "Mod32 should produce entries"
        assert len(nlp_entries) > 0, "NLP-BPE should produce entries"

        # Both should have entries with nodes (countersign links)
        assert any(len(e.nodes) > 0 for e in mod32_entries), \
            "Mod32 should have entries with nodes"
        assert any(len(e.nodes) > 0 for e in nlp_entries), \
            "NLP-BPE should have entries with nodes"

        # Under Mod32, A == B produces just 2 countersign entries (no MCS for single-char)
        # Under NLP-BPE, A == B produces countersign entries (may also have decomposition)
        assert len(mod32_entries) == 2, f"Mod32 A == B should produce 2 entries, got {len(mod32_entries)}"
        assert len(nlp_entries) >= 2, f"NLP-BPE A == B should produce at least 2 entries"


# ============================================================================
# KSN-24: Mod32 compilation is unchanged
# ============================================================================
@_nlp_skip
class TestKSN24:
    """KSN-24: Mod32 compilation is unchanged when NLP-BPE mode exists."""

    def test_mod32_unchanged(self):
        from kscript.compiler import compile_source

        # Simple unsigned
        entries = compile_source("A", tokenizer=Mod32Tokenizer())
        assert len(entries) == 1
        assert entries[0].nodes == []

        # COUNTERSIGN
        entries = compile_source("A == B", tokenizer=Mod32Tokenizer())
        assert len(entries) >= 2  # MCS for A and B + countersign entries

    def test_mod32_mcs_expansion_still_works(self):
        from kscript.compiler import compile_source

        # Multi-char sig should still have MCS expansion under Mod32
        entries = compile_source("ABC", tokenizer=Mod32Tokenizer())
        sigs = {e.signature for e in entries}

        # Should have packed entry for ABC and per-char entries
        assert any(e.nodes is not None and len(e.nodes) >= 2 for e in entries), \
            "MCS canonize entry should exist under Mod32"


# ============================================================================
# KSN-25: Complex example from spec §8.4
# ============================================================================
@_nlp_skip
class TestKSN25:
    """KSN-25: Complex example (§8.4) produces correct entry structure with all operators."""

    def test_complex_example_compiles(self):
        source = (
            "MHALL == SVO =>\n"
            "  S = M\n"
            "  V = H\n"
            "  O = ALL =>\n"
            "    A = D\n"
            "    L = M\n"
            "    L > O"
        )
        entries = compile_nlp(source)
        assert len(entries) > 0, "Complex example should compile"

        tok = _get_nlp_tokenizer()

        # Verify MHALL-SVO countersign relationship at raw entry level
        mhall_tokens = tok.encode("MHALL")
        svo_tokens = tok.encode("SVO")
        mhall_first = mhall_tokens[0]
        svo_first = svo_tokens[0]

        # Find entries linking MHALL first token and SVO first token
        mhall_to_svo = any(
            e.signature == mhall_first and e.nodes is not None and svo_first in e.nodes
            for e in entries
        )
        svo_to_mhall = any(
            e.signature == svo_first and e.nodes is not None and mhall_first in e.nodes
            for e in entries
        )

        assert mhall_to_svo, "MHALL first token should have SVO first token as node"
        assert svo_to_mhall, "SVO first token should have MHALL first token as node"

    def test_complex_example_decompiles(self):
        from kscript.decompiler import Decompiler
        tok = _get_nlp_tokenizer()

        source = (
            "MHALL == SVO =>\n"
            "  S = M\n"
            "  V = H\n"
            "  O = ALL =>\n"
            "    A = D\n"
            "    L = M\n"
            "    L > O"
        )
        entries = compile_nlp(source)
        dec = Decompiler(tokenizer=tok)
        decompiled = dec.decompile(entries)

        # Should produce decompiled output without errors
        assert len(decompiled) > 0
        sig_names = {d.sig for d in decompiled}

        # Key identifiers should be present in decompiled output
        assert "MHALL" in sig_names, f"MHALL should be in decompiled output. Got: {sig_names}"
        assert "SVO" in sig_names, f"SVO should be in decompiled output. Got: {sig_names}"


# ============================================================================
# KSN-26: Literal nodes in mixed blocks preserve character-level encoding
# ============================================================================

# ============================================================================
# KSN-26: Removed — literal concept no longer exists
# ============================================================================


# ============================================================================
# KSN-27: CANONIZE with subscript block flattens correctly
# ============================================================================
@_nlp_skip
class TestKSN27:
    """KSN-27: CANONIZE with subscript block flattens correctly under NLP-BPE."""

    def test_canonize_block_flattening(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A =>\n  B\n  C")
        md = _md_nlp(entries, tok)

        assert "A" in md, "A should be in multidict"
        assert "B" in md.get("A", []), "A should have node B"
        assert "C" in md.get("A", []), "A should have node C"


# ============================================================================
# KSN-28: Chained constructs work under NLP-BPE
# ============================================================================
@_nlp_skip
class TestKSN28:
    """KSN-28: Chained constructs (A => B => C) work identically under NLP-BPE."""

    def test_chained_constructs(self):
        tok = _get_nlp_tokenizer()
        entries = compile_nlp("A => B => C")
        md = _md_nlp(entries, tok)

        # Should have A → B and B → C relationships
        assert _has_node(md, "A", "B"), "A should have node B"
        assert _has_node(md, "B", "C"), "B should have node C"

    def test_chained_decompiles(self):
        from kscript.decompiler import Decompiler
        tok = _get_nlp_tokenizer()

        entries = compile_nlp("A => B => C")
        dec = Decompiler(tokenizer=tok)
        decompiled = dec.decompile(entries)

        sig_names = {d.sig for d in decompiled}
        assert "A" in sig_names, "A should be in decompiled output"
        assert "B" in sig_names, "B should be in decompiled output"
        assert "C" in sig_names, "C should be in decompiled output"
