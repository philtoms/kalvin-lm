"""Tests for KScript interpreter with identity/compound KLine semantics."""

import pytest

from kalvin.model import KLine
from kalvin.significance import S1_BIT, build_s1, build_s2, build_s3, has_s1
from kscript import interpret_script, Interpreter


class TestIdentityKLines:
    """Tests for identity KLine creation (single-char identifiers)."""

    def test_single_char_identity_kline(self):
        """Single char 'M' creates identity KLine with S1|token signature."""
        result = interpret_script("M")

        # Should have at least one kline
        assert len(result.model) >= 1

        # Find the M kline
        assert "M" in result.symbol_table
        m_sig = result.symbol_table["M"]
        m_kline = result.model.find_by_key(m_sig)

        assert m_kline is not None
        # Identity kline has S1 bit set
        assert has_s1(m_kline.signature)
        # Identity kline has exactly one node (the token)
        assert len(m_kline.nodes) == 1

    def test_identity_signature_format(self):
        """Identity KLine signature = S1_BIT | token, nodes = [token]."""
        result = interpret_script("A")

        a_sig = result.symbol_table["A"]
        a_kline = result.model.find_by_key(a_sig)

        # The signature should have S1_BIT
        assert a_kline.signature & S1_BIT != 0
        # The lower bits should be the token
        token = a_kline.nodes[0]
        # signature should be S1_BIT | token
        assert a_kline.signature == (S1_BIT | token)

    def test_multiple_identity_klines(self):
        """Multiple single-char identifiers create separate identity klines."""
        result = interpret_script("A B C")

        assert "A" in result.symbol_table
        assert "B" in result.symbol_table
        assert "C" in result.symbol_table


class TestCompoundKLines:
    """Tests for compound KLine creation (multi-char identifiers)."""

    def test_multi_char_compound_kline(self):
        """Multi-char 'ALL' creates compound KLine."""
        result = interpret_script("ALL")

        assert "ALL" in result.symbol_table
        all_sig = result.symbol_table["ALL"]
        all_kline = result.model.find_by_key(all_sig)

        assert all_kline is not None
        # Compound kline has 3 nodes (A, L, L identity signatures)
        assert len(all_kline.nodes) == 3
        # Each node should have S1 bit (identity signature)
        for node in all_kline.nodes:
            assert has_s1(node)

    def test_compound_signature_includes_s1_and_s2(self):
        """Compound KLine signature includes S1 and S2 bits."""
        result = interpret_script("AB")

        ab_sig = result.symbol_table["AB"]
        ab_kline = result.model.find_by_key(ab_sig)

        # Should have S1 bit
        assert has_s1(ab_kline.signature)
        # Should have S2 bits (non-zero S2 portion)
        from kalvin.significance import get_s2
        assert get_s2(ab_kline.signature) > 0

    def test_compound_includes_identity_klines(self):
        """Compound KLine creation also creates identity klines for each char."""
        result = interpret_script("ABC")

        # Should have identity klines for A, B, C (as part of compound creation)
        # They won't be in symbol_table by name, but they should be in model
        # as nodes of the compound kline
        abc_sig = result.symbol_table["ABC"]
        abc_kline = result.model.find_by_key(abc_sig)

        # Each node is an identity signature
        for node_sig in abc_kline.nodes:
            node_kline = result.model.find_by_key(node_sig)
            assert node_kline is not None
            assert has_s1(node_kline.signature)


class TestSignifyRelationships:
    """Tests for operator relationships via signify()."""

    def test_s1_equals_relationship(self):
        """S1 (=) operator establishes countersigned relationship."""
        result = interpret_script("A = B")

        a_sig = result.symbol_table["A"]
        b_sig = result.symbol_table["B"]

        a_kline = result.model.find_by_key(a_sig)
        b_kline = result.model.find_by_key(b_sig)

        # S1 relationship creates bidirectional links
        # After signify(A, B, S1):
        # - Model should have KLine with A's sig and B's nodes
        # - Model should have KLine with B's sig and A's nodes

        # Find klines with A's signature
        a_klines = [k for k in result.model if k.signature == a_sig]
        b_klines = [k for k in result.model if k.signature == b_sig]

        # Should have multiple klines now (original + bidirectional)
        assert len(a_klines) >= 1
        assert len(b_klines) >= 1

    def test_s2_canonical_relationship(self):
        """S2 (=>) operator verifies compound signature."""
        result = interpret_script("AB => A B")

        ab_sig = result.symbol_table["AB"]
        ab_kline = result.model.find_by_key(ab_sig)

        assert ab_kline is not None

    def test_s3_connotative_relationship(self):
        """S3 (<) operator establishes 'is a kind of' relationship."""
        result = interpret_script("M < S")

        m_sig = result.symbol_table["M"]
        s_sig = result.symbol_table["S"]

        m_kline = result.model.find_by_key(m_sig)
        s_kline = result.model.find_by_key(s_sig)

        assert m_kline is not None
        assert s_kline is not None

        # S3 relationship should create bidirectional links
        # Find klines with M's signature
        m_klines = [k for k in result.model if k.signature == m_sig]
        assert len(m_klines) >= 1

    def test_s4_negative_relationship(self):
        """S4 (!=) operator establishes negative relationship."""
        result = interpret_script("A != B")

        a_sig = result.symbol_table["A"]
        b_sig = result.symbol_table["B"]

        a_kline = result.model.find_by_key(a_sig)
        b_kline = result.model.find_by_key(b_sig)

        assert a_kline is not None
        assert b_kline is not None


class TestComplexScripts:
    """Tests for complex multi-line scripts."""

    def test_mhall_equals_svo(self):
        """Test MHALL = SVO pattern."""
        result = interpret_script("MHALL = SVO")

        # Should have compound klines for MHALL and SVO
        assert "MHALL" in result.symbol_table
        assert "SVO" in result.symbol_table

        mhall_sig = result.symbol_table["MHALL"]
        svo_sig = result.symbol_table["SVO"]

        mhall_kline = result.model.find_by_key(mhall_sig)
        svo_kline = result.model.find_by_key(svo_sig)

        # Compound klines should have multiple nodes
        assert len(mhall_kline.nodes) >= 1
        assert len(svo_kline.nodes) >= 1
        # Each node should be an identity signature (has S1 bit)
        for node in mhall_kline.nodes:
            assert has_s1(node)
        for node in svo_kline.nodes:
            assert has_s1(node)

    def test_complex_multiline_script(self):
        """Test complex multi-line script with nested relationships."""
        source = """
            MHALL = SVO =>
                S < M
                V < H
                O < ALL
        """
        result = interpret_script(source)

        # Should have klines for all identifiers
        assert "MHALL" in result.symbol_table
        assert "SVO" in result.symbol_table
        assert "S" in result.symbol_table
        assert "M" in result.symbol_table
        assert "V" in result.symbol_table
        assert "H" in result.symbol_table
        assert "O" in result.symbol_table
        assert "ALL" in result.symbol_table

    def test_load_save_paths(self):
        """Test that load/save paths are captured."""
        result = interpret_script("""
            load /input.bin
            A = B
            save /output.bin
        """)
        assert result.load_paths == ["/input.bin"]
        assert result.save_path == "/output.bin"

    def test_attention_before_relationship(self):
        """Test attention is processed before relationships (yield point semantics)."""
        result = interpret_script("A? => B")

        # A should be attended first (yield), then relationship to B
        assert "A" in result.symbol_table
        assert "B" in result.symbol_table

        # Attention point should be recorded for A
        assert len(result.attention_klines) == 1
        # Check that A was processed (model updated)
        # The relationship should also be processed
        assert "B" in result.symbol_table

    def test_attention_after_relationship(self):
        """Test attention after relationship."""
        result = interpret_script("A => B?")

        # Both should be processed
        assert "A" in result.symbol_table
        assert "B" in result.symbol_table

        # Attention point should be recorded for B
        assert len(result.attention_klines) == 1
