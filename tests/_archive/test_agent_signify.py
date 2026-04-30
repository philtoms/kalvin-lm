"""Tests for Agent._signify - internal significance calculation of a single KLine.

_signify evaluates a KLine against a frame to determine its significance level:
- S1: Identity, countersigned, or fully canonized (full overlap between sig and nodes)
- S2: Partially canonized, or underfit/overfit (overlap between sig and nodes)
- S3: Connotation (no overlap between sig and nodes)
  - S4: Unsigned (nodes is None)

"""

from __future__ import annotations

import pytest

from kalvin.abstract import KLine
from kalvin.agent import Agent
from kalvin.model import Model
from kalvin.significance import Int32Significance
from kalvin.mod_tokenizer import Mod32Tokenizer

_sig = Int32Significance()
_tok = Mod32Tokenizer()

# ---------------------------------------------------------------------------
# Symbolic signatures (pack=True → non-literal, bits 1-31 only)
#
# Single characters map to unique bit positions with no collision with
# significance bits (32-63), matching the pattern in test_graph.py.
# ---------------------------------------------------------------------------
A = _tok.encode("A", pack=True)[0]    # 0x0002
B = _tok.encode("B", pack=True)[0]    # 0x0004
C = _tok.encode("C", pack=True)[0]    # 0x0008
D = _tok.encode("D", pack=True)[0]    # 0x0010
E = _tok.encode("E", pack=True)[0]    # 0x0020
F = _tok.encode("F", pack=True)[0]    # 0x0040
G = _tok.encode("G", pack=True)[0]    # 0x0080

# Composite signatures (OR of constituent character bits)
AB = A | B                            # 0x0006
ABC = A | B | C                       # 0x000E
AF = A | F                            # 0x0042

# Literal tokens (pack=False → LITERAL_BIT set, is_literal returns True)
LIT_A = _tok.encode("A", pack=False)[0]   # 0x00083
LIT_B = _tok.encode("B", pack=False)[0]   # 0x00085


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent(
    model: Model | None = None,
) -> tuple[Agent, Model]:
    """Build an Agent with Mod32Tokenizer and return (agent, frame).

    The returned *frame* is the same Model passed to ``_signify`` in each test.
    """
    frame = model or Model()
    agent = Agent(model=frame, tokenizer=_tok)
    return agent, frame


# ===================================================================
# 1. Identity → S1
#    is_identity(sig, nodes) is True when both are ints with equal
#    stripped (token) bits.
# ===================================================================

class TestSignifyIdentity:
    """sig and nodes are equal ints (same token bits) → S1."""

    def test_exact_match(self):
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_sig_with_significance_bits_stripped(self):
        """Significance bits in sig are stripped before comparison."""
        agent, frame = _agent()
        sig = _sig.S1 | A  # S1 bit in high 32, token in low 32
        kline = KLine(signature=sig, nodes=A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_different_token_bits_not_identity(self):
        """Different token values fall through to later checks.

        sig ≠ nodes (stripped) → not identity.  Nodes is int (signed) but
        not in frame and not literal → falls through to canonization.
        make_signature handles int nodes gracefully, producing S3 (connotation)
        when there is no bit overlap.
        """
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=B)
        # A & B = 0 → no overlap → S3 (connotation)
        assert agent._signify(kline, frame) == _sig.S3


# ===================================================================
# 2. Unsigned → S4
#    nodes is None (is_unsigned returns True).
# ===================================================================

class TestSignifyUnsigned:
    """nodes is None → S4."""

    def test_none_nodes(self):
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=None)
        assert agent._signify(kline, frame) == _sig.S4

    def test_none_nodes_high_sig(self):
        agent, frame = _agent()
        kline = KLine(signature=_sig.S2 | A, nodes=None)
        assert agent._signify(kline, frame) == _sig.S4


# ===================================================================
# 3. Undersigned / Countersigned → S1
#    nodes is a single int (is_signed = True) and either:
#      a) the node exists in the frame, or
#      b) the tokenizer considers it a literal (LITERAL_BIT set).
# ===================================================================

class TestSignifyUndersigned:
    """nodes is a single int; resolved via frame lookup or literal check."""

    def test_in_frame(self):
        """Node found in frame → S1."""
        frame = Model([KLine(signature=A, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=_sig.S2 | A, nodes=A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_is_literal(self):
        """Node not in frame but tokenizer says literal → S1.

        Uses different token values for sig and nodes so identity check
        doesn't short-circuit.  The literal node (LIT_B) has LITERAL_BIT
        set, so is_literal returns True.
        """
        agent, frame = _agent()
        kline = KLine(signature=_sig.S2 | A, nodes=LIT_B)
        frame.add(kline)
        assert agent._signify(kline, frame) == _sig.S1

    def test_in_frame_and_literal(self):
        """Both conditions true — find_kline succeeds AND is_literal.

        Uses a literal node value (LIT_B) as a KLine signature in the
        frame so both conditions in the undersigned check are True.
        """
        frame = Model([KLine(signature=LIT_B, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=_sig.S3 | A, nodes=LIT_B)
        frame.add(kline)
        assert agent._signify(kline, frame) == _sig.S1

    def test_neither_in_frame_nor_literal_raises(self):
        """Signed int node not in frame and not literal falls through to
        canonization.  make_signature handles int nodes gracefully.

        In normal usage this path is unreachable because the kline would
        already exist in the frame (rationalise checks before calling
        _signify).  With Mod32Tokenizer the int node is handled without
        error — producing S3 when there is no bit overlap.

        Note: sig and nodes must have *different* token bits so that
        is_identity returns False (strip compares token bits only).
        """
        agent, frame = _agent()
        kline = KLine(signature=B, nodes=A)  # different symbolic tokens
        # B & A = 0 → no overlap → S3 (connotation)
        assert agent._signify(kline, frame) == _sig.S3


# ===================================================================
# 4. Canonization – fully canonized → S1
#    nodes is a list, node_sig (OR of non-literal nodes via make_signature)
#    == sig, and every node is resolvable (node_sig is in frame, or all
#    nodes are literals).
# ===================================================================

class TestSignifyCanonizedS1:
    """Canonized nodes that are fully grounded → S1."""

    def test_ns_in_frame(self):
        """node_sig (OR of all nodes) exists as a key in the frame → S1."""
        frame = Model([KLine(signature=AB, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=AB, nodes=[A, B])
        assert agent._signify(kline, frame) == _sig.S1

    def test_all_nodes_literal(self):
        """All nodes are literal tokens → S1.

        make_signature skips literal nodes (returns 0), so sig must also
        be 0 for the canonization path to activate.  Since every node
        is literal, the inner loop's ``not is_literal(n)`` is False for
        all nodes → S1.
        """
        agent, frame = _agent()
        kline = KLine(signature=0, nodes=[LIT_A, LIT_B])
        assert agent._signify(kline, frame) == _sig.S1

    def test_ns_in_frame_single_node(self):
        """Single-node list where node_sig matches sig and is in frame → S1.

        Identity is skipped because sig is int, nodes is list
        (is_identity needs both int).  make_signature([A]) = A = sig,
        A is in frame → S1.
        """
        frame = Model([KLine(signature=A, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=A, nodes=[A])
        assert agent._signify(kline, frame) == _sig.S1


# ===================================================================
# 5. Canonization – partially canonized → S2
#    nodes is a list, node_sig == sig, node_sig is NOT in frame, and at least one
#    node is not a literal.
# ===================================================================

class TestSignifyCanonizedS2:
    """Partially canonized nodes → S2."""

    def test_ns_not_in_frame_non_literal_node(self):
        agent, frame = _agent()
        kline = KLine(signature=AB, nodes=[A, B])
        assert agent._signify(kline, frame) == _sig.S2

    def test_mixed_literal_and_non_literal(self):
        """One literal, one non-literal, node_sig not in frame → S2.

        make_signature([A, LIT_B]) = A (literal skipped).  sig=A,
        nodes_sig=A → canonized path.  A not in frame and A not literal
        → S2.
        """
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=[A, LIT_B])
        assert agent._signify(kline, frame) == _sig.S2

    def test_single_non_literal_node(self):
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=[A])
        assert agent._signify(kline, frame) == _sig.S2

    def test_ns_in_frame_overrides_non_literal(self):
        """Even with non-literal nodes, if node_sig is in frame → S1 (not S2)."""
        frame = Model([KLine(signature=AB, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=AB, nodes=[A, B])
        assert agent._signify(kline, frame) == _sig.S1


# ===================================================================
# 6. Underfit → S2
#    nodes is a list, node_sig != sig (from make_signature), sig & node_sig != 0.
# ===================================================================

class TestSignifyUnderfitS2:
    """sig and node_sig share bits but node_sig != sig → S2."""

    def test_basic_underfit(self):
        """sig has extra bits beyond nodes → underfit → S2.

        AF = A|F has bits 1 and 6; nodes [F] → node_sig = F (bit 6 only).
        sig & node_sig = F (non-zero), sig != node_sig → S2.
        """
        agent, frame = _agent()
        kline = KLine(signature=AF, nodes=[F])
        assert agent._signify(kline, frame) == _sig.S2

    def test_sig_extra_bits(self):
        """sig = ABC (3 bits), node_sig = AB (2 bits) → S2.

        make_signature([B, A]) = B|A = AB.  sig = ABC != node_sig.
        sig & node_sig = AB (non-zero) → S2.
        """
        agent, frame = _agent()
        kline = KLine(signature=ABC, nodes=[B, A])
        assert agent._signify(kline, frame) == _sig.S2


# ===================================================================
# 7. Connotation → S3
#    nodes is a list, node_sig != sig, and sig & node_sig == 0 (no bit overlap).
# ===================================================================

class TestSignifyConnotationS3:
    """No overlap between sig and nodes → S3 (connotation)."""

    def test_no_overlap(self):
        agent, frame = _agent()
        # D = 0x0010, A = 0x0002 → no shared bits
        kline = KLine(signature=D, nodes=[A])
        assert agent._signify(kline, frame) == _sig.S3

    def test_no_overlap_multiple_nodes(self):
        agent, frame = _agent()
        # E = 0x0020, nodes [A, B, C] = 0x000E → no shared bits
        kline = KLine(signature=E, nodes=[A, B, C])
        assert agent._signify(kline, frame) == _sig.S3

    def test_empty_node_list(self):
        """Empty list: make_signature returns 0, sig != 0,
        sig & 0 = 0 → S3.
        """
        agent, frame = _agent()
        kline = KLine(signature=D, nodes=[])
        assert agent._signify(kline, frame) == _sig.S3

    def test_zero_sig_empty_nodes(self):
        """Both zero: node_sig=0 == sig=0 → canonization path,
        empty node list → loop doesn't execute → S1.
        """
        agent, frame = _agent()
        kline = KLine(signature=0, nodes=[])
        assert agent._signify(kline, frame) == _sig.S1


# ===================================================================
# 8. Significance bits in sig during canonization
# ===================================================================

class TestSignifyWithSignificanceBits:
    """sig may carry significance-level bits; canonization compares
    the full 64-bit values."""

    def test_sig_has_s2_bit(self):
        """sig = S2 | A, nodes = [A]: make_signature([A]) = A.
        sig != node_sig (S2 bit present).  sig & node_sig = A (non-zero) → S2.
        """
        agent, frame = _agent()
        kline = KLine(signature=_sig.S2 | A, nodes=[A])
        result = agent._signify(kline, frame)
        assert result == _sig.S2

    def test_sig_has_s3_bit(self):
        """sig = S3 | A, nodes = [A]: make_signature([A]) = A.
        sig != node_sig (S3 bit present).  sig & node_sig = A (non-zero) → S2.
        """
        agent, frame = _agent()
        kline = KLine(signature=_sig.S3 | A, nodes=[A])
        result = agent._signify(kline, frame)
        assert result == _sig.S2


# ===================================================================
# 9. Precedence / ordering verification
# ===================================================================

class TestSignifyPrecedence:
    """Verify that checks are evaluated in the correct order."""

    def test_identity_before_unsigned(self):
        """Even if nodes were None-ish, identity check runs first.

        But identity requires both args to be ints, so None can't match.
        This test just ensures unsigned still works."""
        agent, frame = _agent()
        kline = KLine(signature=0, nodes=None)
        assert agent._signify(kline, frame) == _sig.S4

    def test_signed_before_canonization(self):
        """A signed int node that IS in frame returns S1, not S2/S3."""
        frame = Model([KLine(signature=A, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=A, nodes=A)
        # Identity catches this first (same stripped value) → S1
        assert agent._signify(kline, frame) == _sig.S1

    def test_canonization_before_underfit(self):
        """If node_sig == sig, we never reach the underfit check."""
        frame = Model([KLine(signature=AB, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=AB, nodes=[A, B])
        # node_sig = make_signature([A,B]) = AB == sig, node_sig in frame → S1
        assert agent._signify(kline, frame) == _sig.S1

    def test_underfit_before_s3(self):
        """sig & node_sig != 0 yields S2, not S3 (connotation)."""
        agent, frame = _agent()
        kline = KLine(signature=AF, nodes=[F])
        # make_signature([F]) = F.  AF != F, AF & F = F (non-zero) → S2
        assert agent._signify(kline, frame) == _sig.S2


# ===================================================================
# 10. Dynamic tests – significance rises as model grows
#
#    The canonization and undersigned branches query the frame,
#    so adding klines between _signify calls can raise the result.
#
#    Rising paths:
#      Canonization (node_sig == sig):
#        S2 (node_sig not grounded) → S1 (node_sig grounded in frame)
#      Undersigned (int node):
#        TypeError (node missing) → S1 (node grounded in frame)
#
#    Static paths (no frame involvement): S4 unsigned, S2 underfit,
#    S3 connotation (no overlap).
# ===================================================================


class TestDynamicCanonizationRise:
    """Canonized klines rise from S2 → S1 when node_sig is added to frame.
    """

    def test_two_node_s2_then_s1_by_adding_ns(self):
        """S2 → S1 when node_sig is added to frame.

        kline(sig=AB, nodes=[A, B]): node_sig = make_signature([A,B]) = AB
        Empty frame → S2.  Add AB to frame → S1.
        """
        agent, frame = _agent()
        kline = KLine(signature=AB, nodes=[A, B])

        # Step 1: empty frame → S2
        assert agent._signify(kline, frame) == _sig.S2

        # Step 2: ground node_sig in frame
        frame.add(KLine(signature=AB, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_three_node_s2_then_s1_by_adding_ns(self):
        """Three-node canonized kline rises from S2 → S1.

        kline(sig=ABC, nodes=[A, B, C]): node_sig = make_signature([A,B,C]) = ABC
        """
        agent, frame = _agent()
        kline = KLine(signature=ABC, nodes=[A, B, C])

        assert agent._signify(kline, frame) == _sig.S2

        frame.add(KLine(signature=ABC, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_single_node_s2_then_s1_by_adding_ns(self):
        """Single-node list rises from S2 → S1.

        kline(sig=A, nodes=[A]): node_sig = make_signature([A]) = A.
        Identity is skipped because sig is int, nodes is list
        (is_identity needs both int).
        """
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=[A])

        assert agent._signify(kline, frame) == _sig.S2

        frame.add(KLine(signature=A, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_s2_persists_until_ns_is_grounded(self):
        """Adding unrelated klines to frame does NOT raise significance."""
        agent, frame = _agent()
        kline = KLine(signature=AB, nodes=[A, B])

        assert agent._signify(kline, frame) == _sig.S2

        # Add unrelated klines
        frame.add(KLine(signature=C, nodes=[]))
        frame.add(KLine(signature=D, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S2

        # Now add the actual node_sig
        frame.add(KLine(signature=AB, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_multiple_klines_rise_together_when_ns_shared(self):
        """Several klines with the same node_sig all rise when node_sig is grounded."""
        agent, frame = _agent()

        k1 = KLine(signature=AB, nodes=[A, B])
        k2 = KLine(signature=AB, nodes=[B, A])  # different order

        assert agent._signify(k1, frame) == _sig.S2
        assert agent._signify(k2, frame) == _sig.S2

        frame.add(KLine(signature=AB, nodes=[]))

        assert agent._signify(k1, frame) == _sig.S1
        assert agent._signify(k2, frame) == _sig.S1

    def test_sequential_grounding_multiple_ns(self):
        """Different klines with different node_sig values rise independently."""
        agent, frame = _agent()

        ka = KLine(signature=A, nodes=[A])      # node_sig = A
        kab = KLine(signature=AB, nodes=[A, B])  # node_sig = AB
        kabc = KLine(signature=ABC, nodes=[A, B, C])  # node_sig = ABC

        # All start at S2
        assert agent._signify(ka, frame) == _sig.S2
        assert agent._signify(kab, frame) == _sig.S2
        assert agent._signify(kabc, frame) == _sig.S2

        # Ground A — only ka rises (ab and abc still missing their node_sig)
        frame.add(KLine(signature=A, nodes=[]))
        assert agent._signify(ka, frame) == _sig.S1
        assert agent._signify(kab, frame) == _sig.S2
        assert agent._signify(kabc, frame) == _sig.S2

        # Ground AB — kab rises, kabc still S2
        frame.add(KLine(signature=AB, nodes=[]))
        assert agent._signify(kab, frame) == _sig.S1
        assert agent._signify(kabc, frame) == _sig.S2

        # Ground ABC — kabc rises
        frame.add(KLine(signature=ABC, nodes=[]))
        assert agent._signify(kabc, frame) == _sig.S1

    def test_rise_does_not_affect_static_underfit(self):
        """Grounding a kline in the frame does not change underfit results.

        Underfit path (node_sig != sig, sig & node_sig != 0) is pure bit math —
        no frame lookup.
        """
        agent, frame = _agent()
        kline_underfit = KLine(signature=AF, nodes=[F])

        assert agent._signify(kline_underfit, frame) == _sig.S2

        # Add unrelated and overlapping klines
        frame.add(KLine(signature=F, nodes=[]))
        frame.add(KLine(signature=AF, nodes=[]))

        # Underfit result unchanged — make_signature([F]) = F, F != AF
        assert agent._signify(kline_underfit, frame) == _sig.S2

    def test_rise_does_not_affect_static_s3_no_overlap(self):
        """Grounding klines does not change S3 no-overlap results.

        Connotation path (node_sig != sig, sig & node_sig == 0) is pure bit math.
        """
        agent, frame = _agent()
        kline = KLine(signature=E, nodes=[A, B])

        assert agent._signify(kline, frame) == _sig.S3

        frame.add(KLine(signature=E, nodes=[]))
        frame.add(KLine(signature=A, nodes=[]))

        # Still S3 — make_signature([A,B]) = AB, E & AB = 0 (pure bit math)
        assert agent._signify(kline, frame) == _sig.S3


class TestDynamicUndersignedRise:
    """Undersigned (S3) klines become S1 when node is added to frame.
    """

    def test_s3_then_s1_by_adding_node(self):
        """Signed kline goes from S3 (connotation) → S1 when grounded.
        """
        agent, frame = _agent()
        kline = KLine(signature=B, nodes=A)  # sig≠nodes tokens

        # Step 1: node not in frame → falls through → S3 (no overlap)
        assert agent._signify(kline, frame) == _sig.S3

        # Step 2: not grounded, still S3
        frame.add(kline)
        assert agent._signify(kline, frame) == _sig.S3

        # Step 3: ground the node
        frame.add(KLine(signature=A, nodes=B))
        assert agent._signify(kline, frame) == _sig.S1

    def test_multiple_undersigned_rise_independently(self):
        """Different undersigned klines with different nodes rise
        independently as their respective nodes are grounded.
        """
        agent, frame = _agent()

        ka = KLine(signature=A, nodes=B)  # references B
        kb = KLine(signature=B, nodes=A)  # references A

        # Both S3 initially (no overlap → connotation)
        assert agent._signify(ka, frame) == _sig.S3
        assert agent._signify(kb, frame) == _sig.S3

        # Ground A — ka still S3, kb rises to S1
        frame.add(KLine(signature=A, nodes=B))
        assert agent._signify(ka, frame) == _sig.S3
        assert agent._signify(kb, frame) == _sig.S1

        # Ground B — ka kb rise
        frame.add(KLine(signature=B, nodes=A))
        assert agent._signify(ka, frame) == _sig.S1
        assert agent._signify(kb, frame) == _sig.S1


class TestDynamicRiseStability:
    """Verify that once a kline reaches S1, further model additions
    don't change its significance — S1 is stable.
    """

    def test_s1_remains_s1_after_more_additions(self):
        """A canonized kline at S1 stays S1 regardless of new klines."""
        agent, frame = _agent()
        kline = KLine(signature=AB, nodes=[A, B])

        # Ground node_sig → S1
        frame.add(KLine(signature=AB, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

        # Add many more klines
        for i in range(1, 20):
            frame.add(KLine(signature=i, nodes=[]))

        assert agent._signify(kline, frame) == _sig.S1

    def test_identity_s1_never_changes(self):
        """Identity klines are always S1 regardless of frame state."""
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=A)

        assert agent._signify(kline, frame) == _sig.S1

        frame.add(KLine(signature=0xDEAD, nodes=[0xBEEF]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_unsigned_s4_never_changes(self):
        """Unsigned klines are always S4 regardless of frame state."""
        agent, frame = _agent()
        kline = KLine(signature=A, nodes=None)

        assert agent._signify(kline, frame) == _sig.S4

        frame.add(KLine(signature=A, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S4
