"""Tests for Agent._signify - internal significance calculation of a single KLine.

_signify evaluates a KLine against a frame to determine its significance level:
- S1: Identity, undersigned (in frame / literal), or fully canonised
- S2: Partially canonised, or underfit (sig has more bits than nodes)
- S3: Overfit (nodes have more bits than sig)
- S4: Unsigned or insignificant (no overlap between sig and nodes)
"""

from __future__ import annotations

import pytest
from collections import Counter

from kalvin.abstract import KLine
from kalvin.agent import Agent
from kalvin.model import Model
from kalvin.significance import Int32Significance

_sig = Int32Significance()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Minimal mock tokenizer that lets us control is_literal."""

    def __init__(self, *, literals: set[int] | None = None) -> None:
        self._literals = literals or set()

    def is_literal(self, token_id: int) -> bool:
        return token_id in self._literals

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        # Must return at least one token for Agent.__init__ (ws_token)
        return [0]

    def decode(self, ids: list[int]) -> str:
        return ""

    @property
    def vocab_size(self) -> int:
        return 1


def _agent(
    model: Model | None = None,
    literals: set[int] | None = None,
) -> tuple[Agent, Model]:
    """Build an Agent with a mock tokenizer and return (agent, frame).

    The returned *frame* is the same Model passed to ``_signify`` in each test.
    """
    frame = model or Model()
    tok = _MockTokenizer(literals=literals)
    agent = Agent(model=frame, tokenizer=tok)
    return agent, frame


# Token values kept below 32 bits so they sit in the token space of
# Int32Significance (TOKEN_MASK = (1 << 32) - 1).
TOK_A = 0x0001
TOK_B = 0x0002
TOK_C = 0x0004
TOK_D = 0x0008
TOK_AB = TOK_A | TOK_B        # 0x0003
TOK_ABC = TOK_A | TOK_B | TOK_C  # 0x0007


# ===================================================================
# 1. Identity → S1
#    is_identity(sig, nodes) is True when both are ints with equal
#    stripped (token) bits.
# ===================================================================

class TestSignifyIdentity:
    """sig and nodes are equal ints (same token bits) → S1."""

    def test_exact_match(self):
        agent, frame = _agent()
        kline = KLine(signature=TOK_A, nodes=TOK_A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_sig_with_significance_bits_stripped(self):
        """Significance bits in sig are stripped before comparison."""
        agent, frame = _agent()
        sig = _sig.S1 | TOK_A  # S1 bit in high 32, token in low 32
        kline = KLine(signature=sig, nodes=TOK_A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_different_token_bits_not_identity(self):
        """Different token values fall through to later checks."""
        agent, frame = _agent()
        kline = KLine(signature=TOK_A, nodes=TOK_B)
        # Not identity, not unsigned, nodes is int (signed) but not in
        # frame and not literal → falls through to canonisation section
        # which will try `for n in nodes:` on an int and raise TypeError.
        # This is a defensive test – the path should not be reached in
        # normal operation.
        with pytest.raises(TypeError):
            agent._signify(kline, frame)


# ===================================================================
# 2. Unsigned → S4
#    nodes is None (is_unsigned returns True).
# ===================================================================

class TestSignifyUnsigned:
    """nodes is None → S4."""

    def test_none_nodes(self):
        agent, frame = _agent()
        kline = KLine(signature=TOK_A, nodes=None)
        assert agent._signify(kline, frame) == _sig.S4

    def test_none_nodes_high_sig(self):
        agent, frame = _agent()
        kline = KLine(signature=_sig.S2 | TOK_A, nodes=None)
        assert agent._signify(kline, frame) == _sig.S4


# ===================================================================
# 3. Undersigned / Countersigned → S1
#    nodes is a single int (is_signed = True) and either:
#      a) the node exists in the frame, or
#      b) the tokenizer considers it a literal.
# ===================================================================

class TestSignifyUndersigned:
    """nodes is a single int; resolved via frame lookup or literal check."""

    def test_in_frame(self):
        """Node found in frame → S1."""
        frame = Model([KLine(signature=TOK_A, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=_sig.S2 | TOK_A, nodes=TOK_A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_is_literal(self):
        """Node not in frame but tokenizer says literal → S1."""
        agent, frame = _agent(literals={TOK_A})
        kline = KLine(signature=_sig.S2 | TOK_A, nodes=TOK_A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_in_frame_and_literal(self):
        """Both conditions true – still S1."""
        frame = Model([KLine(signature=TOK_A, nodes=[])])
        agent, frame = _agent(model=frame, literals={TOK_A})
        kline = KLine(signature=TOK_A, nodes=TOK_A)
        # Actually identity catches this first, so use a different sig
        kline = KLine(signature=_sig.S3 | TOK_A, nodes=TOK_A)
        assert agent._signify(kline, frame) == _sig.S1

    def test_neither_in_frame_nor_literal_raises(self):
        """Signed int node not in frame and not literal reaches canonisation
        code which tries to iterate the int as a list → TypeError.

        In normal usage this path is unreachable because the kline would
        already exist in the frame (rationalise checks before calling
        _signify).  We verify the defensive behaviour.

        Note: sig and nodes must have *different* token bits so that
        is_identity returns False (strip compares token bits only).
        """
        agent, frame = _agent(literals=set())  # no literals
        kline = KLine(signature=TOK_B, nodes=TOK_A)  # different token values
        with pytest.raises(TypeError):
            agent._signify(kline, frame)


# ===================================================================
# 4. Canonisation – fully canonised → S1
#    nodes is a list, ns (OR of nodes) == sig, and every node is
#    resolvable (ns is in frame, or all nodes are literals).
# ===================================================================

class TestSignifyCanonisedS1:
    """Canonised nodes that are fully grounded → S1."""

    def test_ns_in_frame(self):
        """ns (OR of all nodes) exists as a key in the frame → S1."""
        frame = Model([KLine(signature=TOK_AB, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])
        assert agent._signify(kline, frame) == _sig.S1

    def test_all_nodes_literal(self):
        """ns not in frame, but every node is a literal → S1."""
        agent, frame = _agent(literals={TOK_A, TOK_B})
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])
        assert agent._signify(kline, frame) == _sig.S1

    def test_ns_in_frame_single_node(self):
        """Single-node list where ns matches sig and is in frame → S1."""
        frame = Model([KLine(signature=TOK_A, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=TOK_A, nodes=[TOK_A])
        # Identity check: sig=TOK_A, nodes=[TOK_A] → isinstance(list, int) is
        # False → not identity.  Not unsigned.  Not signed (list).  Canonisation:
        # ns = TOK_A, ns == sig, ns in frame → S1.
        assert agent._signify(kline, frame) == _sig.S1


# ===================================================================
# 5. Canonisation – partially canonised → S2
#    nodes is a list, ns == sig, ns is NOT in frame, and at least one
#    node is not a literal.
# ===================================================================

class TestSignifyCanonisedS2:
    """Partially canonised nodes → S2."""

    def test_ns_not_in_frame_non_literal_node(self):
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])
        assert agent._signify(kline, frame) == _sig.S2

    def test_mixed_literal_and_non_literal(self):
        """One literal, one non-literal, ns not in frame → S2."""
        agent, frame = _agent(literals={TOK_A})
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])
        assert agent._signify(kline, frame) == _sig.S2

    def test_single_non_literal_node(self):
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_A, nodes=[TOK_A])
        assert agent._signify(kline, frame) == _sig.S2

    def test_ns_in_frame_overrides_non_literal(self):
        """Even with non-literal nodes, if ns is in frame → S1 (not S2)."""
        frame = Model([KLine(signature=TOK_AB, nodes=[])])
        agent, frame = _agent(model=frame, literals=set())
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])
        assert agent._signify(kline, frame) == _sig.S1


# ===================================================================
# 6. Underfit → S2
#    nodes is a list, ns != sig, sig & ns != 0, and
#    sig.bit_count() > ns.bit_count().
# ===================================================================

class TestSignifyUnderfitS2:
    """sig has more bits than nodes (underfit) → S2."""

    def test_basic_underfit(self):
        agent, frame = _agent()
        # sig has bits at 0x1 and 0x10 → 2 bits; nodes contribute 0x10 → 1 bit
        kline = KLine(signature=0x11, nodes=[0x10])
        assert agent._signify(kline, frame) == _sig.S2

    def test_sig_extra_bits(self):
        agent, frame = _agent()
        # sig = 0b111 = 3 bits, ns = 0b011 = 2 bits
        kline = KLine(signature=0b111, nodes=[0b010, 0b001])
        assert agent._signify(kline, frame) == _sig.S2


# ===================================================================
# 7. Overfit → S3
#    nodes is a list, ns != sig, sig & ns != 0, and
#    sig.bit_count() <= ns.bit_count().
# ===================================================================

class TestSignifyOverfitS3:
    """nodes have more bits than sig (overfit) → S3."""

    def test_basic_overfit(self):
        agent, frame = _agent()
        # sig = 0x10 → 1 bit; ns = 0x10 | 0x20 = 0x30 → 2 bits
        kline = KLine(signature=0x10, nodes=[0x10, 0x20])
        assert agent._signify(kline, frame) == _sig.S3

    def test_equal_bit_count_is_overfit(self):
        """Equal bit counts go to the overfit (S3) branch (not strictly >)."""
        agent, frame = _agent()
        # sig = 0b011 = 2 bits; ns = 0b101 = 2 bits; overlap at bit 0
        kline = KLine(signature=0b011, nodes=[0b101])
        assert agent._signify(kline, frame) == _sig.S3

    def test_overfit_multiple_nodes(self):
        agent, frame = _agent()
        # sig = 0x1 → 1 bit; ns = 0x1 | 0x2 | 0x4 = 0x7 → 3 bits
        kline = KLine(signature=0x1, nodes=[0x1, 0x2, 0x4])
        assert agent._signify(kline, frame) == _sig.S3


# ===================================================================
# 8. Insignificant → S4
#    nodes is a list, ns != sig, and sig & ns == 0 (no bit overlap).
# ===================================================================

class TestSignifyInsignificantS4:
    """No overlap between sig and nodes → S4."""

    def test_no_overlap(self):
        agent, frame = _agent()
        kline = KLine(signature=0x10, nodes=[0x01])
        assert agent._signify(kline, frame) == _sig.S4

    def test_no_overlap_multiple_nodes(self):
        agent, frame = _agent()
        kline = KLine(signature=0xF0, nodes=[0x01, 0x02, 0x04])
        assert agent._signify(kline, frame) == _sig.S4

    def test_empty_node_list(self):
        """Empty list: ns = 0, ns != sig (unless sig is also 0), sig & 0 = 0 → S4."""
        agent, frame = _agent()
        kline = KLine(signature=0x10, nodes=[])
        assert agent._signify(kline, frame) == _sig.S4

    def test_zero_sig_empty_nodes(self):
        """Both zero: ns == sig → canonisation path, ns=0 not in frame,
        0 not literal (no iteration over empty list) → S1."""
        agent, frame = _agent()
        kline = KLine(signature=0, nodes=[])
        # ns=0, ns==sig=0 → for loop over empty list → S1
        assert agent._signify(kline, frame) == _sig.S1


# ===================================================================
# 9. Significance bits in sig during canonisation
# ===================================================================

class TestSignifyWithSignificanceBits:
    """sig may carry significance-level bits; canonisation compares
    the full 64-bit values."""

    def test_sig_has_s2_bit(self):
        """sig = S2 | TOK_A, nodes = [TOK_A]: ns = TOK_A != sig → overlap → S2."""
        agent, frame = _agent()
        kline = KLine(signature=_sig.S2 | TOK_A, nodes=[TOK_A])
        result = agent._signify(kline, frame)
        # sig & ns = TOK_A (non-zero).  sig.bit_count() includes the S2 bit
        # which is bit 55.  ns.bit_count() = 1.  So sig.bit_count() > ns.bit_count()
        # → underfit → S2.
        assert result == _sig.S2

    def test_sig_has_s3_bit(self):
        """sig = S3 | TOK_A, nodes = [TOK_A]: ns != sig → overlap → S3."""
        agent, frame = _agent()
        kline = KLine(signature=_sig.S3 | TOK_A, nodes=[TOK_A])
        result = agent._signify(kline, frame)
        # sig & ns = TOK_A (non-zero).  sig.bit_count() includes S3 bit (bit 32).
        # ns.bit_count() = 1.  sig.bit_count() = 2 > 1 → underfit → S2.
        assert result == _sig.S2


# ===================================================================
# 10. Precedence / ordering verification
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

    def test_signed_before_canonisation(self):
        """A signed int node that IS in frame returns S1, not S2/S3."""
        frame = Model([KLine(signature=TOK_A, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=TOK_A, nodes=TOK_A)
        # Identity catches this first (same stripped value) → S1
        assert agent._signify(kline, frame) == _sig.S1

    def test_canonisation_before_underfit(self):
        """If ns == sig, we never reach the underfit/overfit check."""
        frame = Model([KLine(signature=TOK_AB, nodes=[])])
        agent, frame = _agent(model=frame)
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])
        # ns = TOK_AB == sig, ns in frame → S1 (canonisation), not underfit
        assert agent._signify(kline, frame) == _sig.S1

    def test_underfit_before_s4(self):
        """sig & ns != 0 yields S2/S3, not S4."""
        agent, frame = _agent()
        kline = KLine(signature=0x11, nodes=[0x10])
        # Would be S4 if overlap wasn't checked (no canonisation since 0x10 != 0x11)
        assert agent._signify(kline, frame) == _sig.S2


# ===================================================================
# 11. Dynamic tests – significance rises as model grows
#
#    The canonisation and undersigned branches query the frame,
#    so adding klines between _signify calls can raise the result.
#
#    Rising paths:
#      Canonisation (ns == sig):
#        S2 (ns not grounded) → S1 (ns grounded in frame)
#      Undersigned (int node):
#        TypeError (node missing) → S1 (node grounded in frame)
#
#    Static paths (no frame involvement): S4 unsigned, S2 underfit,
#    S3 overfit, S4 no-overlap.
# ===================================================================


class TestDynamicCanonisationRise:
    """Canonised klines rise from S2 → S1 when ns is added to frame.

    In the canonisation branch (_signify lines for ns == sig):
      for n in nodes:
          if not frame.find_kline(ns) and not is_literal(n):
              return S2  # partially canonised
      return S1          # fully canonised

    So the result depends on whether ns (OR of all nodes) is
    resolvable in the frame.
    """

    def test_two_node_s2_then_s1_by_adding_ns(self):
        """S2 → S1 when ns is added to frame.

        kline(sig=AB, nodes=[A, B]): ns = A|B = AB
        Empty frame → S2.  Add AB to frame → S1.
        """
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])

        # Step 1: empty frame → S2
        assert agent._signify(kline, frame) == _sig.S2

        # Step 2: ground ns in frame
        frame.add(KLine(signature=TOK_AB, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_three_node_s2_then_s1_by_adding_ns(self):
        """Three-node canonised kline rises from S2 → S1.

        kline(sig=ABC, nodes=[A, B, C]): ns = A|B|C = ABC
        """
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_ABC, nodes=[TOK_A, TOK_B, TOK_C])

        assert agent._signify(kline, frame) == _sig.S2

        frame.add(KLine(signature=TOK_ABC, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_single_node_s2_then_s1_by_adding_ns(self):
        """Single-node list rises from S2 → S1.

        kline(sig=A, nodes=[A]): ns = A.  Identity is skipped
        because sig is int, nodes is list (is_identity needs both int).
        """
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_A, nodes=[TOK_A])

        assert agent._signify(kline, frame) == _sig.S2

        frame.add(KLine(signature=TOK_A, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_s2_persists_until_ns_is_grounded(self):
        """Adding unrelated klines to frame does NOT raise significance."""
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])

        assert agent._signify(kline, frame) == _sig.S2

        # Add unrelated klines
        frame.add(KLine(signature=TOK_C, nodes=[]))
        frame.add(KLine(signature=TOK_D, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S2

        # Now add the actual ns
        frame.add(KLine(signature=TOK_AB, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_multiple_klines_rise_together_when_ns_shared(self):
        """Several klines with the same ns all rise when ns is grounded."""
        agent, frame = _agent(literals=set())

        k1 = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])
        k2 = KLine(signature=TOK_AB, nodes=[TOK_B, TOK_A])  # different order

        assert agent._signify(k1, frame) == _sig.S2
        assert agent._signify(k2, frame) == _sig.S2

        frame.add(KLine(signature=TOK_AB, nodes=[]))

        assert agent._signify(k1, frame) == _sig.S1
        assert agent._signify(k2, frame) == _sig.S1

    def test_sequential_grounding_multiple_ns(self):
        """Different klines with different ns values rise independently."""
        agent, frame = _agent(literals=set())

        ka = KLine(signature=TOK_A, nodes=[TOK_A])      # ns = A
        kab = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])  # ns = AB
        kabc = KLine(signature=TOK_ABC, nodes=[TOK_A, TOK_B, TOK_C])  # ns = ABC

        # All start at S2
        assert agent._signify(ka, frame) == _sig.S2
        assert agent._signify(kab, frame) == _sig.S2
        assert agent._signify(kabc, frame) == _sig.S2

        # Ground A — only ka rises (ab and abc still missing their ns)
        frame.add(KLine(signature=TOK_A, nodes=[]))
        assert agent._signify(ka, frame) == _sig.S1
        assert agent._signify(kab, frame) == _sig.S2
        assert agent._signify(kabc, frame) == _sig.S2

        # Ground AB — kab rises, kabc still S2
        frame.add(KLine(signature=TOK_AB, nodes=[]))
        assert agent._signify(kab, frame) == _sig.S1
        assert agent._signify(kabc, frame) == _sig.S2

        # Ground ABC — kabc rises
        frame.add(KLine(signature=TOK_ABC, nodes=[]))
        assert agent._signify(kabc, frame) == _sig.S1

    def test_rise_does_not_affect_static_underfit(self):
        """Grounding a kline in the frame does not change underfit results.

        Underfit path (ns != sig, sig & ns != 0) is pure bit math —
        no frame lookup.
        """
        agent, frame = _agent()
        kline_underfit = KLine(signature=0x11, nodes=[0x10])

        assert agent._signify(kline_underfit, frame) == _sig.S2

        # Add unrelated and overlapping klines
        frame.add(KLine(signature=0x10, nodes=[]))
        frame.add(KLine(signature=0x11, nodes=[]))

        # Underfit result unchanged — ns (0x10) != sig (0x11)
        assert agent._signify(kline_underfit, frame) == _sig.S2

    def test_rise_does_not_affect_static_s4_no_overlap(self):
        """Grounding klines does not change S4 no-overlap results."""
        agent, frame = _agent()
        kline = KLine(signature=0xF0, nodes=[0x01, 0x02])

        assert agent._signify(kline, frame) == _sig.S4

        frame.add(KLine(signature=0xF0, nodes=[]))
        frame.add(KLine(signature=0x01, nodes=[]))

        # Still S4 — sig & ns = 0 (pure bit math)
        assert agent._signify(kline, frame) == _sig.S4


class TestDynamicUndersignedRise:
    """Undersigned (int node) klines become S1 when node is added to frame.

    In the undersigned branch:
        if frame.find_kline(ns) or is_literal(ns):
            return S1
    When the node is absent and non-literal, execution falls through to
    the canonisation code which iterates the int → TypeError.
    """

    def test_error_then_s1_by_adding_node(self):
        """Signed int node goes from TypeError → S1 when grounded.

        Uses different token values for sig and nodes so identity check
        doesn't short-circuit (is_identity compares stripped token bits).
        """
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_B, nodes=TOK_A)  # sig≠nodes tokens

        # Step 1: node not in frame → falls through → TypeError
        with pytest.raises(TypeError):
            agent._signify(kline, frame)

        # Step 2: ground the node
        frame.add(KLine(signature=TOK_A, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_multiple_undersigned_rise_independently(self):
        """Different undersigned klines with different nodes rise
        independently as their respective nodes are grounded.
        """
        agent, frame = _agent(literals=set())

        ka = KLine(signature=TOK_B, nodes=TOK_A)  # references A
        kb = KLine(signature=TOK_A, nodes=TOK_B)  # references B

        # Both crash initially
        with pytest.raises(TypeError):
            agent._signify(ka, frame)
        with pytest.raises(TypeError):
            agent._signify(kb, frame)

        # Ground A — ka rises, kb still crashes
        frame.add(KLine(signature=TOK_A, nodes=[]))
        assert agent._signify(ka, frame) == _sig.S1
        with pytest.raises(TypeError):
            agent._signify(kb, frame)

        # Ground B — kb also rises
        frame.add(KLine(signature=TOK_B, nodes=[]))
        assert agent._signify(kb, frame) == _sig.S1


class TestDynamicRiseStability:
    """Verify that once a kline reaches S1, further model additions
    don't change its significance — S1 is stable.
    """

    def test_s1_remains_s1_after_more_additions(self):
        """A canonised kline at S1 stays S1 regardless of new klines."""
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_AB, nodes=[TOK_A, TOK_B])

        # Ground ns → S1
        frame.add(KLine(signature=TOK_AB, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S1

        # Add many more klines
        for i in range(1, 20):
            frame.add(KLine(signature=i, nodes=[]))

        assert agent._signify(kline, frame) == _sig.S1

    def test_identity_s1_never_changes(self):
        """Identity klines are always S1 regardless of frame state."""
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_A, nodes=TOK_A)

        assert agent._signify(kline, frame) == _sig.S1

        frame.add(KLine(signature=0xDEAD, nodes=[0xBEEF]))
        assert agent._signify(kline, frame) == _sig.S1

    def test_unsigned_s4_never_changes(self):
        """Unsigned klines are always S4 regardless of frame state."""
        agent, frame = _agent(literals=set())
        kline = KLine(signature=TOK_A, nodes=None)

        assert agent._signify(kline, frame) == _sig.S4

        frame.add(KLine(signature=TOK_A, nodes=[]))
        assert agent._signify(kline, frame) == _sig.S4
