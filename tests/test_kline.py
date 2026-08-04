"""Tests for KLine — specs/kline.md conformance."""

from kalvin.kline import KDbg, KLine, is_canon, is_identity, is_misfit, is_terminal, is_unknown
from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()


class TestKLineConstruction:
    """KLine construction with normalized nodes."""

    def test_empty_kline(self):
        k = KLine(0)
        assert k.signature == 0
        assert k.nodes == []

    def test_nodes_none_normalized(self):
        k = KLine(0, None)
        assert k.nodes == []

    def test_nodes_int_normalized(self):
        k = KLine(5, 42)
        assert k.nodes == [42]

    def test_nodes_list_preserved(self):
        k = KLine(5, [1, 2, 3])
        assert k.nodes == [1, 2, 3]

    def test_empty_nodes_list(self):
        k = KLine(0, [])
        assert k.nodes == []
        assert len(k) == 0

    def test_single_node_kline(self):
        k = KLine(7, [3])
        assert len(k) == 1
        assert k.nodes == [3]

    def test_multi_node_kline(self):
        k = KLine(7, [1, 2, 4])
        assert len(k) == 3

    def test_dbg(self):
        k = KLine(0, [], dbg=KDbg(label="hello"))
        assert k.dbg.label == "hello"


class TestKLineEquality:
    """Equality: signature + node sequence."""

    def test_equal_klines(self):
        a = KLine(5, [1, 2, 3])
        b = KLine(5, [1, 2, 3])
        assert a == b

    def test_unequal_signature(self):
        a = KLine(5, [1, 2])
        b = KLine(6, [1, 2])
        assert a != b

    def test_unequal_nodes(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [2, 1])
        assert a != b

    def test_unequal_node_count(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [1, 2, 3])
        assert a != b

    def test_not_equal_to_other_type(self):
        k = KLine(5, [1])
        assert k != 42
        assert k != "string"
        assert k is not None

    def test_empty_klines_equal(self):
        a = KLine(0, [])
        b = KLine(0, [])
        assert a == b

    def test_empty_klines_unequal_sig(self):
        a = KLine(0, [])
        b = KLine(1, [])
        assert a != b


class TestKLineHash:
    """Hashable for use in sets/dicts."""

    def test_hash_equal_klines(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [1, 2])
        assert hash(a) == hash(b)

    def test_in_set(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [1, 2])
        s = {a, b}
        assert len(s) == 1

    def test_in_dict(self):
        a = KLine(5, [1, 2])
        d = {a: "value"}
        b = KLine(5, [1, 2])
        assert d[b] == "value"


class TestKLineNodeAccess:
    """Node access via .nodes and len()."""

    def test_nodes_returns_list(self):
        k = KLine(5, [1, 2, 3])
        assert isinstance(k.nodes, list)

    def test_len(self):
        assert len(KLine(0, [])) == 0
        assert len(KLine(0, [1])) == 1
        assert len(KLine(0, [1, 2, 3])) == 3

class TestKDbgOp:
    """KDbg.op field for operator provenance."""

    def test_default_op_is_unknown(self):
        dbg = KDbg()
        assert dbg.op == "UNKNOWN"

    def test_op_set_on_construction(self):
        dbg = KDbg(op="COUNTERSIGNS")
        assert dbg.op == "COUNTERSIGNS"

    def test_repr_includes_op_when_not_identity(self):
        dbg = KDbg(op="COUNTERSIGNS")
        assert "op=COUNTERSIGNS" in repr(dbg)

    def test_repr_omits_default_op(self):
        dbg = KDbg(op="UNKNOWN")
        assert "op=" not in repr(dbg)

    def test_truthy_with_only_op(self):
        dbg = KDbg(op="COUNTERSIGNS")
        assert bool(dbg) is True


class TestStructuralPredicates:
    """is_terminal / is_unknown / is_identity / is_canon / is_misfit — specs/kline.md §Structural Predicates."""

    # ── is_unknown ───────────────────────────────────────────────────────
    def test_kl20_is_unknown_empty(self):
        assert is_unknown(KLine(0xFF, [])) is True

    def test_kl20_is_unknown_false_for_self_referential(self):
        assert is_unknown(KLine(0xFF, [0xFF])) is False

    # ── is_terminal (genus: empty Unknown / self-ref Identity) ─
    def test_kl20a_is_terminal_empty(self):
        assert is_terminal(KLine(0xFF, [])) is True

    def test_kl21a_is_terminal_self_referential(self):
        assert is_terminal(KLine(0xFF, [0xFF])) is True

    def test_kl26a_is_terminal_compound_word_self_ref(self):
        # A §11.3 compound-word is a self-referential identity: its
        # signature is the OR-reduction of its subword tokens.
        packed = 0b110
        assert is_terminal(KLine(packed, [packed])) is True

    def test_is_terminal_canon_shaped_is_not_terminal(self):
        assert is_terminal(KLine(0b110, [0b100, 0b010])) is False

    # ── is_identity (strict: decodable terminals only) ──────────────────
    def test_kl20b_is_identity_empty_is_false(self):
        # The empty form is an Unknown, not an Identity.
        assert is_identity(KLine(0xFF, [])) is False

    def test_kl21_is_identity_self_referential(self):
        assert is_identity(KLine(0xFF, [0xFF])) is True

    def test_kl22_is_identity_single_different_node(self):
        assert is_identity(KLine(0xFF, [0x01])) is False

    def test_is_identity_compound_word_self_ref(self):
        # A §11.3 compound-word is a self-referential identity.
        packed = 0b110
        assert is_identity(KLine(packed, [packed])) is True

    def test_is_identity_canon_shaped_is_not_identity(self):
        assert is_identity(KLine(0b110, [0b100, 0b010])) is False

    # ── is_canon (non-terminal, signature == signature_of(nodes)) ───────
    def test_kl23_is_canon_genuine(self):
        # sig 0b110 = OR(0b100, 0b010); neither node is the signature.
        assert is_canon(KLine(0b110, [0b100, 0b010]), signifier) is True

    def test_kl24_is_canon_self_referential_is_not_canon(self):
        assert is_canon(KLine(0xFF, [0xFF]), signifier) is False

    def test_kl25_is_canon_empty_is_not_canon(self):
        assert is_canon(KLine(0xFF, []), signifier) is False

    def test_kl27_is_canon_compound_word_self_ref_is_not_canon(self):
        # A compound-word is a self-referential identity (a terminal), so it
        # is not a canon.
        packed = 0b110
        assert is_canon(KLine(packed, [packed]), signifier) is False

    def test_is_canon_mismatched_sig(self):
        assert is_canon(KLine(0b100, [0b110]), signifier) is False

    # ── is_misfit (non-terminal, signature != signature_of(nodes)) ──────
    def test_kl28_is_misfit_genuine(self):
        assert is_misfit(KLine(0b110, [0b001, 0b010]), signifier) is True

    def test_kl29_is_misfit_canon_is_false(self):
        assert is_misfit(KLine(0b110, [0b100, 0b010]), signifier) is False

    def test_kl30_is_misfit_empty_is_false(self):
        assert is_misfit(KLine(0xFF, []), signifier) is False

    def test_kl31_is_misfit_self_referential_is_false(self):
        assert is_misfit(KLine(0xFF, [0xFF]), signifier) is False

    def test_kl31a_is_misfit_single_node_connote_denote(self):
        # {A: [B]} — signature != signature_of([B]) → a connote/denote misfit.
        assert is_misfit(KLine(0b100, [0b010]), signifier) is True
