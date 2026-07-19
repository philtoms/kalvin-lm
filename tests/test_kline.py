"""Tests for KLine — specs/kline.md conformance."""

from kalvin.kline import COMPOUND_BIT, KDbg, KLine, is_canon, is_identity
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

    def test_default_op_is_identity(self):
        dbg = KDbg()
        assert dbg.op == "IDENTITY"

    def test_op_set_on_construction(self):
        dbg = KDbg(op="COUNTERSIGNS")
        assert dbg.op == "COUNTERSIGNS"

    def test_repr_includes_op_when_not_identity(self):
        dbg = KDbg(op="COUNTERSIGNS")
        assert "op=COUNTERSIGNS" in repr(dbg)

    def test_repr_omits_identity_op(self):
        dbg = KDbg(op="IDENTITY")
        assert "op=" not in repr(dbg)

    def test_truthy_with_only_op(self):
        dbg = KDbg(op="COUNTERSIGNS")
        assert bool(dbg) is True


class TestStructuralPredicates:
    """is_identity / is_canon — specs/kline.md §Structural Predicates."""

    def test_kl20_is_identity_empty(self):
        assert is_identity(KLine(0xFF, [])) is True

    def test_kl21_is_identity_self_referential(self):
        assert is_identity(KLine(0xFF, [0xFF])) is True

    def test_kl22_is_identity_single_different_node(self):
        assert is_identity(KLine(0xFF, [0x01])) is False

    def test_is_identity_compound_word(self):
        # A §11.3 compound-word: signature carries COMPOUND_BIT, nodes don't.
        # Structurally canon-shaped (multi-node) but semantically an identity.
        nodes = [0b100, 0b010]
        sig = (0b110 | COMPOUND_BIT)
        assert is_identity(KLine(sig, nodes)) is True

    def test_is_identity_compound_word_bit_on_node_is_not_identity(self):
        # If a node also carries the bit, it is not a compound-word identity
        # (the bit is signature-only by construction).
        nodes = [0b100 | COMPOUND_BIT, 0b010]
        sig = (0b110 | COMPOUND_BIT)
        assert is_identity(KLine(sig, nodes)) is False

    def test_is_identity_compound_word_no_bit_is_not_identity(self):
        # Without the bit, a canon-shaped kline is just a canon candidate.
        assert is_identity(KLine(0b110, [0b100, 0b010])) is False

    def test_kl23_is_canon_genuine(self):
        # sig 0b110 = OR(0b100, 0b010); neither node is the signature.
        assert is_canon(KLine(0b110, [0b100, 0b010]), signifier) is True

    def test_kl24_is_canon_self_referential_is_not_canon(self):
        assert is_canon(KLine(0xFF, [0xFF]), signifier) is False

    def test_kl25_is_canon_empty_is_not_canon(self):
        assert is_canon(KLine(0xFF, []), signifier) is False

    def test_is_canon_mismatched_sig(self):
        assert is_canon(KLine(0b100, [0b110]), signifier) is False

    def test_is_canon_compound_word_is_not_canon(self):
        # A compound-word is an identity, so it is filtered out of is_canon.
        nodes = [0b100, 0b010]
        sig = (0b110 | COMPOUND_BIT)
        assert is_canon(KLine(sig, nodes), signifier) is False
