import pytest
from kalvin.abstract import KLine, KNone
from kalvin.model import Model



class TestKLine:
    def test_create_kline(self):
        """Test creating a KLine with int signature and list of KNode ints."""
        signature = 0x123456789ABCDEF0
        nodes = [0x1000, 0x2000]

        kl = KLine(signature=signature, nodes=nodes)

        assert kl.signature == signature
        assert kl.nodes == [0x1000, 0x2000]

    def test_create_factory(self):
        """Test KLine.create factory method combines significance and token."""
        significance = 0xFF00
        token = 0x00FF
        nodes = [0x1000, 0x2000]

        kl = KLine.create(significance=significance, token=token, nodes=nodes)

        assert kl.signature == 0xFFFF  # significance | token
        assert kl.nodes == [0x1000, 0x2000]

    def test_create_factory_with_zero_significance(self):
        """Test KLine.create with zero significance."""
        significance = 0x0000
        token = 0x1234
        nodes = []

        kl = KLine.create(significance=significance, token=token, nodes=nodes)

        assert kl.signature == 0x1234

    def test_create_factory_with_zero_token(self):
        """Test KLine.create with zero token."""
        significance = 0xFF00
        token = 0x0000
        nodes = [0x100]

        kl = KLine.create(significance=significance, token=token, nodes=nodes)

        assert kl.signature == 0xFF00

    def test_store_in_list(self):
        """Test storing KLine objects in a list."""
        kl_list = []

        kl1 = KLine(signature=0x1000000000000000, nodes=[])
        kl2 = KLine(signature=0x1000000000000001, nodes=[])
        kl3 = KLine(signature=0x2000000000000000, nodes=[])

        kl_list.append(kl1)
        kl_list.append(kl2)
        kl_list.append(kl3)

        assert len(kl_list) == 3
        assert kl_list[0].signature == 0x1000000000000000
        assert kl_list[1].signature == 0x1000000000000001
        assert kl_list[2].signature == 0x2000000000000000

    def test_nested_klines_structure(self):
        """Test nested KLine structure with node references."""
        leaf1 = KLine(signature=0x0100, nodes=[])
        leaf2 = KLine(signature=0x0200, nodes=[])
        leaf3 = KLine(signature=0x0300, nodes=[])

        intermediate = KLine(signature=0x0010, nodes=[0x0100, 0x0200])
        root = KLine(signature=0x0001, nodes=[0x0010, 0x0300])

        assert len(root.nodes) == 2
        assert root.nodes[0] == 0x0010
        assert root.nodes[1] == 0x0300

        kl_list = [root, intermediate, leaf1, leaf2, leaf3]
        assert len(kl_list) == 5


class TestModelAddKLine:
    def test_add_new_key(self):
        """Adding a kline with new key succeeds."""
        model = Model()
        kl = KLine(signature=0x1000, nodes=[])

        result = model.add(kl)

        assert result is True
        assert len(model) == 1
        assert model[kl.signature] == kl

    def test_add_duplicate_key_different_nodes(self):
        """Adding kline with same key but different nodes succeeds."""
        kl1 = KLine(signature=0x1000, nodes=[0x0100])
        kl2 = KLine(signature=0x1000, nodes=[0x0200])
        model = Model([kl1])

        result = model.add(kl2)

        assert result is True
        assert len(model) == 2

    def test_add_exact_duplicate_same_key_different_instance(self):
        """Adding kline with same key and nodes creates a new entry."""
        kl1 = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        kl2 = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        model = Model([kl1])

        result = model.add(kl2)

        # Duplicate is added (same signature allowed, different KLine instance)
        assert result is True
        assert len(model) == 2

    def test_add_duplicate_empty_nodes_same_key(self):
        """Adding kline with same key and empty nodes creates a new entry."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x1000, nodes=[])
        model = Model([kl1])

        result = model.add(kl2)

        # Duplicate is added (same signature allowed)
        assert result is True
        assert len(model) == 2

    def test_multiple_keys_all_added(self):
        """Multiple klines with different keys are all added."""
        model = Model()
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        kl3 = KLine(signature=0x3000, nodes=[])

        assert model.add(kl1) is True
        assert model.add(kl2) is True
        assert model.add(kl3) is True
        assert len(model) == 3


class TestModelQuery:
    def test_no_match_returns_empty(self):
        """If no kline matches, result is empty."""
        kl1 = KLine(signature=0x0001, nodes=[])
        kl2 = KLine(signature=0x0002, nodes=[])
        query = KLine(signature=0xFF00, nodes=[])
        model = Model([kl1, kl2])

        assert list(model.query(query)) == []

    def test_single_match(self):
        """If match found, it's returned."""
        matching = KLine(signature=0xFF00, nodes=[])
        non_matching = KLine(signature=0x0001, nodes=[])
        query = KLine(signature=0xFF00, nodes=[])
        model = Model([non_matching, matching])

        assert list(model.query(query)) == [matching]

    def test_all_matches_returned(self):
        """All matching klines are returned."""
        match1 = KLine(signature=0xFF00, nodes=[])
        match2 = KLine(signature=0xFF01, nodes=[])
        query = KLine(signature=0xFF00, nodes=[])
        non_matching = KLine(signature=0x0001, nodes=[])
        model = Model([non_matching, match1, match2])

        assert list(model.query(query)) == [match2, match1]

    def test_reverse_insertion_order(self):
        """Results follow reverse insertion order (newest first)."""
        match1 = KLine(signature=0xFF00, nodes=[])
        match2 = KLine(signature=0xFF01, nodes=[])
        match3 = KLine(signature=0xFF02, nodes=[])
        query = KLine(signature=0xFF00, nodes=[])
        model = Model([match1, match2, match3])

        assert list(model.query(query)) == [match3, match2, match1]


class TestModelExpand:
    def test_depth_one_returns_kline_only(self):
        """depth=1 returns the kline without expansion."""
        key_child = 0x0010
        parent = KLine(signature=0xFF00, nodes=[key_child])
        child = KLine(signature=key_child, nodes=[])
        model = Model([parent, child])

        results = list(model.expand(parent, depth=1))

        assert len(results) == 1
        assert results[0] == parent

    def test_depth_expands_children(self):
        """depth=2 expands direct children."""
        key_child1 = 0x0010
        key_child2 = 0x0020

        child1 = KLine(signature=key_child1, nodes=[])
        child2 = KLine(signature=key_child2, nodes=[])
        parent = KLine(signature=0xFF00, nodes=[key_child1, key_child2])
        model = Model([parent, child1, child2])

        results = list(model.expand(parent, depth=2))

        assert len(results) == 3
        assert results[0] == parent
        assert child1 in results
        assert child2 in results

    def test_depth_limits_expansion(self):
        """Depth parameter limits how many levels of children are expanded."""
        key_grandchild = 0x0100
        key_child = 0x0010
        key_parent = 0xF000

        grandchild = KLine(signature=key_grandchild, nodes=[])
        child = KLine(signature=key_child, nodes=[key_grandchild])
        parent = KLine(signature=key_parent, nodes=[key_child])
        model = Model([parent, child, grandchild])

        # depth=1: only parent, no child expansion
        results = list(model.expand(parent, depth=1))
        assert len(results) == 1
        assert results[0] == parent

        # depth=2: parent + child, no grandchild
        results = list(model.expand(parent, depth=2))
        assert len(results) == 2
        assert results[0] == parent
        assert results[1] == child

        # depth=3: parent + child + grandchild
        results = list(model.expand(parent, depth=3))
        assert len(results) == 3
        assert results[0] == parent
        assert results[1] == child
        assert results[2] == grandchild

    def test_depth_zero_returns_empty(self):
        """depth=0 returns empty."""
        matching = KLine(signature=0xFF00, nodes=[])
        model = Model([matching])

        assert list(model.expand(matching, depth=0)) == []

    def test_cycle_detection_stops_expansion(self):
        """Circular references stop expansion."""
        key_a = 0x0001
        key_b = 0x0002

        kl_a = KLine(signature=key_a, nodes=[key_b])
        kl_b = KLine(signature=key_b, nodes=[key_a])
        model = Model([kl_a, kl_b])

        results = list(model.expand(kl_a, depth=100))

        assert len(results) == 2
        assert kl_a in results
        assert kl_b in results

    def test_self_reference_stops_expansion(self):
        """Self-referencing KLine stops expansion."""
        key = 0xFF00
        kl = KLine(signature=key, nodes=[key])
        model = Model([kl])

        results = list(model.expand(kl, depth=100))

        assert len(results) == 1
        assert results[0] == kl

    def test_nested_hierarchy_expansion(self):
        """Test deeply nested hierarchy is expanded correctly."""
        key_leaf1 = 0x1000
        key_leaf2 = 0x2000
        key_leaf3 = 0x3000
        key_intermediate = 0x0010

        leaf1 = KLine(signature=key_leaf1, nodes=[])
        leaf2 = KLine(signature=key_leaf2, nodes=[])
        leaf3 = KLine(signature=key_leaf3, nodes=[])
        intermediate = KLine(signature=key_intermediate, nodes=[key_leaf1, key_leaf2])
        root = KLine(signature=0xFF00, nodes=[key_intermediate, key_leaf3])
        model = Model([root, intermediate, leaf1, leaf2, leaf3])

        results = list(model.expand(root, depth=3))

        assert len(results) == 5
        assert root in results
        assert intermediate in results
        assert leaf1 in results
        assert leaf2 in results
        assert leaf3 in results

    def test_cyclic_children_stops_expansion(self):
        """Cyclic children (child references ancestor) stop expansion."""
        key_root = 0xFF00
        key_child = 0x0010
        key_grandchild = 0x0100

        grandchild = KLine(signature=key_grandchild, nodes=[key_root])
        child = KLine(signature=key_child, nodes=[key_grandchild])
        root = KLine(signature=key_root, nodes=[key_child])
        model = Model([root, child, grandchild])

        results = list(model.expand(root, depth=10))

        assert len(results) == 3
        assert root in results
        assert child in results
        assert grandchild in results

    def test_cyclic_grandchildren_stops_expansion(self):
        """Cyclic grandchildren (grandchild references parent) stop expansion."""
        key_root = 0xFF00
        key_child = 0x0010
        key_grandchild = 0x0100

        grandchild = KLine(signature=key_grandchild, nodes=[key_child])
        child = KLine(signature=key_child, nodes=[key_grandchild])
        root = KLine(signature=key_root, nodes=[key_child])
        model = Model([root, child, grandchild])

        results = list(model.expand(root, depth=10))

        assert len(results) == 3
        assert root in results
        assert child in results
        assert grandchild in results

    def test_kline_with_no_nodes(self):
        """Expanding a leaf kline returns just itself."""
        leaf = KLine(signature=0xFF00, nodes=[])
        model = Model([leaf])

        results = list(model.expand(leaf, depth=3))

        assert results == [leaf]

    def test_missing_child_nodes_skipped(self):
        """Child nodes not in the model are skipped."""
        parent = KLine(signature=0xFF00, nodes=[0x0010, 0x0020])
        model = Model([parent])  # children not added

        results = list(model.expand(parent, depth=2))

        assert results == [parent]

    def test_shared_child_deduplicated(self):
        """Shared child across multiple parents is only yielded once."""
        key_shared = 0x0010
        shared = KLine(signature=key_shared, nodes=[])
        parent = KLine(signature=0xFF00, nodes=[key_shared, key_shared])
        model = Model([parent, shared])

        results = list(model.expand(parent, depth=2))

        assert results == [parent, shared]


class TestModelIterators:
    def test_getitem_access(self):
        """Can access KLines by index."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        model = Model([kl1, kl2])

        assert model[kl1.signature] == kl1
        assert model[kl2.signature] == kl2

    def test_find_kline(self):
        """Can find KLine by its signature."""
        kl1 = KLine(signature=0x1000, nodes=[0x0100])
        kl2 = KLine(signature=0x2000, nodes=[0x0200])
        model = Model([kl1, kl2])

        found = model.find_kline(0x1000)
        assert found == kl1

        found = model.find_kline(0x3000)
        assert not found.signature

    def test_find_signed_klines(self):
        """Can find all KLines with same signature."""
        kl1 = KLine(signature=0x1000, nodes=[0x0100])
        kl2 = KLine(signature=0x1000, nodes=[0x0200])
        model = Model([kl1, kl2])

        found = model.find_signed_klines(0x1000)
        assert kl1 in found
        assert kl2 in found
        assert len(found) == 2


class TestGetAllDescendants:
    """Tests for Model.get_all_descendants method."""

    def test_no_descendants(self):
        """KLine with no nodes returns empty set."""
        kline = KLine(signature=0x1000, nodes=[])
        model = Model([kline])

        descendants = model.get_all_descendants(0x1000)

        assert descendants == set()

    def test_direct_children_only(self):
        """Returns direct children when no deeper hierarchy."""
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200, 0x0300])
        model = Model([kline])

        descendants = model.get_all_descendants(0x1000)

        assert descendants == {0x0100, 0x0200, 0x0300}

    def test_nested_descendants(self):
        """Recursively collects all descendants at any depth."""
        grandchild = KLine(signature=0x0010, nodes=[0x0001])
        child = KLine(signature=0x0100, nodes=[0x0010])
        parent = KLine(signature=0x1000, nodes=[0x0100])
        model = Model([parent, child, grandchild])

        descendants = model.get_all_descendants(0x1000)

        assert descendants == {0x0100, 0x0010, 0x0001}

    def test_cycle_detection(self):
        """Handles cycles without infinite recursion."""
        # A -> B -> A (cycle)
        # Descendants of A: B (direct child), A (via B's reference back to A)
        kline_a = KLine(signature=0x1000, nodes=[0x2000])
        kline_b = KLine(signature=0x2000, nodes=[0x1000])
        model = Model([kline_a, kline_b])

        descendants = model.get_all_descendants(0x1000)

        # A's descendants include B and A (via the cycle back from B)
        assert descendants == {0x1000, 0x2000}

    def test_nonexistent_key(self):
        """Returns empty set for nonexistent key."""
        model = Model([])

        descendants = model.get_all_descendants(0x1000)

        assert descendants == set()

    def test_multiple_branches(self):
        """Collects descendants from all branches."""
        leaf1 = KLine(signature=0x0100, nodes=[])
        leaf2 = KLine(signature=0x0200, nodes=[])
        leaf3 = KLine(signature=0x0300, nodes=[])
        branch1 = KLine(signature=0x0010, nodes=[0x0100, 0x0200])
        branch2 = KLine(signature=0x0020, nodes=[0x0300])
        root = KLine(signature=0x1000, nodes=[0x0010, 0x0020])
        model = Model([root, branch1, branch2, leaf1, leaf2, leaf3])

        descendants = model.get_all_descendants(0x1000)

        assert descendants == {0x0010, 0x0020, 0x0100, 0x0200, 0x0300}
