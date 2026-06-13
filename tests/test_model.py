"""Tests for Model — specs/model.md conformance."""

from kalvin.kline import KLine
from kalvin.model import KLineStore, Model, _TierChain
from kalvin.signature import make_signature
from kalvin.stm import STM


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


# ── Removed Methods ──────────────────────────────────────────────────


class TestRemovedMethods:
    """MOD-R1/R2/R3: verify old methods no longer exist on Model."""

    def test_add_removed(self):
        assert not hasattr(Model, "add")

    def test_promote_removed(self):
        assert not hasattr(Model, "promote")

    def test_refresh_stm_removed(self):
        assert not hasattr(Model, "refresh_stm")


# ── Cascade Write API ────────────────────────────────────────────────


class TestAddStm:
    """Tests for add_stm() — STM-only write with always-refresh FIFO."""

    def test_add_stm_and_find(self):
        """MOD-23: add_stm writes to STM, discoverable via find."""
        m = make_model()
        k = KLine(5, [1, 2])
        m.add_stm(k)
        assert m.find(5) is k

    def test_add_stm_returns_none(self):
        """add_stm is void (returns None)."""
        m = make_model()
        result = m.add_stm(KLine(5, [1]))
        assert result is None

    def test_add_stm_always_refreshes_fifo(self):
        """MOD-24: add_stm always refreshes FIFO position (remove-if-present then add)."""
        m = make_model(stm_bound=3)
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        k3 = KLine(3, [3])
        m.add_stm(k1)
        m.add_stm(k2)
        m.add_stm(k3)
        # Refresh k1 — moves it to most-recent position
        m.add_stm(k1)
        k4 = KLine(4, [4])
        m.add_stm(k4)  # should evict k2 (oldest after refresh)
        assert m.stm_contains(k1) is True
        assert m.stm_contains(k2) is False
        assert m.stm_contains(k4) is True

    def test_add_stm_eviction(self):
        """MOD-27: STM evicts oldest when bound exceeded."""
        m = make_model(stm_bound=2)
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        k3 = KLine(3, [3])
        m.add_stm(k1)
        m.add_stm(k2)
        m.add_stm(k3)
        assert m.stm_contains(k1) is False  # evicted
        assert m.stm_contains(k2) is True
        assert m.stm_contains(k3) is True

    def test_add_stm_does_not_write_frame(self):
        """add_stm writes to STM only, not Frame."""
        m = make_model()
        k = KLine(5, [1])
        m.add_stm(k)
        assert len(m) == 0  # Frame is empty
        assert m.stm_contains(k) is True


class TestAddFrame:
    """Tests for add_frame() — writes Frame + STM with literal dedup."""

    def test_add_frame_and_find(self):
        """MOD-25: add_frame writes to Frame, discoverable via find."""
        m = make_model()
        k = KLine(5, [1, 2])
        m.add_frame(k)
        assert m.find(5) is k

    def test_add_frame_returns_none(self):
        """add_frame is void (returns None)."""
        m = make_model()
        result = m.add_frame(KLine(5, [1]))
        assert result is None

    def test_add_frame_writes_frame_and_stm(self):
        """MOD-25: add_frame writes to both Frame and STM."""
        m = make_model()
        k = KLine(5, [1])
        m.add_frame(k)
        assert len(m) == 1  # Frame has the kline
        assert m.stm_contains(k) is True

    def test_add_frame_monotonic(self):
        """MOD-32: Frame is monotonic for non-literals."""
        m = make_model()
        k1 = KLine(5, [1])
        k2 = KLine(5, [2])
        m.add_frame(k1)
        m.add_frame(k2)
        assert len(m) == 2


class TestAddLtm:
    """Tests for add_ltm() — writes LTM + Frame + STM with literal dedup."""

    def test_add_ltm_and_find(self):
        """MOD-26: add_ltm writes to LTM, discoverable via find."""
        m = make_model()
        k = KLine(5, [1])
        m.add_ltm(k)
        assert m.find(5) is k

    def test_add_ltm_returns_none(self):
        """add_ltm is void (returns None)."""
        m = make_model()
        result = m.add_ltm(KLine(5, [1]))
        assert result is None

    def test_add_ltm_writes_all_three_tiers(self):
        """MOD-26: add_ltm writes to LTM, Frame, and STM."""
        m = make_model()
        k = KLine(5, [1])
        m.add_ltm(k)
        assert len(m) == 1  # Frame
        assert m.stm_contains(k) is True
        assert m.find(5) is k

    def test_add_ltm_frame_retains(self):
        """MOD-33: LTM is additive — Frame retains the kline."""
        m = make_model()
        k = KLine(5, [1])
        m.add_ltm(k)
        assert len(m) == 1  # Frame has the kline
        # KLine is in Frame (iterable)
        assert any(kl.signature == 5 for kl in m)

    def test_add_ltm_no_precondition(self):
        """add_ltm has no precondition — any kline may be added."""
        m = make_model()
        k = KLine(5, [1])
        m.add_ltm(k)  # never added before, still works
        assert m.find(5) is k


# ── Read Operations ──────────────────────────────────────────────────


class TestModelExists:
    def test_exists_true(self):
        m = make_model()
        k = KLine(5, [1, 2])
        m.add_frame(k)
        assert m.exists(k) is True

    def test_exists_false(self):
        m = make_model()
        assert m.exists(KLine(5, [1])) is False

    def test_exists_different_nodes(self):
        m = make_model()
        m.add_frame(KLine(5, [1, 2]))
        assert m.exists(KLine(5, [1, 3])) is False


class TestModelFind:
    def test_find_by_signature(self):
        m = make_model()
        k = KLine(7, [1, 2, 4])
        m.add_frame(k)
        assert m.find(7) is k

    def test_find_none(self):
        m = make_model()
        assert m.find(42) is None

    def test_find_most_recent(self):
        m = make_model()
        k1 = KLine(7, [1])
        k2 = KLine(7, [2])
        m.add_frame(k1)
        m.add_frame(k2)
        found = m.find(7)
        assert found is k2  # Most recently added


class TestModelFindAll:
    def test_find_all_multiple(self):
        m = make_model()
        k1 = KLine(7, [1])
        k2 = KLine(7, [2])
        m.add_frame(k1)
        m.add_frame(k2)
        results = m.find_all(7)
        assert len(results) == 2

    def test_find_all_empty(self):
        m = make_model()
        assert m.find_all(42) == []


class TestModelLen:
    def test_len_empty(self):
        m = make_model()
        assert len(m) == 0

    def test_len_counts_frame_only(self):
        """add_frame() writes to both STM and Frame. len counts Frame entries."""
        m = make_model()
        m.add_frame(KLine(5, [1]))
        assert len(m) == 1  # add_frame writes to Frame

    def test_len_after_add_ltm(self):
        """add_ltm writes to LTM + Frame + STM. len only counts Frame."""
        m = make_model()
        k = KLine(5, [1])
        m.add_ltm(k)
        assert len(m) == 1  # Frame has the kline

    def test_len_excludes_ltm(self):
        """LTM entries don't add to len (Frame count)."""
        m = make_model()
        m.add_frame(KLine(5, [1]))
        m.add_frame(KLine(6, [2]))
        assert len(m) == 2
        m.add_ltm(KLine(5, [1]))
        m.add_ltm(KLine(6, [2]))
        assert len(m) == 4  # non-literal: always accepted, Frame gets both


class TestModelWhere:
    def test_where_signature_overlap(self):
        m = make_model()
        k1 = KLine(0b110, [0b10, 0b100])
        k2 = KLine(0b001, [0b001])
        m.add_frame(k1)
        m.add_frame(k2)
        results = m.where(0b010)
        assert k1 in results
        assert k2 not in results


class TestModelGraphTraversal:
    def test_resolve(self):
        m = make_model()
        k = KLine(5, [10, 20])
        m.add_frame(k)
        assert m.resolve(5) is k

    def test_query_expand(self):
        m = make_model()
        parent = KLine(5, [10, 20])
        child1 = KLine(10, [30])
        child2 = KLine(20, [])
        m.add_frame(parent)
        m.add_frame(child1)
        m.add_frame(child2)
        expanded = m.query_expand(parent, depth=2)
        assert child1 in expanded
        assert child2 in expanded

    def test_query_expand_depth_1_returns_empty(self):
        m = make_model()
        k = KLine(5, [10])
        m.add_frame(k)
        assert m.query_expand(k, depth=1) == []

    def test_descendants(self):
        m = make_model()
        root = KLine(5, [10, 20])
        child = KLine(10, [30])
        m.add_frame(root)
        m.add_frame(child)
        desc = m.descendants(5)
        assert 10 in desc
        assert 20 in desc
        assert 30 in desc

    def test_cycle_detection(self):
        m = make_model()
        a = KLine(1, [2])
        b = KLine(2, [1])
        m.add_frame(a)
        m.add_frame(b)
        desc = m.descendants(1)
        assert 1 in desc
        assert 2 in desc


class TestModelThreeTier:
    def test_base_read_through(self):
        """Four-tier: base klines discoverable through all tiers."""
        base = make_model()
        k = KLine(5, [1])
        base.add_frame(k)

        frame = Model(base=base)
        assert frame.find(5) is k

    def test_add_frame_goes_to_stm_and_frame(self):
        """add_frame() writes to both STM and Frame."""
        base = make_model()
        session = Model(base=base)
        k = KLine(5, [1])
        session.add_frame(k)
        assert len(base) == 0
        assert len(session) == 1  # Frame has the kline
        assert session.find(5) is k  # discoverable via STM (priority) or Frame


# ── STM Interface Tests ─────────────────────────────────────────────


class TestModelStmContains:
    def test_stm_contains_true(self):
        """stm_contains returns True for a KLine in STM."""
        m = make_model()
        k = KLine(5, [1, 2])
        m.add_frame(k)
        assert m.stm_contains(k) is True

    def test_stm_contains_false(self):
        """stm_contains returns False for a never-added KLine."""
        m = make_model()
        assert m.stm_contains(KLine(99, [1])) is False

    def test_stm_contains_after_eviction(self):
        """stm_contains returns False after STM eviction."""
        m = make_model(stm_bound=2)
        k0 = KLine(0, [0])
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        m.add_frame(k0)
        m.add_frame(k1)
        m.add_frame(k2)
        # k0 evicted (bound=2, 3 added)
        assert m.stm_contains(k0) is False
        assert m.stm_contains(k1) is True
        assert m.stm_contains(k2) is True


class TestModelIterStm:
    def test_iter_stm_empty(self):
        """Empty model yields nothing."""
        m = make_model()
        assert list(m.iter_stm()) == []

    def test_iter_stm_returns_added(self):
        """Added klines appear in the iterator."""
        m = make_model()
        k1 = KLine(5, [1])
        k2 = KLine(6, [2])
        m.add_frame(k1)
        m.add_frame(k2)
        result = list(m.iter_stm())
        assert k1 in result
        assert k2 in result

    def test_iter_stm_insertion_order(self):
        """Klines appear in insertion order."""
        m = make_model()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        k3 = KLine(3, [3])
        m.add_frame(k1)
        m.add_frame(k2)
        m.add_frame(k3)
        assert list(m.iter_stm()) == [k1, k2, k3]


class TestModelFourTier:
    """Tests for four-tier lookup semantics: STM → Frame → LTM → Base."""

    def test_stm_priority_over_frame(self):
        """Same sig kline in STM and Frame → find() returns STM version."""
        m = make_model()
        k1 = KLine(5, [1])
        k2 = KLine(5, [2])
        m.add_frame(k1)  # goes to STM + Frame
        m.add_frame(k2)  # goes to STM + Frame, k1 evicted from STM but still in Frame
        # k2 is most recent in STM
        found = m.find(5)
        assert found is k2

    def test_frame_priority_over_ltm(self):
        """Same sig kline in Frame and LTM → find() returns Frame version."""
        m = make_model()
        k_frame = KLine(5, [1])
        k_ltm = KLine(5, [2])
        m.add_frame(k_frame)  # Frame + STM
        m.add_ltm(k_ltm)  # LTM + Frame + STM
        found = m.find(5)
        # Both are in Frame; most recent wins (k_ltm was added later)
        assert found is k_ltm

    def test_ltm_priority_over_base(self):
        """Same sig kline in LTM and Base → find() returns LTM version."""
        base = Model()
        k_base = KLine(5, [1])
        base.add_frame(k_base)

        m = Model(base=base)
        k_ltm = KLine(5, [2])
        m.add_ltm(k_ltm)  # LTM
        found = m.find(5)
        assert found is k_ltm  # LTM takes priority over Base

    def test_find_all_merges_tiers(self):
        """find_all returns deduplicated results across all tiers."""
        m = make_model()
        k1 = KLine(7, [1])
        k2 = KLine(7, [2])
        m.add_frame(k1)  # Frame
        m.add_ltm(k2)  # LTM + Frame + STM
        results = m.find_all(7)
        assert len(results) == 2
        assert k1 in results
        assert k2 in results

    def test_klines_dedup_across_tiers(self):
        """klines() returns each unique kline once across tiers."""
        m = make_model()
        k = KLine(5, [1])
        m.add_frame(k)  # STM + Frame
        m.add_ltm(k)  # LTM (same kline, non-literal → always accepted)
        results = m.klines()
        # Same kline object in STM, Frame, and LTM → deduplicated
        count = sum(1 for kl in results if kl.signature == 5 and kl.nodes == [1])
        assert count == 1

    def test_construction_with_ltm_model(self):
        """Model(ltm=existing_model) populates LTM tier from the model's klines."""
        prev_session = Model()
        k1 = KLine(5, [1])
        k2 = KLine(6, [2])
        prev_session.add_frame(k1)
        prev_session.add_frame(k2)

        m = Model(ltm=prev_session)
        # LTM should have the klines from prev_session
        assert m.find(5) is k1
        assert m.find(6) is k2
        assert len(m) == 0  # Frame is empty, LTM doesn't count


class TestKLineStore:
    """Unit tests for KLineStore in isolation."""

    def test_add_and_find(self):
        store = KLineStore()
        k = KLine(0xA0, [1, 2])
        store.add(k)
        assert store.find(0xA0) is k

    def test_add_multiple_same_sig(self):
        store = KLineStore()
        k1 = KLine(0xB0, [1])
        k2 = KLine(0xB0, [2])
        store.add(k1)
        store.add(k2)
        # find returns most recent
        assert store.find(0xB0) is k2

    def test_find_all(self):
        store = KLineStore()
        k1 = KLine(0xC0, [1])
        k2 = KLine(0xC0, [2])
        k3 = KLine(0xC0, [3])
        store.add(k1)
        store.add(k2)
        store.add(k3)
        result = store.find_all(0xC0)
        assert result == [k1, k2, k3]

    def test_contains(self):
        store = KLineStore()
        k = KLine(0xD0, [10, 20])
        store.add(k)
        assert store.contains(k) is True
        assert store.contains(KLine(0xD0, [99])) is False

    def test_len(self):
        store = KLineStore()
        assert len(store) == 0
        store.add(KLine(1, [1]))
        assert len(store) == 1
        store.add(KLine(2, [2]))
        assert len(store) == 2

    def test_iter(self):
        store = KLineStore()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        k3 = KLine(3, [3])
        store.add(k1)
        store.add(k2)
        store.add(k3)
        assert list(store) == [k1, k2, k3]

    def test_reversed(self):
        store = KLineStore()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        k3 = KLine(3, [3])
        store.add(k1)
        store.add(k2)
        store.add(k3)
        assert list(reversed(store)) == [k3, k2, k1]

    def test_all_klines(self):
        store = KLineStore()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        store.add(k1)
        store.add(k2)
        assert store.all_klines() == [k1, k2]

    def test_find_missing_returns_none(self):
        store = KLineStore()
        assert store.find(0xDEAD) is None


class TestTierChainContains:
    """Tests for _TierChain.contains across tier types."""

    def test_kline_store_only_found(self):
        store = KLineStore()
        k = KLine(0xA0, [1, 2])
        store.add(k)
        chain = _TierChain([store])
        assert chain.contains(k) is True

    def test_kline_store_only_not_found(self):
        store = KLineStore()
        chain = _TierChain([store])
        assert chain.contains(KLine(0xA0, [1])) is False

    def test_mixed_stm_and_store_found_in_second(self):
        stm = STM(bound=10)
        store = KLineStore()
        k = KLine(0xA0, [1, 2])
        store.add(k)
        chain = _TierChain([stm, store])
        assert chain.contains(k) is True

    def test_model_base_uses_exists(self):
        """Chain with a Model tier uses Model.exists() for contains."""
        base = Model()
        k = KLine(5, [1])
        base.add_frame(k)
        chain = _TierChain([base])
        assert chain.contains(k) is True


class TestTierChainFindFirst:
    """Tests for _TierChain.find_first across tier types."""

    def test_single_store_returns_most_recent(self):
        store = KLineStore()
        k1 = KLine(0xB0, [1])
        k2 = KLine(0xB0, [2])
        store.add(k1)
        store.add(k2)
        chain = _TierChain([store])
        assert chain.find_first(0xB0) is k2

    def test_single_stm_returns_most_recent(self):
        stm = STM(bound=10)
        k1 = KLine(0xB0, [1])
        k2 = KLine(0xB0, [2])
        stm.add(k1)
        stm.add(k2)
        chain = _TierChain([stm])
        assert chain.find_first(0xB0) is k2

    def test_mixed_chain_first_tier_wins(self):
        stm = STM(bound=10)
        store = KLineStore()
        k_stm = KLine(0xB0, [1])
        k_store = KLine(0xB0, [2])
        stm.add(k_stm)
        store.add(k_store)
        chain = _TierChain([stm, store])
        assert chain.find_first(0xB0) is k_stm

    def test_no_match_returns_none(self):
        store = KLineStore()
        chain = _TierChain([store])
        assert chain.find_first(0xDEAD) is None


class TestTierChainFindAll:
    """Tests for _TierChain.find_all with dedup across tiers."""

    def test_dedup_across_two_stores(self):
        """Same kline object in two stores → appears once."""
        store1 = KLineStore()
        store2 = KLineStore()
        k = KLine(0xC0, [1])
        store1.add(k)
        store2.add(k)
        chain = _TierChain([store1, store2])
        results = chain.find_all(0xC0)
        assert len(results) == 1
        assert results[0] is k

    def test_multiple_matches_per_tier(self):
        store = KLineStore()
        k1 = KLine(0xC0, [1])
        k2 = KLine(0xC0, [2])
        store.add(k1)
        store.add(k2)
        chain = _TierChain([store])
        results = chain.find_all(0xC0)
        assert len(results) == 2

    def test_empty_when_no_matches(self):
        store = KLineStore()
        chain = _TierChain([store])
        assert chain.find_all(0xDEAD) == []


class TestTierChainAllKlines:
    """Tests for _TierChain.all_klines with ordering and dedup."""

    def test_stm_first_ordering(self):
        """STM entries appear before KLineStore entries."""
        stm = STM(bound=10)
        store = KLineStore()
        k_stm = KLine(1, [1])
        k_store = KLine(2, [2])
        stm.add(k_stm)
        store.add(k_store)
        chain = _TierChain([stm, store])
        results = chain.all_klines()
        assert results[0] is k_stm
        assert results[1] is k_store

    def test_two_stores_first_tier_before_second(self):
        store1 = KLineStore()
        store2 = KLineStore()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        store1.add(k1)
        store2.add(k2)
        chain = _TierChain([store1, store2])
        results = chain.all_klines()
        assert results[0] is k1
        assert results[1] is k2

    def test_dedup_same_in_two_tiers(self):
        store1 = KLineStore()
        store2 = KLineStore()
        k = KLine(1, [1])
        store1.add(k)
        store2.add(k)
        chain = _TierChain([store1, store2])
        results = chain.all_klines()
        assert len(results) == 1
        assert results[0] is k

    def test_model_base_uses_klines_path(self):
        """Chain with Model base uses Model.klines() via iter(self._tier.klines())."""
        base = Model()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        base.add_frame(k1)
        base.add_frame(k2)
        chain = _TierChain([base])
        results = chain.all_klines()
        assert len(results) == 2

    def test_empty_chain_returns_empty(self):
        chain = _TierChain([])
        assert chain.all_klines() == []


class TestTierChainFindByNodesFirst:
    """Tests for _TierChain.find_by_nodes_first across tier types."""

    def test_stm_returns_last_match(self):
        stm = STM(bound=10)
        k1 = KLine(0x10, [1, 2])
        k2 = KLine(0x20, [1, 2])
        stm.add(k1)
        stm.add(k2)
        chain = _TierChain([stm])
        nodes_sig = make_signature([1, 2])
        result = chain.find_by_nodes_first(nodes_sig)
        assert result is k2

    def test_store_scans_reversed(self):
        store = KLineStore()
        k1 = KLine(0x10, [3, 4])
        k2 = KLine(0x20, [3, 4])
        store.add(k1)
        store.add(k2)
        chain = _TierChain([store])
        nodes_sig = make_signature([3, 4])
        result = chain.find_by_nodes_first(nodes_sig)
        assert result is k2

    def test_mixed_first_tier_wins(self):
        stm = STM(bound=10)
        store = KLineStore()
        k_stm = KLine(0x10, [5, 6])
        k_store = KLine(0x20, [5, 6])
        stm.add(k_stm)
        store.add(k_store)
        chain = _TierChain([stm, store])
        nodes_sig = make_signature([5, 6])
        result = chain.find_by_nodes_first(nodes_sig)
        assert result is k_stm

    def test_no_match_returns_none(self):
        store = KLineStore()
        chain = _TierChain([store])
        nodes_sig = make_signature([99])
        assert chain.find_by_nodes_first(nodes_sig) is None
