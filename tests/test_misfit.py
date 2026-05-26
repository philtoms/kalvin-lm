"""Tests for misfit module — direct function calls, not Model forwarding."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.misfit import classify_misfit, generate_expansions


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


class TestClassifyMisfit:
    def test_canonical(self):
        """S == N → (False, False)."""
        k = KLine(10, [10])  # make_sig([10]) = 10
        assert classify_misfit(k) == (False, False)

    def test_underfitting(self):
        """S & ~N != 0 → (True, False)."""
        # sig=0b110, nodes=[0b100] → nodes_sig=0b100
        # underfit: 0b110 & ~0b100 = 0b110 & 0x..011 = 0b010 ≠ 0
        # overfit: 0b100 & ~0b110 = 0b100 & 0x..001 = 0 → False
        k = KLine(0b110, [0b100])
        assert classify_misfit(k) == (True, False)

    def test_overfitting(self):
        """N & ~S != 0 → (False, True)."""
        # sig=0b100, nodes=[0b110] → nodes_sig=0b110
        # underfit: 0b100 & ~0b110 = 0 → False
        # overfit: 0b110 & ~0b100 = 0b010 ≠ 0 → True
        k = KLine(0b100, [0b110])
        assert classify_misfit(k) == (False, True)

    def test_dual_misfit(self):
        """Both conditions → (True, True)."""
        # sig=0b101, nodes=[0b110] → nodes_sig=0b110
        # underfit: 0b101 & ~0b110 = 0b101 & 0x..001 = 0b001 ≠ 0 → True
        # overfit: 0b110 & ~0b101 = 0b110 & 0x..010 = 0b010 ≠ 0 → True
        k = KLine(0b101, [0b110])
        assert classify_misfit(k) == (True, True)


class TestGenerateExpansions:
    def _make_model_with_klines(self) -> Model:
        m = Model()
        # Canonical klines that can serve as contributors
        m.add(KLine(0b010, [0b010]))  # canonical
        m.add(KLine(0b001, [0b001]))  # canonical
        m.promote_all()
        return m

    def test_underfit_expansion_adds_nodes(self):
        """Underfit expansion yields proposals with added nodes."""
        m = self._make_model_with_klines()
        # sig=0b110, nodes=[0b100] → nodes_sig=0b100
        # gap = 0b110 & ~0b100 = 0b010
        # contributor with sig=0b010 should be found
        k = KLine(0b110, [0b100])
        underfit_gap = 0b010
        overfit_mask = 0
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask))
        assert len(results) >= 1
        proposal, companions = results[0]
        assert proposal.signature == 0b110  # signature stays the same
        assert 0b010 in proposal.nodes or any(
            0b010 in n if isinstance(n, int) else False for n in proposal.nodes
        )
        assert companions == []  # no companions for underfit

    def test_overfit_expansion_removes_nodes(self):
        """Overfit expansion yields trimmed kline + companion."""
        m = Model()
        # sig=0b100, nodes=[0b110] → nodes_sig=0b110
        # excess = 0b110 & ~0b100 = 0b010
        k = KLine(0b100, [0b110])
        underfit_gap = 0
        overfit_mask = 0b010
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask))
        assert len(results) == 1
        proposal, companions = results[0]
        assert proposal.signature == 0b100
        assert 0b110 not in proposal.nodes  # excess node removed
        assert len(companions) == 1
        assert companions[0].nodes == [0b110]  # removed node forms companion

    def test_dual_expansion_atomic_replace(self):
        """Dual expansion yields replacement + companion."""
        m = self._make_model_with_klines()
        # sig=0b101, nodes=[0b110] → nodes_sig=0b110
        # gap = 0b101 & ~0b110 = 0b001
        # excess = 0b110 & ~0b101 = 0b010
        k = KLine(0b101, [0b110])
        underfit_gap = 0b001
        overfit_mask = 0b010
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask))
        # Should produce underfit, overfit, and dual expansions
        assert len(results) >= 2
        # Find the dual expansion (has companions from excess removal AND added nodes from gap fill)
        dual_results = [r for r in results if len(r[1]) > 0]
        assert len(dual_results) >= 1
        # At least one dual result should have contributor nodes added
        has_replacement = any(0b010 in r[0].nodes or 0b001 in r[0].nodes for r in dual_results)
        assert has_replacement

    def test_no_gap_no_expansion(self):
        """No gap and no excess → no expansion proposals."""
        m = Model()
        k = KLine(10, [10])  # canonical
        results = list(generate_expansions(m, k, 0, 0))
        assert len(results) == 0
