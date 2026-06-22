"""Tests for misfit module — direct function calls, not Model forwarding."""

from kalvin.kline import KLine
from kalvin.misfit import classify_misfit, generate_expansions
from kalvin.model import Model
from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


class TestClassifyMisfit:
    def test_identity_self_referential(self):
        """Identity kline {S: [S]} → (False, False).

        make_signature([S]) == S, so both S & ~S (underfit) and S & ~S
        (overfit) are zero. classify_misfit does not distinguish identity
        from canon — see is_identity (KL-21) / is_canon (KL-24).
        """
        k = KLine(10, [10])  # make_sig([10]) = 10
        assert classify_misfit(k, signifier) == (False, False)

    def test_canonical(self):
        """Genuine canon {S: [A, B]} with S == A|B → (False, False).

        nodes_sig = make_signature([A, B]) == S, so neither S & ~nodes_sig
        (underfit) nor nodes_sig & ~S (overfit) is non-zero. See is_canon
        (KL-23).
        """
        k = KLine(0b110, [0b100, 0b010])  # make_sig([0b100, 0b010]) = 0b110
        assert classify_misfit(k, signifier) == (False, False)

    def test_underfitting(self):
        """S & ~N != 0 → (True, False)."""
        # sig=0b110, nodes=[0b100] → nodes_sig=0b100
        # underfit: 0b110 & ~0b100 = 0b110 & 0x..011 = 0b010 ≠ 0
        # overfit: 0b100 & ~0b110 = 0b100 & 0x..001 = 0 → False
        k = KLine(0b110, [0b100])
        assert classify_misfit(k, signifier) == (True, False)

    def test_overfitting(self):
        """N & ~S != 0 → (False, True)."""
        # sig=0b100, nodes=[0b110] → nodes_sig=0b110
        # underfit: 0b100 & ~0b110 = 0 → False
        # overfit: 0b110 & ~0b100 = 0b010 ≠ 0 → True
        k = KLine(0b100, [0b110])
        assert classify_misfit(k, signifier) == (False, True)

    def test_dual_misfit(self):
        """Both conditions → (True, True)."""
        # sig=0b101, nodes=[0b110] → nodes_sig=0b110
        # underfit: 0b101 & ~0b110 = 0b101 & 0x..001 = 0b001 ≠ 0 → True
        # overfit: 0b110 & ~0b101 = 0b110 & 0x..010 = 0b010 ≠ 0 → True
        k = KLine(0b101, [0b110])
        assert classify_misfit(k, signifier) == (True, True)


class TestGenerateExpansions:
    def _make_model_with_klines(self) -> Model:
        m = Model()
        # Identity (self-referential) klines that serve as contributors by signature
        m.add_to_frame(KLine(0b010, [0b010]))  # identity (self-referential: {S:[S]})
        m.add_to_frame(KLine(0b001, [0b001]))  # identity (self-referential: {S:[S]})
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
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask, signifier))
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
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask, signifier))
        assert len(results) == 1
        proposal, companions = results[0]
        assert proposal.signature == 0b100
        assert 0b110 not in proposal.nodes  # excess node removed
        assert len(companions) == 1
        assert companions[0].nodes == [0b110]  # removed node forms companion

    def test_dual_expansion_atomic_replace(self):
        """Dual misfit yields ONLY atomic replacements (excess swapped for gap-filler).

        The new logic performs an atomic swap per gap-filling contributor and
        does NOT also emit separate underfit-only or overfit-only proposals.
        """
        m = self._make_model_with_klines()
        # sig=0b101, nodes=[0b110] → nodes_sig=0b110
        # gap = 0b101 & ~0b110 = 0b001
        # excess = 0b110 & ~0b101 = 0b010
        k = KLine(0b101, [0b110])
        underfit_gap = 0b001
        overfit_mask = 0b010
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask, signifier))

        # Exactly one gap-filling contributor (the 0b001 kline) → one atomic swap.
        assert len(results) == 1
        proposal, companions = results[0]

        # Proposal keeps the original signature and contains the gap-filler node.
        assert proposal.signature == 0b101
        assert 0b001 in proposal.nodes
        # The excess node was removed (swapped out), not retained.
        assert 0b110 not in proposal.nodes

        # Every dual proposal carries exactly one companion built from the
        # removed excess nodes — there are no companion-free underfit-only
        # proposals in the dual path.
        assert all(len(comps) == 1 for _, comps in results)
        assert companions[0].nodes == [0b110]
        assert companions[0].signature == 0b110  # make_signature([0b110])

    def test_dual_expansion_one_swap_per_contributor(self):
        """Dual path yields one atomic swap per gap-filling contributor."""
        m = self._make_model_with_klines()
        # sig=0b111, nodes=[0b110] → nodes_sig=0b110
        # gap = 0b111 & ~0b110 = 0b001  → only the 0b001 kline contributes
        k = KLine(0b111, [0b110])
        results = list(generate_expansions(m, k, underfit_gap=0b001, overfit_mask=0b010, signifier=signifier))
        assert len(results) == 1
        proposal, companions = results[0]
        assert 0b001 in proposal.nodes
        assert 0b110 not in proposal.nodes
        assert companions[0].nodes == [0b110]

    def test_no_gap_no_expansion(self):
        """No gap and no excess → no expansion proposals."""
        m = Model()
        k = KLine(10, [10])  # identity (self-referential: {S:[S]})
        results = list(generate_expansions(m, k, 0, 0, signifier))
        assert len(results) == 0
