"""Tests for misfit module — direct function calls, not Model forwarding.

Test values use ``t(bits) = bits << 32`` so the type word (upper 32 bits) is
populated. Misfit classification operates on the type word only (masked,
consistent with ``signifies``); low-bit-only values would have an empty type
word and classify trivially as canonical. See specs/signifier.md
§classify_misfit.
"""

from kalvin.kline import KLine
from kalvin.misfit import generate_expansions
from kalvin.model import Model
from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()


def t(bits: int) -> int:
    """Place type-word bits in the upper 32 bits of a uint64."""
    return bits << 32


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


class TestClassifyMisfit:
    def test_identity_self_referential(self):
        """Identity kline {S: [S]} → (False, False).

        make_signature([S]) == S, so neither residual direction is non-zero.
        classify_misfit does not distinguish identity from canon — see
        is_identity (KL-21) / is_canon (KL-24).
        """
        k = KLine(t(10), [t(10)])  # make_sig([t(10)]) = t(10)
        assert signifier.classify_misfit(k.signature, k.nodes) == (False, False)

    def test_canonical(self):
        """Genuine canon {S: [A, B]} with S == A|B → (False, False).

        nodes_sig = make_signature([A, B]) == S, so neither residual is
        non-zero. See is_canon (KL-23).
        """
        k = KLine(t(0b110), [t(0b100), t(0b010)])
        assert signifier.classify_misfit(k.signature, k.nodes) == (False, False)

    def test_underfitting(self):
        """Signature over-claims → (True, False).

        sig=t(0b110), nodes=[t(0b100)] → nodes_sig=t(0b100).
        residual(sig, nodes_sig) = t(0b010) ≠ 0 → underfit.
        residual(nodes_sig, sig) = 0 → not overfit.
        """
        k = KLine(t(0b110), [t(0b100)])
        assert signifier.classify_misfit(k.signature, k.nodes) == (True, False)

    def test_overfitting(self):
        """Nodes over-deliver → (False, True).

        sig=t(0b100), nodes=[t(0b110)] → nodes_sig=t(0b110).
        residual(sig, nodes_sig) = 0 → not underfit.
        residual(nodes_sig, sig) = t(0b010) ≠ 0 → overfit.
        """
        k = KLine(t(0b100), [t(0b110)])
        assert signifier.classify_misfit(k.signature, k.nodes) == (False, True)

    def test_dual_misfit(self):
        """Both residuals non-zero → (True, True).

        sig=t(0b101), nodes=[t(0b110)] → nodes_sig=t(0b110).
        residual(sig, nodes_sig) = t(0b001) ≠ 0 → underfit.
        residual(nodes_sig, sig) = t(0b010) ≠ 0 → overfit.
        """
        k = KLine(t(0b101), [t(0b110)])
        assert signifier.classify_misfit(k.signature, k.nodes) == (True, True)

    def test_bpe_id_difference_ignored(self):
        """Same type word, differing BPE ids → (False, False).

        Masking excludes BPE-token-id residuals: two values differing only
        in the low 32 bits have a zero type-word residual. (SIG-23.)
        """
        # type word 0b100 in both; BPE ids 5 and 9 differ.
        k = KLine(t(0b100) | 5, [t(0b100) | 9])
        assert signifier.classify_misfit(k.signature, k.nodes) == (False, False)


class TestGenerateExpansions:
    def _make_model_with_klines(self) -> Model:
        m = Model(signifier=signifier)
        # Identity (self-referential) klines that serve as contributors by signature
        m.add_to_frame(KLine(t(0b010), [t(0b010)]))  # {S:[S]}
        m.add_to_frame(KLine(t(0b001), [t(0b001)]))  # {S:[S]}
        return m

    def test_underfit_expansion_adds_nodes(self):
        """Underfit expansion yields proposals with added nodes."""
        m = self._make_model_with_klines()
        # sig=t(0b110), nodes=[t(0b100)] → nodes_sig=t(0b100)
        # gap = residual(t(0b110), t(0b100)) = t(0b010)
        # contributor with sig=t(0b010) should be found via signifies
        k = KLine(t(0b110), [t(0b100)])
        underfit_gap = signifier.residual(t(0b110), t(0b100))  # t(0b010)
        overfit_mask = 0
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask, signifier))
        assert len(results) >= 1
        proposal, companions = results[0]
        assert proposal.signature == t(0b110)  # signature stays the same
        assert t(0b010) in proposal.nodes
        assert companions == []  # no companions for underfit

    def test_overfit_expansion_removes_nodes(self):
        """Overfit expansion yields trimmed kline + companion."""
        m = Model(signifier=signifier)
        # sig=t(0b100), nodes=[t(0b110)] → nodes_sig=t(0b110)
        # excess = residual(t(0b110), t(0b100)) = t(0b010)
        k = KLine(t(0b100), [t(0b110)])
        underfit_gap = 0
        overfit_mask = signifier.residual(t(0b110), t(0b100))  # t(0b010)
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask, signifier))
        assert len(results) == 1
        proposal, companions = results[0]
        assert proposal.signature == t(0b100)
        assert t(0b110) not in proposal.nodes  # excess node removed
        assert len(companions) == 1
        assert companions[0].nodes == [t(0b110)]  # removed node forms companion

    def test_dual_expansion_atomic_replace(self):
        """Dual misfit yields ONLY atomic replacements (excess swapped for gap-filler).

        The new logic performs an atomic swap per gap-filling contributor and
        does NOT also emit separate underfit-only or overfit-only proposals.
        """
        m = self._make_model_with_klines()
        # sig=t(0b101), nodes=[t(0b110)] → nodes_sig=t(0b110)
        # gap = residual(t(0b101), t(0b110)) = t(0b001)
        # excess = residual(t(0b110), t(0b101)) = t(0b010)
        k = KLine(t(0b101), [t(0b110)])
        underfit_gap = signifier.residual(t(0b101), t(0b110))  # t(0b001)
        overfit_mask = signifier.residual(t(0b110), t(0b101))  # t(0b010)
        results = list(generate_expansions(m, k, underfit_gap, overfit_mask, signifier))

        # Exactly one gap-filling contributor (the t(0b001) kline) → one atomic swap.
        assert len(results) == 1
        proposal, companions = results[0]

        # Proposal keeps the original signature and contains the gap-filler node.
        assert proposal.signature == t(0b101)
        assert t(0b001) in proposal.nodes
        # The excess node was removed (swapped out), not retained.
        assert t(0b110) not in proposal.nodes

        # Every dual proposal carries exactly one companion built from the
        # removed excess nodes — there are no companion-free underfit-only
        # proposals in the dual path.
        assert all(len(comps) == 1 for _, comps in results)
        assert companions[0].nodes == [t(0b110)]
        assert companions[0].signature == t(0b110)  # make_signature([t(0b110)])

    def test_dual_expansion_one_swap_per_contributor(self):
        """Dual path yields one atomic swap per gap-filling contributor."""
        m = self._make_model_with_klines()
        # Genuine dual: sig type word 0b101, nodes type word 0b110.
        # nodes_sig = t(0b110). residual(sig, nodes_sig) = t(0b001) (underfit);
        # residual(nodes_sig, sig) = t(0b010) (overfit). Only the t(0b001)
        # kline contributes; the t(0b110) node is the excess swapped out.
        k = KLine(t(0b101), [t(0b110)])
        results = list(generate_expansions(
            m, k,
            underfit_gap=signifier.residual(t(0b101), t(0b110)),
            overfit_mask=signifier.residual(t(0b110), t(0b101)),
            signifier=signifier,
        ))
        assert len(results) == 1
        proposal, companions = results[0]
        assert t(0b001) in proposal.nodes
        assert t(0b110) not in proposal.nodes
        assert companions[0].nodes == [t(0b110)]

    def test_no_gap_no_expansion(self):
        """No gap and no excess → no expansion proposals."""
        m = Model(signifier=signifier)
        k = KLine(t(10), [t(10)])  # identity (self-referential: {S:[S]})
        results = list(generate_expansions(m, k, 0, 0, signifier))
        assert len(results) == 0
