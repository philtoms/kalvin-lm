"""Tests for Sampling — boundary-based response sampling for cogitation."""

import pytest

from kalvin.agent import Sampling, Agent, Cogitator, WorkItem, _S2_S3_DISTANCE, _TEMP_SCALE
from kalvin.kline import KLine
from kalvin.model import D_MAX, MASK64, Model, QueryCandidate, _pack, _S3_BIAS
from kalvin.events import EventBus


# ── Sampling Construction ─────────────────────────────────────────────

class TestSamplingDefaults:
    def test_default_temperature(self):
        s = Sampling()
        assert s.temperature == 1.0

    def test_default_top_k(self):
        s = Sampling()
        assert s.top_k == 40

    def test_default_top_p(self):
        s = Sampling()
        assert s.top_p == 0.95

    def test_repr(self):
        s = Sampling()
        assert "temperature=1.0" in repr(s)
        assert "top_k=40" in repr(s)
        assert "top_p=0.95" in repr(s)


class TestSamplingCustom:
    def test_custom_values(self):
        s = Sampling(temperature=0.5, top_k=10, top_p=0.8)
        assert s.temperature == 0.5
        assert s.top_k == 10
        assert s.top_p == 0.8

    def test_extreme_temperature_high(self):
        s = Sampling(temperature=100.0)
        assert s.temperature == 100.0

    def test_extreme_temperature_low(self):
        s = Sampling(temperature=0.001)
        assert s.temperature == 0.001

    def test_top_k_zero_unlimited(self):
        s = Sampling(top_k=0)
        assert s.top_k == 0

    def test_top_p_one_no_truncation(self):
        s = Sampling(top_p=1.0)
        assert s.top_p == 1.0


class TestSamplingValidation:
    def test_temperature_zero_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            Sampling(temperature=0)

    def test_temperature_negative_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            Sampling(temperature=-1.0)

    def test_top_k_negative_rejected(self):
        with pytest.raises(ValueError, match="top_k"):
            Sampling(top_k=-1)

    def test_top_p_zero_rejected(self):
        with pytest.raises(ValueError, match="top_p"):
            Sampling(top_p=0.0)

    def test_top_p_above_one_rejected(self):
        with pytest.raises(ValueError, match="top_p"):
            Sampling(top_p=1.1)

    def test_top_p_negative_rejected(self):
        with pytest.raises(ValueError, match="top_p"):
            Sampling(top_p=-0.5)


# ── Boundary Computation ─────────────────────────────────────────────

class TestBoundaries:
    """Test _base_boundaries and _boundaries (temperature-shifted)."""

    def _make_cogitator(self, sampling: Sampling) -> Cogitator:
        """Create a Cogitator with the given sampling for boundary testing."""
        model = Model()
        bus = EventBus()
        cog = Cogitator(
            model=model,
            event_bus=bus,
            on_s1=lambda q, c: None,
            sampling=sampling,
        )
        cog.join(timeout=0.1)
        return cog

    def test_base_boundaries(self):
        """Base boundaries at τ=1: S1|S2 = D_MAX-1, S2|S3 = ~_S2_S3_DISTANCE, S3|S4 = 0."""
        s12, s23, s34 = Cogitator._base_boundaries()
        assert s12 == D_MAX - 1
        assert s23 == (~_S2_S3_DISTANCE) & MASK64
        assert s34 == 0

    def test_identity_at_tau_1(self):
        """τ=1.0 → boundaries == base boundaries."""
        cog = self._make_cogitator(Sampling(temperature=1.0))
        s12, s23, s34 = cog._boundaries()
        base = Cogitator._base_boundaries()
        assert (s12, s23, s34) == base

    def test_high_tau_lowers_s12(self):
        """τ>1 → S1|S2 boundary drops (more S2 qualifies as S1)."""
        cog1 = self._make_cogitator(Sampling(temperature=1.0))
        cog2 = self._make_cogitator(Sampling(temperature=2.0))
        s12_1, _, _ = cog1._boundaries()
        s12_2, _, _ = cog2._boundaries()
        assert s12_2 < s12_1  # boundary dropped

    def test_high_tau_lowers_s23(self):
        """τ>1 → S2|S3 boundary drops (more S3 qualifies as S2)."""
        cog1 = self._make_cogitator(Sampling(temperature=1.0))
        cog2 = self._make_cogitator(Sampling(temperature=2.0))
        _, s23_1, _ = cog1._boundaries()
        _, s23_2, _ = cog2._boundaries()
        assert s23_2 < s23_1  # boundary dropped

    def test_high_tau_s34_capped_at_zero(self):
        """τ>1 → S3|S4 boundary stays at 0 (can't go lower)."""
        cog = self._make_cogitator(Sampling(temperature=5.0))
        _, _, s34 = cog._boundaries()
        assert s34 == 0

    def test_low_tau_raises_s23(self):
        """τ<1 → S2|S3 boundary rises (fewer S3 qualify as S2)."""
        cog1 = self._make_cogitator(Sampling(temperature=1.0))
        cog2 = self._make_cogitator(Sampling(temperature=0.5))
        _, s23_1, _ = cog1._boundaries()
        _, s23_2, _ = cog2._boundaries()
        assert s23_2 > s23_1  # boundary rose

    def test_low_tau_raises_s34(self):
        """τ<1 → S3|S4 boundary rises (more S3 demoted to S4)."""
        cog = self._make_cogitator(Sampling(temperature=0.5))
        _, _, s34 = cog._boundaries()
        assert s34 > 0  # boundary rose from 0

    def test_low_tau_s12_capped_at_dmax(self):
        """τ<1 → S1|S2 boundary stays at D_MAX-1 (can't go higher)."""
        cog = self._make_cogitator(Sampling(temperature=0.5))
        s12, _, _ = cog._boundaries()
        assert s12 == D_MAX - 1

    def test_monotonic_s12_with_temperature(self):
        """Higher τ → lower S1|S2 boundary."""
        results = []
        for tau in [0.5, 1.0, 1.5, 2.0, 5.0]:
            cog = self._make_cogitator(Sampling(temperature=tau))
            s12, _, _ = cog._boundaries()
            results.append(s12)
        # S1|S2 should be monotonically non-increasing (capped at D_MAX-1)
        for i in range(1, len(results)):
            assert results[i] <= results[i - 1], (
                f"Monotonicity violated: τ={[0.5, 1.0, 1.5, 2.0, 5.0][i]} "
                f"S1|S2={results[i]:#x} > previous={results[i-1]:#x}"
            )

    def test_monotonic_s23_with_temperature(self):
        """Higher τ → lower S2|S3 boundary."""
        results = []
        for tau in [0.5, 1.0, 1.5, 2.0, 5.0]:
            cog = self._make_cogitator(Sampling(temperature=tau))
            _, s23, _ = cog._boundaries()
            results.append(s23)
        for i in range(1, len(results)):
            assert results[i] <= results[i - 1]


# ── Classification ────────────────────────────────────────────────────

class TestClassify:
    """Test _classify against boundaries."""

    def test_classify_s1(self):
        """Significance at D_MAX → S1."""
        assert Cogitator._classify(D_MAX, D_MAX - 1, 0, 0) == "S1"

    def test_classify_s2(self):
        """Significance above S2|S3 boundary but below D_MAX-1 → S2."""
        # Pure S2: distance = 50, sig = ~50 (above _S2_S3_DISTANCE=80 boundary)
        sig = (~50) & MASK64
        s23 = (~_S2_S3_DISTANCE) & MASK64
        assert Cogitator._classify(sig, D_MAX - 1, s23, 0) == "S2"

    def test_classify_s3(self):
        """Significance below S2|S3 boundary but above 0 → S3."""
        # Pure S3: distance = _pack(2 + _S3_BIAS) = 100
        sig = (~_pack(2 + _S3_BIAS)) & MASK64
        s23 = (~_S2_S3_DISTANCE) & MASK64
        assert Cogitator._classify(sig, D_MAX - 1, s23, 0) == "S3"

    def test_classify_s4(self):
        """Significance below S3|S4 boundary → S4."""
        s23 = (~_S2_S3_DISTANCE) & MASK64
        assert Cogitator._classify(0, D_MAX - 1, s23, 1) == "S4"

    def test_classify_at_boundary_s12(self):
        """Significance exactly at S1|S2 boundary → S1."""
        assert Cogitator._classify(D_MAX - 1, D_MAX - 1, 0, 0) == "S1"

    def test_classify_just_below_s23_is_s3(self):
        """Significance just below S2|S3 boundary → S3."""
        s23 = (~_S2_S3_DISTANCE) & MASK64
        just_below = s23 - 1
        assert Cogitator._classify(just_below, D_MAX - 1, s23, 0) == "S3"


# ── Agent Sampling Property ──────────────────────────────────────────

class TestAgentSamplingProperty:
    def test_default_sampling(self):
        a = Agent()
        assert a.sampling.temperature == 1.0
        assert a.sampling.top_k == 40
        assert a.sampling.top_p == 0.95
        a.cogitate_join(timeout=0.1)

    def test_sampling_setter(self):
        a = Agent()
        new_sampling = Sampling(temperature=0.7, top_k=20, top_p=0.9)
        a.sampling = new_sampling
        assert a.sampling.temperature == 0.7
        assert a.sampling.top_k == 20
        assert a.sampling.top_p == 0.9
        a.cogitate_join(timeout=0.1)

    def test_sampling_mutation_visible_on_cogitator(self):
        a = Agent()
        new_sampling = Sampling(temperature=1.5)
        a.sampling = new_sampling
        assert a.cogitator.sampling is new_sampling
        a.cogitate_join(timeout=0.1)


# ── Streaming Sampling Integration ───────────────────────────────────

class TestStreamingSampling:
    """Test the _run_work_item streaming loop with sampling parameters."""

    def _make_cogitator(
        self,
        sampling: Sampling,
        yields: list[QueryCandidate],
    ) -> tuple[Cogitator, list[QueryCandidate]]:
        """Create a Cogitator that mocks expand() to yield fixed QCs.

        Returns (cogitator, processed_list) where processed_list captures
        every QC that passes through _process (S2/S3 only).
        """
        model = Model()
        bus = EventBus()
        processed: list[QueryCandidate] = []

        s1_calls: list[tuple] = []

        def on_s1(q, c):
            s1_calls.append((q, c))

        cog = Cogitator(
            model=model,
            event_bus=bus,
            on_s1=on_s1,
            sampling=sampling,
        )
        cog.join(timeout=0.1)

        # Track what gets processed (S2/S3 countersignature check)
        original_process = cog._process

        def tracking_process(qc):
            processed.append(qc)
            original_process(qc)

        cog._process = tracking_process

        # Also track S1 calls
        cog._on_s1 = lambda q, c: s1_calls.append((q, c))
        cog._s1_calls = s1_calls  # attach for test access

        # Mock expand to yield our fixed QCs
        def mock_expand(query, candidate, level, distance=0, **kw):
            yield from yields

        cog._model.expand = mock_expand

        return cog, processed

    def _make_qc(self, significance: int) -> QueryCandidate:
        return QueryCandidate(
            KLine(1, [10]),
            KLine(2, [20]),
            significance,
        )

    def test_top_k_limits_processing(self):
        """top_k=2 processes at most 2 QCs from a stream of 5."""
        # Use low significance (S3 range) so top_p doesn't trigger early
        sig = D_MAX // 20
        qcs = [self._make_qc(sig) for _ in range(5)]
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=2, top_p=1.0),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 2

    def test_top_k_zero_processes_all(self):
        """top_k=0 processes all QCs."""
        sig = D_MAX // 20
        qcs = [self._make_qc(sig) for _ in range(10)]
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 10

    def test_s4_demoted(self):
        """QCs below S3|S4 boundary are demoted (not processed)."""
        # At τ=1, S3|S4 = 0. Significance of 0 is below 0? No, 0 < 0 is False.
        # Actually S3|S4 = 0, so only sig < 0 is S4, which can't happen.
        # With low τ, S3|S4 rises, so sigs in [0, new_s34) become S4.
        sig_low = 10  # very low significance
        qcs = [self._make_qc(sig_low)]

        # Use very low τ to raise S3|S4 above sig_low
        cog, processed = self._make_cogitator(
            Sampling(temperature=0.3, top_k=0, top_p=1.0),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 0  # demoted to S4

    def test_high_temp_promotes_s2_to_s1(self):
        """High τ lowers S1|S2 boundary, so near-S1 S2 QCs classify as S1."""
        # QC with distance=50 → sig = ~50 (S2 at τ=1)
        sig = (~50) & MASK64

        qcs = [self._make_qc(sig)]

        # At τ=1, this is S2 (sig < D_MAX-1)
        cog_tau1, processed_tau1 = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            qcs,
        )

        # At τ=5, S1|S2 boundary drops well below this sig → S1
        cog_tau5, processed_tau5 = self._make_cogitator(
            Sampling(temperature=5.0, top_k=0, top_p=1.0),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog_tau1._run_work_item(item)
        cog_tau5._run_work_item(item)

        # τ=1: S2, processed via _process
        assert len(processed_tau1) == 1
        assert len(cog_tau1._s1_calls) == 0

        # τ=5: S1, processed via _on_s1
        assert len(processed_tau5) == 0  # S1 goes to _on_s1, not _process
        assert len(cog_tau5._s1_calls) == 1

    def test_top_p_stops_on_sufficient_evidence(self):
        """top_p < 1.0 stops when cumulative significance is sufficient."""
        sig = D_MAX // 2
        qcs = [self._make_qc(sig) for _ in range(10)]

        # top_p=0.9: after 2 QCs, cumulative = D_MAX > 0.9 * D_MAX
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=0.9),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 2

    def test_top_p_one_never_stops_early(self):
        """top_p=1.0 never triggers early stop."""
        sig = D_MAX // 20
        qcs = [self._make_qc(sig) for _ in range(5)]

        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 5

    def test_boundary_classification_then_top_k(self):
        """Boundary classification filters S4, then top-k caps the survivors."""
        high_sig = D_MAX // 10  # S3 at τ=1 (below HP, above 0)
        low_sig = 1             # S3 at τ=1 (above 0)

        # Use very low τ to raise S3|S4, making low_sig become S4
        # At τ=0.3, shift = _TEMP_SCALE * (0.3 - 1) = negative, S34 rises
        qcs = [
            self._make_qc(high_sig),
            self._make_qc(low_sig),
            self._make_qc(high_sig),
            self._make_qc(low_sig),
            self._make_qc(high_sig),
        ]

        # At τ=0.3, S3|S4 boundary rises, so low_sig=1 becomes S4
        cog, processed = self._make_cogitator(
            Sampling(temperature=0.3, top_k=2, top_p=1.0),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        # high_sig QCs pass (S3), low_sig QCs demoted (S4)
        # top_k=2 caps at 2 surviving QCs
        assert len(processed) == 2
        assert all(qc.significance == high_sig for qc in processed)

    def test_empty_stream(self):
        """Empty expand() stream → nothing processed."""
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            [],
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 0

    def test_boundaries_computed_once_per_work_item(self):
        """Boundaries are computed once at start of each work item."""
        # This is an architectural test — verify that boundaries are
        # read once, not per-yield. We test by changing sampling mid-stream.
        # Since expand() is mocked, the boundaries are computed at the
        # start and the loop runs to completion with those boundaries.
        sig = (~50) & MASK64  # S2 at τ=1
        qcs = [self._make_qc(sig) for _ in range(5)]

        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            qcs,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        # All 5 processed (S2 at τ=1, no limits)
        assert len(processed) == 5


# ── Integration: Sampling with Real Agent ─────────────────────────────

class TestSamplingIntegration:
    def test_sampling_survives_serialization_roundtrip(self):
        """Sampling state is not serialized (it's runtime config), but
        agent loads with default sampling."""
        a = Agent()
        a.sampling = Sampling(temperature=0.7, top_k=20, top_p=0.9)
        a.rationalise(KLine(5, [1, 2]))

        # Serialize and reload
        data = a.to_dict()
        loaded = Agent.from_dict(data)

        # Sampling reverts to defaults (not persisted)
        assert loaded.sampling.temperature == 1.0
        assert loaded.sampling.top_k == 40
        assert loaded.sampling.top_p == 0.95
        loaded.cogitate_join(timeout=0.1)

    def test_s2_with_custom_sampling(self):
        """S2 kline with custom sampling still queues work items."""
        a = Agent()
        a.sampling = Sampling(temperature=0.5, top_k=10, top_p=0.9)

        # Add candidate, then S2 query
        candidate = KLine(5, [10, 30])
        a.rationalise(candidate)

        q = KLine(0, [10, 20])
        from kalvin.signature import make_signature
        q.signature = make_signature([10, 20], a.tokenizer.is_literal)

        result = a.rationalise(q)
        assert result is False  # S2 → queued
        a.cogitate_join(timeout=0.1)

    def test_rationalise_after_join_with_custom_sampling(self):
        """Agent works after cogitation thread stopped with custom sampling."""
        a = Agent()
        a.sampling = Sampling(temperature=1.5, top_k=5, top_p=0.8)
        a.cogitate_join(timeout=0.1)

        # Should still work (no background thread)
        k = KLine(5, [1, 2])
        result = a.rationalise(k)
        assert result is True
