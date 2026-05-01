"""Tests for Sampling — response sampling parameters for cogitation."""

import pytest

from kalvin.agent import Sampling, Agent, Cogitator, WorkItem
from kalvin.kline import KLine
from kalvin.model import D_MAX, MASK64, Model, QueryCandidate
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


# ── Temperature Adjustment ───────────────────────────────────────────

class TestAdjust:
    def _make_cogitator(self, sampling: Sampling) -> Cogitator:
        """Create a Cogitator with the given sampling for testing _adjust."""
        model = Model()
        bus = EventBus()
        cog = Cogitator(
            model=model,
            event_bus=bus,
            on_s1=lambda q, c: None,
            sampling=sampling,
        )
        # Stop the background thread immediately — we only need _adjust
        cog.join(timeout=0.1)
        return cog

    def test_identity_at_tau_1(self):
        """τ=1.0 → adjust(sig) == sig for any significance."""
        cog = self._make_cogitator(Sampling(temperature=1.0))
        for sig in [D_MAX - 1, 0x8000_0000_0000_0000, 0x0000_0000_FFFF_FFFF, 1]:
            assert cog._adjust(sig) == sig, f"Identity failed at sig={sig:#x}"

    def test_conservative_lowers_significance(self):
        """τ<1 → adjust(sig) ≤ sig for non-max significance."""
        cog = self._make_cogitator(Sampling(temperature=0.5))
        sig = 0x0000_0000_FFFF_FFFF
        adjusted = cog._adjust(sig)
        assert adjusted < sig

    def test_exploratory_raises_significance(self):
        """τ>1 → adjust(sig) ≥ sig for non-max significance."""
        cog = self._make_cogitator(Sampling(temperature=2.0))
        sig = 0x0000_0000_FFFF_FFFF
        adjusted = cog._adjust(sig)
        assert adjusted > sig

    def test_s1_at_identity_only(self):
        """Max significance (D_MAX - 1) stays same at τ=1, changes otherwise.

        S1 has distance=1. Dividing by τ≠1 changes this distance.
        This is correct — temperature modulates the entire significance
        landscape, including near-S1 connotations.
        """
        cog = self._make_cogitator(Sampling(temperature=1.0))
        assert cog._adjust(D_MAX - 1) == D_MAX - 1

        # Conservative τ: S1 significance drops (distance grows)
        cog = self._make_cogitator(Sampling(temperature=0.5))
        assert cog._adjust(D_MAX - 1) < D_MAX - 1

        # Exploratory τ: S1 significance rises (distance shrinks → 0)
        cog = self._make_cogitator(Sampling(temperature=2.0))
        assert cog._adjust(D_MAX - 1) == D_MAX  # distance=1/2=0 → ~0 = D_MAX

    def test_zero_significance_at_low_tau(self):
        """Very low significance stays near-zero under conservative τ."""
        cog = self._make_cogitator(Sampling(temperature=0.1))
        sig = 1  # almost no significance
        adjusted = cog._adjust(sig)
        assert adjusted == 0  # distance = D_MAX-1, / 0.1 → clamped to D_MAX → ~D_MAX = 0

    def test_monotonic_with_temperature(self):
        """Higher τ → higher adjusted significance (for fixed input)."""
        sig = 0x0000_0000_FFFF_FFFF
        results = []
        for tau in [0.5, 1.0, 1.5, 2.0, 5.0]:
            cog = self._make_cogitator(Sampling(temperature=tau))
            results.append(cog._adjust(sig))
        # Results should be monotonically non-decreasing
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1], (
                f"Monotonicity violated: τ={[0.5, 1.0, 1.5, 2.0, 5.0][i]} "
                f"adjusted={results[i]:#x} < previous={results[i-1]:#x}"
            )

    def test_adjust_works_in_distance_space(self):
        """Verify the distance-space arithmetic explicitly."""
        cog = self._make_cogitator(Sampling(temperature=2.0))
        sig = ~100 & MASK64  # significance corresponding to distance=100
        # At τ=2, distance becomes 100/2=50
        expected = ~50 & MASK64
        assert cog._adjust(sig) == expected


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
        publish_threshold: int = 0,
    ) -> tuple[Cogitator, list[QueryCandidate]]:
        """Create a Cogitator that mocks expand() to yield fixed QCs.

        Returns (cogitator, processed_list) where processed_list captures
        every QC that passes through _process.

        publish_threshold defaults to 0 (all QCs pass) for testing streaming.
        """
        model = Model()
        bus = EventBus()
        processed: list[QueryCandidate] = []

        def on_s1(q, c):
            pass

        cog = Cogitator(
            model=model,
            event_bus=bus,
            on_s1=on_s1,
            sampling=sampling,
        )
        # Stop the real background thread
        cog.join(timeout=0.1)

        # Track what gets processed
        original_process = cog._process

        def tracking_process(qc):
            processed.append(qc)
            original_process(qc)

        cog._process = tracking_process

        # Mock expand to yield our fixed QCs
        def mock_expand(query, candidate, level, distance=0, **kw):
            yield from yields

        cog._model.expand = mock_expand

        # Override publish threshold for streaming tests
        cog.publish_threshold = publish_threshold

        return cog, processed

    def _make_qc(self, significance: int) -> QueryCandidate:
        return QueryCandidate(
            KLine(1, [10]),
            KLine(2, [20]),
            significance,
        )

    def test_top_k_limits_processing(self):
        """top_k=2 processes at most 2 QCs from a stream of 5."""
        sig = D_MAX // 10
        qcs = [self._make_qc(sig) for _ in range(5)]
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=2, top_p=0.999),
            qcs,
            publish_threshold=0,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 2

    def test_top_k_zero_processes_all(self):
        """top_k=0 processes all QCs."""
        sig = D_MAX // 20
        qcs = [self._make_qc(sig) for _ in range(10)]
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=0.999),
            qcs,
            publish_threshold=0,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 10

    def test_temperature_gates_low_significance(self):
        """Low temperature skips QCs below publish threshold."""
        low_sig = D_MAX // 10
        qcs = [self._make_qc(low_sig)]

        # Publish threshold above the QC's significance at τ=1
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            qcs,
            publish_threshold=D_MAX // 2,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 0

    def test_high_temperature_passes_low_significance(self):
        """High temperature raises QC above publish threshold."""
        sig = D_MAX // 4
        qcs = [self._make_qc(sig)]

        # At τ=1, significance D_MAX//4 is below D_MAX//2 threshold
        cog_tau1, processed_tau1 = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            qcs,
            publish_threshold=D_MAX // 2,
        )

        # At τ=10, distance shrinks → significance rises above threshold
        cog_tau10, processed_tau10 = self._make_cogitator(
            Sampling(temperature=10.0, top_k=0, top_p=1.0),
            qcs,
            publish_threshold=D_MAX // 2,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog_tau1._run_work_item(item)
        cog_tau10._run_work_item(item)

        assert len(processed_tau1) == 0
        assert len(processed_tau10) == 1

    def test_top_p_stops_on_sufficient_evidence(self):
        """top_p < 1.0 stops when cumulative significance is sufficient."""
        sig = D_MAX // 2
        qcs = [self._make_qc(sig) for _ in range(10)]

        # top_p=0.9: after 2 QCs, cumulative = D_MAX > 0.9 * D_MAX
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=0.9),
            qcs,
            publish_threshold=0,
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
            publish_threshold=0,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 5

    def test_temperature_then_top_k(self):
        """Temperature gates first, then top-k caps the survivors."""
        high_sig = D_MAX // 5
        low_sig = 1  # well below threshold
        qcs = [
            self._make_qc(high_sig),
            self._make_qc(low_sig),
            self._make_qc(high_sig),
            self._make_qc(low_sig),
            self._make_qc(high_sig),
        ]

        # threshold between high and low, top_k=2
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=2, top_p=0.999),
            qcs,
            publish_threshold=high_sig // 2,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        # First high-sig QC: count=1, process
        # First low-sig QC: skipped (temperature gate)
        # Second high-sig QC: count=2, process → top_k reached → break
        assert len(processed) == 2
        assert all(qc.significance == high_sig for qc in processed)

    def test_empty_stream(self):
        """Empty expand() stream → nothing processed."""
        cog, processed = self._make_cogitator(
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            [],
            publish_threshold=0,
        )

        item = WorkItem(KLine(1, [10]), KLine(2, [20]), "S2")
        cog._run_work_item(item)

        assert len(processed) == 0

    def test_publish_threshold_default(self):
        """The default publish threshold is D_MAX - 1 (S1 level)."""
        fresh_cog = Cogitator(
            model=Model(), event_bus=EventBus(),
            on_s1=lambda q, c: None,
        )
        fresh_cog.join(timeout=0.1)
        assert fresh_cog.publish_threshold == D_MAX - 1


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
