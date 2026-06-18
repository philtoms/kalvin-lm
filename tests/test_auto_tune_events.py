"""Tests for auto-tune event enrichment (participants.auto_tune.events)."""

from __future__ import annotations

import pytest

from kalvin.expand import D_MAX, boundaries, classify
from kalvin.kline import KLine
from training.participants.auto_tune.events import (
    _build_kline_display,
    _build_significance,
    _to_kline,
    enrich_event,
)

# ── Fixtures / helpers ────────────────────────────────────────────────


def _make_kline(sig: int, nodes: list[int]) -> KLine:
    return KLine(signature=sig, nodes=nodes)


# ── 1. Progress enrichment ────────────────────────────────────────────


class TestProgressEnrichment:
    def test_basic_progress(self):
        frame = {
            "role": "supervisor",
            "action": "progress",
            "message": {
                "lesson_label": "Lesson 1",
                "lessons_total": 5,
                "lessons_completed": 2,
                "status": "lesson_complete",
            },
        }
        result = enrich_event(frame, seq=42)

        assert result["type"] == "progress"
        assert result["seq"] == 42
        assert result["status"] == "lesson_complete"
        assert result["lesson"] == "Lesson 1"
        assert result["lessons_total"] == 5
        assert result["lessons_completed"] == 2

    def test_progress_with_none_label(self):
        frame = {
            "role": "supervisor",
            "action": "progress",
            "message": {
                "lesson_label": None,
                "lessons_total": 3,
                "lessons_completed": 0,
                "status": "started",
            },
        }
        result = enrich_event(frame, seq=1)

        assert result["lesson"] is None
        assert result["type"] == "progress"


# ── 2 & 3. Rationalise enrichment ────────────────────────────────────


class TestRationaliseEnrichment:
    def test_ground_high_significance(self):
        query = _make_kline(0x42, [1, 2])
        proposal = _make_kline(0x42, [1, 2])
        frame = {
            "role": "supervisor",
            "action": "event",
            "message": {
                "kind": "ground",
                "query": query,
                "proposal": proposal,
                "significance": 0xFFFF_FFFF_FFFF_FFFF,
            },
        }
        result = enrich_event(frame, seq=10)

        assert result["type"] == "rationalise"
        assert result["seq"] == 10
        assert result["kind"] == "ground"
        assert result["significance"]["level"] == "S1"
        assert result["significance"]["raw"] == 0xFFFF_FFFF_FFFF_FFFF
        assert "query" in result
        assert "proposal" in result
        assert "raw" in result["query"]
        assert "source" in result["query"]

    def test_frame_lower_significance(self):
        query = _make_kline(0x80, [3, 4])
        proposal = _make_kline(0x90, [5, 6])
        frame = {
            "role": "supervisor",
            "action": "event",
            "message": {
                "kind": "frame",
                "query": query,
                "proposal": proposal,
                "significance": 0x8000_0000_0000_0000,
            },
        }
        result = enrich_event(frame, seq=11)

        assert result["type"] == "rationalise"
        assert result["kind"] == "frame"
        # 0x8000... is below the S2|S3 boundary, so should not be S1
        assert result["significance"]["level"] != "S1"
        assert result["significance"]["raw"] == 0x8000_0000_0000_0000

    def test_rationalise_with_dict_message(self):
        """Test that a RationaliseEvent-like dict works."""
        frame = {
            "role": "supervisor",
            "action": "event",
            "message": {
                "kind": "ground",
                "query": {"signature": 0x42, "nodes": [1, 2]},
                "proposal": {"signature": 0x42, "nodes": [1, 2]},
                "significance": 0xFFFF_FFFF_FFFF_FFFF,
            },
        }
        result = enrich_event(frame, seq=5)

        assert result["type"] == "rationalise"
        assert result["significance"]["level"] == "S1"


# ── 4. Ratify request enrichment ─────────────────────────────────────


class TestRatifyRequestEnrichment:
    def test_basic_ratify(self):
        frame = {
            "role": "supervisor",
            "action": "ratify_request",
            "message": {
                "query": {"signature": 0x42, "nodes": [1, 2]},
                "proposal": {"signature": 0xFF, "nodes": [3, 4]},
                "significance": 0xFFFF_FFFF_FFFF_FFFF,
            },
        }
        result = enrich_event(frame, seq=99)

        assert result["type"] == "ratify_request"
        assert result["seq"] == 99
        assert result["significance"]["level"] == "S1"
        assert result["query"]["raw"]["signature"] == 0x42
        assert result["proposal"]["raw"]["signature"] == 0xFF
        assert "source" in result["query"]
        assert "source" in result["proposal"]

    def test_ratify_with_kline_instances(self):
        query = _make_kline(0x42, [1, 2])
        proposal = _make_kline(0xFF, [3, 4])
        frame = {
            "role": "supervisor",
            "action": "ratify_request",
            "message": {
                "query": query,
                "proposal": proposal,
                "significance": 0x8000_0000_0000_0000,
            },
        }
        result = enrich_event(frame, seq=100)

        assert result["type"] == "ratify_request"
        assert result["query"]["raw"]["nodes"] == [1, 2]


# ── 5. Escalation enrichment ─────────────────────────────────────────


class TestEscalationEnrichment:
    def test_basic_escalation(self):
        frame = {
            "role": "supervisor",
            "action": "notify",
            "message": {
                "reason": "budget_exhaustion",
                "detail": "Token budget exceeded",
                "lesson_position": 3,
            },
        }
        result = enrich_event(frame, seq=7)

        assert result["type"] == "escalation"
        assert result["seq"] == 7
        assert result["reason"] == "budget_exhaustion"
        assert result["detail"] == "Token budget exceeded"
        assert result["lesson_position"] == 3

    def test_escalation_low_confidence(self):
        frame = {
            "role": "supervisor",
            "action": "notify",
            "message": {
                "reason": "low_confidence",
                "detail": "No proposal above threshold",
                "lesson_position": 1,
            },
        }
        result = enrich_event(frame, seq=8)

        assert result["type"] == "escalation"
        assert result["reason"] == "low_confidence"


# ── 6. Significance normalisation ────────────────────────────────────


class TestSignificanceObject:
    def test_max_significance(self):
        sig = _build_significance(D_MAX)
        assert sig["raw"] == D_MAX
        assert sig["normalised"] == 1.0
        assert sig["level"] == "S1"

    def test_zero_significance(self):
        sig = _build_significance(0)
        assert sig["raw"] == 0
        assert sig["normalised"] == 0.0
        # s34 boundary is 0, so sig=0 satisfies sig >= s34 → S3
        # S4 is only for negative values, which can't happen with uint64
        assert sig["level"] == "S3"

    def test_midrange_normalisation(self):
        # raw = D_MAX // 2 has distance = D_MAX // 2, which is deep S3.
        # Band-anchored normalization is asymptotic: deep S3 is
        # a small but non-zero value in (0.0, 0.50), never clamped to 0.
        raw = D_MAX // 2
        sig = _build_significance(raw)
        assert 0.0 < sig["normalised"] < 0.50  # S3 → asymptotic, non-zero

    def test_level_classification_matches_classify(self):
        s12, s23, s34 = boundaries()
        raw = 0x8000_0000_0000_0000
        sig = _build_significance(raw)
        expected_level = classify(raw, s12, s23, s34)
        assert sig["level"] == expected_level

    def test_s23_boundary(self):
        """At the S2|S3 boundary, level should be S2, normalised should be 0.50."""
        _, s23, _ = boundaries()
        sig = _build_significance(s23)
        assert sig["level"] == "S2"
        # s23 = ~S2_S3_DISTANCE, so distance = S2_S3_DISTANCE (100) → S2 floor → 0.50
        assert sig["normalised"] == pytest.approx(0.50, abs=1e-6)

    def test_s2_less_than_s1(self):
        """S2 normalised significance must be strictly less than S1."""
        from kalvin.expand import MASK64

        s1_sig = D_MAX  # S1, distance=0
        s2_sig = (~2) & MASK64  # S2, distance=2
        s1 = _build_significance(s1_sig)
        s2 = _build_significance(s2_sig)
        assert s1["level"] == "S1"
        assert s2["level"] == "S2"
        assert s2["normalised"] < s1["normalised"]
        assert s2["normalised"] < 1.0

    def test_s1_range_near_one(self):
        """S1 (distance 0) normalises to exactly 1.0; distance 1 is now the top of S2."""
        sig = _build_significance(D_MAX)  # distance=0
        assert sig["normalised"] == 1.0
        assert sig["level"] == "S1"
        # Distance 1 (D_MAX - 1) is the top of S2 (≈ 0.9950), no longer S1.
        sig = _build_significance(D_MAX - 1)  # distance=1
        assert sig["normalised"] == pytest.approx(0.995, abs=1e-4)
        assert sig["level"] == "S2"


# ── 7. KLine Display Object ──────────────────────────────────────────


class TestKLineDisplayObject:
    def test_raw_fields(self):
        kline = _make_kline(0x42, [1, 2])
        display = _build_kline_display(kline)

        assert display["raw"]["signature"] == 0x42
        assert display["raw"]["nodes"] == [1, 2]

    def test_source_is_string(self):
        kline = _make_kline(0x42, [1, 2])
        display = _build_kline_display(kline)

        assert isinstance(display["source"], str)
        assert len(display["source"]) > 0


# ── 8. Dict-to-KLine conversion ──────────────────────────────────────


class TestDictToKLine:
    def test_plain_dict_input(self):
        frame = {
            "role": "supervisor",
            "action": "event",
            "message": {
                "kind": "ground",
                "query": {"signature": 0x42, "nodes": [1, 2]},
                "proposal": {"signature": 0x42, "nodes": [1, 2]},
                "significance": D_MAX,
            },
        }
        result = enrich_event(frame, seq=3)

        assert result["type"] == "rationalise"
        assert result["query"]["raw"]["signature"] == 0x42
        assert result["proposal"]["raw"]["nodes"] == [1, 2]

    def test_to_kline_with_kline_instance(self):
        kline = _make_kline(0x42, [1, 2])
        assert _to_kline(kline) is kline

    def test_to_kline_with_dict(self):
        result = _to_kline({"signature": 0x42, "nodes": [1, 2]})
        assert isinstance(result, KLine)
        assert result.signature == 0x42
        assert result.nodes == [1, 2]

    def test_to_kline_raises_on_bad_type(self):
        with pytest.raises(TypeError, match="Cannot convert"):
            _to_kline("not a kline")


# ── 9. Seq counter ───────────────────────────────────────────────────


class TestSeqCounter:
    def test_seq_in_progress(self):
        frame = {
            "role": "supervisor",
            "action": "progress",
            "message": {
                "lesson_label": None,
                "lessons_total": 1,
                "lessons_completed": 0,
                "status": "started",
            },
        }
        for seq_val in [0, 1, 42, 1000]:
            result = enrich_event(frame, seq=seq_val)
            assert result["seq"] == seq_val

    def test_seq_in_escalation(self):
        frame = {
            "role": "supervisor",
            "action": "notify",
            "message": {
                "reason": "test",
                "detail": "",
                "lesson_position": 0,
            },
        }
        result = enrich_event(frame, seq=999)
        assert result["seq"] == 999


# ── Unknown action ────────────────────────────────────────────────────


class TestUnknownAction:
    def test_unknown_action_raises(self):
        frame = {
            "role": "supervisor",
            "action": "unknown_action",
            "message": {},
        }
        with pytest.raises(ValueError, match="Unknown action"):
            enrich_event(frame, seq=1)
