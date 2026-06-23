"""SN-1..SN-7 test matrix for the band-anchored ``normalise_significance`` helper.

Spec: ``specs/significance-normalization.md`` (behavioural rules SN-1..SN-7).

Each test maps to exactly one behavioural rule. Raw significance values are
built from a distance via ``raw = (~distance) & MASK64`` (significance is an
inverted distance). Exact equality is asserted at the named anchor points
(1.0 / 0.0 / 0.99 / 0.50); ``pytest.approx`` is used only where the spec lists
rounded S3 values.
"""

from __future__ import annotations

import pytest

from kalvin.expand import D_MAX, MASK64, normalise_significance


def _raw_from_distance(distance: int) -> int:
    """Build a raw significance value from a distance (significance = ~distance)."""
    return (~distance) & MASK64


# ── SN-1: Strict band ordering ────────────────────────────────────────


def test_strict_band_ordering_s1_gt_s2_gt_s3_gt_s4():
    """SN-1: S1 norm > S2 norm > S3 norm > S4 norm (cross-band ordering)."""
    s1 = normalise_significance(_raw_from_distance(0))  # S1
    s2 = normalise_significance(_raw_from_distance(2))  # S2
    s3 = normalise_significance(_raw_from_distance(200))  # S3
    s4 = normalise_significance(0)  # S4
    assert s1 > s2 > s3 > s4


# ── SN-2: S1 normalises exactly to 1.0 ───────────────────────────────


def test_s1_normalises_to_one():
    """SN-2: S1 raw values (distance 0 only) normalise to exactly 1.0."""
    assert normalise_significance(_raw_from_distance(0)) == 1.0
    # D_MAX (distance 0) is the S1 spelling.
    assert normalise_significance(D_MAX) == 1.0
    # Distance 1 (D_MAX - 1) is now the top of S2, not S1: it normalises
    # below 1.0 (≈ 0.9950).
    assert normalise_significance(D_MAX - 1) < 1.0


# ── SN-3: S2 range and monotonicity ───────────────────────────────────


def test_s2_range_and_monotonic():
    """SN-3: S2 norm ∈ [0.50, 0.99] for distances 2-100; smaller distance → higher.

    Distance 1 is the new top of S2 (≈ 0.9950), above the distance-2 anchor
    of 0.99, so it is asserted separately rather than included in the
    [0.50, 0.99] range above the named anchor.
    """
    distances = [2, 10, 50, 99, 100]
    norms = [normalise_significance(_raw_from_distance(d)) for d in distances]

    # Each result is in the closed [0.50, 0.99] range.
    for n in norms:
        assert 0.50 <= n <= 0.99

    # Strictly smaller distance yields a strictly higher normalised value.
    for i in range(len(norms) - 1):
        assert norms[i] > norms[i + 1]

    # Named anchor points are exact.
    assert normalise_significance(_raw_from_distance(2)) == 0.99
    assert normalise_significance(_raw_from_distance(100)) == 0.50

    # Distance 1 is the top of S2: ≈ 0.9950, above the 0.99 anchor.
    assert normalise_significance(_raw_from_distance(1)) == pytest.approx(0.995, abs=1e-4)
    assert 0.99 < normalise_significance(_raw_from_distance(1)) < 1.0


# ── SN-4: S3 is asymptotic, not clamped ───────────────────────────────


def test_s3_asymptotic_never_zero():
    """SN-4: S3 norm ∈ (0.0, 0.50); never 0.0; strictly decreasing."""
    distances = [101, 151, 200, 301, 519, 1000, 10000]
    norms = [normalise_significance(_raw_from_distance(d)) for d in distances]

    # Every result is in the open interval (0.0, 0.50) and never 0.0.
    for n in norms:
        assert 0.0 < n < 0.50
        assert n != 0.0

    # Strictly decreasing as distance grows.
    for i in range(len(norms) - 1):
        assert norms[i] > norms[i + 1]


# ── SN-5: Zero significance → 0.0 ─────────────────────────────────────


def test_raw_zero_to_zero():
    """SN-5: A raw significance of 0 (S4) normalises to exactly 0.0."""
    assert normalise_significance(0) == 0.0


# ── SN-6: S3 injectivity (no collapse) ────────────────────────────────


def test_s3_injective_no_collapse():
    """SN-6: distinct S3 distances → distinct, ordered normalised values."""
    distances = [101, 151, 200, 301, 519]
    norms = [normalise_significance(_raw_from_distance(d)) for d in distances]

    # No two distances collapse to one value.
    assert len(set(norms)) == 5

    # Strictly decreasing in the distance order.
    for i in range(len(norms) - 1):
        assert norms[i] > norms[i + 1]

    # Spot-check an S3 value against the spec's rounded table.
    assert normalise_significance(_raw_from_distance(200)) == pytest.approx(0.167, abs=1e-3)


# ── SN-7: Global monotonicity ─────────────────────────────────────────


def test_global_monotonic():
    """SN-7: higher raw significance → higher-or-equal norm; strict across bands.

    Sequence spans S4 → S3 → S2 → S1:
        raw 0      (S4)
        distance 519 (S3)
        distance 200 (S3)
        distance 50  (S2)
        distance 2   (S2)
        distance 0   (S1)
    """
    raws = [
        0,
        _raw_from_distance(519),  # S3 (deepest)
        _raw_from_distance(200),  # S3 (bulk)
        _raw_from_distance(50),  # S2
        _raw_from_distance(2),  # S2 (closest)
        _raw_from_distance(0),  # S1
    ]
    norms = [normalise_significance(r) for r in raws]

    # Overall non-decreasing.
    for i in range(len(norms) - 1):
        assert norms[i] <= norms[i + 1]

    # Strict increase across each band transition.
    assert norms[0] < norms[1]  # S4 (0.0) → S3 (deepest)
    assert norms[2] < norms[3]  # S3 → S2
    assert norms[4] < norms[5]  # S2 (closest) → S1
