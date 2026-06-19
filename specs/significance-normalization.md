# Significance Normalization Specification

## Overview

Normalized significance is a float64 value in `[0.0, 1.0]` representing how
strongly a kline relates to existing knowledge. Normalization is **band-
anchored**: each significance band (S1, S2, S3, S4) owns a fixed sub-range of
`[0.0, 1.0]`, and S3 uses an asymptotic curve so its unbounded distance range
maps to a non-empty, ordered interval without ever being clamped. The raw
significance value and the band classification are unchanged by normalization;
normalization is a display/analysis projection of raw significance.

## Dependencies

### Expand (@model spec — §Significance Semantics)

- Significance is `(~distance) & MASK64` — an inverted distance.
- S1|S2 boundary: distance = 0.
- S2|S3 boundary: distance = `S2_S3_DISTANCE` (= 100).
- S3|S4 boundary: raw significance = 0 (distance = `D_MAX`).
- The per-node unresolved penalty (`UNRESOLVED_PENALTY`) is decoupled
  from `S2_S3_DISTANCE`; it affects how many unresolved nodes a pair can absorb
  before crossing into S3, but it does **not** alter any band boundary.

### Agent (@agent spec)

- The trainer log displays normalized significance.
- Auto-tune event enrichment includes normalized significance.

## Definition

### Normalization Formula

Normalization is a single function of the raw significance value. The function
is implemented once and shared by all consumers (see §Consumers).

```
normalise(raw_sig):
    if raw_sig == 0:                  return 0.0          # S4
    distance = (~raw_sig) & MASK64
    if distance == 0:                return 1.0          # S1 (distance 1 now falls into S2)
    if distance <= S2_S3_DISTANCE:                          # S2 (linear)
        return S2_FLOOR + (S2_TOP - S2_FLOOR)
                   * (S2_S3_DISTANCE - distance) / (S2_S3_DISTANCE - 2)
    # S3: asymptotic in (0.0, S2_FLOOR), never clamped
    delta = distance - S2_S3_DISTANCE
    return S2_FLOOR * S3_K / (S3_K + delta)
```

### Constants

| Constant         | Value | Meaning                                                                      |
| ---------------- | ----- | ---------------------------------------------------------------------------- | ------------------------------------------------- |
| `S2_S3_DISTANCE` | 100   | S2                                                                           | S3 distance boundary (from @model spec)           |
| `S2_TOP`         | 0.99  | normalised anchor at distance 2 (the closest S2 is now distance 1, ≈ 0.9950) |
| `S2_FLOOR`       | 0.50  | normalised value at the S2                                                   | S3 boundary (distance 100); also the S3 asymptote |
| `S3_K`           | 50    | S3 decay rate (single tunable; smaller compresses deep S3 faster)            |

### Normalized Value Ranges

| Band | Distance range  | Normalized range  | Curve                                                                          |
| ---- | --------------- | ----------------- | ------------------------------------------------------------------------------ |
| S1   | 0               | `1.0` (exact)     | constant                                                                       |
| S2   | 1–100           | `[0.50, ≈0.9950]` | linear (closer → higher; `0.99` is the distance-2 anchor, distance 1 ≈ 0.9950) |
| S3   | 101 → `D_MAX-1` | `(0.0, 0.50)`     | asymptotic, strictly decreasing, never 0                                       |
| S4   | `D_MAX` (raw 0) | `0.0`             | constant                                                                       |

### Worked S3 values (S3_K = 50)

| distance | normalized            |
| -------- | --------------------- |
| 101      | 0.490                 |
| 151      | 0.248                 |
| 200      | 0.167                 |
| 301      | 0.100                 |
| 519      | 0.053                 |
| → ∞      | → 0.0 (never reached) |

## Behavioural Rules

**SN-1: Strict band ordering.** For any two raw values in different bands, the
higher band normalizes strictly higher: S1 > S2 > S3 > S4.

**SN-2: S1 normalizes exactly to 1.0.** Any S1 raw value (distance 0 only)
normalizes to exactly `1.0`.

**SN-3: S2 range and monotonicity.** Every S2 raw value normalizes into
`[0.50, ≈0.9950]` (the closest S2 is distance 1, ≈ 0.9950; distance 2 is the
`0.99` anchor), and smaller S2 distance yields a strictly higher normalized
value.

**SN-4: S3 is asymptotic, not clamped.** Every S3 raw value normalizes into
`(0.0, 0.50)`. No S3 value normalizes to `0.0` (only S4 does). Normalized S3
strictly decreases as distance grows.

**SN-5: Zero significance → 0.0.** A raw significance of 0 (S4 / unresolvable)
normalizes to exactly `0.0`.

**SN-6: S3 injectivity (granularity).** No two distinct S3 distances produce
the same normalized value. Every distinct S3 raw value yields a distinct,
ordered normalized value.

**SN-7: Global monotonicity.** Across the whole raw range, higher raw
significance produces a higher-or-equal normalized value (equal only within S1
and within S4, which are constant).

## Consumers

Two sites produce normalized significance and must use the single shared
`normalise` function:

1. **Trainer log** (`src/training/trainer/trainer.py`) — `sig_norm` in the frame event
   log line. The log line additionally shows the raw distance for debug
   readability (e.g. `→ 0.17 (d=200)`).
2. **Auto-tune event enrichment** (`src/training/participants/auto_tune/events.py`) —
   `normalised` field in the significance object.

The shared implementation lives in the significance module
(`src/kalvin/expand.py`) and is imported by both consumers.

## Out of Scope

- Changes to raw significance computation (distance accumulation, boundaries).
- Changes to the band classification (`classify()`) or routing.
- Changes to the distance packing (linear S3 distance, `_S3_BIAS = 1`) — already
  settled by the `s3-distance` auto-tune.
- Display format beyond the normalized value and raw-distance annotation.

## Test Matrix

| ID   | Criterion                                                                       | Category             |
| ---- | ------------------------------------------------------------------------------- | -------------------- |
| SN-1 | S1 norm > S2 norm > S3 norm > S4 norm (cross-band ordering)                     | ordering             |
| SN-2 | S1 norm == 1.0 (distance 0 only)                                                | S1                   |
| SN-3 | S2 norm ∈ [0.50, ≈0.9950] (closest S2 is distance 1); smaller distance → higher | S2 range + monotonic |
| SN-4 | S3 norm ∈ (0.0, 0.50); never 0.0; strictly decreasing                           | S3 asymptotic        |
| SN-5 | raw 0 → 0.0                                                                     | zero                 |
| SN-6 | distinct S3 distances → distinct normalized values (no collapse)                | S3 granularity       |
| SN-7 | global monotonic (higher raw → higher-or-equal norm)                            | monotonic            |
