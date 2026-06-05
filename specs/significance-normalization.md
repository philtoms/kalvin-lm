# Significance Normalization Specification

## Overview

Normalized significance is a float64 value in [0.0, 1.0] representing how
strongly a kline relates to existing knowledge. The normalization must be
**meaningful**: S1 (fully grounded) must produce values ≥ 0.99, S2 (partially
understood) must produce values strictly less than S1 and less than 1.0.

## Dependencies

### Expand (@model spec, @agent spec)

- Significance is `(~distance) & MASK64` — an inverted distance.
- S1|S2 boundary: `D_MAX - 1` (distance ≤ 1).
- S2|S3 boundary: `~S2_S3_DISTANCE` (distance = S2_S3_DISTANCE = 100).
- S3|S4 boundary: 0.

### Agent (@agent spec)

- The trainer log displays normalized significance.
- Auto-tune event enrichment includes normalized significance.

## Definition

### Normalization Formula

```
normalised(raw_sig):
  if raw_sig == 0: return 0.0
  distance = (~raw_sig) & MASK64
  return max(0.0, 1.0 - distance / S2_S3_DISTANCE)
```

### Normalized Value Ranges

| Band | Distance range | Normalized range | Meaning |
|------|---------------|------------------|---------|
| S1 | 0–1 | 1.00–0.99 | Fully grounded |
| S2 | 2–100 | 0.98–0.00 | Partially understood |
| S3 | >100 | 0.00 (clamped) | Connotation only |
| S4 | 0 (raw=0) | 0.00 | Completely novel |

### Behavioural Rules

**SN-1: S2 < S1.** For any S2 significance value, the normalized value is
strictly less than any S1 normalized value.

**SN-2: S2 < 1.0.** No S2 significance value normalizes to 1.0.

**SN-3: S1 ≥ 0.99.** S1 significance values (distance 0 or 1) normalize to
values ≥ 0.99.

**SN-4: S3/S4 clamped to 0.0.** Significance values at or below the S2|S3
boundary normalize to 0.0.

**SN-5: Zero significance → 0.0.** A raw significance of 0 (S4/unresolvable)
normalizes to exactly 0.0.

**SN-6: Monotonic within bands.** Within S1 and S2, higher raw significance
produces higher normalized values (preserving the ordering).

### Rationale

The previous formula `raw / D_MAX` produces identical float64 values (1.0)
for S1 and S2 because distances 0–100 are vanishingly small relative to
D_MAX (2⁶⁴). Normalizing the inverted distance against S2_S3_DISTANCE
produces meaningful float64 values that distinguish bands visually in logs
and event streams.

## Consumers

Two sites produce normalized significance:

1. **Trainer log** (`src/trainer/trainer.py`) — `sig_norm` in the frame event
   log line.
2. **Auto-tune event enrichment** (`src/participants/auto_tune/events.py`) —
   `normalised` field in the significance object.

Both must use the same formula.

## Out of Scope

- Changes to raw significance computation (distance, boundaries).
- Changes to the band classification (`classify()`).
- Display format changes (decimal places, log format).

## Test Matrix

| ID | Criterion | Category |
|----|-----------|----------|
| SN-1 | S2 normalized < S1 normalized | ordering |
| SN-2 | S2 normalized < 1.0 | ordering |
| SN-3 | S1 normalized ≥ 0.99 | S1 range |
| SN-4 | S3 normalized == 0.0 | clamping |
| SN-5 | raw 0 → 0.0 | zero |
| SN-6 | Higher S2 raw → higher normalized | monotonic |
