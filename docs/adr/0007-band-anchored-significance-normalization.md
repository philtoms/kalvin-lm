# Band-anchored significance normalization (S3 is asymptotic, not clamped)

**Status:** accepted. Supersedes design decision **DD-1** in
`plans/impl/expand-robustness.md` ("Normalize against `S2_S3_DISTANCE`, not
`D_MAX`"). DD-1's normalization formula `max(0.0, 1.0 - distance/100)`
collapses every S3 value to `0.0`; this ADR replaces it with a band-anchored
scheme that gives each of S1/S2/S3 a guaranteed, ordered, visible sub-range and
makes S3 asymptotic so its unbounded distance range never collapses.

## The problem

Significance is inverted distance: `significance = (~distance) & MASK64`,
smaller distance = more significant. Bands are fixed distance thresholds:

| Band | Distance range | Meaning |
|------|----------------|---------|
| S1 | 0–1 | fully grounded |
| S2 | 2–100 (`S2_S3_DISTANCE`) | partial |
| S3 | 101 → `D_MAX-1` | connotation only |
| S4 | `D_MAX` (raw sig = 0) | completely novel |

DD-1 normalized as `max(0.0, 1.0 - distance/100)`. That allocates the **entire**
`[0.0, 1.0]` float range to S1+S2 and clamps **every** S3 value to `0.0`.

### Empirical evidence (auto-tune `harness.log` rational events)

| Bucket | Count | Notes |
|--------|-------|-------|
| S1 (fast path) | 128 | distance 0 |
| S4 (no significance) | 120 | raw sig 0 |
| **S3** | **~57** | **101, 102, 141, 146, 170, 181, 200 (×28), 211, 266, 301, 370, 428, 519** |
| S2 (boundary only) | 19 | all at distance 100 (one unresolved node) |

Two findings follow directly:

1. **S3 dominates the rationals (~75%)** and spans a **5× distance range**
   (101 → 519). DD-1 erases all of that ordering into a single `0.0`.
2. **The S2 sub-range is barely used.** Real S2 events cluster at the 100
   boundary; distances 2–99 almost never occur. DD-1 nonetheless reserved the
   whole `[0.0, 0.98]` span for S2 — precision spent where there is no data.

So DD-1 is doubly wrong: it starves the band that dominates (S3) *and* wastes
precision on a band (S2) that rarely populates its low distances.

## The decision

Distance computation is **unchanged** — the `s3-distance` auto-tune settled
that (linear S3 distance, `_S3_BIAS = 1`). The fix belongs entirely in
**normalization**. Normalization becomes **band-anchored**: each band owns a
fixed sub-range of `[0.0, 1.0]`, and S3 uses an asymptotic curve so its
unbounded distance range maps injectively into an open interval without any
clamp.

```
S1  → 1.0                                                    # distance 0–1
S2  → [0.50, 0.99]   linear in distance [2, 100]              # closer S2 → higher
S3  → (0.0, 0.50)    s3 = 0.50 · k / (k + (distance - 100))   # asymptotic, k = 50
S4  → 0.0                                                    # raw sig 0
```

### S3 curve (k = 50)

| distance | band | normalised |
|----------|------|------------|
| 0 | S1 | 1.000 |
| 2 | S2 | 0.990 |
| 50 | S2 | 0.750 |
| 100 | S2 (floor) | 0.500 |
| 101 | S3 | 0.490 |
| 151 | S3 | 0.248 |
| 200 | S3 (bulk of observed data) | 0.167 |
| 301 | S3 | 0.100 |
| 519 | S3 (deepest observed) | 0.053 |
| → ∞ | S3 | → 0.0 (never reached) |

Every distinct S3 distance yields a distinct, ordered, non-zero normalised
value. `k` is the single tunable (smaller `k` compresses deep S3 faster); 50 is
the default.

### Properties guaranteed

- S1 = 1.0 exactly.
- S2 < S1; S2 strictly monotonic in distance.
- S3 < S2; S3 strictly monotonic in distance.
- S3 is **injective** in distance — no two S3 distances collapse to one value.
- S3 never clamps to 0.0 (only S4 does); stable as the topology grows.

## Discovered inconsistencies (fixed alongside this ADR)

1. **Spec ↔ code drift on distance packing.** `specs/model.md` and
   `plans/impl/model.md` still document `_S3_BIAS = 9` and **quadratic**
   packing `_pack(d) = d²`. The code is `_S3_BIAS = 1`, **linear**, with no
   `_pack` — the `s3-distance` session replaced it but never updated the docs.
   Both are corrected to match the code.
2. **Normalization-site inconsistency.** DD-1 claimed both consumers used the
   same formula. They did not: `events.py` used `1.0 - d/100`; `trainer.py` had
   been switched by the `s3-distance` session to **raw integer distance**
   logging (`→ 200`). This ADR re-converges both sites on one source of truth
   (see consequences).

## Consequences

1. **Single normalization source.** A `normalise_significance(raw_sig) -> float`
   helper is added to the significance module (`src/kalvin/expand.py`,
   re-exported) and used by **both** consumers:
   - `src/participants/auto_tune/events.py` — `normalised` field.
   - `src/trainer/trainer.py` — `sig_norm` in the frame event log line.
2. **Trainer log keeps raw distance too.** The one-line log shows the band-
   anchored normalised value **and** the raw distance (e.g.
   `→ 0.17 (d=200)`), preserving the debug-readability the `s3-distance`
   session introduced while regaining cross-event comparability.
3. **Spec rewrite.** `specs/significance-normalization.md` is rewritten:
   band-anchored formula, revised normalised-value-ranges table, and a revised
   test matrix. **SN-4 changes** from "S3 clamped to 0.0" to "S3 asymptotic in
   (0.0, 0.50)"; a new **SN-6** asserts S3 injectivity (no two distances
   collapse).
4. **DD-1 superseded.** `plans/impl/expand-robustness.md` marks DD-1
   superseded by this ADR and records the normalization task.
5. **Distance semantics unchanged.** `classify()`, the boundaries
   (`S1|S2 = D_MAX-1`, `S2|S3 = ~100`, `S3|S4 = 0`), and the raw significance
   computation are untouched. Normalization is display/analysis only; routing
   and classification continue to use raw significance. This ADR therefore does
   **not** affect agent routing or auto-countersign behaviour.
