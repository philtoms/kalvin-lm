# sig8 — prototype of the 8-bit compositional significance redesign

This is a **throwaway prototype** to validate the design decisions locked in
the significance grill (see the handoff doc), before they get committed to
`specs/model.md` and implemented in `src/kalvin/expand.py`.

It is **not** wired into production. It re-implements the aggregation model
in isolation so the math and the data flow can be stress-tested.

## What it exercises

1. **8-bit inverted distance** in the low byte of an int, `& 0xFF` masking.
2. **Global linear inverted distance** in `(0x00, 0xFF)` — no reshape at the
   S2|S3 boundary (Q3).
3. **`S2_S3_BOUNDARY`** = lowest S2 byte (inclusive); only it is configurable
   (Q4, Q5).
4. **Band sentinels** `SIG_S1=0xFF`, `SIG_S2=0xFE`, `SIG_S3=BOUNDARY-1`,
   `SIG_S4=0x00` (Q5).
5. **One quantity, two uses** (Q7): routing classifies; cogitation computes.
6. **Saturation guards on the two limits only** (Q9): distance accumulation
   can never yield `0xFF` (only distance 0 does) nor `0x00` (only a structural
   unresolvable does).
7. **Accounted fraction** as the primary signal (Q10): `Σ aᵢ / total_node_slots`.
8. **Per-node accountedness** (Q11/12): matched/grounded → 1.0; resolvable in
   `h` hops → `decay(h)`; unresolvable → 0.0.
9. **`h` = the actual reentrant hop count** from expand (Q13-revised).
10. **Two pluggable seams** (the Q16 decision this prototype validates):
    `DecayFunction` (leaves) and `ComposeFunction` (aggregation), over a
    **hybrid compose-on-return** evaluation: topology captured on descent,
    decay + compose applied on the return phase.

## The shape under test

The production `expand()` already recurses with `yield from expand(...)` on
the way down and yields the terminal `QueryCandidate` on the way out. The
prototype mirrors this: per-node hop counts are captured during descent, and
the compose-on-return replaces the sum-and-invert pattern at the terminal.

Run the demo:

```bash
python -m prototype.sig8.demo
```

Run the unit checks:

```bash
python -m pytest prototype/sig8/test_sig8.py -q
```
