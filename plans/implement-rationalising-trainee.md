# Implement the Rationalising Trainee — Plan

> **Status: implemented.** Kept as a build record, not an active spec. The
> authoritative current contract is `@specs/dialogue-cogitation.md` (itself a
> working sketch). This is the most speculative part of the sub-project and is
> expected to be reshaped by discovery.

## What was built

`src/training/dialogue/rationalise.py` — the `Rationaliser` engine: a
**stateless** object that derives each turn from `(state, incoming)` and
returns a `(batch, observations)` pair. The `RationalisingTrainee` actor
(`actors.py`) owns the `RationaliserState`, wraps each batch value in a
`RationaliseEvent`, and exposes the observations via `drain_observations`.
The state holds a minimal model of what K has grounded (`grounded`, keyed by
signature) and a work-list. The engine is **stateless about its own
emissions**: it may re-derive a proposal on successive turns, and the actor
is the single deduplication point — it records every proposal it has
published (by `(signature, nodes)`) and drops re-derivations, so K never
repeats itself. (An earlier build kept an `asked` set in the engine to
dedup identity asks; that responsibility moved to the actor so the engine's
state is a pure model of grounding.)

The turn produces two channels: the **batch** (dialogue emissions — speech
acts for T) and the **observations** (every S1 grounding K performs, for
white-box verification). Grounding does not emit into the dialogue.

The mechanism has two cogitation dispatch paths:

- **S3 countersignature** — a single-node relationship whose operands both
  have seen canons: pair the operands left-to-right at group size 1, emit every
  unresolved pairing in one batch. Once every pairing is resolved the entry
  grounds through the normal groundable path; grounding a countersignable
  kline grounds its reciprocal too (both directions of the reciprocal pair
  end up grounded at S1, observed not emitted).

  **Canonical-level reciprocal** — the S3 path's natural completion: when
  cogitation finds every operand pairing of the relationship entry resolved,
  it grounds the **entry itself** at S1 (e.g. `MHALL:[SVO]`); the existing
  `_is_countersignable` reciprocal in `_ground` then mirrors `SVO:[MHALL]`.
  Fired from `cogitate`: the entry drives its own completion, so it fires on
  any cogitation pass after the last pairing resolves — independent of which
  T query triggered the turn. Reuses `_countersignature_proposals` (the
  empty-pairings result is the completion signal) and `_ground`'s reciprocal
  mirroring; no new pairing logic.
- **S2 misfit canonisation** — shape one multi-node proposal by recombining
  grounded klines (node-expansion + node-graft), sourcing every substituted
  node from grounded klines (no invention), and emit it at S2.

The turn is two stages inside `_Turn`: `route` then `cogitate`.

- **Routing** (`_Turn.route`) only pops an **S4** query's pending identity
  ask; **every other query (S1, S2, S3)** is appended to the work-list. An
  **S2** misfit additionally appends its unrecognised nodes and signature as
  identity placeholders. Routing does no grounding and emits nothing.
- **Cogitation** (`_Turn.cogitate`) is a single LIFO pass over the
  work-list. Each entry is resolved in priority order: a **promotable**
  entry (a single-node relationship whose reciprocal is grounded) or a
  **groundable** entry (an identity whose signature is grounded, or a canon
  whose nodes are all grounded) is grounded at S1 (observed, not emitted)
  and dropped; a countersignable entry whose pairings are unresolved takes
  the S3 path (emitting the unresolved pairings); a multi-node misfit takes
  the S2 path; an identity that remains ungroundable is emitted as an S4
  ask. Grounding — whether reached from routing's promotion cascade or from
  cogitation — flows through `_ground`, which grounds the reciprocal of any
  countersignable kline as part of its own bookkeeping, so the two paths
  cannot disagree on the reciprocal. A misfit entry persists, and any entry
  that matches neither dispatch predicate is skipped and re-scanned on
  every subsequent turn until grounding flips it promotable/countersignable.

The runner verifies K's observations **white-box** (model B subset check)
against the script's `events` section (decoded via `decode_events`): every
asserted grounding must be observed at least once; extra observations are not
policed. Grounding assertions apply only to an observable trainee (one
exposing `drain_observations`); the table actors are unaffected.

## Spec References

- `@specs/dialogue-cogitation.md` — the two paths and their boundaries.
- `@specs/dialogue-driven-training.md` — the actor contract this satisfies.

## Terminology note

The plan formerly carried temporary scaffolding terms (grounded vs ratified;
relationships; submit/emit/respond). These are not glossary entries and were
kept only to separate trainer-side from trainee-side rationalisation during
the build. See `@CONTEXT.md` for the real terms.
