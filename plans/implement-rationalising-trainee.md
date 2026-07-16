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
signature) and a work-list.

The turn produces two channels: the **batch** (dialogue emissions — speech
acts for T) and the **observations** (every S1 grounding K performs, for
white-box verification). Grounding does not emit into the dialogue.

The mechanism has two cogitation paths:

- **S3 countersignature** — a single-node relationship whose operands both
  have seen canons: pair the operands left-to-right at group size 1, emit every
  unresolved pairing in one batch, then on resolution ground both directions
  of the reciprocal pair at S1 (observed, not emitted).
- **S2 misfit origination** — a multi-node misfit: shape one proposal by
  recombining grounded klines (node-expansion + node-graft), sourcing every
  substituted node from grounded klines (no invention), and emit it at S2.

Entry rule: S1 grounds/cleans, S4 pops the identity ask; only S2/S3 reach
cogitation. Cogitation is LIFO; a misfit entry persists, and any entry that
matches neither dispatch predicate is skipped and re-scanned on every
subsequent turn (no bound) until `_promote`'s cascade grounds it or an
operand canon arrives and flips it countersignable.

- **S1 canon-countersignature** — a new routing branch, ahead of `_promote`:
an S1 relationship `{S: nodes}` with more than one node, all grounded. K
  does not ground the misfit; it computes `C = make_signature(nodes)`,
  grounds `{C: nodes}` (promoting C), grounds `{S:[C]}` and `{C:[S]}`, and
  drops the work-list entries under S. Each ground is observed; nothing is
  emitted into the dialogue. 1:1 shapes (connotations, denotations),
  identities, and self-canons fall through to `_promote`. K does not look up
  C; `make_signature` computes it and it is admitted whether seen or not.

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
