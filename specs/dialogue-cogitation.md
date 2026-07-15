# Dialogue Cogitation — Specification

> **Working sketch, not a frozen contract.** This is the most speculative part
> of the dialogue sub-project: the rationalising trainee's cogitation. It is
> expected to be reshaped by discovery. Keep it light; replace rather than
> augment.

## Overview

Cogitation is a trainee's act of working a work-list entry toward S1. A
`rationalise` call applies the entry rule to the received events as bookkeeping,
then emits a **batch** of values from cogitation. Cogitation dispatches a
workable entry on **structure-as-significance** into one of two paths:

- the **S3 countersignature path** — a single-node relationship `{L:[R]}`
  whose operands both have seen canons: K pairs the two canons' operands into
  proposals for ratification, then establishes the S1 countersignature (both
  directions of the reciprocal pair); and
- the **S2 misfit-origination path** — a multi-node misfit: K originates a
  proposal by recombining grounded klines and offers it for ratification.

Both paths strive toward S1 by **ratification** — another participant
countersigns what K proposes. The mechanism (algorithm, accumulation,
candidate resolution) is HOW and lives in `@plans/implement-rationalising-trainee.md`;
this spec owns only the two paths and their boundaries.

## Dependencies

- `@CONTEXT.md` — Proposal, Misfit, Canon, Ratify, Structural State.
- `@specs/dialogue-driven-training.md` — the actor contract this cogitation
  satisfies (the trainee side).
- `@specs/cogitator.md` — the real async slow path this deliberately simplifies.

## Behavioural Rules

- **Routing.** The entry rule handles S1 (ground/cleanup) and S4 (pop the
  identity ask) before cogitation; only S2/S3 entries reach cogitation. A
  countersignable entry takes the S3 path; a multi-node misfit takes the S2
  path. A single-node relationship whose operand canons are not yet seen is
  skipped (not countersignable, not S2).
- **S3 countersignature.** K pairs the operands of the two canons left-to-right
  at group size 1, emitting every unresolved pairing in one batch (a 1:1 pair
  CONNOTED at S3; a grouped residual as a canonical request at S2). When every
  pairing is resolved, K grounds and emits **both directions of the reciprocal
  pair** at S1.
- **S2 misfit origination.** Only a misfit-origination entry originates a
  misfit. Every substituted node must be a node of a grounded kline (no
  invention). A grounded kline is a candidate iff it shares a **node value**
  with the entry's nodes; a canon under the entry's own signature is never
  admitted. A shaped proposal already grounded is dropped, not emitted.
- **Termination.** K cogitates the work-list LIFO. A misfit entry persists; K
  has no notion of closing an S2 entry. A run terminates because an emission
  matches the table's terminal row, not because K decided it was done.

## Test Matrix

Cogitation is exercised end-to-end by the canonical MHALL run
(`tests/test_dialogue_smoke.py`). Isolated mechanism tests were removed to keep
the sub-project exploratory; add them as fresh behaviours are discovered, not
to defend the current mechanism.

## Out of Scope

- **Trainer-side cogitation.** The Trainer may originate misfits under the
  identical boundaries; the sole asymmetry is the ratifier (K-originated
  misfits ratified by T; T-originated by the supervisor). Deferred.
- **S3-stall → S2 migration.** When a 1:1 S3 pair is not ratified the entry
  stalls and may migrate. Deferred until real stall behaviour is observed.
- **Supervisor escalation** when no candidate admits.
