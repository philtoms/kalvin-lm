# Dialogue Cogitation — Specification

> **Working sketch, not a frozen contract.** This is the most speculative part
> of the dialogue sub-project: the rationalising trainee's cogitation. It is
> expected to be reshaped by discovery. Keep it light and lean; replace rather than
> augment.

## Overview

Cogitation is a trainee's act of working a work-list entry toward S1. A
`rationalise` call applies the entry rule to the received events as bookkeeping,
then emits a **batch** of values from cogitation. Cogitation dispatches a
workable entry on **structure-as-significance** into one of two paths:

- the **S3 countersignature path** — a single-node relationship `{L:[R]}`
  whose operands both have seen canons: K pairs the two canons' operands into
  proposals (each a CONNOTES at S3, including a grouped residual synthesised
  into a left-operand signature), then establishes the S1 countersignature (both
  directions of the reciprocal pair); and
- the **S2 similar-fit-proposal path** — a multi-node misfit: K originates a
  proposal by recombining grounded klines and offers it for ratification.

Both paths strive toward S1 by **ratification** — another participant
countersigns what K proposes. The mechanism (algorithm, accumulation,
candidate resolution) is HOW and lives in `@plans/implement-rationalising-trainee.md`;
this spec owns only the two paths and their boundaries.

## Dependencies

- `@CONTEXT.md` — Proposal, Misfit, Canon, Ratify, Structural Relationships.
- `@specs/dialogue-driven-training.md` — the actor contract this cogitation
  satisfies (the trainee side).
- `@specs/cogitator.md` — the real async slow path this deliberately simplifies.

## Behavioural Rules

- **Dialogue vs grounding.** Cogitation produces two channels: a **batch**
  of dialogue emissions (speech acts for T — S4 identity asks, S3 connotation
  proposals, S2 similar-fit proposals) and **observations** of K's internal
  S1 groundings (every kline K grounds). Grounding does not emit into the
  dialogue; observations are surfaced for white-box verification.
- **Emission deduplication.** K never publishes the same proposal twice. The
  engine is **stateless about its own emissions** — it may re-derive a
  proposal on successive turns (an S2 misfit persists in the work-list until
  ratified; an unresolved identity re-surfaces) — and the **actor** is the
  single deduplication point: it records every proposal it has published (by
  `(signature, nodes)`) and drops any re-derivation. When the engine's whole
  batch is duplicates, the actor emits nothing and the runner sees a PASS — K
  waits for the trainer. Dedup lives in the actor, not the engine, so the
  engine's state can stay a pure model of what K has grounded.
- **Routing.** Routing significance is **calculated from structure** (the
  objective structural relationship), not read from the producer's surface
  stamp — the same structural derivation as cogitation's dispatch. Routing
  does one thing per query: an **S4** pops the pending identity ask;
  **every other query (S1, S2, S3)** is appended to the work-list for
  cogitation. An **S2** misfit additionally unpacks its unrecognised nodes
  and signature onto the work-list as identity placeholders. Routing emits
  no dialogue batch and performs no grounding — promotion is a per-entry act
  of cogitation, not a routing pre-pass.
- **Cogitation.** Each work-list entry is resolved in one pass, LIFO. An
  entry that is **promotable** (a single-node relationship whose reciprocal
  is grounded) or **groundable** (an identity whose signature is grounded, or
  a canon whose nodes are all grounded) is grounded at S1 — observed, not
  emitted — and dropped. A countersignable entry takes the S3 path; a
  multi-node misfit takes the S2 path. Both of these are emitted as
  proposals. Identities that remain ungroundable are batched up and emitted
  as a single S4 request.
- **S3 countersignature.** K pairs the operands of the two canons left-to-right
  at group size 1, emitting every unresolved pairing in one batch. Every pairing
  — whether a 1:1 pair or a grouped residual — is a CONNOTES proposal at S3:
  when one side reaches a single node, the other side's residual is synthesised
  into a signature (substituted for that pair's left operand) so the grouped
  pair takes the same S3 connotation path as a 1:1 pair. When every pairing is
  resolved, K grounds **both directions of the reciprocal pair** at S1
  (observed, not emitted into the dialogue).
- **S2 similar fit proposal.** Only a misfit entry proposes a similar fit.
  Every substituted node must be a node of a grounded kline (no invention).
  A grounded kline is a candidate iff it shares a **node value** with the
  entry's nodes; a canon under the entry's own signature is never admitted.
  A shaped proposal already grounded is dropped, not emitted. Because the
  misfit entry persists until ratified, the engine re-derives the same
  proposal on successive turns; the actor's emission deduplication (above)
  ensures K publishes it once and then waits.
- **Work-list.** An entry that matches neither dispatch path is not
  discarded — it persists and is re-tried on later turns. Only grounding
  (the entry's nodes becoming seen) or an operand's canon arriving makes it
  fire; until then it emits nothing.

## Test Matrix

Cogitation is exercised end-to-end by the canonical MHALL run
(`tests/test_dialogue_smoke.py`). Isolated mechanism tests were removed to keep
the sub-project exploratory; add them as fresh behaviours are discovered, not
to defend the current mechanism.

## Out of Scope

- **Supervisor escalation** when no candidate admits.
