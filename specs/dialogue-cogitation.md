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
  proposals for ratification, then establishes the S1 countersignature (both
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
- **Routing.** The routing rule handles S1 (ground/cleanup), S4 (pop the
  identity ask), and retrospective promotion before handing over to cogitation;
  only S2/S3 entries and their ungrounded identities reach cogitation.
  An S1 relationship `{S: nodes}` with more than one node, all grounded,
  is an exception: it takes the **canon-countersignature** branch below,
  not promotion. (1:1 shapes — connotations, denotations — promote directly.)
- **Cogitation.** A countersignable entry takes the S3 path; a multi-node misfit
  takes the S2 path. Both of these are emitted as proposals. Identities are batched
  up and emitted as a single request.
- **S3 countersignature.** K pairs the operands of the two canons left-to-right
  at group size 1, emitting every unresolved pairing in one batch (a 1:1 pair
  CONNOTES at S3; a grouped residual as a canonical request at S2). When every
  pairing is resolved, K grounds **both directions of the reciprocal pair** at
  S1 (observed, not emitted into the dialogue).
- **S2 similar fit proposal.** Only a misfit entry proposes a similar fit.
  Every substituted node must be a node of a grounded kline (no invention).
  A grounded kline is a candidate iff it shares a **node value** with the
  entry's nodes; a canon under the entry's own signature is never admitted.
  A shaped proposal already grounded is dropped, not emitted.
- **Work-list.** An entry that matches neither path is not discarded — it
  persists and is re-tried on later turns. Only grounding (the entry's nodes
  becoming seen) or an operand's canon arriving makes it fire; until then it
  emits nothing.
- **S1 canon-countersignature.** When T ratifies a misfit at S1 — an S1
  relationship `{S: nodes}` with more than one node, all grounded — K does
  not ground the misfit. K computes `C = make_signature(nodes)` (the canon
  the ratified nodes form), promotes C by grounding `{C: nodes}`, then
  establishes the S1 countersignature by grounding `{S: [C]}` and `{C: [S]}`.
  Each ground is observed; nothing is emitted into the dialogue. K drops
  the pending work-list entries under S. K never searches for an existing
  canon; it computes C and admits it whether or not it was previously seen.

## Test Matrix

Cogitation is exercised end-to-end by the canonical MHALL run
(`tests/test_dialogue_smoke.py`). Isolated mechanism tests were removed to keep
the sub-project exploratory; add them as fresh behaviours are discovered, not
to defend the current mechanism.

## Out of Scope

- **Supervisor escalation** when no candidate admits.
