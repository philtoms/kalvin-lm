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

Cogitation is the evolving mechanism by which agents become aligned with the
script. The list below is a maintenance-oriented inventory of **current**
cogitation behaviour — what the engine does today, one line each. It is not a
contract: each entry is expected to change as the mechanism evolves. The code
(`src/training/dialogue/rationalise.py`) is the source of truth for how.

- **Two channels.** Each turn emits a dialogue **batch** (S4 asks, S3/S2
  proposals, S1/S2 replies) and **observations** of K's S1 groundings.
- **Identities.** Three shapes for one lexical item: the S4 ask `X:[]`, the
  self-referential `X:[X]`, and the compound `X:[COMPOUND_TOKEN, x, y]`; all
  key by signature alone.
- **Self-identity forging.** A bare `X:[]` the engine can't answer is forged
  by the supervisor (`synthesize`) as `X:[X]` at S1.
- **Emission dedup.** The actor (not the engine) drops any proposal it has
  already published; a fully-duplicate batch yields a PASS.
- **Routing.** Significance is derived from structure. S4 pops the framed
  identity ask; S1/S2/S3 are appended to the work-list; an S2 misfit also
  unpacks its unknown nodes/signature as identity asks.
- **S1 identity fast path.** An incoming S1 identity grounds whenever its
  signature has been seen (framed, pending, or grounded).
- **Replies (role-neutral).** The engine answers an S4 ask with the canon
  (S1 if all nodes grounded, else S2) or the compound identity (S1), and
  ratifies an S3 proposal at S1. Actors filter their role's bands and may
  apply kline-level protocol corrections.
- **Cogitation pass.** One LIFO pass: ground promotable/groundable entries
  (observed, dropped); countersignable entries take the S3 path; multi-node
  misfits take the S2 path; ungroundable identities batch into one S4 ask.
- **S3 countersignature.** Pair two canons' operands left-to-right; emit each
  unresolved pairing as an S3 CONNOTES (residuals synthesised into a
  signature). Each grounded pairing yields its operand-level reciprocal; once
  all pairings resolve, the entry grounds at S1 (canonical reciprocal).
- **S2 similar-fit proposal.** A misfit entry proposes by recombining grounded
  klines that share a node value (no invention; its own canon excluded).
- **Work-list persistence.** Entries that fire no path persist and retry on
  later turns.
- **Frame.** Records emissions to dedup and to match replies to asks;
  identities match by signature across shapes.
- **State injection.** A `RationaliserState` may be empty or loaded from a
  saved snapshot (a grounded prior); the engine stays oracle-free either way.

## Test Matrix

Cogitation is exercised end-to-end by the canonical MHALL run
(`tests/test_dialogue_smoke.py`). Isolated mechanism tests were removed to keep
the sub-project exploratory; add them as fresh behaviours are discovered, not
to defend the current mechanism.

- **DDT-3** — canonical MHALL run with table actors covers the whole exchange
  (zero displacement): the core loop is wired correctly.
- **DDT-4** — when two canons' operands are reciprocally paired (every CONNOTES
  pairing resolved), K grounds both directions of the canonical reciprocal
  pair (`{A:[B]}` and `{B:[A]}`) at S1. Pinned by
  `test_canon_reciprocal_grounded_when_all_operand_pairings_resolve`.

## Out of Scope

- **Supervisor escalation** when no candidate admits.
