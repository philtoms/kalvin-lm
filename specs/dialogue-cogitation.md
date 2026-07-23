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
  of dialogue emissions (speech acts — S4 identity asks, S3 connotation
  proposals, S2 similar-fit proposals, and replies: S1 ratifications and S2
  canon/identity replies) and **observations** of K's internal S1 groundings
  (every kline K grounds). Grounding that is not a reply is observed only;
  observations are surfaced for white-box verification.
- **Identities.** An identity is one lexical item with three shapes: the
  **S4 ask** ``X:[]`` (unrecognised) and two **S1 groundings** —
  self-referential ``X:[X]`` and compound ``X:[COMPOUND_TOKEN, x, y]``. The
  compound shape is the grounding that decodes back into text, so it **must**
  be grounded (not discarded) whenever it is encountered; its subwords are
  how the item is reconstructed. All three shapes are the same identity —
  the frame and grounding key them by signature alone, not by shape.
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
  cogitation, except an **S1 identity reply** which grounds directly (see
  Identities). An **S2** misfit additionally unpacks its unrecognised nodes
  and signature onto the work-list as identity placeholders. Routing emits
  no dialogue batch and performs no grounding other than the S1 identity
  fast path — promotion of other entries is a per-entry act of cogitation,
  not a routing pre-pass.
- **S1 identity fast path.** An incoming S1 identity (a reply) grounds at S1
  whenever its signature has been **seen** — framed as an ask, pending on
  the work-list, or already grounded — not only when an ask-shape is
  currently framed. This is required by the route-all-then-cogitate
  ordering: the ask an S1 reply answers may be emitted by the same turn's
  cogitation, after routing, so the frame alone is a stale view within the
  turn. Grounding consumes the framed ask and the pending work-list identity
  for that signature.
- **Replies (role-neutral).** The engine replies to an incoming query from
  its own state, in addition to its trainee-side proposals: an **S4 identity
  ask** about ``X`` is answered with ``X``'s canon (S1 when every node is
  grounded, else S2 — the engine's structural bookkeeping) or, failing a
  canon, ``X``'s compound identity at S1; an **S3 proposal** is ratified at
  S1. These are earned (no oracle): they read the engine's grounded model
  only. The engine is role-neutral; protocol corrections at the kline level
  are the **actor's** job, not the engine's. The actor filters its role's
  bands (the trainee keeps S2/S3/S4 and suppresses S1/S2 replies; the
  trainer keeps S1/S2 and suppresses S3/S4) and may intervene on individual
  klines — e.g. a trainer proposes canons at S2 to direct the listener to
  cogitate over their nodes, regardless of the engine's S1/S2 stamp. An ask
  the engine cannot answer from state (e.g. a CONNOTES gloss) is left for
  the actor's other paths (escalation or scripted fallback).
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
  pair takes the same S3 connotation path as a 1:1 pair. Each pairing K grounds
  yields its **operand-level reciprocal** at S1 (e.g. `Mary:[Subject]` grounds
  `Subject:[Mary]`). The relationship entry itself persists on the work-list
  across turns; when cogitation finds **every** pairing resolved, it grounds
  the **entry itself** at S1 — the **canonical-level reciprocal** (e.g.
  `MHALL:[SVO]`), whose own reciprocal (`SVO:[MHALL]`) is mirrored by the same
  operand-level-reciprocal rule. Both reciprocals are observed, not emitted
  into the dialogue. Because the entry drives its own completion, no T query
  is required to trigger it — any cogitation pass after the last pairing
  resolves completes the countersignature.
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
- **Frame (emission memory).** The frame records what the engine has put in
  play, to deduplicate emissions and to match replies to asks. Identities are
  keyed by signature alone — any shape recognises any other — so an S4 ask
  the engine emitted matches the S1 reply it later receives. Non-identities
  key on structural significance, as before.
- **State injection.** A ``RationaliserState`` may be constructed empty or
  loaded from a saved snapshot (a **grounded prior**), so an actor can lead
  from a populated model rather than an empty one. A trainer loaded with a
  prior earned by running as a trainee against the supervisor earns its
  canon/identity replies from that state instead of escalating. The engine
  itself is oracle-free either way: the prior is state, not a live oracle.

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
