# Trainer Satisfaction Logic (Paced Loop) — Specification

## Overview

This spec defines the trainer's behaviour under the new **paced-loop** training
policy, replacing the batch-submit + auto-countersign loop in
`@specs/harness-server.md` §Trainer Participant.

Under the paced loop, the trainer no longer submits a whole lesson's compiled
klines to Kalvin in one batch. Instead it partitions the lesson into **prompted**
and **withheld** klines, prompts the primary one kline at a time, and drives a
single depth-first cascade by responding to Kalvin's gap-requests. Withheld
Identities and Canons are supplied as **ratification** (submit-grounding) when
Kalvin requests their signatures, never prompted proactively.

An entry is **learned** when it is grounded at its declared band — understood by
Kalvin and in LTM, with reported significance agreeing with declared significance
(`@CONTEXT.md` §Learned). Grounding, not S1, is the satisfaction signal.

## Dependencies

- `@specs/harness-server.md` — Trainer participant, bus actions, supervisor messages.
- `@specs/agent.md` — RationaliseEvent shape, significance on the KValue.
- `@specs/kscript.md` — compiled-entry declared bands (`op` field), structural
  states (COUNTERSIGNED, CANONIZED, CONNOTED, UNDERSIGNED, IDENTITY).
- `@specs/kline.md` — `is_canon`, `is_identity`, KLine equality.
- `@specs/signifier.md` — `make_signature`, `residual` (misfit diagnosis).
- `@specs/stub-kagent.md` — the bootstrap trainee the satisfaction logic is
  validated against; the contract the real Kalvin must reproduce.
- `@CONTEXT.md` — Learned, Canon, Ratify, Scaffolding.

## Definitions

### Partition

On lesson compile, the trainer partitions compiled entries into two sets.

**Withheld** — held by the trainer, supplied only as ratification in response to a
Kalvin signature request:

- **Identities** — every compiled entry with `op = IDENTITY`.
- **Canons** — every compiled relationship kline where
  `signature == make_signature(nodes)` and it is not identity (`is_canon`).
  This includes the authored semantic canons (C_MHALL, C_SVO, C_ALL) **and** the
  tokenizer's subword canons (`{Mary:[Ma,ry]}`, etc.). Both are withheld; subword
  canons are not filtered out (see §Subword Canons).

**Prompted** — every other compiled entry: COUNTERSIGNED, CONNOTED, UNDERSIGNED,
and CANONIZED-but-not-Canon. These are submitted one at a time.

### Held Index

A lookup from **signature → list of withheld klines** with that signature, ordered
by the **pull priority**:

1. **Canons** (authored and subword alike) — decomposition first.
2. **Relations** — held relations whose signature matches.
3. **Identities** — the fallback atom.

Within a priority tier, klines are ordered by compiled order. The pull for a
signature returns *all* held klines for that signature, in priority order — a
signature may carry both a subword canon and an authored relation, and both are
supplied (canon first, relation second).

### Declared Band

The band the author asserted via the written token, read from the compiled entry's
`op`:

| `op` | Declared band |
|------|---------------|
| COUNTERSIGNED | S1 |
| CANONIZED-which-is-Canon | S2 |
| CANONIZED-not-Canon | (n/a — treated as prompted, see below) |
| CONNOTED, UNDERSIGNED | S3 |
| IDENTITY | S4 |

A CANONIZED entry that is not a Canon (e.g. `A => B` where `A ≠ B|…`) is prompted,
not withheld, and its declared band is S3 (it carries no OR-reduction structure).

### Ratification

The trainer's act of endorsing a Kalvin request by supplying a withheld kline.
Two delivery forms:

- **Canon-ratify** — Kalvin requests signature X; the trainer submits a withheld
  Canon `{X:[operands]}`. Kalvin grounds it at S2.
- **Identity-ratify** — Kalvin requests signature X; no canon/relation is held for
  X, so the trainer submits the identity `X:[]`. Kalvin grounds it at S4.

Relations in the held index (the prompted relations pulled by signature) are *also*
delivered as submission; the trainer does not distinguish "withheld relation" from
"prompted relation" at delivery — a relation pulled on a signature request is
submitted and grounded at S3 like any prompted relation.

### Satisfaction

An entry is **satisfied** when it has been grounded at its declared band by a
Kalvin event whose `proposal` kline equals the entry's kline (KLine equality, KV-2)
and whose `proposal.significance` equals the entry's declared band. Concretely:

- a **grounded event** (frame at band) whose proposal equals the entry and whose
  significance == declared band → **satisfied**.
- the **primary's countersign** (frame at S1) whose proposal equals the primary →
  **satisfied**.

The lesson is complete when every compiled entry is satisfied (`satisfied ⊇
submitted`), where `submitted` now spans all four bands, not just S1.

## Behavioural Rules

### Partition & Hold

1. On lesson compile, the trainer partitions entries into Prompted and Withheld
   (Identities + Canons) per §Partition.
2. The trainer builds the Held Index (signature → ordered withheld list) per
   §Held Index.
3. Withheld klines are **not** submitted to Kalvin at lesson start; they enter the
   model only via ratification.

### Prompt the Primary

4. The trainer submits exactly one prompted kline to start the lesson: the primary
   countersigned relation (`{MHALL:[SVO]}` and its reciprocal). This is the only
   proactive prompt.
5. Submitting the primary kicks off a **single cascade**. The trainer does not
   proactively prompt further prompted klines during the cascade.

### Drive the Cascade (request/response)

6. On a Kalvin **request** event (a `frame` at S4 whose proposal is an ungrounded
   identity `X:[]`), the trainer looks up signature X in the Held Index and submits
   every held kline for X, in pull-priority order (canon → relation → identity).
   This is ratification.
7. If signature X has no held entry, the trainer cannot auto-ratify the request —
   it escalates the proposal to the supervisor (`@specs/supervisor-decision.md` SD-1).
   The request carries `{X: []}` as its proposal; the supervisor answers scaffold
   (write a kline for X) or continue.
8. The trainer submits one kline per Kalvin request-response turn (paced), not in
   a batch. Each submission yields the next request/ground events from Kalvin,
   which drive the next trainer response.

### Canon-First Ordering

9. When a signature carries both a Canon and a Relation, the Canon is submitted
   before the Relation (decompose before relating). The relation is submitted in
   a subsequent turn, once the canon's operands are in flight.

### Grounding = Satisfaction

10. On a Kalvin **ground** event (a `frame` at structural band) whose proposal
    equals a compiled entry and whose significance == that entry's declared band,
    the trainer marks the entry **satisfied**.
11. Satisfaction is keyed on the entry's KLine (signature + nodes), and band-equal.
    An entry declared S3 satisfied by a frame at S2 is **not** satisfied (band
    mismatch → divergence, see §Divergences).
12. The primary is satisfied by a frame at S1 (the countersign), emitted by Kalvin
    once both the primary and its reciprocal are grounded.

### Lesson Completion

13. The lesson is complete when `satisfied ⊇ submitted` (every compiled entry
    grounded at its declared band). The trainer emits `progress: lesson_complete`
    and submits the next lesson.

### Subword Canons

14. Subword canons (tokenizer BPE decompositions) are **not filtered out** of the
    Withheld set. They are withheld, ratifiable, and grounded at S2 like any Canon.
    They are future-proofing: Kalvin needs subword structure to reconstruct words
    for novel queries (e.g. "the little girl" → reconstruct "Mary").
15. The Held Index includes subword canons; the pull priority applies uniformly.

## Stalls and Divergences

### Stall (no held kline for a request)

16. [removed] — folded into `@specs/supervisor-decision.md` SD-1. A request for which
    the trainer holds no kline is a proposal (`{X: []}`) the trainer cannot
    auto-ratify; it is escalated to the supervisor like any other unresolvable
    proposal. There is no distinct `ungroundable_request` event or escalation reason.
17. [removed] — the delegated-mode / default-mode distinction is retired; there is
    one escalation path (`@specs/supervisor-decision.md`).

Real-Kalvin divergence from the table (Kalvin requests something the table doesn't
prescribe, or never requests what the table expects) is deferred to the Kalvin grill
— the stub never diverges, so this branch is unreachable in the bootstrap dialogue.

### Divergence (band mismatch)

18. If Kalvin grounds an entry at a band different from its declared band, the
    trainer records a **divergence**: reported significance ≠ declared significance.
    Two sub-cases:
    - **Under-reach** (reported < declared): Kalvin reports a lower band than
      declared. Likely an ungrounded operand the trainer hasn't yet ratified; the
      trainer does not mark satisfied and continues the cascade.
    - **Over-reach** (reported > declared): Kalvin reports a higher band than
      declared. The trainer records the entry satisfied (Kalvin did understand it)
      and emits a distinct supervisor signal (`progress: over_reach`) noting the
      divergence, so a supervisor can decide whether the over-reach is a discovery
      or an author/curriculum error. The entry is learned; the signal is
      informational.
19. The stub never produces a divergence, so §Divergences is exercised only by
    real Kalvin. The logic is specified here for completeness and to fix the
    trainer's behaviour when the Kalvin grill introduces real responses.

## Out of Scope

- The definition of *how* Kalvin produces its requests and grounds (cogitation,
  expand, misfit) — owned by the Kalvin grill, validated against the same table.
- The global event-kind change (`ground`/`frame` → significance). This spec keys on
  `proposal.significance`, forward-compatible with the change.
- Reactive-scaffolding generation (the LLMSupervisor) and the decision contract —
  owned by `@specs/supervisor-decision.md`. The paced loop submits klines and marks
  satisfaction; what happens when it cannot auto-ratify a proposal is the
  supervisor-decision contract.
- The `ratify_request` enrichment fields (`misfit`, `curriculum_context`) — owned by
  `@specs/supervisor-decision.md`, emitted on every escalated proposal.
- Goal-based curriculum generation.

## Canonical Example

The reference dialogue for "Mary had a little lamb" (single cascade, canon-first,
atom reuse, subword canons grounded) is the table produced in the design grill. It
grounds, in order: the primary `{MHALL:[SVO]}` (S1); the canons C_MHALL, C_ALL,
C_SVO (S2); the subword canons `{Mary:[Ma,ry]}`, `{had:[h,ad]}`, etc. (S2); the
relations `{M:[S]}`, `{H:[V]}`, `{A:[D]}`, `{Lᵢ:[Mo]}`, `{Lₐ:[O]}`, `{ALL:[O]}`
(S3); and every identity atom (S4). Every compiled entry is grounded at its
declared band; the lesson completes on the primary's S1 countersign.

## Test Matrix

| ID  | Criterion | Origin ref |
|-----|-----------|------------|
| TS-1 | Lesson compile partitions entries into Prompted and Withheld (Identities + Canons) | §Partition |
| TS-2 | Withheld set includes subword canons (no filtering) | §Subword Canons, §Partition |
| TS-3 | Held Index maps signature → ordered (canon, relation, identity) klines | §Held Index |
| TS-4 | Only the primary is prompted at lesson start | §Prompt the Primary |
| TS-5 | Submitting the primary starts a single cascade; no further proactive prompts | §Prompt the Primary |
| TS-6 | On a request for signature X, the trainer submits every held kline for X in priority order | §Drive the Cascade |
| TS-7 | Canon is submitted before Relation for a shared signature | §Canon-First Ordering |
| TS-8 | An entry is satisfied by a ground event at its declared band (band-equal) | §Satisfaction |
| TS-9 | An entry grounded at the wrong band is a divergence, not satisfaction | §Divergences |
| TS-10 | A ground at S4 satisfies an identity (S4 is learned, not a failure) | §Satisfaction, §Grounding |
| TS-11 | The primary is satisfied by a frame at S1 (Kalvin-emitted countersign) | §Satisfaction |
| TS-12 | Lesson completes when satisfied ⊇ submitted across all four bands | §Lesson Completion |
| TS-13 | [removed] — relocated to `@specs/supervisor-decision.md` SD-1 (escalation of an unresolvable proposal) | §Stall |
| TS-14 | Over-reach (reported > declared) satisfies the entry and emits `progress: over_reach` | §Divergences |
| TS-15 | Under-reach (reported < declared) does not satisfy; cascade continues | §Divergences |
| TS-16 | Ratification delivers withheld klines via submission, not via countersign | §Ratification |
| TS-17 | Atom reuse: a Canon whose operands are grounded is satisfied with no new primes | §Canonical Example |
