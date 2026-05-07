# Kalvin — Extended Cogitation: S2 Expansion

## Overview

Extended cogitation gives the Cogitator the ability to reshape S2 klines whose
signatures don't match their nodes. Instead of discarding S2 results that fail
countersignature, the Cogitator attempts to **expand** them toward canonical
status — adding missing nodes, removing redundant ones, or both — and emits
the result as a proposal for the teacher to ratify.

This is the mechanism by which Kalvin performs **self-directed study**: it
works through its own partial understanding, generating proposals that a
teacher can confirm or redirect. Extended cogitation depends on structural
grounding (see `docs/roadmap.md`, Challenge 6) being implemented first.

---

## Misfit Classification

Given a candidate kline with signature `S` and nodes signature
`N = make_signature(nodes)`:

| Condition            | Classification | Meaning                                       |
| -------------------- | -------------- | --------------------------------------------- |
| `S == N`             | Canonical (S1) | No expansion needed                           |
| `S & ~N != 0`        | Underfitting   | Signature promises bits the nodes don't deliver |
| `N & ~S != 0`        | Overfitting    | Nodes carry bits the signature doesn't capture  |
| Both                 | Dual misfit    | Both conditions hold simultaneously            |

The **underfit gap** is `S & ~N` — the bits the candidate needs but doesn't
have. The **overfit excess** is `N & ~S` — the bits the candidate carries
but doesn't need.

A kline may be both underfitting and overfitting at the same time. The
Cogitator must handle both conditions for such klines.

---

## Expansion Operations

The Cogitator attempts three kinds of expansion, depending on misfit
classification. Order of operations between underfit and overfit expansion
is not specified; any order that satisfies the constraints is acceptable.

### Underfit Expansion — Add Nodes

Compute `gap = S & ~N`. Search the model for klines whose signatures
contribute to the gap. For each candidate addition, verify that
`make_signature(addition_nodes)` exists in the model. Construct the
expanded kline:

```
{S: [original_nodes + addition_nodes]}
```

Verify the expanded kline moves toward canonical: `make_signature(expanded_nodes)`
is closer to `S`.

### Overfit Expansion — Remove Nodes

Identify nodes whose bits contribute to `N & ~S`. Remove those nodes from
the candidate. Verify that `make_signature(removed_nodes)` exists in the
model — the removed nodes must have a home. Construct the trimmed kline:

```
{S: [remaining_nodes]}
```

The removed nodes must either form a new kline whose signature already
exists in the model, or complete an existing underfitting kline.

### Dual Expansion — Replace Nodes

When a kline is both underfitting and overfitting, treat as a single
atomic replacement: swap a subset of overfit nodes for a subset that
fills the underfit gap. All generated signatures — the removed group,
the added group, and the result — must exist in the model.

---

## Universal Constraint

**Every signature generated during expansion must already exist in the model.**
This applies to:

- `make_signature(added_nodes)` — must exist in the model
- `make_signature(removed_nodes)` — must exist in the model
- `make_signature(expanded_nodes)` — must exist in the model (this is the
  proposal kline itself, whose ratification requires a countersigned kline
  already present)

This constraint guarantees:

1. **No invention** — the Cogitator never creates new concepts, only
   recombines existing ones.
2. **No data loss** — removed nodes always have a destination that the
   model already recognises.
3. **Ratifiability** — the reciprocal kline required for countersignature
   is already in the model.

When the Cogitator cannot satisfy this constraint (the required signatures
don't exist), it produces no proposals. The teacher, observing no `frame`
event for that query before the `done` event, infers that scaffolding is
needed and provides the missing klines in a subsequent training round.

---

## The Extended Cogitator Pipeline

Extended cogitation adds a new phase after countersignature checking in the
existing `_process` method:

```
_process(QueryCandidate(query, candidate, significance)):

  # Phase 1: Countersignature (existing behaviour)
  if model.is_countersigned(query, candidate):
    model.add(candidate)
    on_s1(query, candidate)
    return

  # Phase 2: S2 Expansion (new)
  candidate_sig = candidate.signature
  nodes_sig = make_signature(candidate.nodes)

  if candidate_sig == nodes_sig:
    return  # canonical — nothing to expand

  # Determine misfit
  underfit_gap = candidate_sig & ~nodes_sig
  overfit_mask = nodes_sig & ~candidate_sig

  # Generate expansion proposals
  for expansion in generate_expansions(candidate, underfit_gap, overfit_mask):
    if validate_expansion(expansion):   # all signatures exist in model
      emit_proposal(expansion)          # frame event for teacher
```

`generate_expansions()` may yield multiple proposals per work item. The
Cogitator explores the model for different ways to satisfy the expansion
constraints, subject to the existing sampling controls (top-k budget,
top-p evidence threshold, temperature boundary shifts).

### Reentrant Rationalisation

Extended cogitation preserves the existing distinction between rationalisation
and cogitation. The ability to re-enter rationalisation through cogitation —
thus generating deeper cogitation tails — is retained unchanged.

### Proposal Emission

Each valid expansion is emitted as a `frame` event on the event bus — the
same mechanism used for S1 proposals. The teacher evaluates it against its
expectation using kline equality (MVP). No additional event type or metadata
is required; the proposal is a kline, and the teacher compares it to what
it expected.

The teacher is responsible for determining what changed if it needs to. The
MVP only requires: does the proposed kline match the expectation? If yes,
countersign (ratify). If no, scaffold.

---

## S2 Klines from Training

KScript's `=>` operator naturally produces S2 klines:

| KScript           | Compiled kline                      | Misfit type |
| ----------------- | ----------------------------------- | ----------- |
| `AB => C`         | `{AB: [C]}` — sig `A\|B`, nodes `C` | Underfitting |
| `A => B C`        | `{A: [B, C]}` — sig `A`, nodes `B\|C` | Overfitting |
| `WDMH => MHALL`   | `{WDMH: [M, H, A, L, L]}`          | Dual misfit |

The training pipeline creates these deliberately:

- **Underfitting klines act as templates** — they have a known concept
  (signature) with holes to fill. They allow Kalvin to match a question
  with a question+answer structure.
- **Overfitting klines act as sequencers or planners** — they carry
  step-by-step structure under a single goal signature. They allow Kalvin
  to rationalise a query in discrete steps.

The Cogitator's expansion process fills templates and decomposes sequencers,
turning S2 partial understanding into proposals that, once ratified, become
S1 grounded knowledge.

---

## Dependence on Structural Grounding

Extended cogitation requires structural grounding for two reasons:

1. **Promotion after ratification** — when the teacher countersigns an
   expansion proposal, all participating STM klines must be promoted to
   frame (not just the ratified kline), including the added/removed node
   groups and any S4 identity klines involved.

2. **Frame richness** — the expansion search requires a model populated
   with the signatures it needs to find. Structural grounding ensures that
   frames hold S4–S1, giving the Cogitator more graph topology to traverse
   and more candidate signatures to match against.

---

## Temperature

For the MVP, temperature does not affect expansion. Future work may allow
temperature to influence candidate selection during expansion — higher τ
broadening the search for node additions, lower τ restricting it to close
matches.

---

## Relationship to Other Documents

| Document                       | Relationship                                            |
| ------------------------------ | ------------------------------------------------------- |
| `specs/agent.md`               | Cogitation spec updated with S2 expansion phase         |
| `specs/overview.md`            | Overview updated with expansion in cogitation section   |
| `specs/significance.md`        | S2 description references expansion                     |
| `docs/learning.md`             | Self-study now includes S2 expansion                    |
| `docs/training-loop.md`        | Training loop references S2 expansion proposals         |
| `docs/training-schedule.md`    | KScript `=>` documented as producing S2 templates       |
| `docs/roadmap.md`              | Extended cogitation added as a challenge after Phase A  |
