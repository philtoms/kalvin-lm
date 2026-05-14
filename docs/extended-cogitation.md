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
grounding (see `docs/roadmap.md`, Challenge 1) being implemented first.

---

## Misfit Classification

Given a candidate kline with signature `S` and nodes signature
`N = make_signature(nodes)`:

| Condition     | Classification | Meaning                                         |
| ------------- | -------------- | ----------------------------------------------- |
| `S == N`      | Canonical (S1) | No expansion needed                             |
| `S & ~N != 0` | Underfitting   | Signature promises bits the nodes don't deliver |
| `N & ~S != 0` | Overfitting    | Nodes carry bits the signature doesn't capture  |
| Both          | Dual misfit    | Both conditions hold simultaneously             |

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

Every expansion proposal must satisfy the **Ratification Constraint**
(described below).

### Underfit Expansion — Add Nodes

Compute `gap = S & ~N`. Search the model for klines whose signatures
contribute to the gap. Construct the expanded kline:

```
{S: [original_nodes + addition_nodes]}
```

Verify the expanded kline moves toward canonical: `make_signature(expanded_nodes)`
is closer to `S`.

The proposed expanded kline is emitted as a proposal. The teacher ratifies
it only if it matches a training expectation.

### Overfit Expansion — Remove Nodes

Identify nodes whose bits contribute to `N & ~S`. Remove those nodes from
the candidate. Construct the trimmed kline:

```
{S: [remaining_nodes]}
```

The Cogitator also constructs a **companion kline** from the removed
nodes:

```
{make_signature(removed_nodes): [removed_nodes]}
```

Both the trimmed kline and the companion kline are emitted as proposals.
The teacher ratifies each independently against its training expectations.

### Dual Expansion — Replace Nodes

When a kline is both underfitting and overfitting, treat as a single
atomic replacement: swap a subset of overfit nodes for a subset that
fills the underfit gap.

The replacement kline and the removed-group companion kline are both
emitted as proposals. The teacher ratifies each independently against
its training expectations.

---

## Ratification Constraint

**Every expansion proposal must be ratified by the teacher against a
training expectation.** The Cogitator emits proposals; the teacher decides
which ones enter the model. This is the sole constraint on expansion.

### Proposals Emitted per Expansion Type

| Expansion type | Proposals emitted                                                    |
| -------------- | -------------------------------------------------------------------- |
| Added nodes    | The expanded kline                                                   |
| Removed nodes  | The trimmed kline **and** the companion kline from removed nodes     |
| Dual misfit    | The replacement kline **and** the companion kline from removed nodes |

Each proposal is emitted as an independent `frame` event. The teacher
ratifies (or rejects) each one individually.

### Guarantees

1. **No invention** — the model only acquires klines the teacher expects.
2. **No orphan nodes** — every removed-node group is proposed as a
   companion kline, giving the teacher the opportunity to ratify it.
3. **No separate existence check** — because only ratified klines enter
   the model, every signature in the model is there by teacher approval.
   The former "must exist in model" constraint is an implicit consequence.

When the teacher rejects all proposals for a query — observed as no
`frame` event before the `done` event — it infers that scaffolding is
needed and provides the missing klines in a subsequent training round.

---

## The Extended Cogitator Pipeline

The Cogitator's `_process` method handles S2 expansion. Countersignature
(ratification) is checked upstream in `rationalise()` Phase 3 (Assess),
before candidates are selected — so `_process` only needs to handle
expansion:

```
_process(QueryCandidate(query, candidate, significance)):

  # S2 Expansion
  candidate_sig = candidate.signature
  nodes_sig = make_signature(candidate.nodes)

  if candidate_sig == nodes_sig:
    return  # canonical — nothing to expand

  # Determine misfit
  underfit_gap = candidate_sig & ~nodes_sig
  overfit_mask = nodes_sig & ~candidate_sig

  # Generate and emit expansion proposals
  for expansion in generate_expansions(candidate, underfit_gap, overfit_mask):
    emit_proposal(expansion.proposal)            # frame event for teacher
    if expansion.has_removed_nodes:
      emit_proposal(make_companion(expansion))   # companion kline
```

`generate_expansions()` may yield multiple proposals per work item. The
Cogitator explores the model for different ways to satisfy the expansion
constraints. The Cogitator processes all yields from `model.expand()`
without filtering.

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

| KScript         | Compiled kline                        | Misfit type  |
| --------------- | ------------------------------------- | ------------ |
| `AB => A`       | `{AB: [A]}` — sig `A\|B`, nodes `A`   | Underfitting |
| `A => A B`      | `{A: [A, B]}` — sig `A`, nodes `A\|B` | Overfitting  |
| `WDMH => MHALL` | `{WDMH: [M, H, A, L, L]}`             | Dual misfit  |

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

## Exploration Depth

For the MVP, the Cogitator processes all yields from `model.expand()`
without filtering. Future work may add exploration depth controls to limit
how many expansion proposals are generated per work item.

---

## Relationship to Other Documents

| Document                    | Relationship                                           |
| --------------------------- | ------------------------------------------------------ |
| `specs/agent.md`            | Cogitation spec updated with S2 expansion phase        |
| `specs/overview.md`         | Overview updated with expansion in cogitation section  |
| `specs/significance.md`     | S2 description references expansion                    |
| `docs/learning-and-training.md` | S2 expansion in study context, proposals in training loop, KScript `=>` templates |
| `docs/roadmap.md`           | Extended cogitation is Challenge 2 in the roadmap |
