# Cogitator Specification

## Overview

The Cogitator is the slow path of the rationalisation pipeline. It is a
background thread that drains a backlog of pre-routed work items
(query|candidate|level pairs), expands each through `model.expand()` to
discover connotations, classifies yields against significance boundaries,
and routes results to a `CogitationHandler`. It also performs S2 expansion
— reshaping misfit candidate klines toward canonical status — emitting
proposals for the agent to ratify.

The Cogitator is a thin threading dispatcher. All significance computation
lives in the model's `expand()`; all expansion-proposal logic is delegated.
It receives pre-routed work items from the agent and emits proposals at any
significance level.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Agent (@agent spec)

- Submits pre-routed `WorkItem`s to the Cogitator during Phase 5 of
  rationalisation. The agent is the primary `CogitationHandler`
  implementation.
- Owns the candidate cap and S1-first ordering applied before submission.

### Model (@model spec)

- Provides `expand(Q, C)` generator yielding `QueryCandidate`s
  (query, candidate, significance).
- Provides `add_frame` / `add_ltm` for tiered writes of proposals and
  companions.
- Provides `generate_expansions(candidate)` for S2 expansion proposals and
  companion klines.
- Provides constants `D_MAX` and `MASK64` for significance values.
- Computes significance internally; the Cogitator consumes
  `QueryCandidate.significance` directly without inversion.

### Signature (@signature spec)

- Provides `make_signature(nodes)` used during S2 misfit classification
  (canonical check, underfit gap, overfit excess).

### Kline (@kline spec)

- Klines flow through the Cogitator as query, candidate, proposal, and
  companion values.

## Definition

A Cogitator consists of:

| Component  | Type               | Description                                        |
| ---------- | ------------------ | -------------------------------------------------- |
| model      | Model              | For `expand()`, expansions, tiered writes.         |
| adapter    | KAgentAdapter      | Receives `RationaliseEvent`s (`on_event`).         |
| handler    | CogitationHandler  | Receives S1 results and expansion proposals.       |
| timeout    | float              | Idle seconds before emitting `"done"` (default 2.0). |
| backlog    | queue of WorkItem  | Pending work items.                                |
| thread     | background daemon  | Pulls and processes work items.                    |

## Work Items

A WorkItem is a single query-candidate pair queued for background
processing:

```
WorkItem:
  query:      KLine
  candidate:  KLine
  level:      str   # routing level: "S1", "S2", or "S3"
```

The agent submits one WorkItem per routed candidate (including S1). The
Cogitator expands each non-S1 WorkItem into a sequence of QueryCandidates
via `model.expand()`.

## Query Candidates

A QueryCandidate is a single query-candidate-distance result yielded by
`model.expand()`:

```
QueryCandidate(query, candidate, significance):
  query:        KLine
  candidate:    KLine
  significance: int     # pre-computed by the model (~packed_distance & MASK64)
```

Intermediate yields represent discovered connotations — indirect relationships
between nodes of the query and candidate. The final yield is always the
terminal significance for the original pair.

The model computes significance internally; the Cogitator uses `.significance`
directly without any inversion.

## Processing

The Cogitator processes yields from `model.expand()` until S1 is found.
When S1 is discovered, the work item stops (query fully resolved) and
no further yields are evaluated. If no S1 is found, all yields are processed
and S2/S3 expansion proposals are emitted as frame events regardless of
their computed significance.

### Significance Boundaries

Three boundaries classify yielded significance values:

```
D_MAX ── S1|S2 ──────── S2|S3 ──────────── S3|S4 ── 0
```

| Boundary | Position                    | Meaning                        |
| -------- | --------------------------- | ------------------------------ |
| S1\|S2   | `D_MAX - 1`                 | Only exact S1 qualifies as S1  |
| S2\|S3   | `~_S2_S3_DISTANCE`          | Packed distance threshold (100)  |
| S3\|S4   | `0`                          | Only zero-significance is S4   |

Classification is a cascade: `sig ≥ S1|S2 → S1`, `sig ≥ S2|S3 → S2`,
`sig ≥ S3|S4 → S3`, else S4. Raw significance values are never mutated.

### All Yields Processed

The Cogitator processes all connotations yielded by `expand()` for each
submitted WorkItem. There is no truncation or early stopping within a
single WorkItem — every discovered relationship is evaluated. (Candidate
fan-out is bounded upstream by the agent's candidate cap — @agent spec
§Candidate Cap.)

## CogitationHandler

CogitationHandler is a `@runtime_checkable` Protocol defining the seam
between the Cogitator and its consumers. The Agent is the primary
implementation.

```
CogitationHandler:
  on_s1(query, candidate)                      — called when an S1 (exact) result is discovered
  on_expansion(query, proposal, significance)  — called when an expansion proposal is generated (S2/S3)
```

### S1 Fast-Path

When a work item is pre-routed as S1 (the candidate's routing level is
S1), the Cogitator skips `model.expand()` entirely and calls
`handler.on_s1(query, candidate)` directly. This avoids generating
intermediate S2/S3 connotation yields that would produce unnecessary
expansion proposals.

```
run_work_item(WorkItem(query, candidate, routing_level)):
  if routing_level == S1:
    handler.on_s1(query, candidate)   # Fast-path: skip expand()
    return
  for qc in model.expand(query, candidate):
    if qc.significance >= s12:
      handler.on_s1(query, candidate)
      break
    else:
      process(qc)
```

### Work Item Processing

For S2/S3 work items, the Cogitator expands each WorkItem, processing all
yields from `model.expand()`:

```
process(QueryCandidate(query, candidate, significance)):
  # S2 expansion only
  if candidate is canonical:
    return                        # nothing to expand
  for proposal, companions in model.generate_expansions(candidate):
    model.add_frame(proposal)      # write proposal to Frame + STM
    emit frame event for proposal
    for companion in companions:
      model.add_frame(companion)   # write companion to Frame + STM
      emit frame event for companion
```

All yields are processed until S1 is found, at which point the work item
stops (the query is fully resolved). Raw significance values are
never mutated. Proposals can be emitted at any significance level.

## S2 Expansion

When countersignature fails for an S2 result, the Cogitator attempts to
**expand** the candidate kline toward canonical status by reshaping its
nodes to match its signature. This is the mechanism for self-directed
study — the Cogitator works through partial understanding and emits
proposals for the agent to ratify.

The conceptual description of S2 expansion (misfit types, templates,
sequencers, guarantees) is defined in the vision document
(`docs/kalvin-vision.md`, Study → S2 Expansion). This section specifies
the implementation detail.

### Misfit Classification

Given a candidate kline with signature `S` and nodes signature
`N = make_signature(nodes)`:

| Condition       | Classification | Meaning                                         |
| --------------- | -------------- | ----------------------------------------------- |
| `S == N`        | Canonical (S1) | No expansion needed                             |
| `S & ~N != 0`   | Underfitting   | Signature promises bits the nodes don't deliver |
| `N & ~S != 0`   | Overfitting    | Nodes carry bits the signature doesn't capture  |
| Both            | Dual misfit    | Both conditions hold simultaneously             |

The **underfit gap** is `S & ~N`. The **overfit excess** is `N & ~S`.
A kline may be both underfitting and overfitting at the same time.

### Underfit Expansion — Add Nodes

Compute `gap = S & ~N`. Search the model for klines whose signatures
contribute to the gap. Construct the expanded kline:

```
{S: [original_nodes + addition_nodes]}
```

Verify the expanded kline moves toward canonical: `make_signature(expanded_nodes)`
is closer to `S`. The proposed expanded kline is emitted as a `frame` event.

### Overfit Expansion — Remove Nodes

Identify nodes whose bits contribute to `N & ~S`. Remove those nodes from
the candidate. Construct the trimmed kline:

```
{S: [remaining_nodes]}
```

Construct a **companion kline** from the removed nodes:

```
{make_signature(removed_nodes): [removed_nodes]}
```

Both the trimmed kline and the companion kline are emitted as independent
`frame` events.

### Dual Expansion — Replace Nodes

When a kline is both underfitting and overfitting, perform a single atomic
replacement: swap a subset of overfit nodes for a subset that fills the
underfit gap. Emit the replacement kline and the companion kline from
removed nodes as independent `frame` events.

### Proposals Emitted per Expansion Type

| Expansion type | Proposals emitted                                                |
| -------------- | ---------------------------------------------------------------- |
| Added nodes    | The expanded kline                                               |
| Removed nodes  | The trimmed kline **and** the companion kline from removed nodes |
| Dual misfit    | The replacement kline **and** the companion kline from removed nodes |

Each proposal is an independent `frame` event. The agent ratifies (or
rejects) each one individually.

### Universal Constraint

Every signature generated during expansion must already exist in the model.
This guarantees no invention, no data loss, and ratifiability. When the
constraint cannot be satisfied, no proposal is emitted — the agent infers
scaffolding is needed from the absence of a `frame` event.

All expansion proposals require agent ratification via countersignature.

### S2 Klines from KScript

KScript's `=>` operator naturally produces S2 klines:

| KScript         | Compiled kline                        | Misfit type  |
| --------------- | ------------------------------------- | ------------ |
| `AB => A`       | `{AB: [A]}` — sig `A\|B`, nodes `A`   | Underfitting |
| `A => A B`      | `{A: [A, B]}` — sig `A`, nodes `A\|B` | Overfitting  |
| `WDMH => MHALL` | `{WDMH: [M, H, A, L, L]}`             | Dual misfit  |

Underfit klines act as **templates** (known concept, holes to fill).
Overfit klines act as **sequencers** (step-by-step structure under a single
goal signature). The cogitator fills templates and decomposes sequencers.

### Dependence on Structural Grounding

S2 expansion requires structural grounding for two reasons:

1. **Promotion after ratification** — when the agent countersigns an
   expansion proposal, all participating klines must be cascaded to LTM
   via `add_ltm` (not just the ratified kline), including the added/removed node
   groups and any S4 identity klines involved. `promote_participating` calls
   `add_ltm` in a loop for each participating kline.

2. **Frame richness** — the expansion search requires a model populated
   with the signatures it needs to find. Structural grounding ensures that
   frames hold S4–S1, giving the Cogitator more graph topology to traverse
   and more candidate signatures to match against.

The `promote_participating` function should be reviewed and made fit for
purpose — ensuring all participating klines are cascaded to LTM via
`add_ltm` calls.

### Exploration Depth

For the MVP, the Cogitator processes all yields from `model.expand()`
until S1 is found, at which point the work item stops, without filtering.
Future work may add exploration depth controls to limit how many expansion
proposals are generated per work item.

## S1 Callback

When the Cogitator discovers an S1 — either via the S1 fast-path (pre-routed)
or via boundary classification during expand() — it calls
`handler.on_s1(query, candidate)` on the CogitationHandler unconditionally.
The Agent implementation checks `is_s1(model, candidate)` as a structural
guard — if the candidate is structurally S1 (canonical or countersigned), it
calls `promote_participating(model, query, candidate)` to cascade all
participating STM klines to LTM via `add_ltm`. A frame event is always
published (unconditional) with significance `D_MAX - 1`. The `_run_work_item`
S1 branch delegates entirely to `on_s1`, keeping the dispatcher thin.

## Lifecycle

The Cogitator runs asynchronously. When the backlog has been empty for a
configurable timeout (default: **2 seconds**), the Cogitator emits a
`"done"` event so that subscribers can realign. The Cogitator does **not**
halt on timeout — it resets its idle timer and continues processing new
work items as they arrive. Each timeout expiry emits another `"done"` event.

The Cogitator can only be stopped explicitly via `join(timeout)`.

### Inter-Lesson Drain

See **@cogitator-drain spec** for the inter-lesson drain contract
(`drain()` blocks until the backlog is empty and the current work item
finishes; configurable timeout; thread-safe processing flag).

## Resolved Questions

### Cogitator Receives Pre-Routed Work Items

The Cogitator no longer retrieves candidates or expands graph context.
It receives a WorkItem and computes distance for that single pair.

**Rationale**: Separating routing (agent, fast) from graph expansion
(cogitator, slow) gives a clean workload split. Future iterations can
evolve the Cogitator to perform additional graph expansion and re-routing.

## Test Matrix

> These IDs are relocated from @agent spec and remain stable (cascade rule:
> spec IDs are never renumbered). They keep their original `AGT-` prefix
> for traceability to existing tests.

| ID     | Criterion                                                           | Origin ref |
| ------ | ------------------------------------------------------------------- | ---------- |
| AGT-29 | Countersignature discovery: S2 → S1 via countersignature in cogitation, klines cascaded to LTM | — |
| AGT-30 | Cogitator join: thread stops cleanly                                | — |
| AGT-31 | S2 submits work item: WorkItem queued with correct fields           | — |
| AGT-32 | All yields processed: every QC from `expand()` evaluated            | — |
| AGT-33 | S1 detection: high-significance QC triggers handler.on_s1          | — |
| AGT-34 | S2/S3 expansion: non-canonical QC triggers expansion proposals, proposals written to Frame | — |
| AGT-35 | Proposals at any significance: S2 and S3 proposals emitted as frame events | — |
| AGT-36 | Boundary S1 + structural check: participating klines cascaded to LTM via add_ltm | — |
| AGT-37 | Boundary S1 + structural S1: LTM cascade occurs                      | — |
| AGT-38 | S2 before S1: deferred S2 work items discarded, zero cogitator submissions | auto-tune/s1-batch-dedup |
| AGT-39 | Cogitator break-on-S1: on_s1 called exactly once, no expansion calls after | auto-tune/s1-batch-dedup |
| AGT-40 | S2/S3 event for already-satisfied entry is skipped (satisfaction guard) | auto-tune/direct-cogitation-push |
| AGT-41 | Lesson completion uses satisfaction count, not event count | auto-tune/direct-cogitation-push |
| AGT-42 | Lesson completion does not re-fire on post-completion cogitation events | auto-tune/direct-cogitation-push |

## Out of Scope

The following are explicitly **out of scope** for this spec:

- **Candidate retrieval and routing.** The agent retrieves candidates and
  routes them before submitting work items (@agent spec).
- **Significance computation.** Distance→significance inversion is internal
  to the model's `expand()` (@model spec).
- **Persistence format, tokenisation, model internals.**
- **The Trainer's LLM cogitator.** The reactive-decision LLM agent in
  `src/training/` is a distinct component (see @reactive-scaffolding spec,
  @reactive-delegation spec).

## Referenced By

- **Agent** (@agent spec) — owns and submits work items to the Cogitator,
  and is the primary `CogitationHandler` implementation.
- **Cogitator Drain** (@cogitator-drain spec) — extends the lifecycle with
  inter-lesson drain semantics.
- **Model** (@model spec) — provides `expand()`, `generate_expansions()`,
  and tiered writes consumed by the Cogitator.
