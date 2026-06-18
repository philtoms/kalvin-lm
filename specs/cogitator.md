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
- Provides `add_to_frame` / `add_to_ltm` for tiered writes of proposals and
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
  level:      str   # routing level: "S2" or "S3"
```

The agent submits one WorkItem per routed candidate (S2 or S3). The
Cogitator expands each WorkItem into a sequence of QueryCandidates via
`model.expand()`.

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

### Work Item Levels

Work items arrive routed as **S2** or **S3** only (see @agent spec,
§Routing). Routing does not produce S1 or S4: full node overlap is a
necessary but insufficient condition for S1 (true S1 is structural —
canonical composition or countersignature, established by `expand()` /
`is_s1()`), and identity klines are resolved on the agent's fast path
before any candidate is submitted.

### Work Item Processing

For every work item, the Cogitator expands the pair and classifies each
yield against the significance boundaries:

```
run_work_item(WorkItem(query, candidate, routing_level)):
  for qc in model.expand(query, candidate):
    if qc.significance >= s12:        # S1 — genuine exact match
      handler.on_s1(query, candidate)
      break
    elif qc.significance >= s34:      # S2 or S3
      process(qc)
    # S4 yields are skipped
```

An S1 (distance 1) discovered during expansion is a genuine structural
exact match and triggers `handler.on_s1()`, terminating the loop for that
pair. There is no routing-level S1 short-circuit.

### Work Item Processing

The Cogitator expands each WorkItem, processing all yields from
`model.expand()`:

```
process(QueryCandidate(query, candidate, significance)):
  # S2 expansion only
  if candidate is canonical:
    return                        # nothing to expand
  for proposal, companions in model.generate_expansions(candidate):
    model.add_to_frame(proposal)      # write proposal to Frame + STM
    emit frame event for proposal
    for companion in companions:
      model.add_to_frame(companion)   # write companion to Frame + STM
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
rejects) each one individually. An identity proposal or companion (empty
nodes or self-referential `{S: [S]}`) is not emitted (see Universal
Constraint below).

### Universal Constraint

Every signature generated during expansion must already exist in the model.
This guarantees no invention, no data loss, and ratifiability. When the
constraint cannot be satisfied, no proposal is emitted — the agent infers
scaffolding is needed from the absence of a `frame` event.

A second constraint governs proposal *shape*: **an expansion proposal must
not be identity**. Identity (@CONTEXT.md §Identity) is either empty nodes
`{S: []}` or self-referential `{S: [S]}`, and carries no decomposition
information — so it is never a valid *expansion* proposal. A single removed
node `n` would form the companion `{n: [n]}` (identity), which is dropped
rather than emitted; likewise any proposal that reduces to identity is
dropped. (Note: `{S: [S]}` *is* a legitimate kline state — it is identity —
but it is not something the expander should produce, since the expander's
purpose is to decompose.)

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
   via `add_to_ltm` (not just the ratified kline), including the added/removed node
   groups and any S4 identity klines involved. `promote_participating` calls
   `add_to_ltm` in a loop for each participating kline.

2. **Frame richness** — the expansion search requires a model populated
   with the signatures it needs to find. Structural grounding ensures that
   frames hold S4–S1, giving the Cogitator more graph topology to traverse
   and more candidate signatures to match against.

The `promote_participating` function should be reviewed and made fit for
purpose — ensuring all participating klines are cascaded to LTM via
`add_to_ltm` calls.

### Exploration Depth

For the MVP, the Cogitator processes all yields from `model.expand()`
until S1 is found, at which point the work item stops, without filtering.
Future work may add exploration depth controls to limit how many expansion
proposals are generated per work item.

## S1 Callback

When the Cogitator discovers an S1 — via boundary classification during
expand() (a terminal distance-1 yield) — it calls
`handler.on_s1(query, candidate)` on the CogitationHandler. The Agent
implementation checks `is_s1(model, candidate)` as a structural guard — if
the candidate is structurally S1 (canonical or countersigned), it calls
`promote_participating(model, query, candidate)` to cascade all
participating STM klines to LTM via `add_to_ltm`. A frame event is always
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

The Cogitator is a background thread that processes S2/S3 work items
asynchronously. When a lesson triggers slow-path cogitation, work items may
still be processing when the next lesson begins. These late-arriving events
would be processed with the new lesson's reactor state, consuming its
reactive budget and corrupting entry satisfaction tracking. The drain
contract prevents this cross-lesson spillover.

- **DRN-1.** Before submitting each lesson, the Trainer MUST drain the
  Cogitator backlog. A drain request is sent via the bus to the KAgent
  adapter. The adapter calls `Cogitator.drain()` which blocks until the
  backlog is empty and the current work item finishes.
- **DRN-2.** While a drain is pending, the Trainer MUST NOT submit lesson
  entries. Lesson compilation and submission are deferred until the
  `drained` response is received from the adapter.
- **DRN-3.** When the Cogitator backlog is empty and no work item is
  processing, `drain()` MUST return immediately (negligible overhead for
  lessons that don't trigger slow-path cogitation).
- **DRN-4.** The drain operation has a configurable timeout (default 30
  seconds). If the drain times out, the adapter responds with a `drained`
  message and the Trainer proceeds. This prevents indefinite blocking.
- **DRN-5.** The Cogitator MUST track a `_processing` flag (set before a
  work item begins, cleared after it finishes). `drain()` waits for both
  the backlog to be empty AND the processing flag to be clear.

## Reactive Scaffolding Submission

When the Trainer's LLM agent generates reactive scaffolding in response to
S2/S3 events, the scaffolding must be compiled, validated, and submitted to
Kalvin via the harness bus. The Cogitator sanitises LLM output before
compilation and decompiles klines into human-readable KScript so the LLM
has the name context it needs.

### RS-1 — System prompt uses correct KScript syntax

The cogitation system prompt documents the syntax the KScript lexer
actually supports:

- Identifiers: uppercase letters only (A–Z); no hex literals.
- Operators: `==` (countersign), `=>` (canonize), `=` (undersign),
  `>` (relationship). No `~>`, `<-`, or `->`.
- Comments: parenthesised `(...)` only; no `#` comments.

### RS-2 — Misfit summaries are decompiled to KScript

The cogitate adapter decompiles the event query and proposal klines to
human-readable KScript (e.g. `M > H`) before passing them as
`expectation_summary` and `proposal_summary` to the `MisfitInfo`. If
decompilation fails, the adapter falls back to the raw `repr`.

### RS-3 — Hash-comment stripping before compilation

`_strip_hash_comments()` removes lines starting with `#` from scaffolding
before compilation — a defensive measure for LLMs trained on Python-style
comments. If sanitisation removes all content (scaffolding was entirely
comments), the Cogitator returns `scaffolding=None` without attempting
compilation.

### RS-4 — Submission log line

The Reactor logs `"submitted reactive scaffolding"` at INFO level after
successfully sending reactive scaffolding to the trainee role, giving an
observable end-to-end confirmation.

### Behavioural rules

6. The Cogitator MUST sanitise scaffolding source by removing `#`-prefixed
   comment lines before attempting compilation.
7. If sanitisation removes all content, the Cogitator MUST return
   `scaffolding=None` without attempting compilation.
8. The system prompt MUST only document operators the KScript lexer
   supports.
9. Misfit summaries passed to the LLM MUST use decompiled KScript, not hex
   repr; on decompilation failure they fall back to repr.
10. The Reactor MUST log "submitted reactive scaffolding" after sending
    scaffolding to the trainee role.

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
| AGT-36 | Boundary S1 + structural check: participating klines cascaded to LTM via add_to_ltm | — |
| AGT-37 | Boundary S1 + structural S1: LTM cascade occurs                      | — |
| AGT-38 | S2 before S1: deferred S2 work items discarded, zero cogitator submissions | — |
| AGT-39 | Cogitator break-on-S1: on_s1 called exactly once, no expansion calls after | — |
| AGT-40 | S2/S3 event for already-satisfied entry is skipped (satisfaction guard) | — |
| AGT-41 | Lesson completion uses satisfaction count, not event count | — |
| AGT-42 | Lesson completion does not re-fire on post-completion cogitation events | — |
| AGT-43 | Drain sent before each lesson, even when no S2/S3 expected (DRN-1)              | — |
| AGT-44 | Lesson entries not submitted until `drained` response received (DRN-2)         | — |
| AGT-45 | Empty-backlog drain completes in <10ms (DRN-3)                                  | — |
| AGT-46 | Drain timeout returns False but does not stop the thread (DRN-4)               | — |
| AGT-47 | Processing flag guards against premature drain return (DRN-5)                  | — |
| AGT-48 | Cross-lesson spillover eliminated: lesson N events don't affect lesson N+1 budget | — |
| AGT-49 | System prompt contains no hex literal syntax (RS-1)                            | — |
| AGT-50 | System prompt contains no invalid operators `~>`, `<-`, `->` (RS-1)            | — |
| AGT-51 | `_strip_hash_comments` removes `#` lines, keeps valid KScript (RS-3)           | — |
| AGT-52 | `_strip_hash_comments` returns empty for all-comment input (RS-3)              | — |
| AGT-53 | Cogitator returns None when scaffolding is all comments (RS-3)                  | — |
| AGT-54 | Cogitate adapter decompiles query kline to KScript (RS-2)                      | — |
| AGT-55 | Cogitate adapter decompiles proposal kline to KScript (RS-2)                   | — |
| AGT-56 | Cogitate adapter falls back to repr on decompilation failure (RS-2)            | — |
| AGT-57 | Reactor logs "submitted reactive scaffolding" after bus send (RS-4)            | — |

## Out of Scope

The following are explicitly **out of scope** for this spec:

- **Candidate retrieval and routing.** The agent retrieves candidates and
  routes them before submitting work items (@agent spec).
- **Significance computation.** Distance→significance inversion is internal
  to the model's `expand()` (@model spec).
- **Persistence format, tokenisation, model internals.**
- **The Trainer's LLM cogitator's reactive-scaffolding pipeline**
  (sanitisation, decompilation, submission logging) lives in this spec's
  §Reactive Scaffolding Submission. The reactive-decision *delegation*
  (whether the Cogitator or the supervisor makes the decision) is a
  distinct component (@reactive-delegation spec).

## Referenced By

- **Agent** (@agent spec) — owns and submits work items to the Cogitator,
  and is the primary `CogitationHandler` implementation.
- **Cogitator Drain** — see §Lifecycle › Inter-Lesson Drain in this spec.
- **Reactive Scaffolding Submission** — see §Reactive Scaffolding Submission
  in this spec.
- **Model** (@model spec) — provides `expand()`, `generate_expansions()`,
  and tiered writes consumed by the Cogitator.
