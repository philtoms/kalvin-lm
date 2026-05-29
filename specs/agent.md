# Agent Specification

## Overview

The Agent is the orchestrator of the Kalvin rationalisation pipeline. It
receives input, encodes it into Klines, retrieves candidates from the Model,
routes each query-candidate pair, and integrates results back into the
knowledge graph.

The Agent's principal function is **rationalisation**: determining how a new
Kline relates to existing knowledge and deciding what action to take. The
pipeline is split into a **fast path** (routing — no model calls) and a
**slow path** (cogitation — background graph expansion).

All computational detail is delegated — encoding to the Tokenizer, storage
and lookup to the Model, graph expansion to the Model's expand API.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Kline (@kline spec)

- A Kline is an identified, ordered sequence of zero or more nodes.
- Nodes are opaque uint64 values.
- Literal klines (all nodes are literal tokens) represent exact tokens;
  non-literal klines are composed structures.

### Signature (@signature spec)

- Provides `make_signature(nodes) → KSig` (OR-reduction over all nodes,
  with bit 0 as the literal-content flag).
- Provides bitwise AND matching for candidate retrieval.
- Uses `is_literal` from the @kline spec internally.

### Tokenizer (@tokenizer spec)

- Provides `encode(text) → list[int]` and `decode(nodes) → str`.
- Nodes returned by `encode` are fully typed — the tokenizer handles all
  internal encoding details including type prefix combination.

### Model (@model spec)

- Stores, indexes, and retrieves Klines.
- Provides candidate retrieval via `find`, `find_all`, `query`, `where`.
- Provides `find_by_nodes` for transitive grounding via nodes signature.
- Provides `promote` and `promote_all` for persisting Klines to the base.
- Provides `expand(Q, C)` generator for graph expansion yielding
  connotations and terminal significance.
- Provides `QueryCandidate` named tuple with `.significance` field
  (pre-computed by the model).
- Provides constants `D_MAX` and `MASK64` for significance values.
- Provides `is_countersigned(kline)` to check if a kline is countersigned by any kline in the model.
- Manages a three-tier memory internally (STM → Frame → Base). The agent
  sees a single Model API; tiering is invisible to the agent.
- The model decides how and where Klines are stored. The agent is
  responsible for calling model operations; the model is responsible
  for managing its internal memory tiers.

## Definition

An Agent consists of:

| Component | Type      | Description                            |
| --------- | --------- | -------------------------------------- |
| tokenizer | Tokenizer | Encodes text ↔ nodes.                  |
| model     | Model     | Layered knowledge graph (STM → Frame → |
|           |           | Base). Agent sees a single Model API.  |
| cogitator | Cogitator | Background processor for S2/S3 work    |
|           |           | items.                                 |

## Construction

```
Agent(
    tokenizer  = None,   # defaults to Mod32
    model      = None,   # defaults to empty Model
)
```

- `tokenizer` — a Tokenizer instance. Defaults to Mod32.
- `model` — a Model instance serving as the base knowledge graph. Defaults
  to an empty Model.

A newly constructed Agent contains zero Klines in its model. The Cogitator
is created internally and starts its background thread immediately.

## Rationalisation

Rationalisation is the process of integrating a Kline into the knowledge
graph. It proceeds in phases with a fast/slow split:

```
Rationalise(Q):
  ┌─── FAST PATH ────────────────────────────────────────────┐
  │ 1. PREPARE                                               │
  │    Assign signature if missing.                           │
  ├───────────────────────────────────────────────────────────┤
  │ 2. GROUND CHECK                                          │
  │    Does Q already exist in the model?                     │
  │    → Yes: emit "ground" event, return True.              │
  ├───────────────────────────────────────────────────────────┤
  │ 3. ASSESS                                                │
  │    Evaluate Q's structural grounding:                     │
  │    → Unsigned (no nodes): emit "frame" S4, return True.  │
  │    → All-literal: emit "frame" S1, return True.          │
  │    → Self-grounded: emit "frame" S1, return True.        │
  │    → Countersigned: emit "frame" S1, return True.        │
  ├───────────────────────────────────────────────────────────┤
  │ 4. RETRIEVE CANDIDATES                                    │
  │    candidates = model.where(Q.signature)                  │
  │    → No candidates: emit "frame" S4, return True.        │
  ├───────────────────────────────────────────────────────────┤
  │ 5. ROUTE EACH CANDIDATE                                   │
  │    For each candidate Cᵢ:                                │
  │      level = route(Q, Cᵢ)   ← node membership, no model │
  │      S1 → promote, emit "frame", return True (done)      │
  │      S2 → submit WorkItem(Q, Cᵢ) to cogitator            │
  │      S3 → submit WorkItem(Q, Cᵢ) to cogitator            │
  │    return False                                           │
  └───────────────────────────────────────────────────────────┘

  ┌─── SLOW PATH (Cogitator, background thread) ────────────┐
  │ For each WorkItem(Q, C):                                  │
  │   for qc in model.expand(Q, C):                             │
  │     process(qc)                                              │
  │                                                              │
  │ process(QueryCandidate(query, candidate, significance)):     │
  │   S2 expansion: reshape misfit klines toward canonical,     │
  │   emit proposals for agent ratification.                    │
  └───────────────────────────────────────────────────────────────┘
```

### Phase 1: Prepare

Ensure the Kline has a signature. If `Q.signature == 0`, compute it:

```
Q.signature = make_signature(Q.nodes)   # @signature spec
```

### Phase 2: Ground Check

Test whether an equal Kline already exists in the model. Kline equality
is defined in the @kline spec (same signature, same node sequence).

If grounded:

- Emit a `"ground"` event.
- Return `True` (significant).
- No further processing.

This prevents infinite recursion and avoids re-processing known knowledge.

### Phase 3: Assess

Structural assessment determines whether Q can be fast-tracked without
candidate retrieval or significance computation.

**Unsigned**: If Q has zero nodes, it carries no information. Emit a
`"frame"` event at S4. Return `True`.

**Canonical — all-literal**: If every node in Q is a literal (per
`is_literal` from the @kline spec), Q is a pure token sequence. Because
`make_signature` contributes bit 0 for each literal node, Q's signature
is `1` — a valid canonical signature. Emit a `"frame"` event at S1.
Return `True`.

**Canonical — self-grounded**: If `Q.signature == make_signature(Q.nodes)`
(as defined in the @signature spec) and every node that could resolve does
resolve in the model (exists as a Kline signature), Q is fully grounded.
The signature faithfully represents the nodes — nothing is missing and
nothing is extraneous. Emit a `"frame"` event at S1. Return `True`.

**Countersigned — ratified**: If the model contains a kline whose signature
equals `make_signature(Q.nodes)` and whose sole node equals `Q.signature`,
then Q is countersigned — another kline vouches for it structurally. This is
the **ratification** check. It runs in the fast lane before candidates are
retrieved, because countersignature is a structural property of Q and the
model, not dependent on any particular candidate. Emit a `"frame"` event at
S1. Return `True`.

If none of the above, proceed to candidate retrieval.

### Phase 4: Retrieve Candidates

Candidates are Klines from the model that are potentially significant to Q.
Candidate retrieval uses signature overlap:

```
candidates = model.where(k => (k.signature & Q.signature) != 0)
```

A Kline whose signature shares any set bit with Q's signature is a
candidate. This is a necessary (but not sufficient) condition for
significance — bitwise AND matching pre-filters the model before the more
expensive routing runs.

If no candidates are found, Q is novel. Emit a `"frame"` S4 event,
add Q to the model, and return `True`.

### Phase 5: Route Each Candidate

Add Q to the model. Then for each candidate, perform routing — a fast
node-membership test with no model calls:

#### Routing

```
route(Q, C):
  if Q has no nodes:   return "S4"
  match_count = number of Q.nodes that exist in C.nodes
  if match_count == len(Q.nodes):  return "S1"
  if match_count > 0:              return "S2"
  else:                            return "S3"
```

Routing is a pure function. It checks whether each query node value appears
in the candidate's node sequence. No model function is called.

#### Per-Candidate Action

| Route | Action | Model call? |
| ----- | ------ | ----------- |
| S1    | Promote Q, emit `"frame"` S1, return `True` | No |
| S2    | Submit `WorkItem(Q, C)` to Cogitator | No |
| S3    | Submit `WorkItem(Q, C)` to Cogitator | No |

**S1 short-circuits**: the first candidate that routes as S1 terminates the
loop immediately. No further candidates are routed. No distance is computed.

If all candidates route as S2 or S3, each becomes an individual work item
for the Cogitator. Return `False`.

## Work Items

A WorkItem is a single query-candidate pair queued for background
processing:

```
WorkItem:
  query:      KLine
  candidate:  KLine
```

The agent submits one WorkItem per routed S2/S3 candidate. The Cogitator
expands each WorkItem into a sequence of QueryCandidates via `model.expand()`.

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

The model computes significance internally; callers use `.significance`
directly without any inversion.

## Cogitation

Cogitation is the background processing of individual query-candidate work
items. The Cogitator receives pre-routed work items, expands each through
`model.expand()` to discover connotations, processes all yields, and emits proposals at any significance level.

### Processing

The Cogitator processes all yields from `model.expand()` without filtering.
Every connotation discovered during graph expansion is evaluated. Proposals
can be emitted at any significance level — the agent evaluates them against
expectations regardless of their computed significance.

#### Significance Boundaries

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

#### All Yields Processed

The Cogitator processes all connotations yielded by `expand()`. There is no
truncation or early stopping — every discovered relationship is evaluated.

### Cogitator

```
Cogitator:
  model:      Model
  event_bus:  EventBus
  handler:    CogitationHandler
  timeout:    float (default 2.0)
  backlog:    queue of WorkItem
  thread:     background
```

The Cogitator runs on a background daemon thread. It pulls work items from
the backlog and processes each one.

### CogitationHandler

CogitationHandler is a `@runtime_checkable` Protocol defining the seam
between the Cogitator and its consumers. The Agent is the primary
implementation.

```
CogitationHandler:
  on_s1(query, candidate)                      — called when an S1 (exact) result is discovered
  on_expansion(query, proposal, significance)  — called when an expansion proposal is generated (S2/S3)
```

### Work Item Processing

The Cogitator expands each WorkItem, processing all yields from
`model.expand()`:

```
run_work_item(WorkItem(query, candidate)):
  for qc in model.expand(query, candidate):
    if qc.significance >= s12:
      handler.on_s1(query, candidate)   # S1: delegate to handler
    else:
      process(qc)                 # S2/S3: expansion check

process(QueryCandidate(query, candidate, significance)):
  # S2 expansion only — ratification handled upstream in rationalise()
  if candidate is canonical:
    return                        # nothing to expand
  for proposal, companions in model.generate_expansions(candidate):
    emit frame event for proposal
    emit frame events for companions
```

All yields are processed without filtering. Raw significance values are
never mutated. Proposals can be emitted at any significance level.

### S2 Expansion

When countersignature fails for an S2 result, the Cogitator attempts to
**expand** the candidate kline toward canonical status by reshaping its
nodes to match its signature. This is the mechanism for self-directed
study — the Cogitator works through partial understanding and emits
proposals for the agent to ratify.

The conceptual description of S2 expansion (misfit types, templates,
sequencers, guarantees) is defined in the [origin document]
(`docs/kalvin-origin.md`, Study → S2 Expansion). This section specifies
the implementation detail.

#### Misfit Classification

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

#### Underfit Expansion — Add Nodes

Compute `gap = S & ~N`. Search the model for klines whose signatures
contribute to the gap. Construct the expanded kline:

```
{S: [original_nodes + addition_nodes]}
```

Verify the expanded kline moves toward canonical: `make_signature(expanded_nodes)`
is closer to `S`. The proposed expanded kline is emitted as a `frame` event.

#### Overfit Expansion — Remove Nodes

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

#### Dual Expansion — Replace Nodes

When a kline is both underfitting and overfitting, perform a single atomic
replacement: swap a subset of overfit nodes for a subset that fills the
underfit gap. Emit the replacement kline and the companion kline from
removed nodes as independent `frame` events.

#### Proposals Emitted per Expansion Type

| Expansion type | Proposals emitted                                                |
| -------------- | ---------------------------------------------------------------- |
| Added nodes    | The expanded kline                                               |
| Removed nodes  | The trimmed kline **and** the companion kline from removed nodes |
| Dual misfit    | The replacement kline **and** the companion kline from removed nodes |

Each proposal is an independent `frame` event. The agent ratifies (or
rejects) each one individually.

#### Universal Constraint

Every signature generated during expansion must already exist in the model.
This guarantees no invention, no data loss, and ratifiability. When the
constraint cannot be satisfied, no proposal is emitted — the agent infers
scaffolding is needed from the absence of a `frame` event.

All expansion proposals require agent ratification via countersignature.

#### S2 Klines from KScript

KScript's `=>` operator naturally produces S2 klines:

| KScript         | Compiled kline                        | Misfit type  |
| --------------- | ------------------------------------- | ------------ |
| `AB => A`       | `{AB: [A]}` — sig `A\|B`, nodes `A`   | Underfitting |
| `A => A B`      | `{A: [A, B]}` — sig `A`, nodes `A\|B` | Overfitting  |
| `WDMH => MHALL` | `{WDMH: [M, H, A, L, L]}`             | Dual misfit  |

Underfit klines act as **templates** (known concept, holes to fill).
Overfit klines act as **sequencers** (step-by-step structure under a single
goal signature). The cogitator fills templates and decomposes sequencers.

#### Dependence on Structural Grounding

S2 expansion requires structural grounding for two reasons:

1. **Promotion after ratification** — when the agent countersigns an
   expansion proposal, all participating STM klines must be promoted to
   frame (not just the ratified kline), including the added/removed node
   groups and any S4 identity klines involved.

2. **Frame richness** — the expansion search requires a model populated
   with the signatures it needs to find. Structural grounding ensures that
   frames hold S4–S1, giving the Cogitator more graph topology to traverse
   and more candidate signatures to match against.

#### Exploration Depth

For the MVP, the Cogitator processes all yields from `model.expand()`
without filtering. Future work may add exploration depth controls to limit
how many expansion proposals are generated per work item.

### S1 Callback

When the Cogitator discovers an S1 via boundary classification, it calls
`handler.on_s1(query, candidate)` on the CogitationHandler. The Agent
implementation of this method triggers `agent.rationalise(query)`,
re-rationalising on the cogitation thread.

### Cogitation Lifecycle

The Cogitator runs asynchronously. When the backlog has been empty for a
configurable timeout (default: **2 seconds**), the Cogitator emits a
`"done"` event so that subscribers can realign. The Cogitator does **not**
halt on timeout — it resets its idle timer and continues processing new
work items as they arrive. Each timeout expiry emits another `"done"` event.

The Cogitator can only be stopped explicitly via `join(timeout)`.

## Events

The Agent publishes events during rationalisation for observers to consume.

### Event Types

| Kind     | Trigger                              | Significance |
| -------- | ------------------------------------ | ------------ |
| `ground` | Kline already exists in model        | S1           |
| `frame`  | Kline integrated (new or confirmed)  | S1–S4        |
| `done`   | Cogitation backlog empty for timeout | 0            |

### Event Structure

```
RationaliseEvent:
  kind:          str       # "ground", "frame", "done"
  query:         Kline     # The Kline being rationalised
  proposal:      Kline     # The matching or resulting Kline
  significance:  int       # Significance value
```

Subscribers receive events synchronously in publication order.

## Resolved Questions

### 1. Routing in Agent vs Significance Module

Routing (node-membership classification) is now performed by the agent
directly. The significance computation (distance→significance inversion) is
performed by the model's `expand()` method. This eliminates the
`significance_pipeline`, `compute_significance`, and `SignificanceResult`
abstractions.

**Rationale**: Routing is a fast, model-free operation that directly
determines agent control flow. Significance inversion is now internalized in
the model, keeping the cogitation path simple — it consumes
`QueryCandidate.significance` directly.

### 2. No "Best Candidate" Selection

The previous architecture computed significance for all candidates then
selected the best result. The current architecture routes each candidate
independently. The first S1 terminates immediately. All S2/S3 candidates
are submitted as individual work items.

**Rationale**: Best-candidate selection forced full computation before the
agent could act. The per-candidate routing model enables short-circuit on
S1 and parallel processing of S2/S3.

### 3. Cogitator Receives Pre-Routed Work Items

The Cogitator no longer retrieves candidates or expands graph context.
It receives a WorkItem and computes distance
for that single pair.

**Rationale**: Separating routing (agent, fast) from graph expansion
(cogitator, slow) gives a clean workload split. Future iterations can
evolve the Cogitator to perform additional graph expansion and re-routing.

## Test Matrix

### Routing

| ID    | Criterion                                                     | Origin ref |
| ----- | ------------------------------------------------------------- | ---------- |
| AGT-1 | All nodes match → returns "S1"                                 | — |
| AGT-2 | Some nodes match → returns "S2"                                | — |
| AGT-3 | No nodes match → returns "S3"                                  | — |
| AGT-4 | Empty query → returns "S4"                                     | — |
| AGT-5 | Single node match → returns "S1"                               | — |
| AGT-6 | Routing independent of signature: only candidate nodes matter  | — |

### Rationalisation — Phase 1: Prepare

| ID     | Criterion                                                   | Origin ref |
| ------ | ----------------------------------------------------------- | ---------- |
| AGT-7  | Signature assigned: KLine with sig=0 gets `make_signature(nodes)` | — |
| AGT-8  | Signature preserved: existing non-zero sig unchanged        | — |

### Rationalisation — Phase 2: Ground Check

| ID     | Criterion                                               | Origin ref |
| ------ | ------------------------------------------------------- | ---------- |
| AGT-9  | First rationalise: returns True, adds to model           | — |
| AGT-10 | Duplicate rationalise: returns True, emits "ground" event | — |
| AGT-11 | Different sig same nodes: not a ground (different KLine) | — |

### Rationalisation — Phase 3: Assess

| ID     | Criterion                                                  | Origin ref |
| ------ | ---------------------------------------------------------- | ---------- |
| AGT-12 | Unsigned (no nodes): returns True, emits "frame" S4         | — |
| AGT-13 | All-literal: returns True, emits "frame" S1                | — |
| AGT-14 | Self-grounded canonical: returns True when all nodes resolve | — |
| AGT-15 | Not self-grounded: falls through to Phase 4               | — |

### Rationalisation — Phase 4: Retrieve Candidates

| ID     | Criterion                                         | Origin ref |
| ------ | ------------------------------------------------- | ---------- |
| AGT-16 | No candidates: returns True (S4 novel)             | — |
| AGT-17 | Candidates found: proceeds to routing              | — |

### Rationalisation — Phase 5: Route Each Candidate

| ID     | Criterion                                                           | Origin ref |
| ------ | ------------------------------------------------------------------- | ---------- |
| AGT-18 | First candidate S1: returns True, promotes, no further routing      | — |
| AGT-19 | Later candidate S1: earlier S2/S3 routed, then S1 terminates loop  | — |
| AGT-20 | All S2: returns False, all submitted as WorkItems                   | — |
| AGT-21 | All S3: returns False, all submitted as WorkItems                   | — |
| AGT-22 | S1 short-circuits: `model.expand` never called when S1 found        | — |

### Events

| ID     | Criterion                                  | Origin ref |
| ------ | ------------------------------------------ | ---------- |
| AGT-23 | Subscribe and publish: callback receives event | — |
| AGT-24 | Multiple subscribers: all receive event    | — |
| AGT-25 | Event fields correct: kind, query, proposal, significance | — |
| AGT-26 | Thread safety: publish from another thread | — |
| AGT-27 | Empty bus: no crash on publish with no subscribers | — |
| AGT-28 | Event delivery: all events received in order | — |

### Cogitation

| ID     | Criterion                                                           | Origin ref |
| ------ | ------------------------------------------------------------------- | ---------- |
| AGT-29 | Countersignature discovery: S2 → S1 via countersignature in cogitation | — |
| AGT-30 | Cogitator join: thread stops cleanly                                | — |
| AGT-31 | S2 submits work item: WorkItem queued with correct fields           | — |
| AGT-32 | All yields processed: every QC from `expand()` evaluated            | — |
| AGT-33 | S1 detection: high-significance QC triggers handler.on_s1          | — |
| AGT-34 | S2/S3 expansion: non-canonical QC triggers expansion proposals      | — |
| AGT-35 | Proposals at any significance: S2 and S3 proposals emitted as frame events | — |
| AGT-36 | Boundary S1 + structural check: promotion only on structural S1     | — |
| AGT-37 | Boundary S1 + structural S1: promotion occurs                       | — |

### Serialization

| ID     | Criterion                                         | Origin ref |
| ------ | ------------------------------------------------- | ---------- |
| AGT-38 | JSON round-trip: save/load preserves KLines        | — |
| AGT-39 | Binary round-trip: save/load preserves KLines      | — |
| AGT-40 | Empty agent: serializes and deserializes correctly | — |

## Open Questions

### 1. Candidate Retrieval in the Model

The model spec's `where(predicate)` performs a linear scan, which may be
too slow for large models.

**Recommendation:** Add a `model.candidates_for(signature)` method that
returns all Klines whose signatures share at least one bit with the query
signature. The model can implement this efficiently using an inverted
bit-to-signature index.

### 2. ~~Cogitation Evolution~~ — Resolved

The Cogitator processes all yields from `model.expand()` without filtering.
Proposals can be emitted at any significance level. See §Cogitation.

### 3. Grounding Assessment Formalisation

This spec defines grounding checks (self-grounded, all-literal, unsigned)
as fast-path optimisations. An alternative design would route everything
through routing, with the model's `is_s1` function handling these cases
internally. `is_s1` now performs structural grounding (canonical or
countersigned), which subsumes the earlier resolve-only check.

**Recommendation:** Keep as agent-level fast paths. Grounding is about
structural properties of a single Kline, not about comparison between two
Klines.

## What an Agent is Not

The following are explicitly **out of scope** for this spec:

- **Graph expansion.** The Agent delegates graph expansion to
  the Model's `expand()` generator. The Cogitator invokes it in the background.
- **Tokenization internals.** How text is segmented into tokens is defined
  in the @tokenizer spec. The Agent consumes the tokenizer's output.
- **Model internals.** How Klines are stored, indexed, and retrieved is
  defined in the @model spec. The Agent uses the model's API.
- **Persistence format.** Binary and JSON serialization formats are
  implementation-level concerns.
- **Thread management.** How cogitation runs asynchronously (threads,
  async/await, etc.) is an implementation concern.
- **Debug metadata.** Labels, source text, timestamps, or other diagnostic
  data attached to Klines.

## Referenced By

- **Model** (@model spec) — the Agent stores and retrieves Klines, and
  calls the model's expand and countersignature API. The model computes
  significance internally.
- **Signature** (@signature spec) — the Agent creates signatures during
  the prepare phase and uses bitwise AND matching for candidate retrieval.
- **Tokenizer** (@tokenizer spec) — the Agent uses the tokenizer for
  encoding and decoding.
- **Kline** (@kline spec) — the Agent constructs, compares, and stores
  Klines.
