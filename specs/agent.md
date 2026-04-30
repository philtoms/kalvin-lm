# Agent Specification

## Overview

The Agent is the orchestrator of the Kalvin rationalisation pipeline. It
receives input, encodes it into Klines, retrieves candidates from the Model,
routes each query-candidate pair, and integrates results back into the
knowledge graph.

The Agent's principal function is **rationalisation**: determining how a new
Kline relates to existing knowledge and deciding what action to take. The
pipeline is split into a **fast path** (routing — no model calls) and a
**slow path** (cogitation — background distance computation).

All computational detail is delegated — encoding to the Tokenizer, storage
and lookup to the Model, distance calculation to the Model's distance API.

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
- Depends on the tokenizer's `is_literal` function.

### Tokenizer (@tokenizer spec)

- Provides `encode(text) → list[int]` and `decode(nodes) → str`.
- Provides `is_literal(node) → bool`.
- Nodes returned by `encode` are fully typed — the tokenizer handles all
  internal encoding details including type prefix combination.

### Model (@model spec)

- Stores, indexes, and retrieves Klines.
- Provides candidate retrieval via `find`, `find_all`, `query`, `where`.
- Provides `find_by_nodes` for transitive grounding via nodes signature.
- Provides `promote` and `promote_all` for persisting Klines to the base.
- Provides `distance(Q, C, level)` for packed distance computation.
- Provides `is_countersigned(Q, C)` for mutual-reference detection.
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
  ├───────────────────────────────────────────────────────────┤
  │ 4. RETRIEVE CANDIDATES                                    │
  │    candidates = model.where(Q.signature)                  │
  │    → No candidates: emit "frame" S4, return True.        │
  ├───────────────────────────────────────────────────────────┤
  │ 5. ROUTE EACH CANDIDATE                                   │
  │    For each candidate Cᵢ:                                │
  │      level = route(Q, Cᵢ)   ← node membership, no model │
  │      S1 → promote, emit "frame", return True (done)      │
  │      S2 → submit WorkItem(Q, Cᵢ, "S2") to cogitator     │
  │      S3 → submit WorkItem(Q, Cᵢ, "S3") to cogitator     │
  │    return False                                           │
  └───────────────────────────────────────────────────────────┘

  ┌─── SLOW PATH (Cogitator, background thread) ────────────┐
  │ For each WorkItem(Q, C, level):                          │
  │   distance = model.distance(Q, C, level)                 │
  │   significance = ~distance                               │
  │   if model.is_countersigned(Q, C):                       │
  │     add C to model                                       │
  │     re-rationalise Q                                     │
  └───────────────────────────────────────────────────────────┘
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
`tokenizer.is_literal`), Q is a pure token sequence. Because
`make_signature` contributes bit 0 for each literal node, Q's signature
is `1` — a valid canonical signature. Emit a `"frame"` event at S1.
Return `True`.

**Canonical — self-grounded**: If `Q.signature == make_signature(Q.nodes)`
(as defined in the @signature spec) and every node that could resolve does
resolve in the model (exists as a Kline signature), Q is fully grounded.
The signature faithfully represents the nodes — nothing is missing and
nothing is extraneous. Emit a `"frame"` event at S1. Return `True`.

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
| S2    | Submit `WorkItem(Q, C, "S2")` to Cogitator | No |
| S3    | Submit `WorkItem(Q, C, "S3")` to Cogitator | No |

**S1 short-circuits**: the first candidate that routes as S1 terminates the
loop immediately. No further candidates are routed. No distance is computed.

If all candidates route as S2 or S3, each becomes an individual work item
for the Cogitator. Return `False`.

## Work Items

A WorkItem is a single query-candidate-level tuple queued for background
processing:

```
WorkItem:
  query:      KLine
  candidate:  KLine
  level:      str    # "S2" or "S3"
```

The agent submits one WorkItem per routed S2/S3 candidate. The Cogitator
processes each independently.

## Cogitation

Cogitation is the background processing of individual query-candidate work
items. The Cogitator receives pre-routed work items, computes deep
significance via model distance, and checks for countersignature.

### Cogitator

```
Cogitator:
  model:      Model
  event_bus:  EventBus
  on_s1:      callback(query, candidate)
  timeout:    float (default 2.0)
  backlog:    queue of WorkItem
  thread:     background
```

The Cogitator runs on a background daemon thread. It pulls work items from
the backlog and processes each one.

### Work Item Processing

```
process(WorkItem(query, candidate, level)):
  1. distance = model.distance(query, candidate, level)
  2. significance = (~distance) & MASK64
  3. if model.is_countersigned(query, candidate):
       model.add(candidate)
       on_s1(query, candidate)    # triggers re-rationalisation
```

The MVP preserves the routed level (S2 or S3) without re-routing based on
the computed significance. Countersignature is the only mechanism that can
promote to S1 during cogitation.

### S1 Callback

When the Cogitator discovers an S1 via countersignature, it calls the
`on_s1` callback, which triggers `agent.rationalise(query)` on the agent.
This re-rationalisation runs on the cogitation thread.

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
  value:         Kline     # The matching or resulting Kline
  significance:  int       # Significance value
```

Subscribers receive events synchronously in publication order.

## Resolved Questions

### 1. Routing in Agent vs Significance Module

Routing (node-membership classification) is now performed by the agent
directly. The significance module is reduced to constants (`D_MAX`,
`MASK64`). This eliminates the `significance_pipeline`,
`compute_significance`, and `SignificanceResult` abstractions.

**Rationale**: Routing is a fast, model-free operation that directly
determines agent control flow. Wrapping it in a separate pipeline added
indirection without benefit. Distance computation is now only invoked in
the Cogitator's slow path.

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
It receives a WorkItem with a pre-routed level and computes distance
for that single pair.

**Rationale**: Separating routing (agent, fast) from distance computation
(cogitator, slow) gives a clean workload split. Future iterations can
evolve the Cogitator to perform additional graph expansion and re-routing.

## Open Questions

### 1. Candidate Retrieval in the Model

The model spec's `where(predicate)` performs a linear scan, which may be
too slow for large models.

**Recommendation:** Add a `model.candidates_for(signature)` method that
returns all Klines whose signatures share at least one bit with the query
signature. The model can implement this efficiently using an inverted
bit-to-signature index.

### 2. Cogitation Evolution

The MVP Cogitator processes work items individually with no graph expansion
or pass tracking. Future iterations may add:

- Graph expansion via `model.query(depth=D_cogitate)` to find additional
  candidates for each work item.
- Pass tracking to limit re-processing of the same query.
- Significance-based re-routing (using the computed distance to adjust the
  level).

### 3. Grounding Assessment Formalisation

This spec defines grounding checks (self-grounded, all-literal, unsigned)
as fast-path optimisations. An alternative design would route everything
through routing, with the model's `is_s1` function handling these cases
internally.

**Recommendation:** Keep as agent-level fast paths. Grounding is about
structural properties of a single Kline, not about comparison between two
Klines.

## What an Agent is Not

The following are explicitly **out of scope** for this spec:

- **Distance computation.** The Agent delegates distance calculation to
  the Model's `distance()` API. The Cogitator invokes it in the background.
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
  calls the model's distance and countersignature API.
- **Signature** (@signature spec) — the Agent creates signatures during
  the prepare phase and uses bitwise AND matching for candidate retrieval.
- **Tokenizer** (@tokenizer spec) — the Agent uses the tokenizer for
  encoding and decoding.
- **Kline** (@kline spec) — the Agent constructs, compares, and stores
  Klines.
