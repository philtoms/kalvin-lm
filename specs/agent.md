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
  │   if model.is_countersigned(query, candidate):               │
  │     add candidate to model                                   │
  │     re-rationalise query                                     │
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
`model.expand()` to discover connotations, applies **response sampling** to
the stream of yields, and checks surviving results for countersignature.

### Sampling

Response sampling parameters control how the Cogitator consumes the
`expand()` generator stream. Values follow LLM convention.

```
Sampling:
  temperature: float   # (0, ∞). Default 1.0.
  top_k:       int     # [0, ∞). 0 = unlimited. Default 40.
  top_p:       float   # (0, 1.0]. Default 0.95.
```

| Parameter     | Streaming role        | Mechanism                              |
| ------------- | --------------------- | -------------------------------------- |
| `temperature` | Boundary shift        | Shift S1\|S2, S2\|S3, S3\|S4 boundaries |
| `top_k`       | Exploration budget   | Cap processed connotations per work item |
| `top_p`       | Evidence threshold   | Stop when cumulative significance ≥ p   |

Applied in order per-yield: boundary classification (S4 demotion), then
top-p (evidence check), then top-k (budget check). O(1) state per work item.

#### Significance Boundaries

Three boundaries classify yielded significance values:

```
D_MAX ── S1|S2 ──────── S2|S3 ──────────── S3|S4 ── 0
```

Base positions (τ = 1):

| Boundary | Position                    | Meaning                        |
| -------- | --------------------------- | ------------------------------ |
| S1\|S2   | `D_MAX - 1`                 | Only exact S1 qualifies as S1  |
| S2\|S3   | `~_S2_S3_DISTANCE`          | Packed distance threshold (100)  |
| S3\|S4   | `0`                          | Only zero-significance is S4   |

Classification is a cascade: `sig ≥ S1|S2 → S1`, `sig ≥ S2|S3 → S2`,
`sig ≥ S3|S4 → S3`, else S4. Raw significance values are never mutated.

#### Temperature

Shifts all three boundaries by the same amount, with capping:

| Direction | S1\|S2         | S2\|S3       | S3\|S4     | Effect                      |
| --------- | -------------- | ------------ | ---------- | --------------------------- |
| τ > 1     | drops ↓        | drops ↓      | capped at 0 | More S2→S1, more S3→S2     |
| τ < 1     | capped at D_MAX-1 | rises ↑   | rises ↑    | Fewer S2→S1, more S3→S4    |

The shift function is: `shift = _TEMP_SCALE × (τ - 1)`. This is
exploratory — alternative shift functions may be tuned later.

#### Top-k

Maximum number of connotations processed per work item. After `k` results
pass the S4 demotion gate, the generator is exhausted. 0 means unlimited.

#### Top-p

Cumulative significance threshold. After processing, if the running sum of
significance reaches `top_p × D_MAX`, the generator is exhausted.
When `top_p = 1.0`, no early stopping occurs.

### Cogitator

```
Cogitator:
  model:      Model
  event_bus:  EventBus
  on_s1:      callback(query, candidate)
  timeout:    float (default 2.0)
  sampling:   Sampling (default: temperature=1.0, top_k=40, top_p=0.95)
  backlog:    queue of WorkItem
  thread:     background
```

The Cogitator runs on a background daemon thread. It pulls work items from
the backlog and processes each one.

### Work Item Processing

The Cogitator expands each WorkItem into a stream of QueryCandidates.
Boundaries are computed once per work item. Each yield is classified
against the boundaries:

```
run_work_item(WorkItem(query, candidate)):
  (s12, s23, s34) = boundaries(sampling.temperature)
  evidence_target = sampling.top_p × D_MAX
  count = 0, cumulative = 0

  for qc in model.expand(query, candidate):
    band = classify(qc.significance, s12, s23, s34)

    if band == "S4":
      continue                    # demote: below S3|S4 boundary

    count += 1
    cumulative += qc.significance

    if band == "S1":
      on_s1(query, candidate)     # promote immediately
    else:
      process(qc)                 # S2/S3: countersignature check

    if cumulative >= evidence_target:
      break                       # sufficient evidence
    if 0 < sampling.top_k <= count:
      break                       # budget exhausted

process(QueryCandidate(query, candidate, significance)):
  if model.is_countersigned(query, candidate):
    model.add(candidate)
    on_s1(query, candidate)       # triggers re-rationalisation
```

Boundaries are computed once at the start of each work item. Raw significance
is never mutated — temperature acts on the boundaries, not on the values.
A high τ lowers S1|S2, allowing near-S1 S2 connotations to be classified
as S1 and immediately published.

### S1 Callback

When the Cogitator discovers an S1 via countersignature, it calls the
`on_s1` callback, which triggers `agent.rationalise(query)` on the agent.
This re-rationalisation runs on the cogitation thread and may inject new
work items into the backlog.

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

## Open Questions

### 1. Candidate Retrieval in the Model

The model spec's `where(predicate)` performs a linear scan, which may be
too slow for large models.

**Recommendation:** Add a `model.candidates_for(signature)` method that
returns all Klines whose signatures share at least one bit with the query
signature. The model can implement this efficiently using an inverted
bit-to-signature index.

### 2. ~~Cogitation Evolution~~ — Resolved

Top-k/top-p/temperature sampling has been added to the Cogitator. The
`expand()` stream is now consumed with per-yield temperature gating, top-k
budget capping, and top-p cumulative evidence early stopping. See §Cogitation.

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
