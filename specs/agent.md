# Agent Specification

## Overview

The Agent is the orchestrator of the Kalvin rationalisation pipeline. It
receives input, encodes it into Klines, retrieves candidates from the Model,
routes each query-candidate pair, and integrates results back into the
model.

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

### Signature (@signature spec)

- Provides `make_signature(nodes) → KSig` (plain OR-reduction of raw node values).
- Provides bitwise AND matching for candidate retrieval.

### Tokenizer (@tokenizer spec)

- Provides `encode(text) → list[int]` and `decode(nodes) → str`.
- Nodes returned by `encode` are fully typed — the tokenizer handles all
  internal encoding details including type prefix combination.

### Model (@model spec)

- Stores, indexes, and retrieves Klines.
- Provides candidate retrieval via `find`, `find_all`, `query`, `where`.
- Provides `find_by_nodes` for transitive grounding via nodes signature.
- Provides `add_to_stm`, `add_to_frame`, `add_to_ltm` for tiered writes based on
  significance outcome (STM only, STM+Frame, or STM+Frame+LTM).
- Provides `expand(Q, C)` generator for graph expansion yielding
  connotations and terminal significance.
- Provides `QueryCandidate` named tuple with `.significance` field
  (pre-computed by the model).
- Provides constants `D_MAX` and `MASK64` for significance values.
- Provides `is_countersigned(kline)` to check if a kline is countersigned by any kline in the model.
- Manages a four-tier memory internally (STM → Frame → LTM → Base). The
  agent selects the appropriate write method based on significance outcome;
  tier cascade semantics are handled by the model.
- The model decides how and where Klines are stored. The agent is
  responsible for calling model operations; the model is responsible
  for managing its internal memory tiers.

## Definition

An Agent consists of:

| Component | Type      | Description                            |
| --------- | --------- | -------------------------------------- |
| tokenizer | Tokenizer | Encodes text ↔ nodes.                  |
| model     | Model     | Layered memory (STM → Frame → |
|           |           | LTM → Base). Agent sees a single API. |
| cogitator | Cogitator | Background slow-path processor. See    |
|           |           | @cogitator spec.                       |

## Construction

```
Agent(
    tokenizer      = None,   # defaults to NLPTokenizer; NLP data is mandatory
    model          = None,   # defaults to empty Model
    max_candidates = 8,      # cap on S2/S3 candidates submitted per entry
)
```

- `tokenizer` — a Tokenizer instance. Defaults to an `NLPTokenizer` (NLP data is mandatory; construction raises if unavailable).
- `model` — a Model instance serving as the base memory. Defaults
  to an empty Model.
- `max_candidates` — maximum number of S2/S3 candidates submitted to
  the Cogitator per entry (see §Candidate Cap).

A newly constructed Agent contains zero Klines in its model. The Cogitator
is created internally and starts its background thread immediately.

## Rationalisation

Rationalisation is the process of integrating a Kline into the
model. It proceeds in phases with a fast/slow split:

```
Rationalise(Q):
  ┌─── FAST PATH ────────────────────────────────────────────┐
  │ 1. PREPARE                                               │
  │    Assign signature if missing.                           │
  ├───────────────────────────────────────────────────────────┤
  │ 2. GROUND CHECK                                          │
  │    Does Q already exist in the model?                     │
  │    → Yes: add_to_stm(Q), emit "ground" event, return True.  │
  ├───────────────────────────────────────────────────────────┤
  │ 3. ASSESS                                                │
  │    Evaluate Q's structural grounding:                     │
  │    → Unsigned (no nodes): add_to_ltm(Q), emit "frame" S4,   │
  │       return True.                                       │
  │    → Self-grounded: add_to_ltm(Q), emit "frame" S1,         │
  │       return True.                                       │
  │    → Countersigned: add_to_ltm(Q), emit "frame" S1,         │
  │       return True.                                       │
  ├───────────────────────────────────────────────────────────┤
  │ 4. RETRIEVE CANDIDATES                                    │
  │    candidates = model.where(Q.signature)                  │
  │    → No candidates: add_to_ltm(Q), emit "frame" S4,         │
  │       return True.                                       │
  ├───────────────────────────────────────────────────────────┤
  │ 5. ROUTE EACH CANDIDATE                                   │
  │    add_to_stm(Q)                                              │
  │    Sort candidates: S1, then S2, then S3; within each    │
  │    tier by node overlap (descending). Truncate S2/S3     │
  │    candidates to max_candidates.                          │
  │    For each candidate Cᵢ:                                │
  │      level = route(Q, Cᵢ)   ← node membership, no model │
  │      S1 → add_to_ltm cascade, emit "frame", return True     │
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

- Call `model.add_to_stm(Q)` to register the event in STM.
- Emit a `"ground"` event.
- Return `True` (significant).
- No further processing.

This prevents infinite recursion and avoids re-processing known knowledge.

### Phase 3: Assess

Structural assessment determines whether Q can be fast-tracked without
candidate retrieval or significance computation.

**Unsigned**: If Q has zero nodes, it carries no information. Call
`model.add_to_ltm(Q)`. Emit a `"frame"` event at S4. Return `True`.

**Canonical — self-grounded**: If `Q.signature == make_signature(Q.nodes)`
(as defined in the @signature spec) and every node that could resolve does
resolve in the model (exists as a Kline signature), Q is fully grounded.
The signature faithfully represents the nodes — nothing is missing and
nothing is extraneous. Call `model.add_to_ltm(Q)`. Emit a `"frame"` event at
S1. Return `True`.

**Countersigned — ratified**: If the model contains a kline whose signature
equals `make_signature(Q.nodes)` and whose sole node equals `Q.signature`,
then Q is countersigned — another kline vouches for it structurally. This is
the **ratification** check. It runs in the fast lane before candidates are
retrieved, because countersignature is a structural property of Q and the
model, not dependent on any particular candidate. Call `model.add_to_ltm(Q)`.
Emit a `"frame"` event at S1. Return `True`.

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

If no candidates are found, Q is novel. Call `model.add_to_ltm(Q)`. Emit a
`"frame"` S4 event, and return `True`.

### Phase 5: Route Each Candidate

Add Q to STM via `model.add_to_stm(Q)`. Route each candidate, then submit
**all** candidates to the Cogitator as work items.

#### Routing

```
route(Q, C):
  match_count = number of Q.nodes that exist in C.nodes
  if match_count > 0:  return "S2"
  else:                return "S3"
```

Routing is a pure function. It checks whether any query node value appears
in the candidate's node sequence. No model function is called.

Routing distinguishes only **S2** (at least one overlapping node) from
**S3** (no overlapping node). It does **not** route S1 or S4:

- **S1 is not a routing outcome.** Full node overlap is a necessary but
  insufficient condition for S1; true S1 is a structural property
  established by `expand()` / `is_s1()` (canonical composition or
  countersignature), not by node membership. Routing a pair as "S1" purely
  on overlap would publish S1 significance for pairs that are not
  structurally grounded.
- **S4 is not a routing outcome.** Identity klines (empty nodes) are
  resolved on the fast path in `rationalise` before any candidate is
  submitted to the Cogitator, so an empty query never reaches routing.

#### Candidate Ordering

Candidates are sorted before submission: **S2 candidates first**, then S3.
Within each tier, candidates are sorted by node overlap count (descending),
so the closest matches are expanded first.

#### Candidate Cap

After sorting, the S2/S3 portion of the candidate list is truncated to
`max_candidates` (default 8). Only the top-K candidates are submitted to
the Cogitator. This bounds the per-entry fan-out regardless of model
density.

The cap does **not** affect:
- Structural S1 resolution on the fast path — klines that are canonical
  with all nodes grounded, or countersigned, are resolved in `rationalise`
  before any candidate is submitted.
- S4 (novel) routing — an S4 entry has no candidates, so the cap is
  irrelevant.

#### Per-Candidate Action

| Route | Action | Model call? |
| ----- | ------ | ----------- |
| S2    | Submit `WorkItem(Q, C)` to Cogitator | No |
| S3    | Submit `WorkItem(Q, C)` to Cogitator | No |

All candidates are submitted as work items for expansion. The Cogitator
classifies each expansion yield against the significance boundaries; a
yield that computes to S1 (distance 1) is a genuine structural exact match
and triggers `handler.on_s1()` (see §Cogitation).

Return `False`.

## Cogitation

The slow path — background processing of pre-routed work items, S2/S3
expansion, significance-boundary classification, and the
Cogitator/CogitationHandler/WorkItem/QueryCandidate contracts — is defined
in the **@cogitator spec**. This spec owns only the agent's role in the
seam: it submits one `WorkItem` per routed candidate during Phase 5, and
it is the primary `CogitationHandler` implementation (`on_s1`,
`on_expansion`).

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
independently (S2/S3) and submits all to the Cogitator as work items for
expansion; an S1 discovered during expansion terminates the pair.

**Rationale**: Best-candidate selection forced full computation before the
agent could act. The per-candidate routing model with S1-first ordering
enables immediate S1 resolution and parallel processing of S2/S3.

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
| AGT-9  | First rationalise: returns True, kline in model via add_to_ltm   | — |
| AGT-10 | Duplicate rationalise: returns True, emits "ground" event, STM refreshed | — |
| AGT-11 | Different sig same nodes: not a ground (different KLine) | — |

### Rationalisation — Phase 3: Assess

| ID     | Criterion                                                  | Origin ref |
| ------ | ---------------------------------------------------------- | ---------- |
| AGT-12 | Unsigned (no nodes): returns True, emits "frame" S4, kline in LTM | — |
| AGT-14 | Self-grounded canonical: returns True when all nodes resolve, kline in LTM | — |
| AGT-15 | Not self-grounded: falls through to Phase 4               | — |

### Rationalisation — Phase 4: Retrieve Candidates

| ID     | Criterion                                         | Origin ref |
| ------ | ------------------------------------------------- | ---------- |
| AGT-16 | No candidates: returns True (S4 novel), kline in LTM       | — |
| AGT-17 | Candidates found: proceeds to routing              | — |

### Rationalisation — Phase 5: Route Each Candidate

| ID     | Criterion                                                           | Origin ref |
| ------ | ------------------------------------------------------------------- | ---------- |
| AGT-18 | All candidates submitted to cogitator as work items            | — |
| AGT-19 | Candidates sorted S2-first, then by overlap count (descending)   | — |
| AGT-19a | S2/S3 candidates truncated to `max_candidates` (default 8) after sort | — |
| AGT-19b | Within the cap, S2 candidates prioritised over S3 candidates       | — |
| AGT-19c | Within the same level, higher node-overlap candidates prioritised  | — |
| AGT-19d | Structural S1 resolution on the fast path unaffected by the cap   | — |
| AGT-19e | S4 (novel) routing unaffected by the cap                           | — |
| AGT-20 | All S2: returns False, all submitted as WorkItems                   | — |
| AGT-21 | All S3: returns False, all submitted as WorkItems                   | — |
| AGT-22 | S1 discovered during expansion: `on_s1` called, expansion breaks    | — |
| AGT-22a | Slow path query: kline in STM only (not Frame or LTM)         | — |

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

Relocated to **@cogitator spec** (IDs AGT-29 through AGT-42, kept stable).
See @cogitator spec §Test Matrix.

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

The Cogitator processes yields from `model.expand()` until S1 is found.
Proposals can be emitted at any significance level. See @cogitator spec.

### 3. Grounding Assessment Formalisation

This spec defines grounding checks (self-grounded, identity)
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
- **Cogitator** (@cogitator spec) — the Agent owns the Cogitator, submits
  pre-routed work items, and is the primary CogitationHandler.
