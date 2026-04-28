# Agent Specification

## Overview

The Agent is the orchestrator of the Kalvin rationalisation pipeline. It
receives input, encodes it into Klines, retrieves candidates from the Model,
delegates significance computation, and integrates results back into the
knowledge graph.

The Agent's principal function is **rationalisation**: determining how a new
Kline relates to existing knowledge and deciding what action to take. All
computational detail is delegated — encoding to the Tokenizer, storage and
lookup to the Model, distance and match testing to the Significance pipeline.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Kline (@kline spec)

- A Kline is an identified, ordered sequence of zero or more nodes.
- Nodes are opaque uint64 values.
- Literal klines represent exact tokens; non-literal klines are composed
  structures.

### Signature (@signature spec)

- Provides `make_signature(nodes) → KSig` (OR-reduction of non-literal nodes).
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
- Provides the significance API: `is_s1`, `s2_distance`, `s3_distance`.
- Manages a three-tier memory internally (STM → Frame → Base). The agent
  sees a single Model API; tiering is invisible to the agent.
- The model decides how and where Klines are stored. The agent is
  responsible for calling model operations; the model is responsible
  for managing its internal memory tiers.

### Significance (@significance spec)

- Computes `significance(Q, C)` between a query Kline Q and a candidate C.
- Returns a `(significance: uint64, level: S1|S2|S3|S4)` pair.
- Routing is based on per-node S1 testing; distance is inverted to yield
  significance.

## Definition

An Agent consists of:

| Component | Type      | Description                            |
| --------- | --------- | -------------------------------------- |
| tokenizer | Tokenizer | Encodes text ↔ nodes.                  |
| model     | Model     | Layered knowledge graph (STM → Frame → |
|           |           | Base). Agent sees a single Model API.  |
| activity  | Counter   | Tracks Kline access frequency.         |

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

A newly constructed Agent contains zero Klines in its model.

## Rationalisation

Rationalisation is the process of integrating a Kline into the knowledge
graph. It proceeds in phases:

```
Rationalise(Q):
  ┌─────────────────────────────────────────────────┐
  │ 1. PREPARE                                      │
  │    Assign signature if missing.                  │
  ├─────────────────────────────────────────────────┤
  │ 2. GROUND CHECK                                 │
  │    Does Q already exist in the model?            │
  │    → Yes: emit "ground" event, done.             │
  ├─────────────────────────────────────────────────┤
  │ 3. ASSESS                                       │
  │    Evaluate Q's structural grounding:            │
  │    → Unsigned (no nodes): emit "frame" S4, done. │
  │    → All-literal: emit "frame" S1, done.         │
  │    → Otherwise: proceed to retrieval.            │
  ├─────────────────────────────────────────────────┤
  │ 4. RETRIEVE CANDIDATES                           │
  │    Find candidate Klines from the model.         │
  ├─────────────────────────────────────────────────┤
  │ 5. COMPUTE SIGNIFICANCE                         │
  │    For each candidate, run significance pipeline.│
  ├─────────────────────────────────────────────────┤
  │ 6. INTEGRATE                                    │
  │    Add Q to model. Act on significance result.   │
  │    → S1: confirm.                                │
  │    → S4: novel.                                  │
  │    → S2/S3: queue for cogitation.               │
  └─────────────────────────────────────────────────┘
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

**All-literal**: If every node in Q is a literal (per `tokenizer.is_literal`),
Q is a pure token sequence. Emit a `"frame"` event at S1. Return `True`.

**Self-grounded**: If `Q.signature == make_signature(Q.nodes)` (as defined
in the @signature spec) and every non-literal node resolves in the model
(exists as a Kline signature in the model), Q is fully grounded. Emit a
`"frame"` event at S1. Return `True`.

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
expensive significance pipeline runs.

If no candidates are found, the result is S4 (novel). Add Q to the model,
emit a `"frame"` event at S4. Return `True`.

### Phase 5: Compute Significance

For each candidate Cᵢ, run the significance pipeline defined in the
@significance spec:

```
for each candidate Cᵢ:
    (significance, level) = significance_pipeline(Q, Cᵢ, model)
```

The significance pipeline performs per-node S1 testing, routes to the
appropriate distance function, and inverts the distance to yield a
significance value.

Collect all `(Cᵢ, significance, level)` triples.

### Phase 6: Integrate

Add Q to the model via `model.add(Q)`. The model manages its internal
memory tiers (STM, frame, base) as it sees fit.

Select the best result (highest significance value). Act based on its level:

| Best level | Action                                      | Return |
| ---------- | ------------------------------------------- | ------ |
| S1         | Promote Q to base. Emit `"frame"` S1 event. | True   |
| S4         | Promote Q to base. Emit `"frame"` S4 event. | True   |
| S2         | Queue Q for cogitation.                     | False  |
| S3         | Queue Q for cogitation.                     | False  |

S1 and S4 are **significants** — the Kline is either confirmed or novel,
and no further processing is needed. S2 and S3 are **rationals** — the
Kline has a partial or weak relationship to existing knowledge and requires
deeper investigation via cogitation.

## Cogitation

Cogitation is the background processing of rational Klines (S2/S3). It
performs deeper graph traversal to find additional candidates that the
initial retrieval may have missed.

### Cogitation Pipeline

```
Cogitate(Q):
  1. Expand Q's graph context:
     candidates = model.query(Q.signature, depth=D_cogitate)

  2. For each candidate Cᵢ:
     (significance, level) = significance_pipeline(Q, Cᵢ, model)

  3. If any candidate achieves S1:
     - Add candidate to model.

  4. Re-rationalise Q.
```

`D_cogitate` is a configurable depth parameter (default: **2**).

Re-rationalisation in step 4 may itself queue Q for further cogitation if
the result remains S2/S3. The Agent must detect cycles: if Q has been
cogitated more than a configurable maximum (default: **3** passes) without
reaching a significant result, it is abandoned and remains at its last
computed level.

### Cogitation Lifecycle

Cogitation runs asynchronously. The Agent manages a backlog queue of
Klines. When the backlog has been empty for a configurable timeout
(default: **2 seconds**), the Agent emits a `"done"` event and stops the
cogitation thread.

The cogitation thread can be stopped explicitly and joined with a timeout.

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

## Open Questions

The following gaps exist across the spec spectrum. They require resolution
before implementation.

### 1. Candidate Retrieval in the Model

The significance spec requires the Agent to provide candidates, and this
spec defines candidate retrieval as signature-overlap filtering. However,
the model spec's `where(predicate)` performs a linear scan, which may be
too slow for large models.

**Options:**

- (a) Add a `model.candidates_for(signature)` method to the model spec
  that returns all Klines whose signatures share at least one bit with
  the query signature. The model can implement this efficiently using an
  inverted bit-to-signature index.
- (b) Accept the linear scan for now and optimise later.

**Recommendation:** (a). Candidate retrieval is the hottest path in
rationalisation. A dedicated method allows the model to maintain an
appropriate index.

### 2. Model Significance API Semantics

The model spec defines `is_s1(node, candidate)`, `s2_distance(Q, C)`,
`s3_distance(Q, C)` with semantics marked **TBD**. These are the functions
consumed by the significance pipeline.

**Question:** Who defines the semantics of these functions?

**Options:**

- (a) The model spec itself defines them (they are model-level operations).
- (b) A new "comparison" or "distance" spec defines them.
- (c) The agent spec defines them (they are agent-level concerns).

**Recommendation:** (a). The model owns its data and should define how
comparisons work. The significance spec defines the routing and inversion;
the model spec defines the actual comparison mechanics.

### 3. Grounding Assessment Formalisation

This spec defines grounding checks (self-grounded, all-literal, unsigned)
as fast-path optimisations that bypass the full significance pipeline. An
alternative design would route everything through the significance pipeline,
with the model's `is_s1` function handling these cases internally.

**Question:** Should grounding be a separate agent-level concept, or should
it be unified with the significance pipeline?

**Recommendation:** Keep as agent-level fast paths. Grounding is about
structural properties of a single Kline, not about comparison between two
Klines. Conflating them would make the significance spec more complex
without adding clarity.

### 4. Cogitation Depth and Pass Limit

`D_cogitate` (default 2) and the maximum cogitation passes (default 3) are
configurable but their interaction with model size and Kline complexity is
not analysed.

**Recommendation:** Make both configurable. Document that deeper cogitation
is more expensive but may resolve more rationals.

## What an Agent is Not

The following are explicitly **out of scope** for this spec:

- **Significance computation.** The Agent orchestrates the pipeline but
  delegates distance calculation and inversion to the @significance spec.
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

- **Significance** (@significance spec) — the Agent provides query Klines,
  candidates, and consumes significance results.
- **Model** (@model spec) — the Agent stores and retrieves Klines, and
  calls the model's significance API.
- **Signature** (@signature spec) — the Agent creates signatures during
  the prepare phase and uses bitwise AND matching for candidate retrieval.
- **Tokenizer** (@tokenizer spec) — the Agent uses the tokenizer for
  encoding and decoding.
- **Kline** (@kline spec) — the Agent constructs, compares, and stores
  Klines.
