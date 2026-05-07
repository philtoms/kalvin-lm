# Kalvin — The Training Loop

## Overview

The training loop is a dialogue between a **teacher** (human or automated) and a **kalvin instance**, mediated by three existing mechanisms: `kalvin.rationalise()`, the **event bus**, and **countersignature**.

The teacher does not need new APIs. It uses `rationalise()` to submit queries and listens to events to receive proposals. Ratification is countersignature — the teacher rationalises the reciprocal kline, creating a mutual cross-reference that makes the proposal structurally S1.

This document describes the training loop as it will be implemented. It builds on the concepts defined in `docs/learning.md` (the learning principles) and references the specs in `specs/` for formal definitions.

---

## The Single Change to Current Behavior: Sharpen S1 Promotion

The existing cogitator auto-promotes any result classified as S1, including those that are only S1 because temperature lowered the S1|S2 boundary. Under the training model, temperature-shifted S1 is a **proposal**, not a conclusion.

The change: **only true S1 promotes to frame.**

| Significance | Computed | Temperature-shifted | Action |
|---|---|---|---|
| True S1 | distance = 0 (all nodes match) | N/A | Auto-promote to frame |
| Canonical S1 | all-literal or self-grounded | N/A | Auto-promote to frame |
| t(S2) → S1 | distance > 0, but τ lowers boundary | Yes | Add to STM, emit `frame` event, **do not promote** |
| S2 | S2 range, below τ-shifted boundary | No | Queue for cogitation |
| S3 | S3 range | No | Queue for cogitation |
| S4 | No candidates | N/A | Auto-promote to frame |

This preserves the existing fast-path logic (canonical, grounded, and unsigned results are untouched) while making temperature-shifted S1 a proposal that awaits ratification.

The distinction is: true S1 means the query's nodes *actually match* the candidate's nodes (distance = 0). Temperature-shifted S1 means the significance value falls above the τ-shifted S1|S2 boundary, but the structural distance is non-zero — there is still a gap between what kalvin knows and what it is proposing to understand.

---

## Frames: One per Rationalise Call

Every call to `kalvin.rationalise()` that does not short-circuit (ground check, canonical, unsigned) initiates a new **frame**. A frame is a kalvin instance layered over the previous frame as its base:

```
rationalise(query_1)  →  Frame 1:  STM → Frame → Base
rationalise(query_2)  →  Frame 2:  STM → Frame → [Frame 1 as base]
rationalise(query_3)  →  Frame 3:  STM → Frame → [Frame 2 as base]
```

### Frame Properties

- **Frames share references** to underlying frame data — no deep copying. The frame stack is a linked list of write surfaces over shared structural data.
- **Promotion in a later frame is visible to an earlier frame's cogitation**, because earlier frames' bases contain later frames' promoted data by reference.
- **No frame is ever abandoned.** Each frame's cogitator continues processing independently. The training loop grows as frame count increases. Later frames enrich the model visible to earlier frames' ongoing cogitation.
- **Sampling parameters** (temperature, top_k, top_p) are per-frame, set at construction time. They do not change in-flight.

### Frame Factory

Creating a new frame is constructing a new kalvin instance with the current frame as base:

```
frame_n = Agent(model=Model(base=frame_n_minus_1.model))
```

Or via a convenience method (TBD):

```
frame_n = kalvin.new_frame()
```

### The Frame Stack as Learning Trajectory

The frame stack is not just a mechanism — it is a **record of learning**:

```
Frame 3 (τ=1.5, scaffold for missing concept)
  ↓ base
Frame 2 (τ=1.0, corrective scaffolding)
  ↓ base
Frame 1 (τ=1.0, original query)
  ↓ base
Base model
```

Each frame captures the context of a particular training action: the query, the sampling parameters, the model state at that point. The teacher can walk the stack to understand the learning trajectory.

Future extension: a temporal element added to events so the teacher (or a kalvin-as-tutor) can make judgments about the trajectory — rate of convergence, effectiveness of scaffolding, etc.

---

## KScript: Query, Expectation, and Scaffolding

A training script is a KScript whose structure, by convention, encodes:

- **Top-level kline** — the query and its expectation (e.g., `MHALL => SVO`)
- **Nested klines** — intermediate scaffolding, each with its own query/expectation structure

```
MHALL => SVO          ← top-level query / expectation
  M = H               ← scaffolding: undersign
  S > V               ← scaffolding: connotate
  ALL => A L L        ← scaffolding: canonize with subscript
    A = D             ← deeper scaffolding
```

The teacher compiles the script, extracts the top-level kline as the query, and holds the full compiled set as the expectation structure. The script also carries the scaffolding the teacher is prepared to supply if needed.

No KScript changes are required for the MVP. Sampling parameters are set API-level, per-frame. Future work may add metadata annotations to KScript for encoding `temp`, `top_k`, `top_p` alongside the script.

---

## Ratification: Countersignature as Endorsement

Ratification is not a new operation. It is the existing countersignature mechanism, triggered by the teacher:

1. Kalvin rationalises a query, cogitates, and emits a `frame` event with a proposed kline (e.g., `{A: [B, C]}`)
2. The teacher evaluates the proposal against the expectation
3. If acceptable, the teacher **countersigns** by rationalising the reciprocal kline through `kalvin.rationalise()`
4. This creates the mutual cross-reference — structural S1 — and the kline is now grounded

```
kalvin proposes:       {A: [B, C]}       (emitted as frame event)
teacher countersigns:  rationalise(Kline(B | C, [A]))
                       → creates {BC: A}
                       → mutual cross-reference detected
                       → structural S1 achieved
```

This means:
- **Temperature-promoted S1 is a proposal.** Kalvin says "at this temperature, I think I understand this" — expressed as a `frame` event, added to STM but not promoted.
- **Countersignature is ratification.** The teacher creates the structural relationship that makes it *actually* S1.
- **Canonical S1 (all-literal, self-grounded) is self-ratifying.** No countersignature needed — the kline routes to S1 on its own via the existing fast path.

### MVP: Exact Match

For the MVP, the trainer compares the submitted kline against the expectation using **kline equality** (same signature, same node sequence):

```
event.kline == expectation_kline
```

If they match exactly, the trainer countersigns. If they don't match, the trainer constructs new scaffolding.

Future extension: rationalisation itself becomes reentrant (kalvin-as-tutor), where the "comparison" is itself a rationalisation pass rather than exact equality. This would allow the teacher to accept structurally equivalent but not byte-identical proposals.

---

## The Loop: Step by Step

```
Teacher                              Kalvin
  │                                    │
  │  compile KScript                   │
  │  set sampling params               │
  │  rationalise(query_kline) ────────→│  Frame N created
  │                                    │  rationalise returns True/False
  │  ←────────── RationaliseEvent ─────│  event emitted
  │                                    │
  │  compare event kline               │
  │  against expectation               │
  │                                    │
  ├── If matches (kline == expectation)│
  │     rationalise(reciprocal) ──────→│  Countersignature created
  │                                    │  Structural S1 achieved
  │                                    │  Promoted to frame
  │                                    │
  ├── If doesn't match                 │
  │     extract misaligned info        │
  │     construct new KScript          │
  │     rationalise(scaffold_kline)───→│  Frame N+1 created
  │     (this frame at same or         │  Enriches model
  │      different temperature)        │
  │                                    │
  │     [continue listening for        │  Frame N cogitation
  │      events from Frame N]          │  continues independently
  │                                    │
  └── If no response (timeout)         │
        adjust temperature             │
        rationalise(query_kline) ─────→│  Frame N+1 created
        (same kline, new params)       │  Different boundary settings
```

### Event Correlation

The `RationaliseEvent` carries `query` and `value` klines. The teacher correlates via signature — "I submitted a query with signature X, and I got back a frame event for a query with signature X." For the MVP, the teacher tracks one query per frame and uses signature matching to correlate events.

### The Feedback Loop is Synchronised by Return Value

`rationalise()` returns `True` (fast path: S1 or S4, done) or `False` (slow path: S2/S3, cogitation in progress). The teacher uses this to coordinate:

- `True` → the event has already been emitted. Evaluate immediately.
- `False` → cogitation is running. Listen for events. The frame's cogitator emits `frame` events as it discovers proposals, and `done` when the backlog empties.

---

## Scenarios

### Scenario A — Immediate Success

1. Teacher constructs script, extracts query, calls `rationalise(query)` → Frame 1 created, returns `True`.
2. Kalvin routes query to true S1 (canonical match). Auto-promotes. Emits `frame` event.
3. Teacher receives event. Kline matches expectation exactly.
4. Teacher countersigns: `rationalise(reciprocal_kline)`. Structural S1 confirmed.

```
Frame 1:  MHALL → SVO
  query:     {MHALL: [S, V, O]}
  candidate: {MHALL: [S, V, O]}  (exists in model)
  route:     S1 (all nodes match)
  action:    auto-promote
  teacher:   countersigns → rationalise({SVO: [M, H, A, L, L]})
  result:    structural S1 grounded
```

### Scenario B — Scaffolding Required

1. Teacher constructs script, extracts query, calls `rationalise(query)` → Frame 1 created, returns `False`.
2. Kalvin routes query to S2. Queues for cogitation.
3. Cogitation yields a t(S2)→S1 proposal (temperature-shifted, not true S1). Emits `frame` event. Kline added to STM, **not promoted**.
4. Teacher receives event. Kline does not match expectation.
5. Teacher constructs new script covering the misaligned information. Calls `rationalise(scaffold)` → Frame 2 created over Frame 1.
6. Kalvin rationalises scaffold. Achieves true S1 (or a t(S2) that the teacher accepts). Promoted/countersigned.
7. Meanwhile, Frame 1's cogitation **continues**. Model is now richer (Frame 2's promotions are visible via shared references). Frame 1 eventually yields a new proposal.
8. Teacher evaluates. This time it matches. Countersigns. Frame 1's query is now structurally S1.

```
Frame 1:  MHALL → SVO  (τ=1.0)
  query:     {MHALL: [S, V, O]}
  route:     S2 (some nodes match)
  cogitate:  yields t(S2)→S1 proposal
  teacher:   does not match expectation

Frame 2:  scaffold for missing concept  (τ=1.0, base=Frame 1)
  query:     {S: M}
  route:     S1 (canonical)
  action:    auto-promote
  teacher:   countersigns

Frame 1 (continued):
  cogitation continues with enriched model
  yields new proposal → matches expectation
  teacher countersigns → structural S1
```

### Scenario C — Temperature Adjustment

1. Teacher constructs script, extracts query, calls `rationalise(query)` at τ=1.0 → Frame 1 created, returns `False`.
2. Frame 1 cogitation runs but produces nothing the teacher can accept within a reasonable time.
3. Teacher constructs same query at τ=1.5, calls `rationalise(query)` → Frame 2 created over Frame 1.
4. At higher temperature, boundaries lower. Kalvin yields a t(S2)→S1 proposal. Emits `frame` event.
5. Teacher evaluates. The kline matches expectation. Countersigns via `rationalise(reciprocal)`. Promoted in Frame 2.
6. Frame 1's cogitation continues. Model is now richer (Frame 2's promotions visible via shared references). Frame 1 eventually yields a stronger result — potentially true S1 without temperature assistance.
7. No frames abandoned. Frame stack represents the full learning history.

```
Frame 1:  MHALL → SVO  (τ=1.0)
  query:     {MHALL: [S, V, O]}
  route:     S2
  cogitate:  nothing acceptable within timeout
  teacher:   no action, proceeds to create Frame 2

Frame 2:  MHALL → SVO  (τ=1.5, base=Frame 1)
  query:     {MHALL: [S, V, O]}
  cogitate:  boundaries lowered → t(S2)→S1 proposal
  teacher:   matches expectation → countersigns

Frame 1 (continued):
  cogitation continues with enriched model
  may reach true S1 autonomously
```

### The Temperature Pattern

Temperature is not nested or stacked — there is no push/pop. Each frame simply has its own temperature. Frame 1 at τ=1.0 and Frame 2 at τ=1.5 coexist independently. The teacher's decision to use a different temperature is reflected in the new frame's construction, not in any modification to existing frames.

---

## The Teacher's Responsibilities

The teacher is an external coordinator that:

1. **Compiles training scripts** into kline queries and expectations using KScript.
2. **Manages the frame stack** — creates new frames (kalvin instances) as needed.
3. **Sets sampling parameters** per frame (API-level for MVP).
4. **Subscribes to event buses** for all active frames.
5. **Compares proposals against expectations** using kline equality (MVP) or reentrant rationalisation (future).
6. **Countersigns acceptable proposals** by rationalising the reciprocal kline.
7. **Constructs corrective scaffolding** when proposals don't match expectations.
8. **Adjusts temperature** by creating new frames at different parameter settings.

The teacher does not modify kalvin internals. It does not roll back model state. It does not force promotion. It uses `rationalise()` and listens to events, the same interface available to any observer.

---

## Study: Cogitation Between Training Interactions

Study is not a separate mechanism. It is the existing cogitator's S2/S3 background processing, viewed from the learning perspective. Between teacher interactions:

- Each frame's cogitator continues processing its backlog independently.
- Cogitation may discover countersignature relationships, triggering re-rationalisation.
- The enriched model (from other frames' promotions) may cause earlier frames' cogitation to produce stronger results.
- Study cannot reach true S1 on its own — it can produce t(S2)→S1 proposals that await teacher countersignature, or discover structural S1 via countersignature in the existing sense (mutual cross-reference found during graph expansion).

The distinction between "training" and "study" is a matter of teacher presence, not system behavior. The cogitator runs the same way regardless.

---

## Monotonicity: Constructive Grounding

Monotonicity is a guarantee about **grounded information**, not about significance values. Significance may fluctuate — adding klines changes candidate sets and can alter distance calculations. But:

- Once a kline is promoted to a frame, it stays. Frames are append-only.
- Every scaffolding kline that achieves S1 (true or countersigned) adds permanent structure to the knowledge graph.
- The model can only grow richer — never poorer.

The learning trajectory is monotonic in the sense that the **knowledge base only grows**. Individual significance measurements for a given query may go up or down as the model changes, but the total amount of grounded, ratifiable knowledge only increases.

---

## MVP Implementation Requirements

Four things are needed to implement the training loop:

### 1. Sharpen S1 Promotion in the Cogitator

Temperature-shifted S1 adds to STM and emits events but does not promote. This is a surgical change to `_run_work_item` — the `band == "S1"` path needs to distinguish true S1 from boundary-shifted S1.

Detection: true S1 is when `distance == 0` (all nodes match). Temperature-shifted S1 is when the significance value is above the shifted S1|S2 boundary but the underlying structural distance is non-zero.

### 2. Frame Factory

A method or convention for creating a new kalvin instance with the current frame as base. Options:

- Teacher constructs `Agent(model=Model(base=existing_frame_model))` manually.
- Convenience method: `kalvin.new_frame(temperature=1.5, top_k=40, top_p=0.95)`.

### 3. Event Subscription per Frame

Each frame (Agent instance) has its own event bus. The teacher subscribes to all active frames' event buses simultaneously. Events include the frame context (implicitly, via which bus the event arrived on).

### 4. Kline Equality Comparison

Already exists in the spec. The teacher uses it directly: `event_kline == expectation_kline`.

---

## MVP vs. Future

| Capability | MVP | Future |
|---|---|---|
| Trainer comparison | Exact kline equality | Reentrant rationalisation (kalvin-as-tutor) |
| Sampling parameters | API-level, per-frame | KScript metadata encoding |
| Frame stack | Unbounded, reference-shared | Temporal events, trajectory analysis |
| S1 promotion | Sharpened (true S1 only) | Full ratification semantics |
| Inter-frame enrichment | Implicit via shared references | Explicit enrichment API |
| Monotonicity | Grounding is constructive | Verified convergence metrics |
| Teacher | External coordinator | Kalvin-as-tutor (reentrant) |

---

## Relationship to `docs/learning.md`

This document describes the *mechanism* — how the training loop works in terms of existing APIs and data structures. `docs/learning.md` describes the *principles* — why the system learns, what agency means, and why S1 requires ratification.

The key connections:

| learning.md principle | training-loop.md mechanism |
|---|---|
| "Temperature proposes; the trainer disposes" | t(S2)→S1 proposals in STM, teacher countersigns to ratify |
| "Learning is recursive rationalisation" | Each scaffolding step is a new frame, same pipeline |
| "S1 is ratified, not claimed" | Countersignature by teacher creates structural S1 |
| "Agency must be exercised" | Cogitation (study) runs on all frames independently |
| "The system learns what it doesn't know" | S4 and failed proposals identify gaps for scaffolding |
| "Learning is constructive" | Frame stack is append-only, grounding is permanent |
