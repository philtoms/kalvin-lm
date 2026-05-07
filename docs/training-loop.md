# Kalvin — The Training Loop

## Overview

The training loop is a dialogue between a **teacher** (human or automated) and a **kalvin instance**, mediated by three existing mechanisms: `kalvin.rationalise()`, the **event bus**, and **countersignature**.

The teacher does not need new APIs. It uses `rationalise()` to submit queries and listens to events to receive proposals. Ratification is countersignature — the teacher rationalises the reciprocal kline, creating a mutual cross-reference that makes the proposal structurally S1.

This document describes the training loop as it will be implemented. It builds on the concepts defined in `docs/learning.md` (the learning principles) and references the specs in `specs/` for formal definitions.

---

## The Single Change to Current Behavior: Structural Grounding

The existing cogitator auto-promotes any result classified as S1. Under the training model, S1 is determined by **structure**, not by boundary classification. A kline is S1 (grounded) if its signature fully describes its nodes or it is countersigned by another kline. All significance is recoverable through structure.

The change: **after ratification, promote all STM klines involved in the ratification process to frame** — not just the kline being ratified. Frames will hold klines of all significance levels from S4 through S1.

### What This Means for S4

S4 klines are **identity klines** — the system has never encountered them before. Ratified S4 means the system knows a kline exists, it has identity that can be verified. Promoting S4 klines after ratification is not a contradiction; it is the system recording that it has been exposed to this kline and that exposure has been confirmed by the trainer.

### Why This Matters

Ensuring that all klines involved in the ratification process are available to future rationalisation enhances Kalvin's ability to cogitate beyond simple pattern matching. A frame that holds S4 identity klines alongside S1 grounded klines provides a richer model for the cogitator to traverse — it can discover relationships between novel and grounded structures that would be invisible if only S1 klines were promoted.

### Promotion After Ratification

The sequence is:

1. Teacher submits query. Kalvin rationalises. Klines enter STM at whatever significance the pipeline computes.
2. Cogitation runs. Proposals are emitted via events.
3. Teacher evaluates and **ratifies** by countersigning.
4. **After ratification**, all STM klines that participated in the ratification process are promoted to frame.

This replaces the previous model where only true S1 promoted to frame. The promotion trigger is ratification, not significance level.

| Kline type | How it enters STM | Promoted? | When |
|---|---|---|---|
| Canonical S1 (all-literal, self-grounded) | Fast path | Yes | Immediately (self-ratifying) |
| Countersigned S1 | Cogitation discovers countersignature | Yes | When countersignature detected |
| Teacher-ratified kline | Cogitation → proposal → teacher countersigns | Yes | After teacher ratification |
| S4 identity kline (involved in ratification) | No candidates → novel | Yes | After ratification of related kline |
| S2/S3 kline (involved in ratification) | Cogitation | Yes | After ratification of related kline |
| S2/S3 kline (not involved in ratification) | Cogitation | No | Stays in STM for further cogitation |

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
5. **After ratification**, all STM klines involved in the ratification process are promoted to frame

```
kalvin proposes:       {A: [B, C]}       (emitted as frame event)
teacher countersigns:  rationalise(Kline(B | C, [A]))
                       → creates {BC: A}
                       → mutual cross-reference detected
                       → structural S1 achieved
                       → all participating STM klines promoted to frame
```

This means:
- **Grounding is structural.** S1 is determined by interrogating kline structure — signature describes nodes, or countersigned — not by boundary classification.
- **Countersignature is ratification.** The teacher creates the structural relationship that makes the kline S1.
- **Promotion follows ratification.** All klines involved in the ratification process are promoted, regardless of their individual significance level. This enriches the frame with S4 identity klines and S2/S3 partial klines that participated.
- **Canonical S1 (all-literal, self-grounded) is self-ratifying.** No countersignature needed — the kline routes to S1 on its own via the existing fast path. These promote immediately.

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
2. Kalvin routes query to S1 (canonical match — signature describes nodes). Auto-promotes. Emits `frame` event.
3. Teacher receives event. Kline matches expectation exactly.
4. Teacher countersigns: `rationalise(reciprocal_kline)`. Structural S1 confirmed.
5. All STM klines involved in this ratification are promoted to frame.

```
Frame 1:  MHALL → SVO
  query:     {MHALL: [S, V, O]}
  candidate: {MHALL: [S, V, O]}  (exists in model)
  route:     S1 (all nodes match, signature describes nodes)
  action:    auto-promote (self-ratifying)
  teacher:   countersigns → rationalise({SVO: [M, H, A, L, L]})
  result:    structural S1 grounded, participating klines promoted
```

### Scenario B — Scaffolding Required

1. Teacher constructs script, extracts query, calls `rationalise(query)` → Frame 1 created, returns `False`.
2. Kalvin routes query to S2. Queues for cogitation.
3. Cogitation yields a proposal. Emits `frame` event. Kline added to STM.
4. Teacher receives event. Kline does not match expectation.
5. Teacher constructs new script covering the misaligned information. Calls `rationalise(scaffold)` → Frame 2 created over Frame 1.
6. Kalvin rationalises scaffold. Achieves S1 (signature describes nodes, or countersigned). Teacher ratifies.
7. **After ratification**, all STM klines involved in Frame 2's ratification are promoted to frame — including S4 identity klines and S2/S3 partial klines.
8. Meanwhile, Frame 1's cogitation **continues**. Model is now richer (Frame 2's promotions are visible via shared references). Frame 1 eventually yields a new proposal.
9. Teacher evaluates. This time it matches. Countersigns. Frame 1's query is now structurally S1. All participating klines promoted.

```
Frame 1:  MHALL → SVO  (τ=1.0)
  query:     {MHALL: [S, V, O]}
  route:     S2 (some nodes match)
  cogitate:  yields proposal
  teacher:   does not match expectation

Frame 2:  scaffold for missing concept  (τ=1.0, base=Frame 1)
  query:     {S: M}
  route:     S1 (canonical — signature describes nodes)
  action:    ratify → promote all participating klines
  teacher:   countersigns

Frame 1 (continued):
  cogitation continues with enriched model
  yields new proposal → matches expectation
  teacher countersigns → structural S1, all participating klines promoted
```

### Scenario C — Temperature Adjustment

1. Teacher constructs script, extracts query, calls `rationalise(query)` at τ=1.0 → Frame 1 created, returns `False`.
2. Frame 1 cogitation runs but produces nothing the teacher can accept within a reasonable time.
3. Teacher constructs same query at τ=1.5, calls `rationalise(query)` → Frame 2 created over Frame 1.
4. At higher temperature, boundaries lower. Kalvin yields a proposal. Emits `frame` event.
5. Teacher evaluates. The kline matches expectation. Countersigns via `rationalise(reciprocal)`. All participating klines promoted to Frame 2.
6. Frame 1's cogitation continues. Model is now richer (Frame 2's promotions visible via shared references, including S4 and S2/S3 klines from the ratification). Frame 1 eventually yields a stronger result.
7. No frames abandoned. Frame stack represents the full learning history.

```
Frame 1:  MHALL → SVO  (τ=1.0)
  query:     {MHALL: [S, V, O]}
  route:     S2
  cogitate:  nothing acceptable within timeout
  teacher:   no action, proceeds to create Frame 2

Frame 2:  MHALL → SVO  (τ=1.5, base=Frame 1)
  query:     {MHALL: [S, V, O]}
  cogitate:  boundaries lowered → proposal
  teacher:   matches expectation → countersigns
  promote:   all participating klines (S1, S2, S3, S4)

Frame 1 (continued):
  cogitation continues with enriched model
  may reach S1 autonomously via enriched frame
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
- Study cannot achieve structural S1 on its own — it can produce proposals
  that await teacher countersignature, or discover structural S1 via
  countersignature in the existing sense (mutual cross-reference found during
  graph expansion).
- **S2 expansion** extends study with the ability to reshape partial understanding:
  when countersignature fails, the cogitator attempts to add missing nodes or
  remove redundant ones, generating proposals that move S2 klines toward
  canonical status. See `docs/extended-cogitation.md`.

The distinction between "training" and "study" is a matter of teacher presence, not system behavior. The cogitator runs the same way regardless.

---

## Monotonicity: Constructive Grounding

Monotonicity is a guarantee about **grounded information**, not about significance values. Significance may fluctuate — adding klines changes candidate sets and can alter distance calculations. But:

- Once a kline is promoted to a frame, it stays. Frames are append-only.
- Every kline promoted after ratification adds permanent structure to the knowledge graph — regardless of its individual significance level.
- S4 identity klines, once ratified and promoted, provide verified reference points for future rationalisation and cogitation.
- The model can only grow richer — never poorer.

The learning trajectory is monotonic in the sense that the **knowledge base only grows**. Individual significance measurements for a given query may go up or down as the model changes, but the total amount of grounded, ratifiable knowledge only increases.

---

## MVP Implementation Requirements

Four things are needed to implement the training loop:

### 1. Structural Grounding

S1 (grounded) is determined by structure, not by boundary classification. A kline is S1 if its signature fully describes its nodes or it is countersigned by another kline. After ratification, all STM klines involved in the ratification process are promoted to frame — not just the kline being ratified.

Implementation: modify the ratification path so that promotion occurs after ratification and applies to all participating STM klines. This ensures frames hold klines of all significance levels (S4 identity klines through S1 grounded klines), enriching the model available to future cogitation.

The determination of whether a kline is grounded is a structural interrogation: does the kline's signature fully describe its nodes (`make_signature(nodes) == signature`), or does a countersigned relationship exist?

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
| S1 promotion | Structural grounding, promotion after ratification | Full ratification semantics |
| Inter-frame enrichment | Implicit via shared references | Explicit enrichment API |
| Monotonicity | Grounding is constructive | Verified convergence metrics |
| Teacher | External coordinator | Kalvin-as-tutor (reentrant) |

---

## Relationship to Other Documents

| Document | Relationship |
|----------|-------------|
| `docs/learning.md` | The *principles* — why the system learns, what agency means, and why S1 requires ratification. |
| `docs/training-schedule.md` | The *practice* — what to teach (curriculum design), how to transition from training to operations (operational harness), and the Mary's World (MW) reference scenario. |
| This document | The *mechanism* — how the training loop works in terms of existing APIs and data structures. |
| `docs/extended-cogitation.md` | The *S2 expansion design* — how the Cogitator reshapes partial understanding into proposals. |

The key connections:

| learning.md principle | training-loop.md mechanism |
|---|---|
| "Temperature proposes; the trainer disposes" | Proposals in STM, teacher countersigns to ratify |
| "Learning is recursive rationalisation" | Each scaffolding step is a new frame, same pipeline |
| "S1 is ratified, not claimed" | Countersignature by teacher creates structural S1, all participating klines promoted |
| "Agency must be exercised" | Cogitation (study) runs on all frames independently |
| "The system learns what it doesn't know" | S4 identity klines and failed proposals identify gaps for scaffolding |
| "Learning is constructive" | Frame stack is append-only, all ratified klines promoted regardless of significance |

For the practical perspective on training — curriculum design, priming vs. querying strategies, and the operational harness — see `docs/training-schedule.md`.
