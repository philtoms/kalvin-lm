# Kalvin — The Training Loop

## Overview

The training loop is a dialogue between a **teacher** (human or automated) and a **kalvin agent**, mediated by three existing mechanisms: `agent.rationalise()`, the **event bus**, and **countersignature**.

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

| Kline type                                   | How it enters STM                            | Promoted? | When                                |
| -------------------------------------------- | -------------------------------------------- | --------- | ----------------------------------- |
| Canonical S1 (all-literal, self-grounded)    | Fast path                                    | Yes       | Immediately (self-ratifying)        |
| Countersigned S1                             | Cogitation discovers countersignature        | Yes       | When countersignature detected      |
| Teacher-ratified kline                       | Cogitation → proposal → teacher countersigns | Yes       | After teacher ratification          |
| S4 identity kline (involved in ratification) | No candidates → novel                        | Yes       | After ratification of related kline |
| S2/S3 kline (involved in ratification)       | Cogitation                                   | Yes       | After ratification of related kline |
| S2/S3 kline (not involved in ratification)   | Cogitation                                   | No        | Stays in STM for further cogitation |

---

## Frames: One per Script

A **frame** is the session-level write surface inside the agent's model (STM → Frame → Base). Under the training loop, **one agent (and therefore one frame) is created per training script**. All rationalise calls for that script operate within the scope of this single frame.

```
Script 1 processing:
  rationalise(query_1)  →  Agent:  STM → Frame → Base
  rationalise(query_2)  →  Agent:  STM → Frame → Base   (same agent)
  rationalise(query_3)  →  Agent:  STM → Frame → Base   (same agent)
  ...

Script 2 processing:
  rationalise(query_4)  →  Agent:  STM → Frame → Base   (new agent)
  ...
```

### Why One Frame per Script

The previous design created a new frame for every call to rationalise, resulting in a growing frame stack with reference-sharing between frames. This introduced unnecessary complexity:

- **Frame stack management** — the teacher had to track and subscribe to multiple agent instances.
- **Cross-frame sharing** — promotion in a later frame was intended to be visible to earlier frames' cogitation via shared references, requiring careful coordination.
- **Sampling per frame** — each frame needed its own sampling parameters set at construction time.

In practice, all training loop processing for a given script operates in a single, flat scope. Every rationalise call for that script adds to the same STM, cogitates against the same frame, and promotes to the same frame. No additional sharing mechanism is needed — it is simply the same model.

### Agent Construction

When a new script is processed, the teacher constructs an agent:

```python
agent = Agent(
    model=Model(base=existing_model),
)
```

Or, if this is the first script:

```python
agent = Agent()
```

All queries from the script are then rationalised through this single agent instance. The agent's event bus is the single channel for all proposals and events.

### Promotion is Simply STM → Frame

Within the single-agent model, promotion is the existing mechanism: klines move from STM to frame when ratified. There is no frame stack, no layering, and no cross-frame visibility concern. The model's three-tier lookup (STM → Frame → Base) already provides the correct semantics — recent activity in STM, confirmed knowledge in frame, accumulated wisdom in base.

### Sampling Parameters as Agent Properties

Sampling parameters (temperature, top_k, top_p) are agent-level properties, accessible via `agent.sampling`. They control the cogitator's boundary computation for all work items processed by this agent. They can be adjusted between rationalise calls if desired:

```python
agent.sampling = Sampling(temperature=1.5, top_k=40, top_p=0.95)
```

There is no per-frame or per-call sampling isolation. Sampling parameters apply uniformly to all cogitation within the agent. This is the simplest model and is correct for the training loop: the teacher sets the exploration parameters for the session and adjusts as needed.

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

No KScript changes are required for the MVP. Sampling parameters are set on the agent directly. Future work may add metadata annotations to KScript for encoding `temp`, `top_k`, `top_p` alongside the script.

---

## Ratification: Countersignature as Endorsement

Ratification is not a new operation. It is the existing countersignature mechanism, triggered by the teacher:

1. Kalvin rationalises a query, cogitates, and emits a `frame` event with a proposed kline (e.g., `{A: [B, C]}`)
2. The teacher evaluates the proposal against the expectation
3. If acceptable, the teacher **countersigns** by rationalising the reciprocal kline through `agent.rationalise()`
4. This creates the mutual cross-reference — structural S1 — and the kline is now grounded
5. **After ratification**, all STM klines involved in the ratification process are promoted to frame

```
agent proposes:        {A: [B, C]}       (emitted as frame event)
teacher countersigns:  agent.rationalise(Kline(B | C, [A]))
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
Teacher                              Agent
  │                                    │
  │  compile KScript                   │
  │  construct agent                   │
  │  set sampling params               │
  │  rationalise(query_kline) ────────→│  rationalise returns True/False
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
  │     rationalise(scaffold_kline)───→│  Same agent, same frame
  │     (optionally adjust sampling)   │  Enriches model
  │                                    │
  │     [continue listening for        │  Cogitation continues
  │      events from same agent]       │  on background thread
  │                                    │
  └── If no response (timeout)         │
        adjust sampling params         │
        rationalise(query_kline) ─────→│  Same agent, new params
        (same kline, new sampling)     │
```

### Event Correlation

The `RationaliseEvent` carries `query` and `proposal` klines. The teacher correlates via signature — "I submitted a query with signature X, and I got back a frame event for a query with signature X." For the MVP, the teacher tracks one query at a time and uses signature matching to correlate events.

### The Feedback Loop is Synchronised by Return Value

`rationalise()` returns `True` (fast path: S1 or S4, done) or `False` (slow path: S2/S3, cogitation in progress). The teacher uses this to coordinate:

- `True` → the event has already been emitted. Evaluate immediately.
- `False` → cogitation is running. Listen for events. The cogitator emits `frame` events as it discovers proposals, and `done` when the backlog empties.

---

## Scenarios

### Scenario A — Immediate Success

1. Teacher constructs agent, compiles script, extracts query, calls `rationalise(query)` → returns `True`.
2. Agent routes query to S1 (canonical match — signature describes nodes). Auto-promotes. Emits `frame` event.
3. Teacher receives event. Kline matches expectation exactly.
4. Teacher countersigns: `rationalise(reciprocal_kline)`. Structural S1 confirmed.
5. All STM klines involved in this ratification are promoted to frame.

```
Agent:
  query:     {MHALL: [M, H, A, L, L]}
  candidate: {MHALL: [S, V, O]}  (exists in model)
  route:     S1 (all nodes match, signature describes nodes)
  action:    auto-promote (self-ratifying)
  teacher:   countersigns → rationalise({SVO: [M, H, A, L, L]})
  result:    structural S1 grounded, participating klines promoted
```

### Scenario B — Scaffolding Required

1. Teacher constructs agent, compiles script, extracts query, calls `rationalise(query)` → returns `False`.
2. Agent routes query to S2. Queues for cogitation.
3. Cogitation yields a proposal. Emits `frame` event. Kline added to STM.
4. Teacher receives event. Kline does not match expectation.
5. Teacher constructs new script covering the misaligned information. Calls `rationalise(scaffold)` on the same agent.
6. Agent rationalises scaffold. Achieves S1 (signature describes nodes, or countersigned). Teacher ratifies.
7. **After ratification**, all STM klines involved in the ratification are promoted to frame — including S4 identity klines and S2/S3 partial klines.
8. Meanwhile, the agent's cogitation **continues** on the background thread. The model is now richer (the scaffold's promotions are visible because everything is in the same model). Cogitation eventually yields a new proposal.
9. Teacher evaluates. This time it matches. Countersigns. The original query is now structurally S1. All participating klines promoted.

```
Agent:
  Step 1: rationalise({MHALL: [S, V, O]})  (τ=1.0)
    route:     S2 (some nodes match)
    cogitate:  yields proposal
    teacher:   does not match expectation

  Step 2: rationalise({S: M})  (τ=1.0, same agent)
    route:     S1 (canonical — signature describes nodes)
    action:    ratify → promote all participating klines
    teacher:   countersigns

  Step 3: cogitation continues with enriched model
    yields new proposal → matches expectation
    teacher countersigns → structural S1, all participating klines promoted
```

### Scenario C — Sampling Adjustment

1. Teacher constructs agent, compiles script, extracts query, calls `rationalise(query)` at τ=1.0 → returns `False`.
2. Cogitation runs but produces nothing the teacher can accept within a reasonable time.
3. Teacher adjusts sampling: `agent.sampling = Sampling(temperature=1.5)`.
4. Teacher submits same query at τ=1.5, calls `rationalise(query)` on the same agent.
5. At higher temperature, boundaries lower. Agent yields a proposal. Emits `frame` event.
6. Teacher evaluates. The kline matches expectation. Countersigns via `rationalise(reciprocal)`. All participating klines promoted.
7. Cogitation continues on the background thread. The model is now richer. Further cogitation may yield additional results.

```
Agent:
  Step 1: rationalise({MHALL: [S, V, O]})  (τ=1.0)
    route:     S2
    cogitate:  nothing acceptable within timeout
    teacher:   adjusts sampling

  Step 2: agent.sampling = Sampling(temperature=1.5)
          rationalise({MHALL: [S, V, O]})  (τ=1.5, same agent)
    cogitate:  boundaries lowered → proposal
    teacher:   matches expectation → countersigns
    promote:   all participating klines (S1, S2, S3, S4)
```

### The Sampling Pattern

Sampling parameters are agent-level properties, not per-call or per-frame. They apply uniformly to all cogitation within the agent. The teacher can adjust them between rationalise calls if the current settings are not producing acceptable proposals. There is no nesting, stacking, or isolation — just a flat property on the agent.

---

## The Teacher's Responsibilities

The teacher is an external coordinator that:

1. **Compiles training scripts** into kline queries and expectations using KScript.
2. **Constructs an agent** for each script being processed.
3. **Sets sampling parameters** on the agent as needed (via `agent.sampling`).
4. **Subscribes to the agent's event bus** for proposals and events.
5. **Compares proposals against expectations** using kline equality (MVP) or reentrant rationalisation (future).
6. **Countersigns acceptable proposals** by rationalising the reciprocal kline through the same agent.
7. **Constructs corrective scaffolding** when proposals don't match expectations.
8. **Adjusts sampling** by modifying `agent.sampling` when the current settings are not effective.

The teacher does not modify kalvin internals. It does not roll back model state. It does not force promotion. It does not manage a frame stack. It uses `rationalise()` and listens to events, the same interface available to any observer.

---

## Study: Cogitation Between Training Interactions

Study is not a separate mechanism. It is the existing cogitator's S2/S3 background processing, viewed from the learning perspective. Between teacher interactions:

- The agent's cogitator continues processing its backlog on the background thread.
- Cogitation may discover countersignature relationships, triggering re-rationalisation.
- The enriched model (from prior promotions) may cause cogitation to produce stronger results over time.
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

S1 (grounded) is determined by structure, not by boundary classification. A kline is S1 if its signature fully describes its nodes or it is countersigned by another kline. After ratification, all STM klines involved in the ratification process are promoted to frame — not just the kline being ratified. This ensures frames hold klines of all significance levels (S4 identity klines through S1 grounded klines), enriching the model available to future cogitation.

The determination of whether a kline is grounded is a structural interrogation: does the kline's signature fully describe its nodes (`make_signature(nodes) == signature`), or does a countersigned relationship exist?

### 2. Agent Construction per Script

The teacher constructs one agent per training script. The agent is created with an optional base model for long-term knowledge:

```python
agent = Agent(model=Model(base=existing_model))
```

All queries from the script are rationalised through this single agent instance. No frame factory or convenience method is needed — standard `Agent()` construction is sufficient.

### 3. Event Subscription

The agent has a single event bus. The teacher subscribes to it for all proposals and events. Events are published synchronously in publication order.

### 4. Kline Equality Comparison

Already exists in the spec. The teacher uses it directly: `event_kline == expectation_kline`.

---

## MVP vs. Future

| Capability          | MVP                                                | Future                                          |
| ------------------- | -------------------------------------------------- | ----------------------------------------------- |
| Trainer comparison  | Exact kline equality                               | Reentrant rationalisation (kalvin-as-tutor)     |
| Sampling parameters | Agent-level properties, adjustable between calls   | KScript metadata encoding                       |
| Frame model         | One frame per script (single agent)                | Multi-agent orchestration for complex curricula |
| S1 promotion        | Structural grounding, promotion after ratification | Full ratification semantics                     |
| Monotonicity        | Grounding is constructive                          | Verified convergence metrics                    |
| Teacher             | External coordinator                               | Kalvin-as-tutor (reentrant)                     |

---

## Relationship to Other Documents

| Document                      | Relationship                                                                                                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/learning.md`            | The _principles_ — why the system learns, what agency means, and why S1 requires ratification.                                                                         |
| `docs/training-schedule.md`   | The _practice_ — what to teach (curriculum design), how to transition from training to operations (operational harness), and the Mary's World (MW) reference scenario. |
| This document                 | The _mechanism_ — how the training loop works in terms of existing APIs and data structures.                                                                           |
| `docs/extended-cogitation.md` | The _S2 expansion design_ — how the Cogitator reshapes partial understanding into proposals.                                                                           |

The key connections:

| learning.md principle                        | training-loop.md mechanism                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------ |
| "Temperature proposes; the trainer disposes" | Proposals in STM, teacher countersigns to ratify                                     |
| "Learning is recursive rationalisation"      | Each scaffolding step is rationalised through the same agent                         |
| "S1 is ratified, not claimed"                | Countersignature by teacher creates structural S1, all participating klines promoted |
| "Agency must be exercised"                   | Cogitation (study) runs on the agent's background thread independently               |
| "The system learns what it doesn't know"     | S4 identity klines and failed proposals identify gaps for scaffolding                |
| "Learning is constructive"                   | Frame is append-only, all ratified klines promoted regardless of significance        |
