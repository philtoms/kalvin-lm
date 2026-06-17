# Sub-Plan: Agent — Significance Constants, Events, Agent, Cogitator

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 6–8
**Estimate:** 3 days
**Depends on:** Foundations (KLine, Signature, Tokenizer, STM — see @stm spec), Model

---

## 1. Spec References

See **@agent spec** for full definition (rationalisation pipeline, routing,
events).
See **@cogitator spec** for the slow path (Cogitator, CogitationHandler,
WorkItem, S2 expansion) and `plans/impl/cogitator.md` for its implementation.
Agent test matrix: AGT-1 through AGT-28 plus Serialization.
Cogitation test matrix (AGT-29..AGT-42) is in @cogitator spec.

See **@model spec** §Significance Semantics for constants and distance rules.

### Key API (from spec)

**Agent:**
```python
Agent(tokenizer=None, model=None)
agent.rationalise(kline) → bool
agent._route(query, candidate) → str   # "S1"/"S2"/"S3"/"S4"
```

**Events:**
```python
EventBus.subscribe(callback) → None
EventBus.publish(event) → None
RationaliseEvent(kind, query, proposal, significance)
```

**WorkItem:**
```python
WorkItem(query=KLine, candidate=KLine)
```

**Cogitator:** background daemon thread, processes work items via `model.expand()`,
emits `done` event after idle timeout. Defined in @cogitator spec; implemented
in `src/kalvin/cogitator.py` (see `plans/impl/cogitator.md`).

### Significance Constants (Phase 6)

> **Note:** Constants (`D_MAX`, `MASK64`) are defined in `model.py`.
> See @model spec §Significance Semantics.

### Test Mapping

| Spec ID | Test               | Description                |
| ------- | ------------------ | -------------------------- |
| SGF-1   | D_MAX value        | Equals 0xFFFF_FFFF_FFFF_FFFF |
| SGF-2   | MASK64 value       | Equals 0xFFFF_FFFF_FFFF_FFFF |

---

## 2. Events Implementation (Phase 7)

**Files:** `src/kalvin/events.py`, `tests/test_events.py`
**Depends on:** Kline (for RationaliseEvent)
**Estimate:** 0.5 day

See **@agent spec** §Events for event types, structure, and bus API.
Test matrix: AGT-23 through AGT-27.

### Implementation Skeleton

---

## 3. Agent + Cogitator Implementation (Phase 8)

**Files:** `src/kalvin/agent.py`, `tests/test_agent.py`
**Depends on:** Everything (Cogitator via @cogitator spec)
**Estimate:** 2 days

See **@agent spec** for full definition (rationalisation pipeline phases,
routing, cogitation, work items, S2 expansion).
Test matrix: AGT-1 through AGT-40.

### Routing Implementation — `_route(query, candidate) → str`

```python
@staticmethod
def _route(query: KLine, candidate: KLine) -> str:
    nodes = query.nodes
    if not nodes:
        return "S4"
    candidate_nodes = set(candidate.nodes)
    match_count = sum(1 for n in nodes if n in candidate_nodes)
    if match_count == len(nodes):
        return "S1"
    elif match_count > 0:
        return "S2"
    else:
        return "S3"
```

### Rationalisation Implementation

See **@agent spec** §Rationalisation for the 6-phase pipeline with fast/slow split.

### WorkItem

See **@cogitator spec** §Work Items. Defined in `src/kalvin/cogitator.py`.

### Cogitator Implementation

See **@cogitator spec** §Processing and `plans/impl/cogitator.md`.
`src/kalvin/agent.py` imports `Cogitator`/`CogitationHandler`/`WorkItem`
from `src/kalvin/cogitator.py` and wires itself as the handler.

**Key implementation detail:** expansion logic lives in `src/kalvin/expand.py`.
See `plans/impl/structural-grounding.md` for the full expansion algorithm.

---

## 4. Test Mapping

### Phase 1: Prepare

| Spec ID | Test                | Description                                 |
| ------- | ------------------- | ------------------------------------------- |
| AGT-7   | Signature assigned  | KLine with sig=0 gets make_signature(nodes) |
| AGT-8   | Signature preserved | KLine with existing sig unchanged           |

### Phase 2: Ground Check

| Spec ID | Test                     | Description                        |
| ------- | ------------------------ | ---------------------------------- |
| AGT-9   | First rationalise        | Returns True, adds to model        |
| AGT-10  | Duplicate rationalise    | Returns True, emits "ground" event |
| AGT-11  | Different sig same nodes | Not a ground (different KLine)     |

### Phase 3: Assess

| Spec ID | Test                    | Description                         |
| ------- | ----------------------- | ----------------------------------- |
| AGT-12  | Unsigned (no nodes)     | Returns True, emits "frame" S4      |
| AGT-14  | Self-grounded canonical | Returns True when all nodes resolve |
| AGT-15  | Not self-grounded       | Falls through to Phase 4            |

### Phase 4: Retrieve Candidates

| Spec ID | Test             | Description                          |
| ------- | ---------------- | ------------------------------------ |
| AGT-16  | No candidates    | Agent returns True (S4 novel)        |
| AGT-17  | Candidates found | Proceeds to routing                  |

### Phase 5: Route Each Candidate

| Spec ID | Test                       | Description                                          |
| ------- | -------------------------- | ---------------------------------------------------- |
| AGT-18  | All candidates to cogitator | All candidates (including S1) submitted as WorkItems |
| AGT-19  | S1-first ordering          | Candidates sorted S1-first, then by overlap count    |
| AGT-20  | All S2                     | Returns False, all submitted as WorkItems             |
| AGT-21  | All S3                     | Returns False, all submitted as WorkItems             |
| AGT-22  | S1 fast-path in cogitator  | Skips expand(), calls on_s1 directly                  |
| AGT-38  | S2 before S1: no cogitator submit | ✅ test_s2_before_s1_no_cogitator_submit         |
| AGT-40  | Satisfaction guard         | S2/S3 for satisfied entry skipped                     |
| AGT-41  | Satisfaction-based completion | Lesson complete by satisfied count, not events     |
| AGT-42  | No re-fire on post-completion events | Lesson complete guard                       |

### Routing (`_route`)

| Spec ID | Test                              | Description                                |
| ------- | --------------------------------- | ------------------------------------------ |
| AGT-1   | All nodes match                   | Returns "S1"                               |
| AGT-2   | Some nodes match                  | Returns "S2"                               |
| AGT-3   | No nodes match                    | Returns "S3"                               |
| AGT-4   | Empty query                       | Returns "S4"                               |
| AGT-5   | Single node match                 | Returns "S1"                               |
| AGT-6   | Routing independent of signature  | Only candidate nodes matter, not signature  |

### Events

| Spec ID | Test                  | Description                              |
| ------- | --------------------- | ---------------------------------------- |
| AGT-23  | Subscribe and publish | Callback receives event                  |
| AGT-24  | Multiple subscribers  | All receive event                        |
| AGT-25  | Event fields          | kind, query, proposal, significance correct |
| AGT-26  | Thread safety         | Publish from another thread              |
| AGT-27  | Empty bus             | No crash on publish with no subscribers  |
| AGT-28  | Event delivery        | All events received in order             |

### Cogitation

Relocated to **@cogitator spec** (IDs AGT-29..AGT-42, stable).
See `plans/impl/cogitator.md` §Test Mapping for test functions.

### Serialization

| Spec ID | Test              | Description                           |
| ------- | ----------------- | ------------------------------------- |
| AGT-38  | JSON round-trip   | Save/load preserves KLines            |
| AGT-39  | Binary round-trip | Save/load preserves KLines            |
| AGT-40  | Empty agent       | Serializes and deserializes correctly |
