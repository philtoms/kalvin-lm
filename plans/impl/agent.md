# Sub-Plan: Agent — Significance Constants, Events, Agent, Cogitator

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 6–8
**Estimate:** 3 days
**Depends on:** Foundations (KLine, Signature, Tokenizer, STM — see @stm spec), Model

---

## 1. Spec References

See **@agent spec** for full definition (rationalisation pipeline, routing,
cogitation, events, work items).
Test matrix: AGT-1 through AGT-40.

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
emits `done` event after idle timeout.

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
**Depends on:** Everything
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

See **@agent spec** §Work Items. Named tuple: `(query: KLine, candidate: KLine)`.

### Cogitator Implementation

See **@agent spec** §Cogitation for processing semantics.

**Key implementation detail:** `_process` handles S2 expansion only.
Ratification handled upstream in `rationalise()`.
See `plans/impl/structural-grounding.md` for full expansion algorithm.

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
| AGT-13  | All-literal             | Returns True, emits "frame" S1      |
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
| AGT-18  | First candidate S1         | Returns True, promotes, no further candidates tested |
| AGT-19  | Second candidate S1        | First routes S2, second S1; S2 deferred and discarded, zero cogitator submissions |
| AGT-20  | All S2                     | Returns False, deferred items submitted as WorkItems  |
| AGT-21  | All S3                     | Returns False, deferred items submitted as WorkItems  |
| AGT-22  | S1 short-circuits expansion | model.expand never called when S1 found            |
| AGT-38  | S2 before S1: no cogitator submit | ✅ test_s2_before_s1_no_cogitator_submit         |

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

| Spec ID | Test                        | Description                                    |
| ------- | --------------------------- | ---------------------------------------------- |
| AGT-29  | Countersignature discovery  | S2 → S1 via countersignature in cogitation      |
| AGT-30  | Join                        | Thread stops cleanly                           |
| AGT-31  | S2 submits work item        | WorkItem queued with correct fields            |
| AGT-32  | All yields processed        | Every QC from expand() is evaluated            |
| AGT-33  | S1 detection                | High-significance QC triggers on_s1 callback   |
| AGT-34  | S2/S3 expansion             | Non-canonical QC triggers expansion proposals  |
| AGT-35  | Proposals at any significance | S2 and S3 proposals emitted as frame events |
| AGT-36  | Boundary S1 + structural check | Promotion only on structural S1            |
| AGT-37  | Boundary S1 + structural S1 | Promotion occurs                              |
| AGT-39  | Cogitator break-on-S1       | ✅ test_cogitator_stops_on_s1 — on_s1 exactly once, no expansions after |

### Serialization

| Spec ID | Test              | Description                           |
| ------- | ----------------- | ------------------------------------- |
| AGT-38  | JSON round-trip   | Save/load preserves KLines            |
| AGT-39  | Binary round-trip | Save/load preserves KLines            |
| AGT-40  | Empty agent       | Serializes and deserializes correctly |
