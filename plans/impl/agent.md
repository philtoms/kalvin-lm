# Sub-Plan: Agent — Significance Constants, Events, Agent, Cogitator

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 6–8
**Estimate:** 3 days
**Depends on:** Foundations (KLine, Signature, Tokenizer, STM — see @stm spec), Model

---

## 1. Significance Constants (Phase 6)

> **Note:** The standalone `significance` module has been removed. Constants
> are defined directly in `agent.py`. See `specs/significance.md` for the
> full conceptual specification.

**File:** `src/kalvin/agent.py` (module-level constants)
**Depends on:** Nothing
**Estimate:** 0.5 day

### Spec

Constants for distance/significance arithmetic, defined at the top of
`agent.py`. Routing and distance computation are performed by the Agent
and Cogitator directly.

```python
D_MAX  = 0xFFFF_FFFF_FFFF_FFFF   # maximum distance (S4)
MASK64 = 0xFFFF_FFFF_FFFF_FFFF   # 64-bit mask for inversion
```

**Routing** (performed by `Agent._route(Q, C)`) — pure node-membership test,
no model dependency:

```
route(Q, C):
  if Q has no nodes:   return "S4"
  match_count = |{n ∈ Q.nodes : n ∈ C.nodes}|
  if match_count == len(Q.nodes):  return "S1"
  if match_count > 0:              return "S2"
  else:                            return "S3"
```

**Inversion** (performed by Cogitator):

```python
significance = (~distance) & MASK64
```

### Test Cases

Tests for these constants are inlined in `tests/test_agent.py`.

| Test               | Description                |
| ------------------ | -------------------------- |
| D_MAX value        | Equals 0xFFFF_FFFF_FFFF_FFFF |
| MASK64 value       | Equals 0xFFFF_FFFF_FFFF_FFFF |

---

## 2. Events (Phase 7)

**Files:** `src/kalvin/events.py`, `tests/test_events.py`
**Depends on:** Kline (for RationaliseEvent)
**Estimate:** 0.5 day

### Spec

Pub/sub for rationalisation observers.

**Event types:**

| Kind     | Trigger                  | Significance      |
| -------- | ------------------------ | ----------------- |
| `ground` | KLine already exists     | S1 (all bits set) |
| `frame`  | KLine integrated         | S1–S4             |
| `done`   | Cogitation backlog empty | 0                 |

**Event structure:**

```python
class RationaliseEvent:
    kind: str           # "ground", "frame", "done"
    query: KLine        # The KLine being rationalised
    value: KLine        # The matching or resulting KLine
    significance: int   # Significance value
```

**Event bus:**

```python
class EventBus:
    subscribe(callback) → None
    publish(event) → None   # Thread-safe, synchronous delivery
```

### Test Cases

| Test                  | Description                              |
| --------------------- | ---------------------------------------- |
| Subscribe and publish | Callback receives event                  |
| Multiple subscribers  | All receive event                        |
| Event fields          | kind, query, value, significance correct |
| Thread safety         | Publish from another thread              |
| Empty bus             | No crash on publish with no subscribers  |

---

## 3. Agent + Cogitator (Phase 8)

**Files:** `src/kalvin/agent.py`, `tests/test_agent.py`
**Depends on:** Everything
**Estimate:** 2 days

### Agent Spec

The orchestrator of the rationalisation pipeline with a fast/slow split.

**Structure:**

```python
class Agent:
    tokenizer: KTokenizer   # Default: Mod32Tokenizer
    model: Model             # Three-tier knowledge graph
    cogitator: Cogitator     # Background work-item processor
    events: EventBus         # Pub/sub
```

**Construction:**

```python
Agent(tokenizer=None, model=None)
# Defaults: Mod32Tokenizer, empty Model with tokenizer's is_literal_fn
# Cogitator is created internally and starts immediately.
```

### Routing — `_route(query, candidate) → str`

Inline static method. No model call.

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

### Rationalisation — Fast/Slow Split

```
Phase 1: PREPARE
  If Q.signature == 0 and Q has nodes:
    Q.signature = make_signature(Q.nodes, tokenizer.is_literal)

Phase 2: GROUND CHECK
  If model.exists(Q):
    emit "ground" event, return True (S1)

Phase 3: ASSESS
  If Q has no nodes:
    model.add(Q), emit "frame" S4, return True
  If all nodes are literal:
    model.add(Q), emit "frame" S1, return True
  If Q.signature == make_signature(Q.nodes) AND
     all non-literal nodes resolve in model:
    model.add(Q), emit "frame" S1, return True

Phase 4: RETRIEVE CANDIDATES
  candidates = model.where(Q.signature)   # AND overlap
  If no candidates:
    model.add(Q), emit "frame" S4, return True (novel)

Phase 5: ROUTE EACH CANDIDATE
  model.add(Q)
  For each candidate C:
    level = _route(Q, C)              # no model call
    If S1: promote Q, emit "frame" S1, return True (done)
    If S2: cogitator.submit(WorkItem(Q, C, "S2"))
    If S3: cogitator.submit(WorkItem(Q, C, "S3"))
  return False                        # all routed S2/S3
```

**S1 short-circuits**: the first candidate that routes as S1 terminates the
loop immediately. No further candidates are routed. No distance is computed.

### WorkItem

```python
class WorkItem(NamedTuple):
    query: KLine
    candidate: KLine
    level: str  # "S2" or "S3"
```

### Cogitator Spec

Background work-item processor. Receives pre-routed WorkItems, computes deep
significance, checks countersignature.

```python
class Cogitator:
    model: Model
    event_bus: EventBus
    on_s1: Callable          # callback for S1 discovery
    timeout: float           # default 2.0s
    backlog: queue[WorkItem]
    thread: daemon thread
```

**Processing per work item:**

```
run(WorkItem(query, candidate, level)):
  for qc in model.expand(query, candidate, level):
    process(qc)

process(QueryCandidate(query, candidate, distance)):
  1. significance = (~distance) & MASK64
  2. if model.is_countersigned(query, candidate):
       model.add(candidate)
       on_s1(query, candidate)    # triggers re-rationalisation
```

`model.expand()` yields intermediate `QueryCandidate` items for each
discovered connotation (S2 and S3 indirect relationships), followed by a
terminal `QueryCandidate` with the packed distance for the original pair.
Each is processed identically — countersignature is checked for every
yielded result.

**MVP:** The routed level (S2/S3) is preserved — significance is computed but
not used for re-routing. Countersignature is the only mechanism that can promote
to S1 during cogitation. Connotation expansion via `model.expand()` provides
graph exploration beyond the initial query-candidate pair.

**Lifecycle:**

- Runs in a background daemon thread.
- Pulls work items from a backlog queue.
- When backlog is empty for `timeout` seconds → emit `"done"` event so
  subscribers can realign. Does **not** halt — resets idle timer and
  continues processing new work items.
- Can be stopped via `cogitate_join(timeout)`.

### Countersignature Test

```python
def is_countersigned(Q, C):
    return (C.signature in Q.nodes) and (Q.signature in C.nodes)
```

---

## 4. Test Cases

### Phase 1: Prepare

| Test                | Description                                 |
| ------------------- | ------------------------------------------- |
| Signature assigned  | KLine with sig=0 gets make_signature(nodes) |
| Signature preserved | KLine with existing sig unchanged           |

### Phase 2: Ground Check

| Test                     | Description                        |
| ------------------------ | ---------------------------------- |
| First rationalise        | Returns True, adds to model        |
| Duplicate rationalise    | Returns True, emits "ground" event |
| Different sig same nodes | Not a ground (different KLine)     |

### Phase 3: Assess

| Test                    | Description                         |
| ----------------------- | ----------------------------------- |
| Unsigned (no nodes)     | Returns True, emits "frame" S4      |
| All-literal             | Returns True, emits "frame" S1      |
| Self-grounded canonical | Returns True when all nodes resolve |
| Not self-grounded       | Falls through to Phase 4            |

### Phase 4: Retrieve Candidates

| Test             | Description                          |
| ---------------- | ------------------------------------ |
| No candidates    | Agent returns True (S4 novel)        |
| Candidates found | Proceeds to routing                  |

### Phase 5: Route Each Candidate

| Test                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| First candidate S1         | Returns True, promotes, no further candidates tested |
| Second candidate S1        | First routes S2, second S1, returns True             |
| All S2                     | Returns False, all submitted as WorkItems            |
| All S3                     | Returns False, all submitted as WorkItems            |
| S1 short-circuits expansion | model.expand never called when S1 found            |

### Routing (`_route`)

| Test                              | Description                                |
| --------------------------------- | ------------------------------------------ |
| All nodes match                   | Returns "S1"                               |
| Some nodes match                  | Returns "S2"                               |
| No nodes match                    | Returns "S3"                               |
| Empty query                       | Returns "S4"                               |
| Single node match                 | Returns "S1"                               |
| Routing independent of signature  | Only candidate nodes matter, not signature  |

### WorkItem

| Test              | Description                      |
| ----------------- | -------------------------------- |
| Field access      | query, candidate, level correct  |
| Equality          | Same fields → equal              |

### Cogitation (Cogitator)

| Test                        | Description                                    |
| --------------------------- | ---------------------------------------------- |
| Countersignature discovery  | S2 → S1 via countersignature in cogitation      |
| Join                        | Thread stops cleanly                           |
| S2 submits work item        | WorkItem queued with correct fields            |
| Rationalise after join      | Works without cogitation thread                |

### Events

| Test           | Description                  |
| -------------- | ---------------------------- |
| Event delivery | All events received in order |
| Ground event   | Correct kind + significance  |
| Frame event    | Correct kind + significance  |

### Serialization

| Test              | Description                           |
| ----------------- | ------------------------------------- |
| JSON round-trip   | Save/load preserves KLines            |
| Binary round-trip | Save/load preserves KLines            |
| Empty agent       | Serializes and deserializes correctly |
