# Sub-Plan: Agent — Significance Constants, Events, Agent, Cogitator

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 6–8
**Estimate:** 3 days
**Depends on:** Foundations (KLine, Signature, Tokenizer, STM — see @stm spec), Model

---

## 1. Significance Constants (Phase 6)

> **Note:** The standalone `significance` module has been removed. Constants
> (`D_MAX`, `MASK64`) are defined in `model.py`. See `specs/significance.md`
> for the full conceptual specification.

**File:** `src/kalvin/model.py` (module-level constants)
**Depends on:** Nothing
**Estimate:** 0.5 day

### Spec

Constants for distance/significance arithmetic, defined at the top of
`model.py`. Routing is performed by the Agent, distance computation and
significance inversion by the Model's `expand()`.

```python
D_MAX  = 0xFFFF_FFFF_FFFF_FFFF   # maximum distance, also max significance
MASK64 = 0xFFFF_FFFF_FFFF_FFFF   # 64-bit mask for bitwise inversion
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

**Inversion** (performed by Model's `expand()`):

```python
significance = (~packed_distance) & MASK64
# Inverted inside expand(), callers receive significance directly
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
    proposal: KLine     # The matching or resulting KLine
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
| Event fields          | kind, query, proposal, significance correct |
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
# Defaults: Mod32Tokenizer, empty Model
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
    Q.signature = make_signature(Q.nodes)

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
```

### Cogitator Spec

Background work-item processor. Receives pre-routed WorkItems, computes deep
significance, classifies against boundaries, checks countersignature. Response
sampling controls stream consumption.

```python
class Sampling:
    temperature: float   # (0, ∞). Default 1.0.
    top_k:       int     # [0, ∞). 0 = unlimited. Default 40.
    top_p:       float   # (0, 1.0]. Default 0.95.

class Cogitator:
    model: Model
    event_bus: EventBus
    on_s1: Callable          # callback for S1 discovery
    timeout: float           # default 2.0s
    sampling: Sampling       # response sampling parameters
    backlog: queue[WorkItem]
    thread: daemon thread
```

**Boundary computation:**

```
base_boundaries():
  s12 = D_MAX - 1              # S1|S2: only exact S1 qualifies
  s23 = ~_S2_S3_DISTANCE        # S2|S3: packed distance threshold
  s34 = 0                       # S3|S4: only zero is S4
  return (s12, s23, s34)

boundaries(τ):
  if τ == 1.0: return base_boundaries()
  shift = _TEMP_SCALE × (τ - 1.0)
  s12 = clamp(s12_base - shift, 0, D_MAX - 1)
  s23 = clamp(s23_base - shift, 0, D_MAX - 1)
  s34 = clamp(s34_base - shift, 0, D_MAX - 1)
  return (s12, s23, s34)

classify(sig, s12, s23, s34):
  if sig >= s12: return "S1"
  if sig >= s23: return "S2"
  if sig >= s34: return "S3"
  return "S4"
```

**Processing per work item — boundary-based streaming:**

```
run_work_item(WorkItem(query, candidate)):
  (s12, s23, s34) = boundaries(sampling.temperature)
  evidence_target = sampling.top_p × D_MAX
  count = 0, cumulative = 0

  for qc in model.expand(query, candidate):
    band = classify(qc.significance, s12, s23, s34)

    if band == "S4":
      continue                    # demote

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
    on_s1(query, candidate)       # re-rationalise
```

Boundaries are computed once per work item. Raw significance is never
mutated — temperature acts on the boundaries, not on the values. High τ
lowers S1|S2, allowing near-S1 S2 connotations to be classified as S1.

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
| Field access      | query, candidate correct         |
| Equality          | Same fields → equal              |

### Cogitation (Cogitator)

| Test                        | Description                                    |
| --------------------------- | ---------------------------------------------- |
| Countersignature discovery  | S2 → S1 via countersignature in cogitation      |
| Join                        | Thread stops cleanly                           |
| S2 submits work item        | WorkItem queued with correct fields            |
| Rationalise after join      | Works without cogitation thread                |

### Sampling

| Test                          | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| Default values                | temperature=1.0, top_k=40, top_p=0.95               |
| Validation rejects bad        | temp≤0, top_k<0, top_p∉(0,1]                        |
| Agent.sampling property       | Read/write proxy to cogitator                       |
| Base boundaries               | S1\|S2=D_MAX-1, S2\|S3=HP, S3\|S4=0                 |
| Boundary identity at τ=1      | boundaries() == base_boundaries()                   |
| High τ lowers S1\|S2          | τ>1 → S1\|S2 boundary drops                         |
| High τ lowers S2\|S3          | τ>1 → S2\|S3 boundary drops                         |
| High τ caps S3\|S4 at 0       | τ>1 → S3\|S4 stays at 0                             |
| Low τ raises S2\|S3           | τ<1 → S2\|S3 boundary rises                         |
| Low τ raises S3\|S4           | τ<1 → S3\|S4 boundary rises                         |
| Low τ caps S1\|S2 at D_MAX-1  | τ<1 → S1\|S2 stays at D_MAX-1                       |
| Monotonic S1\|S2 with τ       | Higher τ → lower S1\|S2 boundary                   |
| Classify S1/S2/S3/S4          | Correct classification against boundaries           |
| High τ promotes S2 to S1      | Near-S1 S2 QC classified as S1 at high τ            |
| S4 demotion                   | Low τ raises S3\|S4, QCs below it demoted           |
| Top-k budget                  | k=2 processes at most 2 QCs per work item           |
| Top-k unlimited               | k=0 processes all QCs                               |
| Top-p early stop              | p<1.0 stops before exhausting generator             |
| Top-p no stop                 | p=1.0 never triggers early stop                     |
| Boundary + top-k composition  | Classification filters, then top-k caps             |

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
