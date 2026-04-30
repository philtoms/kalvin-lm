# Kalvin: Build-From-Scratch Implementation Plan

**Purpose:** A self-contained blueprint for implementing the Kalvin rationalisation agent from zero — no reference to any existing codebase.

**Date:** 2026-04-29

---

## 0. What is Kalvin?

Kalvin is an **agent** — a system that receives new information, evaluates how it relates to existing knowledge, and autonomously decides what to do next. This capacity for choice is called **Agency**, and it arises from a mechanism called **Significance**.

The system operates on two fundamental concepts:

- **Nodes** — opaque 64-bit unsigned integers (the atoms).
- **KLines** — identified, ordered sequences of nodes (the structures).

New information arrives as KLines. The agent **rationalises** each KLine against its existing knowledge graph, producing a significance level (S1–S4) that determines the next action. The knowledge graph grows by composing existing nodes into increasingly complex hierarchies.

### The Core Pipeline

```
Input Text → Tokenizer → Nodes → KLine → Agent → Model (Knowledge Graph)
                                         ↓
                              Significance Pipeline
                              (S1/S2/S3/S4 routing)
                                         ↓
                              Integrate / Cogitate / Promote
```

### Significance Levels

| Level | Meaning                    | Agent Action          |
| ----- | -------------------------- | --------------------- |
| S1    | "I fully understand this." | Confirm. Promote.     |
| S2    | "I understand some."       | Queue for cogitation. |
| S3    | "This is reminiscent."     | Queue for cogitation. |
| S4    | "This is completely new."  | Novel. Promote.       |

---

## 1. System Architecture

### 1.1 Component Dependency Graph

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent (orchestrator)                       │
│  Depends on: Tokenizer, Model, Signature, Significance, Events   │
├──────────────────────────────────────────────────────────────────┤
│  Significance Pipeline                                            │
│  Depends on: Model (s2_distance, s3_distance)                    │
│  Routing is self-contained (node-membership, no model call)       │
├──────────────────────────────────────────────────────────────────┤
│  Model (STM → Frame → Base)                                      │
│  Depends on: Kline, Signature, STM                               │
├──────────┬──────────┬──────────────┬──────────────────────────────┤
│ Kline    │ Signature│   Tokenizer  │   Events                     │
│          │          │  (Mod / BPE) │   (EventBus)                 │
├──────────┴──────────┴──────────────┴──────────────────────────────┤
│  Nodes (uint64) — the universal atom                              │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Build Order (Bottom-Up)

Components must be built and tested from the leaves up:

```
Phase 0: Project scaffold       — directories, dependencies, test runner
Phase 1: Kline                  — fundamental data unit
Phase 2: Signature              — OR-reduction identity computation
Phase 3: Tokenizer (Mod + BPE)  — text ↔ node conversion
Phase 4: STM                    — bounded dual-keyed index
Phase 5: Model                  — three-tier knowledge graph
Phase 6: Significance Pipeline  — distance and routing
Phase 7: Events                 — pub/sub for rationalisation
Phase 8: Agent                  — the 6-phase rationalisation orchestrator
Phase 9: Persistence            — serialisation, save/load
```

### 1.3 File Structure

```
kalvin/
├── src/
│   ├── kalvin/
│   │   ├── __init__.py          # Package root
│   │   ├── abstract.py          # Abstract base classes for Kalvin
│   │   ├── kline.py             # KLine data structure
│   │   ├── signature.py         # make_signature, signifies
│   │   ├── tokenizer.py         # BPE tokenizer
│   │   ├── mod_tokenizer.py     # Mod tokenizer (Mod32, Mod64)
│   │   ├── stm.py               # Short-Term Memory
│   │   ├── model.py             # Three-tier Model
│   │   ├── significance.py      # Significance pipeline
│   │   ├── events.py            # EventBus + RationaliseEvent
│   │   └── agent.py             # Agent orchestrator
├── tests/
│   ├── test_kline.py
│   ├── test_signature.py
│   ├── test_tokenizer.py
│   ├── test_stm.py
│   ├── test_model.py
│   ├── test_significance.py
│   ├── test_events.py
│   └── test_agent.py
├── pyproject.toml
└── README.md
```

---

## 2. Bit Layout — The Universal Encoding

All data in Kalvin is represented as uint64 values. The bit layout is the single most important design decision. Every component depends on it.

### 2.1 Node Types

```
┌──────────────────────────────────────────────────────────────────┐
│ PACKED NODE (non-literal, Mod tokenizer)                         │
│ ┌────┬──────────────────────────────────────────────────────┐    │
│ │  0 │ bits 1–N: character bits (N=31 for Mod32, 63 Mod64) │    │
│ └────┴──────────────────────────────────────────────────────┘    │
│ bit 0 = 0,  is_literal = False                                   │
│                                                                   │
│ LITERAL NODE (Mod tokenizer)                                      │
│ ┌──────────────────────┬─────────────────────────────────────┐   │
│ │ code point (upper 32) │ 0xFFFFFFFF (literal mask, lower 32) │   │
│ └──────────────────────┴─────────────────────────────────────┘   │
│ lower 32 bits all set,  is_literal = True                         │
│                                                                   │
│ BPE NODE (with type prefix)                                       │
│ ┌─────────────────────┬──────────────────────────────────────┐   │
│ │ type prefix bits     │ vocabulary index (lower bits)        │   │
│ └─────────────────────┴──────────────────────────────────────┘   │
│ is_literal = False (BPE tokens are never literal)                 │
│                                                                   │
│ SIGNATURE (make_signature output)                                 │
│ ┌────┬──────────────────────────────────────────────────────┐    │
│ │ LC │ OR-reduction of non-literal nodes                    │    │
│ └────┴──────────────────────────────────────────────────────┘    │
│ bit 0 = literal-content flag (LC), 1 if any literal nodes        │
│ bits 1+ = OR of non-literal node values                           │
│ NO bit-pattern test — identified by role, not by pattern         │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Discriminators

```python
# Tokenizer-level test
is_literal(node) = (node & 0xFFFFFFFF) == 0xFFFFFFFF

# Derived properties
is_packed(node)  = not is_literal(node) AND (node & 1) == 0
is_signature(x)  = NO TEST — any uint64 can be a signature
```

### 2.3 Well-Known Values

| Value                   | Name                 | Meaning                                                  |
| ----------------------- | -------------------- | -------------------------------------------------------- |
| `0`                     | `UNSIGNED`           | No nodes. Empty kline. Cannot be found via AND matching. |
| `1`                     | `LITERAL_ONLY`       | Contains literal content only (no non-literal nodes).    |
| `0x8000_0000_0000_0000` | `D_BOUNDARY`         | Midpoint. Separates S2 and S3 distance ranges.           |
| `0xFFFF_FFFF_FFFF_FFFF` | `D_MAX` / `S1_VALUE` | Maximum distance / maximum significance.                 |
| `0x0000_0000_0000_0000` | `S4_VALUE`           | Zero significance / maximum distance.                    |

### 2.4 Why This Layout Works

The 32-bit literal mask (`0xFFFFFFFF` in the lower bits) creates a **wide moat** between literal nodes and everything else:

- **Packed nodes:** bit 0 clear → never confused with literal.
- **Signatures with bit 0 set:** only bit 0 set (not all 32 bits) → `0xFFFFFFFF` mask test returns False.
- **BPE tokens:** vocabulary indices are small numbers → lower 32 bits are never all 1s.
- **Literal nodes:** lower 32 bits are all 1s → unambiguous.

This means a signature value (which may have bit 0 set from the literal-content flag) is never accidentally treated as a literal node, and vice versa.

---

## 3. Component Specifications

### 3.1 KLine (`kline.py`)

The fundamental unit of the knowledge graph.

**Structure:**

```python
class KLine:
    signature: int          # uint64 identity key
    nodes: list[int]        # ordered sequence of uint64 (always a list)
```

**Construction:**

```python
KLine(signature, nodes)
# signature — required uint64
# nodes — accepts: [], [int, ...], or int (normalized to list)
```

**Key rules:**

- `nodes` is always normalized to `list[int]` (empty list for no nodes, never None).
- No `literal` field is stored. Literal status is computed on demand.
- `is_literal()` requires an `is_literal_fn` injected at construction or call time.
- Empty klines (no nodes) are non-literal.
- Equality: same signature AND same node sequence (same length, order, values).
- Hashable: `hash((signature, tuple(nodes)))`.

**Required methods:**

```python
kline.nodes       → list[int]
len(kline)        → int (node count)
kline.is_literal(is_literal_fn) → bool
kline == other    → bool (signature + node sequence equality)
hash(kline)       → int
```

**`is_literal` implementation:**

```python
def is_literal(self, is_literal_fn: Callable[[int], bool]) -> bool:
    if not self.nodes:
        return False
    return all(is_literal_fn(node) for node in self.nodes)
```

**Design decisions:**

- `is_literal_fn` is injected, not imported. Kline doesn't depend on the tokenizer module directly.
- This keeps Kline testable in isolation (pass a lambda for tests).
- `dbg_text: str` may be included as an optional implementation-level field (not spec'd, for debugging only).

---

### 3.2 Signature (`signature.py`)

Pure functions for computing and comparing signatures.

**`make_signature(nodes, is_literal_fn) → int`:**

```python
def make_signature(nodes: list[int], is_literal_fn: Callable[[int], bool]) -> int:
    sig = 0
    for node in nodes:
        if is_literal_fn(node):
            sig |= 1       # bit 0: literal-content flag
        else:
            sig |= node    # non-literal contributes full value
    return sig
```

**Properties:**

- Deterministic and commutative (node order doesn't matter).
- Lossy: order and multiplicity of non-literal nodes lost; literal nodes lose identity entirely (all contribute only bit 0).
- `make_signature([]) == 0` (unsigned).
- `make_signature([literal_A, literal_B]) == 1` (literal-content flag only).
- `make_signature([non_literal]) == non_literal` (identity for single non-literal node).

**`signifies(a, b) → bool`:**

```python
def signifies(a: int, b: int) -> bool:
    return (a & b) != 0
```

The basis for candidate retrieval. Commutative. Vacuous for 0.

**`significance_value(distance) → int`:**

```python
def significance_value(distance: int) -> int:
    return (~distance) & 0xFFFF_FFFF_FFFF_FFFF
```

**Important:** No `is_signature` predicate. Any uint64 may serve as a signature. It is identified by role (the `signature` field of a KLine), not by bit pattern.

---

### 3.3 Tokenizer (`mod_tokenizer.py`, `tokenizer.py`)

Converts between text and nodes. Two types, both conforming to the same interface:

```python
class KTokenizer(ABC):
    vocab_size: int                    # Number of distinct tokens
    encode(text_or_int) → list[int]   # Text → nodes
    decode(ids) → str                 # Nodes → text
    is_literal(node) → bool           # Literal test
```

#### Mod Tokenizer

Maps characters to bit positions. Two encoding modes:

**Packed encoding** (default for multi-char strings):

- All characters OR'd into a single node.
- Bit 0 clear.
- Lossy: order and multiplicity lost.
- `encode("ABC") → [CHAR_BIT['A'] | CHAR_BIT['B'] | CHAR_BIT['C']]`

**Literal encoding** (explicit `pack=False` or `literal=True`):

- One node per character.
- Format: `(codepoint << 32) | 0xFFFFFFFF`
- Preserves order and identity.
- Bypasses vocabulary: any Unicode character can be encoded.
- `encode("AB", pack=False) → [(65<<32)|0xFFFFFFFF, (66<<32)|0xFFFFFFFF]`

**Variants:**

- **Mod32:** 31 character bits (bits 1–31). Default.
- **Mod64:** 63 character bits (bits 1–63).

**Default vocabulary:** `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_=`
Characters not in the explicit vocabulary are assigned the next available bit position (wrapping).

**`is_literal(node) → bool`:**

```python
(node & 0xFFFFFFFF) == 0xFFFFFFFF
```

**Decode:** Auto-detects literal vs packed from the literal mask.

**Design decision — default encode mode:** `encode(text)` should default to **packed** mode (single node). Literal encoding is explicit via `encode(text, pack=False)`. This matches the primary use case: packed nodes for signatures, literal for exact text.

#### BPE Tokenizer

Learns subword vocabulary from a training corpus.

- BPE tokens are **never literal**: `is_literal(node) → False`.
- Raw BPE tokens are sequential IDs (e.g., `15496` for "hello").
- Type prefixes (linguistic properties) are applied at the agent layer: `node = type_prefix | token_id`.
- Training via `train(texts, vocab_size, pattern)`.
- Persistence via `save_to_directory(path, name)` / `from_directory(path, name)`.

**Dependencies:** The BPE tokenizer is optional. It requires `rustbpe` for training and `tiktoken` for inference. These should be optional dependencies, not core requirements.

---

### 3.4 STM (`stm.py`)

Short-Term Memory: a bounded, dual-keyed index over recently added KLines.

**Structure:**

```python
class STM:
    _store: dict[int, list[KLine]]   # key → bucket of KLines
    _order: list[KLine]              # insertion order (FIFO for eviction)
    _dedup: set[tuple[int, tuple[int,...]]]  # (sig, nodes) pairs
    _bound: int                      # default 256
```

**Dual-keyed indexing:**
Each KLine is indexed under two keys:

1. **Signature:** `kline.signature`
2. **Nodes signature:** `make_signature(kline.nodes)`

When both keys are identical, stored under a single key.

**API:**

```python
stm.add(kline, dedup=True) → bool      # Add, returns False if duplicate
stm.get(key) → list[KLine]             # All KLines under key
stm.find_by_signature(sig) → list[KLine]
stm.find_by_nodes(nodes_sig) → list[KLine]
stm.remove(kline) → None               # Remove from all indexes
stm.clear() → None
len(stm) → int
```

**Eviction:** When `add` would exceed the bound, the oldest entry is evicted (FIFO). Eviction removes from the STM index only — the KLine remains in the frame. The evicted KLine's signature and nodes-signature entries are removed.

**Dependency:** Requires `make_signature` from the signature module, and an `is_literal_fn` (injected) for computing nodes signatures.

---

### 3.5 Model (`model.py`)

Three-tier layered knowledge graph: `STM → Frame → Base`.

**Tier roles:**

| Tier  | Purpose               | Bounded   | Lifetime       | Modified by `add`   |
| ----- | --------------------- | --------- | -------------- | ------------------- |
| STM   | Transitive grounding  | Yes (256) | Rolling window | Yes                 |
| Frame | Session write surface | No        | Per-session    | Yes                 |
| Base  | Long-term knowledge   | No        | Persistent     | No (promotion only) |

**Lookup order:** STM → Frame → Base. Callers see a single unified API.

**Construction:**

```python
Model(base=None, stm_bound=256, is_literal_fn=None)
```

**Storage operations:**

```python
model.add(kline) → bool           # Add to STM + Frame
model.exists(kline) → bool       # Check across all tiers
model.find(signature) → KLine|None       # Most recent by sig
model.find_all(signature) → list[KLine]  # All by sig
model.find_by_nodes(nodes_sig) → KLine|None  # By nodes signature
model.remove(signature) → bool   # Remove most recent, never from base
len(model) → int                 # Frame count only
```

**Deduplication:** When `add` receives a literal KLine (all nodes literal per `is_literal`), check for an equal KLine in any tier. If duplicate exists, reject (return False). Non-literal KLines are always accepted.

**Iteration:**

```python
model.klines() → list[KLine]     # All KLines, reverse insertion order, deduped
model.where(predicate) → list[KLine]  # Filtered; if predicate is int, AND match
```

**Graph traversal:**

```python
model.resolve(node) → KLine|None     # node → KLine lookup (= find)
model.expand(kline, depth=2) → list[KLine]  # Graph expansion
model.descendants(node) → set[int]   # Recursive node collection
model.query(signature, depth=1) → list[KLine]  # Find + expand
```

**Expand semantics:**

- `depth=0` → `[]`
- `depth=1` → `[]`
- `depth=2` → direct children of kline
- `depth=N` → N-1 levels deep
- Cycle detection: visited KLines not re-visited.
- Missing nodes skipped.
- Order: node order, depth-first.

**Promotion:**

```python
model.promote(kline) → bool    # Add to base. Returns False if no base.
model.promote_all() → int      # Promote all frame KLines to base.
```

**Significance API (consumed by significance pipeline):**

```python
model.s2_distance(query, candidate) → int   # [1, D_BOUNDARY)
model.s3_distance(query, candidate) → int   # [D_BOUNDARY, D_MAX)
model.is_countersigned(a, b) → bool
```

Routing in the significance pipeline is self-contained (node-membership
testing) and does not call any model function. Only the distance step
consumes `s2_distance` and `s3_distance`. See §4 for the semantics of
these functions.

**Model-internal functions:**

```python
model.is_s1(node) → bool             # checks model.find(node)
model._is_canon(kline) → bool        # sig == make_signature(nodes)
model._edge_hops(sig) → Iterator    # yields (hop_count, next_sig)
```

`is_s1` returns `True` when the node value resolves to a kline in any tier
(`model.find(node) is not None`).

`_is_canon` tests whether a kline is canonical (its signature equals the
`make_signature` reduction of its nodes).

`_edge_hops` is a generator that yields `(hop_count, next_sig)` pairs for
each non-canonical resolution step. The `s2_distance` algorithm consumes
it to find the first hop that lands in the opposing kline's mismatch set.

---

### 3.6 Significance Pipeline (`significance.py`)

Computes how well a candidate KLine answers a query KLine.

**Core formula:** `significance = ~distance` (bitwise NOT on uint64).

**Three-step pipeline: Route → Distance → Invert**

**Step 1: Route** — For each node in Q, test whether that node value exists
in the candidate's node sequence. This is a straightforward membership test
that does **not** call any model function. Count matched nodes and route:

```
match_count = number of nodes in Q that exist in C's node sequence

All candidates absent  → S4 (distance = MAX)
All nodes match         → S1 (distance = 0)
Some nodes match        → S2 (distance = model.s2_distance(Q, C))
No nodes match          → S3 (distance = model.s3_distance(Q, C))
```

Routing is self-contained and pessimistic: any unmatched node prevents S1.

**Step 2: Distance** — Only S2 and S3 routes call the model's distance
functions. S1 sets distance = 0 directly; S4 sets distance = MAX directly.

**Step 3: Invert** — `significance = ~distance`

**Output:**

```python
@dataclass
class SignificanceResult:
    significance: int   # ~distance (uint64)
    distance: int       # raw distance
    level: str          # "S1", "S2", "S3", or "S4"
    match_count: int    # number of nodes matched in routing
    total_nodes: int    # total nodes in query
```

**API:**

```python
significance_pipeline(query, candidates, model) → list[tuple[KLine | None, SignificanceResult]]
compute_significance(query, candidate, model) → SignificanceResult
```

**S4 handling:** The pipeline handles all significance levels including
S4. When `candidates` is empty, it returns `[(None, S4_result)]` instead
of `[]`. Callers pass candidates directly without pre-testing for empty.

**D_boundary:** `0x8000_0000_0000_0000` (midpoint). Configurable hyperparameter.

**Clamping:** Model functions that return out-of-range distances are clamped to the valid range for their level.

**Properties:**

- Three-step pipeline: Route → Distance → Invert. Each step is independent.
- Routing is self-contained: uses node-membership testing, no model function.
- Inverted metric: higher significance = closer match.
- Pessimistic: any unmatched node prevents S1.
- Arithmetic ordering: S1 > S2 > S3 > S4 by unsigned comparison.

---

### 3.7 Events (`events.py`)

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

---

### 3.8 Agent (`agent.py`)

The orchestrator of the rationalisation pipeline.

**Structure:**

```python
class Agent:
    tokenizer: KTokenizer   # Default: Mod32Tokenizer
    model: Model             # Three-tier knowledge graph
    events: EventBus         # Pub/sub
```

**Construction:**

```python
Agent(tokenizer=None, model=None)
# Defaults: Mod32Tokenizer, empty Model with tokenizer's is_literal_fn
```

**Rationalisation — 6-phase pipeline:**

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

Phase 5: COMPUTE SIGNIFICANCE
  results = significance_pipeline(Q, candidates, model)
  # Pipeline: route (node-membership) → distance → invert
  # Handles empty candidates → [(None, S4_result)]

Phase 6: INTEGRATE
  model.add(Q)
  best = max result by significance value
  S1 → promote Q, emit "frame" S1, return True
  S4 → promote Q, emit "frame" S4, return True (novel)
  S2 → queue Q for cogitation, return False
  S3 → queue Q for cogitation, return False
```

**Cogitation:**

Background processing of rational KLines (S2/S3). Runs asynchronously.

```
Cogitate(Q):
  1. candidates = model.query(Q.signature, depth=D_cogitate)
  2. For each candidate C:
     a. (significance, level) = significance_pipeline(Q, C, model)
     b. If is_countersigned(Q, C): upgrade to S1
     c. If S1: add C to model, break
  3. Re-rationalise Q
```

**Countersignature test:**

```python
def is_countersigned(Q, C):
    return (C.signature in Q.nodes) and (Q.signature in C.nodes)
```

**Cogitation parameters:**

- `D_cogitate` (default 2): graph traversal depth.
- `max_passes` (default 3): max re-rationalisation attempts before abandoning.
- `timeout` (default 2s): idle time before emitting "done" and stopping.

**Cogitation lifecycle:**

- Runs in a background thread (daemon).
- Manages a backlog queue.
- When backlog is empty for `timeout` seconds → emit "done", stop thread.
- Can be stopped explicitly via `cogitate_join(timeout)`.
- Pass counter tracks per-KLine cogitation attempts by identity.

---

## 4. Model Significance API — Resolved Semantics

This section defines the semantics of the model functions consumed by the
significance pipeline and used internally for distance computation.

### 4.1 `is_s1(node) → bool` (model-internal)

**Note:** This function is used internally by the model's distance implementations.
It is **not** called by the significance pipeline — routing uses simple
node-membership testing instead (see §3.6).

**Semantics:** A node achieves S1 when its value resolves to a kline in the
model — i.e., `model.find(node) is not None`. This is a stateful test: adding or
removing klines changes the result.

```python
def is_s1(self, node: int) -> bool:
    return self.find(node) is not None
```

**Rationale:** A grounded node — one that corresponds to known structure in the
model — has achieved S1 significance. The test is whether the node value serves
as a signature for some stored kline, regardless of which candidate is being
compared.

### 4.2 `s2_distance(query, candidate) → int`

S2 distance has been selected because there is a node mismatch between query
and candidate klines. The algorithm is a **per-node hop-distance** with
grounding credit.

#### Definitions

- **is_s1(node)** — `model.find(node) is not None`. The node resolves to a
  known kline in any tier.
- **is_canon(kline)** — `kline.signature == make_signature(kline.nodes)`.
  The kline's signature exactly represents its nodes. Canonical klines are
  terminals in the resolution chain.
- **edge_hops(sig)** — the number of non-canonical resolution steps from a
  signature. Follows the chain: resolve `sig` → kline → `make_signature(kline.nodes)`
  → resolve again. Stops at a dead end (unresolvable) or a canonical kline.
  Yields `(hop_count, next_sig)` at each step, where `next_sig` is the
  signature produced by `make_signature(kline.nodes)` at that hop.
- **MAX_HOP** — hyperparameter (default 100). Upper bound on chain depth and
  the penalty for unresolvable mismatched nodes.

```python
def _edge_hops(self, sig: int) -> Iterator[tuple[int, int]]:
    hop_count = 0
    while hop_count < MAX_HOP:
        kline = self.find(sig)
        if kline is None or self._is_canon(kline):
            break
        hop_count += 1
        sig = self._make_sig(kline.nodes)
        yield hop_count, sig
```

#### Algorithm

Starting distance is 0 (all S2 bits clear). Distance grows with mismatch
and discovery. All calculations are additive (positive drives toward S3
boundary) except the grounding credit which is subtractive (drives toward S1).

```python
def s2_distance(self, query: KLine, candidate: KLine) -> int:
    if not query.nodes:
        return 1

    q_set = set(query.nodes)
    c_set = set(candidate.nodes)
    mismatched_q = q_set - c_set
    mismatched_c = c_set - q_set
    matched = q_set & c_set

    distance = 0

    # Mismatched query nodes: find hops that land in mismatched_c
    for n in mismatched_q:
        hop_distance = MAX_HOP
        for hops, match_sig in self._edge_hops(n):
            if match_sig in mismatched_c:
                hop_distance = hops
                break
        distance += hop_distance

    # Mismatched candidate nodes: find hops that land in mismatched_q
    for n in mismatched_c:
        hop_distance = MAX_HOP
        for hops, match_sig in self._edge_hops(n):
            if match_sig in mismatched_q:
                hop_distance = hops
                break
        distance += hop_distance

    # Grounding credit: matched nodes that resolve to known klines
    for n in matched:
        if self.find(n) is not None:
            distance -= 1

    return max(1, min(int(distance), D_BOUNDARY - 1))
```

#### Per-node contributions

| Node state                               | Contribution | Rationale                      |
| ---------------------------------------- | ------------ | ------------------------------ |
| Mismatched, edge_hops = 0 (unresolvable) | +MAX_HOP     | No path to grounded knowledge  |
| Mismatched, edge_hops = N                | +N           | N hops from grounded knowledge |
| Matched + grounded (is_s1)               | −1           | Cancels one hop-distance unit  |
| Matched + ungrounded                     | 0            | Neutral                        |

#### Properties

1. **Starting distance = 0** — all S2 bits clear. Distance grows with
   mismatch and discovery.
2. **Hop distance is the distance** — each mismatched node contributes
   `1..MAX_HOP` based on its edge hop depth. A node 1 hop from grounded
   knowledge contributes 1. An unresolvable node contributes MAX_HOP.
3. **Grounding credit is same-scale** — each matched node that resolves
   subtracts 1, directly offsetting one hop-distance unit.
4. **Bidirectional** — mismatched nodes from both query and candidate
   contribute to distance, ensuring symmetry of information gaps.

### 4.3 `s3_distance(query, candidate) → int`

**Semantics:** Distance is based on **signature bit overlap** between query and candidate, mapped to `[D_BOUNDARY, D_MAX)`.

```python
def s3_distance(self, query: KLine, candidate: KLine) -> int:
    q_sig = self._make_sig(query.nodes)
    c_sig = candidate.signature
    if q_sig == 0:
        return D_MAX - 1
    overlap = bin(q_sig & c_sig).count("1")
    total = bin(q_sig | c_sig).count("1")
    if total == 0:
        return D_MAX - 1
    ratio = overlap / total
    distance = D_BOUNDARY + int((1 - ratio) * (D_MAX - D_BOUNDARY))
    return max(D_BOUNDARY, min(distance, D_MAX - 1))
```

**Rationale:** When no nodes match (S3), we still know the signatures overlap (that's how the candidate was found). The degree of overlap provides a weak signal: more shared bits = more connotational similarity = closer to S2. No overlap (which shouldn't happen given AND-based retrieval, but handled) is maximally far.

### 4.4 `is_countersigned(a, b) → bool`

```python
def is_countersigned(self, a: KLine, b: KLine) -> bool:
    return (b.signature in a.nodes) and (a.signature in b.nodes)
```

Structural test only. Literal nodes cannot match a signature (their `0xFFFFFFFF` lower bits differ from any signature value), so the test naturally considers only non-literal matches.

---

## 5. Open Design Decisions (Resolved)

### 5.1 Candidate Retrieval Efficiency

**Decision:** Implement `model.where(signature)` using linear scan initially. Add a `candidates_for(signature)` method later if profiling shows it's needed.

**Rationale:** For models with <10K KLines, linear scan is fast enough. The inverted bit-index optimization is a performance concern, not a correctness concern. Ship first, optimize later.

**Future optimization:** Maintain an inverted index `bit_position → set[signature]` that maps each set bit to all signatures containing that bit. `candidates_for(sig)` would union the sets for all set bits in `sig`.

### 5.2 All-Literal Signature Collision

All all-literal KLines produce `make_signature = 1`. This means:

- Two different all-literal KLines have the same signature.
- They are distinguishable only by their node sequences.
- The model's signature index will have many KLines under key `1`.

**Decision:** Accept this. All-literal KLines are fast-tracked in Phase 3 (Assess) and never reach candidate retrieval. If they do reach retrieval during cogitation, the significance pipeline's routing will count zero matches (literal nodes won't exist in non-literal candidates' node sequences) and route to S3.

**Mitigation:** Document that all-literal KLines should be promoted to base to avoid repeated processing.

### 5.3 Default Encode Mode

**Decision:** `tokenizer.encode(text)` defaults to **packed** mode. Literal encoding is explicit via `encode(text, pack=False)`.

**Rationale:** Packed mode produces the signature-friendly single-node output. Literal mode is for exact text sequences. The common case (building signatures) uses packed; the less common case (exact text) is explicit.

### 5.4 STM Nodes-Signature Indexing for All-Literal KLines

All all-literal KLines produce `nodes_sig = 1`. STM's `find_by_nodes(1)` returns the most recently added.

**Decision:** Accept this degenerate case. It's correct behavior — same as multiple KLines sharing any signature. Document it.

### 5.5 32-Bit Code Point Limit

Literal encoding: `(codepoint << 32) | 0xFFFFFFFF`. Unicode code points max at U+10FFFF (21 bits). Upper 32 bits provide ample room.

Raw integer literal encoding is limited to 32-bit values (0 to 2³²−1). This is not a constraint for character encoding.

### 5.6 Persistence Format

**Decision:** Support both JSON and binary serialization.

**JSON format:**

```json
{
  "klines": [
    { "signature": 5, "nodes": [1, 2] },
    { "signature": 10, "nodes": [3, 4] }
  ]
}
```

**Binary format:** Packed little-endian:

- `uint32` kline count
- Per kline: `uint64` signature, `uint32` node count, `uint64` \* N nodes

### 5.7 Project Dependencies

**Core (required):**

- Python ≥ 3.10
- pytest (dev)

**Optional:**

- `rustbpe` — BPE tokenizer training
- `tiktoken` — BPE tokenizer inference
- `pyarrow` — parquet data loading for BPE training

**Not required:**

- torch, torchvision, torchaudio (remove from core dependencies)
- numpy, matplotlib, spacy (not used by core system)
- textual (TUI, separate concern)

---

## 6. Implementation Phases

### Phase 0: Project Scaffold

**Estimate:** 0.5 day

**Tasks:**

1. Create project directory structure.
2. Write `pyproject.toml` with minimal dependencies:

   ```toml
   [project]
   name = "kalvin"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = []

   [project.optional-dependencies]
   bpe = ["rustbpe>=0.1.0", "tiktoken>=0.7.0"]
   dev = ["pytest>=8.0.0", "pytest-cov>=4.1.0", "ruff>=0.2.0"]
   ```

3. Create empty `__init__.py` files.
4. Verify test runner works: `pytest tests/ -v`.

**Deliverable:** Empty project with passing test runner.

---

### Phase 1: KLine

**Files:** `src/kalvin/kline.py`, `tests/test_kline.py`
**Depends on:** Nothing (leaf concept; `is_literal_fn` injected)
**Estimate:** 0.5 day

**Implementation:**

```python
# kline.py
KNode = int
KSig = int

class KLine:
    __slots__ = ("signature", "nodes", "dbg_text")

    def __init__(self, signature: int, nodes, dbg_text: str = ""):
        self.signature = signature
        self.nodes = _normalize(nodes)  # Always list[int]
        self.dbg_text = dbg_text

    def is_literal(self, is_literal_fn) -> bool:
        if not self.nodes:
            return False
        return all(is_literal_fn(n) for n in self.nodes)

    def __eq__(self, other): ...
    def __hash__(self): ...
    def __len__(self): ...
    def __repr__(self): ...
```

**Test cases:**

| Test                        | Description                           |
| --------------------------- | ------------------------------------- |
| Construction: empty nodes   | `KLine(5, [])` → nodes = `[]`         |
| Construction: single int    | `KLine(5, 3)` → nodes = `[3]`         |
| Construction: list          | `KLine(5, [1, 2])` → nodes = `[1, 2]` |
| Equality: same sig + nodes  | `KLine(5, [1]) == KLine(5, [1])`      |
| Inequality: different sig   | `KLine(5, [1]) != KLine(6, [1])`      |
| Inequality: different nodes | `KLine(5, [1]) != KLine(5, [2])`      |
| Hash consistency            | Equal KLines have equal hashes        |
| is_literal: all literal     | With `lambda n: True` → True          |
| is_literal: mixed           | With `lambda n: n > 100` → False      |
| is_literal: empty           | Returns False                         |
| Length                      | `len(KLine(5, [1, 2, 3])) == 3`       |

---

### Phase 2: Signature

**Files:** `src/kalvin/signature.py`, `tests/test_signature.py`
**Depends on:** `is_literal_fn` (injected)
**Estimate:** 0.5 day

**Implementation:**

```python
# signature.py
def make_signature(nodes, is_literal_fn) -> int:
    sig = 0
    for node in nodes:
        if is_literal_fn(node):
            sig |= 1
        else:
            sig |= node
    return sig

def signifies(a, b) -> bool:
    return (a & b) != 0

def significance_value(distance) -> int:
    return (~distance) & 0xFFFF_FFFF_FFFF_FFFF
```

**Test cases:**

| Test                                     | Expected                |
| ---------------------------------------- | ----------------------- |
| `make_signature([], fn)`                 | `0`                     |
| `make_signature([literal], fn)`          | `1`                     |
| `make_signature([literal, literal], fn)` | `1` (idempotent)        |
| `make_signature([packed], fn)`           | `packed` (identity)     |
| `make_signature([literal, packed], fn)`  | `1 \| packed`           |
| `make_signature([A, B], fn)`             | `A \| B` (commutative)  |
| `signifies(0, anything)`                 | `False`                 |
| `signifies(1, 1)`                        | `True`                  |
| `signifies(0b110, 0b10)`                 | `True`                  |
| `signifies(0b110, 0b1)`                  | `False`                 |
| `significance_value(0)`                  | `0xFFFF_FFFF_FFFF_FFFF` |
| `significance_value(MAX)`                | `0`                     |

**Note:** Use a mock `is_literal_fn` in tests (e.g., `lambda n: n & 0xFFFFFFFF == 0xFFFFFFFF`).

---

### Phase 3: Tokenizer

**Files:** `src/kalvin/mod_tokenizer.py`, `src/kalvin/tokenizer.py`, `tests/test_tokenizer.py`
**Depends on:** Abstract interface (can be inline or in `abstract.py`)
**Estimate:** 1.5 days

#### 3a. Mod Tokenizer

**Implementation checklist:**

1. Build character-to-bit mapping from alphabet.
2. Packed encode: OR all character bits, bit 0 clear.
3. Literal encode: `(codepoint << 32) | 0xFFFFFFFF` per character.
4. Decode: auto-detect via mask test.
5. `is_literal(node)`: `(node & 0xFFFFFFFF) == 0xFFFFFFFF`.
6. Mod32 variant: 31 character bits.
7. Mod64 variant: 63 character bits.

**Test cases:**

| Test                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| Packed encode single char | `encode("A") == [bit_A]`                                     |
| Packed encode multi-char  | `encode("AB") == [bit_A \| bit_B]`                           |
| Packed round-trip         | `decode(encode("ABC"))` contains A, B, C (order may differ)  |
| Literal encode            | `encode("AB", pack=False)` → two nodes with literal mask     |
| Literal round-trip        | `decode(encode("AB", pack=False)) == "AB"` (order preserved) |
| Empty string              | `encode("") == []`, `decode([]) == ""`                       |
| is_literal: literal node  | `is_literal((65 << 32) \| 0xFFFFFFFF) == True`               |
| is_literal: packed node   | `is_literal(6) == False`                                     |
| is_literal: zero          | `is_literal(0) == False`                                     |
| Literal encoding values   | Verify `('A' << 32) \| 0xFFFFFFFF` for char 'A'              |
| Vocab size                | Matches number of unique characters in alphabet              |
| Characters not in vocab   | Still encodable (assigned next bit)                          |

#### 3b. BPE Tokenizer

**Implementation:** Wrapper around `rustbpe` (training) and `tiktoken` (inference). Mark as optional dependency.

**Test cases:** Basic encode/decode round-trip, `is_literal` always False, error handling for untrained tokenizer.

---

### Phase 4: STM

**Files:** `src/kalvin/stm.py`, `tests/test_stm.py`
**Depends on:** Kline, Signature (for `make_signature`)
**Estimate:** 1 day

**Test cases:**

| Test                          | Description                                                   |
| ----------------------------- | ------------------------------------------------------------- |
| Add and retrieve by signature | `stm.add(kl); stm.find_by_signature(sig) == [kl]`             |
| Add and retrieve by nodes sig | `stm.add(kl); stm.find_by_nodes(nodes_sig) == [kl]`           |
| Bound enforcement             | Add `bound + 1` KLines; first is evicted                      |
| Eviction correctness          | Evicted KLine not in STM but not in frame (frame is separate) |
| Dedup                         | Adding same (sig, nodes) returns False                        |
| Multiple KLines same sig      | All stored under one key                                      |
| Remove                        | Remove clears from all indexes                                |
| Clear                         | Everything removed                                            |
| Empty STM                     | `len(stm) == 0`, all lookups return empty/None                |
| Nodes signature indexing      | Two KLines with different sigs but same nodes_sig both found  |

---

### Phase 5: Model

**Files:** `src/kalvin/model.py`, `tests/test_model.py`
**Depends on:** Kline, Signature, STM
**Estimate:** 2–3 days (largest component)

**Test cases organized by area:**

#### Storage

| Test                      | Description                      |
| ------------------------- | -------------------------------- |
| Add and find              | Add KLine, find by signature     |
| Add returns True          | Normal case                      |
| Literal dedup             | Duplicate literal KLine rejected |
| Non-literal no dedup      | Duplicate non-literal accepted   |
| Exists                    | True after add, False before     |
| Find returns most recent  | Multiple KLines same sig         |
| Find_all                  | Returns all KLines with sig      |
| Find_by_nodes             | Returns by nodes signature       |
| Remove                    | Removes most recent with sig     |
| Remove never touches base | Verify base unchanged            |
| Len                       | Frame count only                 |

#### Three-tier lookup

| Test             | Description                            |
| ---------------- | -------------------------------------- |
| STM priority     | KLine in STM found before frame        |
| Frame fallback   | KLine not in STM found in frame        |
| Base fallback    | KLine not in STM/frame found in base   |
| Cross-tier dedup | Literal KLine in base blocks frame add |

#### Graph traversal

| Test                   | Description             |
| ---------------------- | ----------------------- |
| Resolve                | Node resolves to KLine  |
| Expand depth 0         | Returns empty           |
| Expand depth 2         | Returns direct children |
| Expand cycle detection | No infinite loop        |
| Descendants            | Recursive collection    |
| Query                  | Find + expand           |

#### Promotion

| Test                     | Description                          |
| ------------------------ | ------------------------------------ |
| Promote to base          | KLine appears in base                |
| Promote without base     | Returns False                        |
| Promote all              | All frame KLines promoted            |
| Promote skips base dupes | Existing base KLines not overwritten |

#### Significance API

| Test                      | Description                                        |
| ------------------------- | -------------------------------------------------- |
| is_s1 resolves            | `model.find(node) is not None` → True              |
| is_s1 no resolve          | Node not in model → False                          |
| is_s1 node not signature  | Node in kline.nodes but no kline with that sig     |
| \_is_canon match          | `sig == make_signature(nodes)` → True              |
| \_is_canon mismatch       | `sig != make_signature(nodes)` → False             |
| \_edge_hops unresolvable  | Node doesn't resolve → empty generator                   |
| \_edge_hops canonical     | Resolves to canonical → empty generator                  |
| \_edge_hops chain         | Yields (hop_count, next_sig) at each non-canonical step  |
| s2_distance empty query   | Returns 1                                          |
| s2_distance no resolution | All mismatched unresolvable → MAX_HOP each         |
| s2_distance grounding     | Matched node that resolves → −1 credit             |
| s2_distance edge hops     | Mismatched node with chain → proportional distance |
| s2_distance range         | Value in `[1, D_BOUNDARY)`                         |
| s2_distance clamped       | Large results clamped to D_BOUNDARY - 1            |
| s3_distance range         | Value in `[D_BOUNDARY, D_MAX)`                     |
| is_countersigned          | Mutual reference detected                          |
| Not countersigned         | One-way reference → False                          |

**Note:** `is_s1`, `_is_canon`, and `_edge_hops` are tested as model-internal
functions. The significance pipeline does not call them directly — routing
uses node-membership testing.

---

### Phase 6: Significance Pipeline

**Files:** `src/kalvin/significance.py`, `tests/test_significance.py`
**Depends on:** Kline, Model (mock or real)
**Estimate:** 1 day

**Test cases:**

| Test                      | Description                                               |
| ------------------------- | --------------------------------------------------------- |
| All nodes match           | Returns S1, distance 0, significance MAX                  |
| Some nodes match          | Returns S2, distance in range                             |
| No nodes match            | Returns S3, distance in range                             |
| No candidates             | Returns [(None, S4_result)], distance MAX, significance 0 |
| Empty query               | Returns S4                                                |
| Single node match         | Returns S1                                                |
| Clamping                  | Out-of-range distances clamped                            |
| Best candidate selection  | Pipeline returns all results; caller picks max            |
| Significance ordering     | S1 > S2 > S3 > S4 numerically                             |
| Routing is self-contained | Verify no model.is_s1 calls during routing                |
| Routing uses membership   | Node in candidate.nodes → match; not in → no match        |

---

### Phase 7: Events

**Files:** `src/kalvin/events.py`, `tests/test_events.py`
**Depends on:** Kline (for RationaliseEvent)
**Estimate:** 0.5 day

**Test cases:**

| Test                  | Description                              |
| --------------------- | ---------------------------------------- |
| Subscribe and publish | Callback receives event                  |
| Multiple subscribers  | All receive event                        |
| Event fields          | kind, query, value, significance correct |
| Thread safety         | Publish from another thread              |
| Empty bus             | No crash on publish with no subscribers  |

---

### Phase 8: Agent

**Files:** `src/kalvin/agent.py`, `tests/test_agent.py`
**Depends on:** Everything
**Estimate:** 2 days

**Test cases organized by rationalisation phase:**

#### Phase 1: Prepare

| Test                | Description                                 |
| ------------------- | ------------------------------------------- |
| Signature assigned  | KLine with sig=0 gets make_signature(nodes) |
| Signature preserved | KLine with existing sig unchanged           |

#### Phase 2: Ground check

| Test                     | Description                        |
| ------------------------ | ---------------------------------- |
| First rationalise        | Returns True, adds to model        |
| Duplicate rationalise    | Returns True, emits "ground" event |
| Different sig same nodes | Not a ground (different KLine)     |

#### Phase 3: Assess

| Test                    | Description                         |
| ----------------------- | ----------------------------------- |
| Unsigned (no nodes)     | Returns True, emits "frame" S4      |
| All-literal             | Returns True, emits "frame" S1      |
| Self-grounded canonical | Returns True when all nodes resolve |
| Not self-grounded       | Falls through to Phase 4            |

#### Phase 4: Retrieve candidates

| Test             | Description                                             |
| ---------------- | ------------------------------------------------------- |
| No candidates    | Pipeline returns S4 result; agent handles via Phase 5/6 |
| Candidates found | Proceeds to Phase 5                                     |

#### Phase 5 + 6: Significance + Integrate

| Test                | Description                          |
| ------------------- | ------------------------------------ |
| Best candidate S1   | Returns True, promotes               |
| Best candidate S2   | Returns False, queues cogitation     |
| Best candidate S3   | Returns False, queues cogitation     |
| Multiple candidates | Best (highest significance) selected |

#### Cogitation

| Test                       | Description                |
| -------------------------- | -------------------------- |
| Countersignature discovery | S2 → S1 via cogitation     |
| Pass limit                 | Abandoned after max passes |
| Join                       | Thread stops cleanly       |

#### Events

| Test           | Description                  |
| -------------- | ---------------------------- |
| Event delivery | All events received in order |
| Ground event   | Correct kind + significance  |
| Frame event    | Correct kind + significance  |

#### Serialization

| Test              | Description                           |
| ----------------- | ------------------------------------- |
| JSON round-trip   | Save/load preserves KLines            |
| Binary round-trip | Save/load preserves KLines            |
| Empty agent       | Serializes and deserializes correctly |

---

### Phase 9: Persistence & Polish

**Estimate:** 1 day

**Tasks:**

1. Binary serialization (Agent.to_bytes / from_bytes).
2. JSON serialization (Agent.to_dict / from_dict).
3. File save/load with auto-format detection.
4. Remove any unused dependencies.
5. Integration test: full pipeline from text input to persisted knowledge graph.

---

## 7. Dependency Injection Pattern

Several components require `is_literal_fn` from the tokenizer. To avoid circular dependencies and keep components testable in isolation, use **constructor injection**:

```
Kline.is_literal(is_literal_fn)     # Passed at call time
make_signature(nodes, is_literal_fn) # Passed at call time
Model(is_literal_fn=fn)              # Stored at construction
STM(is_literal_fn=fn)                # Stored at construction
Agent(tokenizer=tok)                 # Uses tok.is_literal
```

The Agent is the composition root — it creates the tokenizer and injects `tokenizer.is_literal` into all downstream components.

---

## 8. Constants Reference

```python
# Bit layout
LITERAL_MASK = 0xFFFF_FFFF           # Lower 32 bits all set
MASK64 = 0xFFFF_FFFF_FFFF_FFFF       # Full 64-bit mask

# Well-known signatures
UNSIGNED = 0                         # No nodes
LITERAL_ONLY = 1                     # All-literal content

# Significance
D_BOUNDARY = 0x8000_0000_0000_0000   # S2/S3 separator (midpoint)
D_MAX = 0xFFFF_FFFF_FFFF_FFFF        # Maximum distance
S1_VALUE = 0xFFFF_FFFF_FFFF_FFFF     # Maximum significance
S4_VALUE = 0x0000_0000_0000_0000     # Minimum significance

# Model
STM_BOUND_DEFAULT = 256              # STM capacity
MAX_HOP = 100                        # S2 edge hop chain depth / unresolvable penalty
D_COGITATE_DEFAULT = 2               # Cogitation traversal depth
MAX_COGITATE_PASSES = 3              # Max re-rationalisation attempts
COGITATE_TIMEOUT = 2.0               # Seconds before "done" event

# Tokenizer
MOD32_BITS = 31                      # Character bit positions
MOD64_BITS = 63
```

---

## 9. Risk Assessment

| Risk                                                    | Severity | Mitigation                                                                             |
| ------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------- |
| Model significance API semantics may need iteration     | High     | S2 distance now uses per-node hop-distance algorithm (§4.2); evolve based on real data |
| All-literal signature collision (sig=1 for all)         | Medium   | Fast-path in Assess phase; accept degenerate indexing                                  |
| Candidate retrieval O(N) scan too slow for large models | Medium   | Profile first; add inverted bit index if needed                                        |
| Cogitation thread safety bugs                           | Medium   | Thorough concurrent testing; keep thread logic simple                                  |
| Literal mask accidental collision                       | Low      | Only `0xFFFFFFFF` lower 32 bits triggers; verified for all node types                  |
| BPE tokenizer optional deps fragile                     | Low      | Make BPE entirely optional; core system works with Mod only                            |

---

## 10. Summary of What Gets Built When

| Phase | Component    | Files                              | Est.       | Depends On |
| ----- | ------------ | ---------------------------------- | ---------- | ---------- |
| 0     | Scaffold     | `pyproject.toml`, dirs             | 0.5d       | —          |
| 1     | KLine        | `kline.py`                         | 0.5d       | —          |
| 2     | Signature    | `signature.py`                     | 0.5d       | —          |
| 3     | Tokenizer    | `mod_tokenizer.py`, `tokenizer.py` | 1.5d       | —          |
| 4     | STM          | `stm.py`                           | 1d         | 1, 2, 3    |
| 5     | Model        | `model.py`                         | 2–3d       | 1, 2, 4    |
| 6     | Significance | `significance.py`                  | 1d         | 1, 5       |
| 7     | Events       | `events.py`                        | 0.5d       | 1          |
| 8     | Agent        | `agent.py`                         | 2d         | 1–7        |
| 9     | Persistence  | `agent.py` (extend)                | 1d         | 8          |
|       | **Total**    |                                    | **12–14d** |            |

Phases 1, 2, and 3 can proceed in parallel (all are leaf components with injected dependencies).

---

## 11. Questions for Clarification

Before starting implementation, confirm the following with the system owner:

1. **BPE tokenizer priority:** Is BPE support needed for MVP, or can we ship with Mod tokenizer only and add BPE later?
   - _Recommendation:_ Mod only for MVP. BPE is a separate concern.

2. **Persistence format:** Is JSON sufficient, or is binary serialization required from day one?
   - _Recommendation:_ JSON for development; binary for production.

3. **is_s1 semantics:** ~~The subsumption check~~ Now resolved: `is_s1(node)` returns `model.find(node) is not None`. A grounded node (one that resolves to a known kline in any tier) achieves S1. See §4.1.

4. **Thread model for cogitation:** Should cogitation use a background thread (as specified), or would async/await be preferred?
   - _Recommendation:_ Background thread for simplicity. The spec's "done" event requires some concurrency primitive.

5. **`dbg_text` field:** Should the debug text field be included in KLine, or kept separate?
   - _Recommendation:_ Include as optional implementation-level field. Not part of equality or hash.

6. **`abstract.py`:** Should there be formal ABC classes for Tokenizer, Model, Agent, or are duck-typed protocols sufficient?
   - _Recommendation:_ Use Python's `ABC` for Tokenizer (multiple implementations). Use duck typing for Model and Agent (single implementation).
