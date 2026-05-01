# Sub-Plan: Foundations — Bit Layout, KLine, Signature, Tokenizer, STM

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 0–4
**Estimate:** 4 days
**Depends on:** Nothing (leaf components)

---

## 0. Project Scaffold (Phase 0)

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

## 1. Bit Layout — The Universal Encoding

All data in Kalvin is represented as uint64 values. The bit layout is the
single most important design decision. Every component depends on it.

### 1.1 Node Types

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

### 1.2 Discriminators

```python
# Tokenizer-level test
is_literal(node) = (node & 0xFFFFFFFF) == 0xFFFFFFFF

# Derived properties
is_packed(node)  = not is_literal(node) AND (node & 1) == 0
is_signature(x)  = NO TEST — any uint64 can be a signature
```

### 1.3 Well-Known Values

| Value                   | Name                 | Meaning                                                  |
| ----------------------- | -------------------- | -------------------------------------------------------- |
| `0`                     | `UNSIGNED`           | No nodes. Empty kline. Cannot be found via AND matching. |
| `1`                     | `LITERAL_ONLY`       | Contains literal content only (no non-literal nodes).    |
| 32                      | `D_PACK_SHIFT`       | Bit position separating S2 and S3 components in packed distance. |
| `0xFFFF_FFFF_FFFF_FFFF` | `D_MAX` / `S1_VALUE` | Maximum distance / maximum significance.                 |
| `0x0000_0000_0000_0000` | `S4_VALUE`           | Zero significance / maximum distance.                    |

### 1.4 Why This Layout Works

The 32-bit literal mask (`0xFFFFFFFF` in the lower bits) creates a **wide moat** between literal nodes and everything else:

- **Packed nodes:** bit 0 clear → never confused with literal.
- **Signatures with bit 0 set:** only bit 0 set (not all 32 bits) → `0xFFFFFFFF` mask test returns False.
- **BPE tokens:** vocabulary indices are small numbers → lower 32 bits are never all 1s.
- **Literal nodes:** lower 32 bits are all 1s → unambiguous.

---

## 2. KLine (Phase 1)

**Files:** `src/kalvin/kline.py`, `tests/test_kline.py`
**Depends on:** Nothing (leaf concept; `is_literal_fn` injected)
**Estimate:** 0.5 day

### Spec

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

### Implementation

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

### Test Cases

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

## 3. Signature (Phase 2)

**Files:** `src/kalvin/signature.py`, `tests/test_signature.py`
**Depends on:** `is_literal_fn` (injected)
**Estimate:** 0.5 day

### Spec

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
- Lossy: order and multiplicity lost; literal nodes lose identity (contribute only bit 0).
- `make_signature([]) == 0` (unsigned).
- `make_signature([literal_A, literal_B]) == 1` (literal-content flag only).
- `make_signature([non_literal]) == non_literal` (identity for single non-literal node).

**`signifies(a, b) → bool`:**

```python
def signifies(a: int, b: int) -> bool:
    return (a & b) != 0
```

The basis for candidate retrieval. Commutative. Vacuous for 0.

**Important:** No `is_signature` predicate. Any uint64 may serve as a signature.
It is identified by role (the `signature` field of a KLine), not by bit pattern.

**Note:** The `significance_value` function has been removed from this module.
Significance inversion (`(~distance) & MASK64`) is performed inline in
`agent.py`. See `specs/significance.md` for the conceptual specification.

### Test Cases

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

---

## 4. Tokenizer (Phase 3)

**Files:** `src/kalvin/mod_tokenizer.py`, `src/kalvin/tokenizer.py`, `tests/test_tokenizer.py`
**Depends on:** Abstract interface (can be inline or in `abstract.py`)
**Estimate:** 1.5 days

### Spec

Converts between text and nodes. Two types, both conforming to the same interface:

```python
class KTokenizer(ABC):
    vocab_size: int                    # Number of distinct tokens
    encode(text_or_int) → list[int]   # Text → nodes
    decode(ids) → str                 # Nodes → text
    is_literal(node) → bool           # Literal test
```

### Mod Tokenizer

Maps characters to bit positions. Two encoding modes:

**Packed encoding** (default):

- All characters OR'd into a single node.
- Bit 0 clear. Lossy: order and multiplicity lost.
- `encode("ABC") → [CHAR_BIT['A'] | CHAR_BIT['B'] | CHAR_BIT['C']]`

**Literal encoding** (`pack=False`):

- One node per character: `(codepoint << 32) | 0xFFFFFFFF`
- Preserves order and identity. Bypasses vocabulary.
- `encode("AB", pack=False) → [(65<<32)|0xFFFFFFFF, (66<<32)|0xFFFFFFFF]`

**Variants:** Mod32 (31 character bits, default), Mod64 (63 character bits).

**Default vocabulary:** `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_=`
Characters not in the explicit vocabulary are assigned the next available bit position.

**`is_literal(node) → bool`:**

```python
(node & 0xFFFFFFFF) == 0xFFFFFFFF
```

**Decode:** Auto-detects literal vs packed from the literal mask.

### BPE Tokenizer

Learns subword vocabulary from a training corpus.

- BPE tokens are **never literal**: `is_literal(node) → False`.
- Raw BPE tokens are sequential IDs. Type prefixes applied at agent layer.
- Optional dependency: requires `rustbpe` + `tiktoken`.

### Test Cases

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
| Vocab size                | Matches number of unique characters in alphabet              |
| Characters not in vocab   | Still encodable (assigned next bit)                          |

---

## 5. STM (Phase 4)

**Files:** `src/kalvin/stm.py`, `tests/test_stm.py`
**Depends on:** Kline, Signature (for `make_signature`)
**Spec:** @stm spec
**Estimate:** 1 day

### Spec

Full definition in **specs/stm.md**. Summary:

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

**Eviction:** When `add` would exceed the bound, the oldest entry is evicted
(FIFO). Eviction removes from the STM index only — the KLine remains in the
frame.

**Dependency:** Requires `make_signature` from the signature module, and an
`is_literal_fn` (injected) for computing nodes signatures.

### Test Cases

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
