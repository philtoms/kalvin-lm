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
# Standalone function — not a tokenizer method
def is_literal(node: int) -> bool:
    """Bit-layout test: lower 32 bits all set = literal node."""
    return (node & 0xFFFF_FFFF) == 0xFFFF_FFFF

# Derived properties
def is_packed(node: int) -> bool:
    return not is_literal(node) and (node & 1) == 0

# No bit-pattern test for signatures
is_signature(x)  = NO TEST — any uint64 can be a signature
```

### 1.3 Well-Known Values

| Value                   | Name                 | Meaning                                                  |
| ----------------------- | -------------------- | -------------------------------------------------------- |
| `0`                     | `UNSIGNED`           | No nodes. Empty kline. Cannot be found via AND matching. |
| `1`                     | `LITERAL_ONLY`       | Contains literal content only (no non-literal nodes).    |
| 9                       | `_S3_BIAS`           | Tier bias for S3 connotation hops before quadratic packing.        |
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
**Depends on:** Nothing (leaf concept; `is_literal` is a standalone function)
**Estimate:** 0.5 day

### Spec Reference

See **@kline spec** for full definition (structure, equality, construction, `is_literal`).
Test matrix: KL-1 through KL-14.

**Key rules (from spec):**

- `nodes` always `list[int]` (empty list for no nodes, never None).
- No `literal` field stored — computed on demand via `is_literal()`.
- Equality: same signature AND same node sequence.
- Hashable: `hash((signature, tuple(nodes)))`.

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

    def is_literal(self) -> bool:
        if not self.nodes:
            return False
        return all(is_literal(n) for n in self.nodes)

    def __eq__(self, other): ...
    def __hash__(self): ...
    def __len__(self): ...
    def __repr__(self): ...
```

### Test Cases

| Spec ID | Test                        | Description                           |
| ------- | --------------------------- | ------------------------------------- |
| KL-1    | Construction: empty nodes   | `KLine(5, [])` → nodes = `[]`         |
| KL-2    | Construction: single int    | `KLine(5, 3)` → nodes = `[3]`         |
| KL-3    | Construction: list          | `KLine(5, [1, 2])` → nodes = `[1, 2]` |
| KL-4    | Equality: same sig + nodes  | `KLine(5, [1]) == KLine(5, [1])`      |
| KL-5    | Inequality: different sig   | `KLine(5, [1]) != KLine(6, [1])`      |
| KL-6    | Inequality: different nodes | `KLine(5, [1]) != KLine(5, [2])`      |
| KL-7    | Hash consistency            | Equal KLines have equal hashes        |
| KL-8    | is_literal: all literal     | All nodes have literal mask → True    |
| KL-9    | is_literal: mixed           | Some non-literal nodes → False        |
| KL-10   | is_literal: empty           | Returns False                         |
| KL-11   | Length                      | `len(KLine(5, [1, 2, 3])) == 3`       |

---

## 3. Signature (Phase 2)

**Files:** `src/kalvin/signature.py`, `tests/test_signature.py`
**Depends on:** `is_literal` (standalone function, same module or imported)
**Estimate:** 0.5 day

### Spec Reference

See **@signature spec** for full definition (creation, properties, bitwise AND matching).
Test matrix: SIG-1 through SIG-10.

**Key functions (from spec):**

- `make_signature(nodes) → int`: OR-reduction with literal-content flag.
- `signifies(a, b) → bool`: `(a & b) != 0`.
- No `is_signature` predicate — any uint64 may serve as a signature.

**Note:** The `significance_value` function has been removed from this module.
Significance inversion is performed inline in `model.py`. See @model spec
§Significance Semantics.

### Test Cases

| Spec ID | Test                                     | Expected                |
| ------- | ---------------------------------------- | ----------------------- |
| SIG-1   | `make_signature([])`                     | `0`                     |
| SIG-2   | `make_signature([literal])`              | `1`                     |
| SIG-3   | `make_signature([literal, literal])`     | `1` (idempotent)        |
| SIG-4   | `make_signature([packed])`               | `packed` (identity)     |
| SIG-5   | `make_signature([literal, packed])`      | `1 \| packed`           |
| SIG-6   | `make_signature([A, B])`                 | `A \| B` (commutative)  |
| SIG-7   | `signifies(0, anything)`                 | `False`                 |
| SIG-8   | `signifies(1, 1)`                        | `True`                  |
| SIG-9   | `signifies(0b110, 0b10)`                 | `True`                  |
| SIG-10  | `signifies(0b110, 0b1)`                  | `False`                 |

---

## 4. Tokenizer (Phase 3)

**Files:** `src/kalvin/mod_tokenizer.py`, `src/kalvin/tokenizer.py`, `tests/test_tokenizer.py`
**Depends on:** Abstract interface (can be inline or in `abstract.py`)
**Estimate:** 1.5 days

### Spec Reference

See **@tokenizer spec** for full definition (interface, Mod/BPE variants, encoding modes).
Test matrix: TOK-1 through TOK-12.

**Key rules (from spec):**

- Packed encoding: all-uppercase-alpha → single node, bit 0 clear, OR-reduced.
- Literal encoding: everything else → one node per character, literal mask `(codepoint << 32) | 0xFFFFFFFF`.
- No `pack` parameter — mode auto-detected from input content.
- `is_literal` is a standalone function, not a tokenizer method.
- Variants: Mod32 (31 bits, default), Mod64 (63 bits).

> **Note:** `is_literal` is no longer part of the tokenizer interface.
> It is a standalone function defined in the node encoding layer (see §1.2).

### BPE Tokenizer

BPE tokens are **never literal**. Raw tokens are sequential IDs; type prefixes
applied at agent layer. Optional dependency: `rustbpe` + `tiktoken`.
See @tokenizer spec §BPE Tokenizer.

### Test Cases

| Spec ID | Test                      | Description                                                  |
| ------- | ------------------------- | ------------------------------------------------------------ |
| TOK-1   | Packed encode single char | `encode("A") == [bit_A]`                                     |
| TOK-2   | Packed encode multi-char  | `encode("AB") == [bit_A \| bit_B]`                           |
| TOK-3   | Packed round-trip         | `decode(encode("ABC"))` contains A, B, C (order may differ)  |
| TOK-4   | Literal encode            | `encode("123")` → two nodes with literal mask                |
| TOK-5   | Literal round-trip        | `decode(encode("123")) == "123"` (order preserved)           |
| TOK-6   | Auto-detection            | `encode("A")` → packed; `encode("1")` → literal              |
| TOK-7   | Empty string              | `encode("") == []`, `decode([]) == ""`                       |
| TOK-8   | is_literal: literal node  | `is_literal((65 << 32) \| 0xFFFFFFFF) == True`               |
| TOK-9   | is_literal: packed node   | `is_literal(6) == False`                                     |
| TOK-10  | is_literal: zero          | `is_literal(0) == False`                                     |
| TOK-11  | Vocab size                | Matches number of unique characters in alphabet              |
| TOK-12  | Characters not in vocab   | Still encodable (assigned next bit)                          |

---

## 5. STM (Phase 4)

**Files:** `src/kalvin/stm.py`, `tests/test_stm.py`
**Depends on:** Kline, Signature (for `make_signature`)
**Spec:** @stm spec
**Estimate:** 1 day

### Spec Reference

Full definition in **@stm spec**. Test matrix: STM-1 through STM-10.

**Summary of key API (from spec):**

```python
stm.add(kline, dedup=True) → bool      # Add, returns False if duplicate
stm.get(key) → list[KLine]             # All KLines under key
stm.find_by_signature(sig) → list[KLine]
stm.find_by_nodes(nodes_sig) → list[KLine]
stm.remove(kline) → None               # Remove from all indexes
stm.clear() → None
len(stm) → int
```

**Dual-keyed indexing:** each KLine indexed under both signature and
`make_signature(kline.nodes)`. FIFO eviction at bound. See @stm spec.

### Test Cases

| Spec ID | Test                          | Description                                                   |
| ------- | ----------------------------- | ------------------------------------------------------------- |
| STM-1   | Add and retrieve by signature | `stm.add(kl); stm.find_by_signature(sig) == [kl]`             |
| STM-2   | Add and retrieve by nodes sig | `stm.add(kl); stm.find_by_nodes(nodes_sig) == [kl]`           |
| STM-3   | Bound enforcement             | Add `bound + 1` KLines; first is evicted                      |
| STM-4   | Eviction correctness          | Evicted KLine not in STM but not in frame (frame is separate) |
| STM-5   | Dedup                         | Adding same (sig, nodes) returns False                        |
| STM-6   | Multiple KLines same sig      | All stored under one key                                      |
| STM-7   | Remove                        | Remove clears from all indexes                                |
| STM-8   | Clear                         | Everything removed                                            |
| STM-9   | Empty STM                     | `len(stm) == 0`, all lookups return empty/None                |
| STM-10  | Nodes signature indexing      | Two KLines with different sigs but same nodes_sig both found  |
