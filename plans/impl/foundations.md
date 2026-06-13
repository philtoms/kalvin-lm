# Sub-Plan: Foundations — Bit Layout, KLine, Signature, Tokenizer, STM

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 0–4
**Estimate:** 4 days
**Depends on:** Nothing (leaf components)

---

> **Archival Note (added 2026-06-13):** This plan was written before the compiled-entry `op` field
> terminology was updated. The code described here used token-name op strings; the current code uses
> structural-state names. The mapping:
>
> - `"UNSIGNED"` → `"IDENTITY"` — the op value for bare-node identity klines was renamed (per the
>   `plans/impl/rename-unsigned-to-identity.md` plan, completed before ADR-0006).
> - `"COUNTERSIGN"` → `"COUNTERSIGNED"`, `"UNDERSIGN"` → `"UNDERSIGNED"`,
>   `"CONNOTATE"` → `"CONNOTED"`, `"CANONIZE"` → `"CANONIZED"` — compiled-entry op values
>   now use past-participle structural-state names (per ADR-0006, implemented in KB-209).
>
> **Old terms present in this file:** `UNSIGNED` (in the well-known values table).
>
> For the authoritative current terminology, see CONTEXT.md glossary entries **Structural State**
> and **Identity**, and `docs/adr/0006-op-is-structural-state-not-token.md`.

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
│ PACKED NODE (Mod tokenizer)                                      │
│ ┌────┬──────────────────────────────────────────────────────┐    │
│ │  0 │ bits 1–N: character bits (N=31 for Mod32, 63 Mod64) │    │
│ └────┴──────────────────────────────────────────────────────┘    │
│ bit 0 = 0                                                        │
│                                                                   │
│ BPE NODE (with type prefix)                                       │
│ ┌─────────────────────┬──────────────────────────────────────┐   │
│ │ type prefix bits     │ vocabulary index (lower bits)        │   │
│ └─────────────────────┴──────────────────────────────────────┘   │
│                                                                   │
│ SIGNATURE (make_signature output)                                 │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ OR-reduction of all nodes                                    │ │
│ └──────────────────────────────────────────────────────────────┘ │
│ NO bit-pattern test — identified by role, not by pattern         │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Discriminators

```python
# No bit-pattern test for signatures
is_signature(x)  = NO TEST — any uint64 can be a signature
```

### 1.3 Well-Known Values

| Value                   | Name                 | Meaning                                                  |
| ----------------------- | -------------------- | -------------------------------------------------------- |
| `0`                     | `UNSIGNED`           | No nodes. Empty kline. Cannot be found via AND matching. |
| 9                       | `_S3_BIAS`           | Tier bias for S3 connotation hops before quadratic packing.        |
| `0xFFFF_FFFF_FFFF_FFFF` | `D_MAX` / `S1_VALUE` | Maximum distance / maximum significance.                 |
| `0x0000_0000_0000_0000` | `S4_VALUE`           | Zero significance / maximum distance.                    |

### 1.4 Why This Layout Works

All nodes go through the tokenizer — there is no branching between encoding
paths. Two node types exist:

- **Packed nodes:** uppercase alpha characters OR'd into a single `uint64` with bit 0 clear. Lossy: order and multiplicity are lost. Used as bitmask signatures for fast overlap-based candidate retrieval.
- **NLP-BPE nodes:** `(nlp_type32 << 32) | bpe_token_id` — grammatically rich tokens carrying POS/DEP/MORPH type bits and BPE vocabulary indices.

Both types contribute their full value to signature OR-reduction without masking or branching.

---

## 2. KLine (Phase 1)

**Files:** `src/kalvin/kline.py`, `tests/test_kline.py`
**Depends on:** Nothing (leaf concept)
**Estimate:** 0.5 day

### Spec Reference

See **@kline spec** for full definition (structure, equality, construction).
Test matrix: KL-1 through KL-7, KL-11.

**Key rules (from spec):**

- `nodes` always `list[int]` (empty list for no nodes, never None).
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
| KL-11   | Length                      | `len(KLine(5, [1, 2, 3])) == 3`       |

---

## 3. Signature (Phase 2)

**Files:** `src/kalvin/signature.py`, `tests/test_signature.py`
**Depends on:** Nothing (pure function)
**Estimate:** 0.5 day

### Spec Reference

See **@signature spec** for full definition (creation, properties, bitwise AND matching).
Test matrix: SIG-1, SIG-4, SIG-6, SIG-7, SIG-9, SIG-10, SIG-14.

**Key functions (from spec):**

- `make_signature(nodes) → int`: OR-reduction of all raw node values. No masking, no branching, no special cases.
- `signifies(a, b) → bool`: `(a & b) != 0`.
- No `is_signature` predicate — any uint64 may serve as a signature.

**Note:** The `significance_value` function has been removed from this module.
Significance inversion is performed inline in `model.py`. See @model spec
§Significance Semantics.

### Test Cases

| Spec ID | Test                                     | Expected                |
| ------- | ---------------------------------------- | ----------------------- |
| SIG-1   | `make_signature([])`                     | `0`                     |
| SIG-4   | `make_signature([packed])`               | `packed` (identity)     |
| SIG-6   | `make_signature([A, B])`                 | `A \| B` (commutative)  |
| SIG-7   | `signifies(0, anything)`                 | `False`                 |
| SIG-9   | `signifies(0b110, 0b10)`                 | `True`                  |
| SIG-10  | `signifies(0b110, 0b1)`                  | `False`                 |
| SIG-14  | Mod32 backward compat                    | `0b10 | 0b100 == 0b110` |

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
- All other strings go through the tokenizer uniformly — no special encoding path.
- No `pack` parameter — mode auto-detected from input content.
- Variants: Mod32 (31 bits, default), Mod64 (63 bits).

### BPE Tokenizer

BPE tokens are sequential vocabulary IDs; type prefixes applied at
agent layer. Optional dependency: `rustbpe` + `tiktoken`.
See @tokenizer spec §BPE Tokenizer.

### Test Cases

| Spec ID | Test                      | Description                                                  |
| ------- | ------------------------- | ------------------------------------------------------------ |
| TOK-1   | Packed encode single char | `encode("A") == [bit_A]`                                     |
| TOK-2   | Packed encode multi-char  | `encode("AB") == [bit_A \| bit_B]`                           |
| TOK-3   | Packed round-trip         | `decode(encode("ABC"))` contains A, B, C (order may differ)  |
| TOK-7   | Empty string              | `encode("") == []`, `decode([]) == ""`                       |
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
