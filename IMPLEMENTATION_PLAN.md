# Kalvin Implementation Plan

**Generated from:** `openspec/` specification suite  
**Existing codebase:** `src/kalvin/` (prototype, non-conforming to specs)  
**Date:** 2026-04-28 (revision 2 — literal encoding rework)

---

## 0. Executive Summary

The OpenSpec defines a **rationalisation pipeline** with six tightly-coupled components:

```
Kline ← Signature ← Tokenizer ← Model ← Significance ← Agent
```

The existing codebase has implementations of all six, but they **diverge substantially** from the specs in architecture, API surface, and semantics. This plan proposes a **layer-by-layer rebuild from the leaves up**, with each layer fully tested against its spec before the layer above it starts.

### Key changes since last plan revision

The specs were reworked around **literal node encoding**, with cascading effects on signatures, tokenizers, and the bit-layout invariants that the rest of the system depends on:

1. **Signature spec: literal nodes now contribute bit 0** (not bit 1). `make_signature` changed from `sig |= 0b10` to `sig |= 1`. All-literal klines now produce signature `1` instead of `0b10`. The `is_signature` predicate has been removed entirely — any uint64 may serve as a signature.
2. **Tokenizer spec: literal encoding completely reworked.** Literal nodes now use `(codepoint << 32) | 0xFFFFFFFF` instead of `(codepoint << 1) | 1`. The `is_literal` test changed from `(node & 1) == 1` to `(node & 0xFFFFFFFF) == 0xFFFFFFFF`. This is a **32-bit literal mask** in the lower bits, with the code point in the upper 32 bits.
3. **Kline spec: signatures no longer described as "non-literal nodes (bit 0 clear)".** They are simply `make_signature` outputs — any uint64 value.
4. **Model/Agent specs: countersignature wording updated** to reference the 32-bit literal mask instead of "bit 0 differs".

These changes resolve the **bit 1 conflict** flagged in the previous plan (§3.1) and the **bit 0 ambiguity** between tokenizer `is_literal` and signature construction. The new design creates a clean separation:

| Concept | Bit pattern | Test |
|---------|------------|------|
| Packed (non-literal) node | Bit 0 clear, upper bits carry data | `!is_literal(node)` |
| Literal node | Lower 32 bits = `0xFFFFFFFF`, upper 32 bits = code point | `is_literal(node)` = `(node & 0xFFFFFFFF) == 0xFFFFFFFF` |
| Signature | OR-reduction of nodes; bit 0 = literal-content flag | No bit-pattern test; identified by role, not by pattern |

---

## 1. Specification → Existing Code: Gap Analysis

### 1.1 Kline — minor wording update

| Spec Requirement | Existing Code | Gap |
|---|---|---|
| `Kline(signature, nodes, literal)` with `literal: bool` | No `literal` param; uses None/int/list subtypes | **Rewrite needed.** |
| `kline.is_literal()` → bool | No equivalent on Kline | **Missing.** |
| `nodes` always a sequence of uint64 | `None`, `int`, or `list[int]` | **Contradiction.** |
| Equality: signature + node sequence (literal excluded) | `equals()` exists but handles subtypes | Close. |
| Signature = `make_signature` output (any uint64) | Existing code assumed signatures have bit 0 clear | **Changed.** Signatures can now have bit 0 set (if kline contains literals). |

**Verdict:** Rewrite. Same as previous plans, but now the Model must handle signatures with bit 0 set.

### 1.2 Signature — **fundamentally reworked again**

| Spec Requirement | Existing Code | Gap |
|---|---|---|
| `make_signature(nodes)` → OR all nodes; literal nodes contribute **bit 0** (`1`) | `ModTokenizer.make_signature()` skips literals entirely | **Critical gap.** New spec: literals contribute bit 0. Previous plan had bit 1. Now it's bit 0. |
| `make_signature([literal_A, literal_B])` → `1` | Would produce `0` | **Changed from previous plan** (was `0b10`). |
| `make_signature([literal, non_literal])` → `1 | non_literal` | Would produce `non_literal` (no bit 0) | **Changed.** |
| No `is_signature` predicate | Previous plan defined `is_signature` as `(value & 1) == 0` | **Removed.** Any uint64 can be a signature. This is a simplification. |
| Bit 0 = literal-content flag | No concept | **New.** Bit 0 set means "contains literal content." |
| `signifies(a, b) → (a & b) != 0` | `KLine.signifies()` exists | ✅ Unchanged. |

**Verdict:** The shift from bit 1 to bit 0 for literal-content is the single most important change in this revision. It eliminates the bit 1 conflict with the Mod tokenizer (since the Mod tokenizer uses bits 1–N for character encoding, and bit 0 was already its LITERAL_BIT flag). But it means signatures can now have bit 0 set, which affects every component that previously relied on "bit 0 clear = signature."

### 1.3 Tokenizer — **literal encoding completely reworked**

| Spec Requirement | Existing Code | Gap |
|---|---|---|
| Literal encoding: `(codepoint << 32) \| 0xFFFFFFFF` | `(codepoint << 1) \| 1` | **Fundamental change.** Literal nodes now occupy the full lower 32 bits as a mask. Code points shift from bit 1 to bit 32. |
| `is_literal(node)` = `(node & 0xFFFFFFFF) == 0xFFFFFFFF` | `(node & 1) == 1` | **Changed.** Was a single-bit test, now a 32-bit mask test. |
| Packed encoding: bits 1–N for characters | Same | ✅ Unchanged. |
| Decoding: detect literal via `(node & 0xFFFFFFFF) == 0xFFFFFFFF` | Detect via `bit 0 == 1` | **Changed.** |
| Literal mask avoids collision with packed nodes and signatures | Previous bit-0-only literal could theoretically collide with a signature value that had bit 0 set | ✅ Now clean: `0xFFFFFFFF` lower bits are unambiguous. |

**Verdict:** The literal encoding rework is the second most important change. It affects:
- Every literal node value in the system (different bit layout).
- `is_literal` implementation (32-bit mask instead of single bit).
- `decode` auto-detection (mask test instead of bit-0 test).
- The relationship between literal nodes and signatures (signatures can now have bit 0 set, but literal nodes have all lower 32 bits set — they're distinguishable).

### 1.4 Model — wording updated

| Spec Requirement | Existing Code | Gap |
|---|---|---|
| All items from previous plans (three-tier, STM, etc.) | Same gaps | Same. |
| `is_countersigned(A, B)` | No equivalent | **Same as last revision.** |
| Countersignature: "literal tokens use a 32-bit mask" (not "bit 0 differs") | N/A | **Wording change.** Same logic — literal nodes still can't equal a signature value, but for a different reason (full 32-bit mask vs. any signature value). |

**Verdict:** Same major rewrite as before. The countersignature implementation is the same but the rationale text changed.

### 1.5 Significance — unchanged

No spec changes since last revision.

### 1.6 Agent — wording updated

| Spec Requirement | Existing Code | Gap |
|---|---|---|
| All-literal canonical test: signature = `1` (not `0b10`) | N/A | **Changed from previous plan.** |
| Countersignature wording: "32-bit mask" instead of "bit 0 differs" | N/A | **Wording change.** |

**Verdict:** Same major rewrite. All-literal kline signature is now `1` instead of `0b10`.

---

## 2. Implementation Phases

### Phase 1: Kline (leaf, no dependencies)
**Files:** `src/kalvin/kline.py`  
**Estimate:** 1 day  
**Spec:** `openspec/kline.md` (minor update)

#### Changes
1. Add `literal: bool` parameter to `__init__`.
2. Normalize `nodes` to always be a `list[int]` (empty list, not None).
3. Add `is_literal()` method that returns the `literal` flag.
4. Add `__eq__` based on signature + node sequence (exclude literal flag).
5. Add `__hash__` for use in sets/dicts.
6. Remove subtype helpers (`is_unsigned`, `is_signed`, `is_canonized`).
7. Remove `signifies()`, `filter()`, `mask()` — belong to Model/Signature.
8. Keep `dbg_text` as implementation-level (not spec'd).

**Note:** Signatures can now be any uint64 (including values with bit 0 set). The Kline doesn't enforce any bit-pattern constraint on its signature field.

#### Tests
- Construction, equality, hash, empty kline, single-node kline.
- Kline with signature that has bit 0 set (from literal-content flag).

---

### Phase 2: Signature (depends on Tokenizer `is_literal`)
**Files:** New `src/kalvin/signature.py`  
**Estimate:** 0.5 day  
**Spec:** `openspec/signature.md` **(updated: bit 0, not bit 1)**

#### Changes
Create `signature.py` with:

```python
def make_signature(nodes: list[int], is_literal_fn) -> int:
    sig = 0
    for node in nodes:
        if is_literal_fn(node):
            sig |= 1       # bit 0: literal-content flag
        else:
            sig |= node    # non-literal contributes full value
    return sig

def signifies(a: int, b: int) -> bool:
    return (a & b) != 0
```

Key differences from previous plan:
- **No `is_signature` predicate.** Removed from spec. Any uint64 can be a signature.
- **Literal nodes contribute bit 0** (`1`), not bit 1 (`0b10`).
- **No `LITERAL_CONTENT_BIT` constant.** It's just `1`.

#### Critical test cases
```python
make_signature([]) == 0                                    # unsigned
make_signature([literal]) == 1                             # literal-content flag only
make_signature([literal, literal]) == 1                    # idempotent
make_signature([non_literal]) == non_literal               # identity
make_signature([literal, non_literal]) == (1 | non_literal)  # mixed; bit 0 set
make_signature([non_literal_A, non_literal_B]) == (A | B)  # commutative

signifies(0, anything) == False                            # unsigned vacuous
signifies(1, 1) == True                                    # all-literal klines match each other
signifies(1, packed_node) == False                         # all-literal vs packed (bit 0 only)
signifies(1 | packed, packed) == True                      # mixed kline matches packed
```

#### Migration impact
- **Every signature value in the system changes** (again).
- All-literal klines: `0` → `1` (not `0b10` as in previous plan).
- Mixed klines: `non_literal_bits` → `1 | non_literal_bits` (bit 0 now set).

---

### Phase 3: Tokenizer (depends on Kline, Signature)
**Files:** `src/kalvin/tokenizer.py`, `src/kalvin/mod_tokenizer.py`  
**Estimate:** 1.5 days (increased — literal encoding rework)  
**Spec:** `openspec/tokenizer.md` **(updated: literal mask encoding)**

This phase has significantly more work than previous plans due to the literal encoding rework.

#### Changes to Mod tokenizer

**1. Packed encoding** — unchanged:
```python
encode("ABC") → [CHAR_BIT['A'] | CHAR_BIT['B'] | CHAR_BIT['C']]  # bit 0 clear
```

**2. Literal encoding** — completely new format:
```python
# OLD: (codepoint << 1) | 1
# NEW: (codepoint << 32) | 0xFFFFFFFF

encode("ABC", literal=True) → [(65 << 32) | 0xFFFFFFFF,
                                (66 << 32) | 0xFFFFFFFF,
                                (67 << 32) | 0xFFFFFFFF]
```

**3. `is_literal` test** — changed:
```python
# OLD: (node & 1) == 1
# NEW: (node & 0xFFFFFFFF) == 0xFFFFFFFF
```

**4. Decode auto-detection** — changed:
```python
# OLD: if bit 0 == 1 → literal
# NEW: if (node & 0xFFFFFFFF) == 0xFFFFFFFF → literal
```

**5. Literal decode** — changed:
```python
# OLD: chr(node >> 1)
# NEW: chr(node >> 32)
```

**6. Raw integer literal encoding** — changed:
```python
# OLD: (value << 1) | 1
# NEW: (value << 32) | 0xFFFFFFFF
```

**7. Remove `make_signature()`** — moved to `signature.py`.

#### Changes to BPE tokenizer
1. Remove `make_signature()`.
2. `is_literal()` → `False` (unchanged).

#### Abstract interface (`KTokenizer` in `abstract.py`)
1. Remove `make_signature` from ABC.
2. `is_literal(node) → bool` — semantics changed from bit-0 test to mask test, but the ABC signature is the same.

#### Interaction between literal mask and signatures

The literal mask `0xFFFFFFFF` ensures literal nodes are unambiguously distinguishable from everything else:

| Value type | Lower 32 bits | Bit 0 | Example |
|------------|--------------|-------|---------|
| Packed node | Not all 1s | 0 | `0b110` (chars A+B) |
| Literal node | `0xFFFFFFFF` | 1 | `0x00000041_FFFFFFFF` (char 'A') |
| Signature (no literals) | Not all 1s | 0 | `0b110` (same as packed — by design) |
| Signature (with literals) | Not all 1s (unless coincidentally) | 1 | `0b111` (bit 0 from literal flag + bits 1-2 from non-literals) |

A signature with bit 0 set (from literal-content flag) is **not** confused with a literal node because:
- Literal node: lower 32 bits = `0xFFFFFFFF` (all set).
- Signature with bit 0: lower 32 bits have only bit 0 set (or a few bits from non-literal contributions), not all 32.

This is the key design insight: the 32-bit literal mask creates a **wide moat** between literal nodes and everything else.

#### Tests
- Mod32: packed encode/decode round-trip (order loss accepted).
- Mod32: literal encode/decode round-trip (order preserved).
- Mod32: literal encoding produces correct values: `('A' << 32) | 0xFFFFFFFF`.
- Mod32: `is_literal` returns True for literal nodes, False for packed.
- Mod32: `is_literal` returns False for signatures with bit 0 set.
- Mod64: same.
- `decode` auto-detection: literal mask triggers literal decode, not packed.
- Empty string → empty list.
- Characters not in vocabulary handled correctly.
- Raw integer literal encoding.
- Round-trip: `decode(encode(text, literal=True)) == text` for all printable ASCII.

---

### Phase 4: Model (depends on Kline, Signature, Tokenizer)
**Files:** `src/kalvin/model.py`, `src/kalvin/stm.py`  
**Estimate:** 2–3 days  
**Spec:** `openspec/model.md` (minor wording update)

Same three-tier architecture as before. No structural changes from previous plan.

#### `is_countersigned` implementation
```python
def is_countersigned(self, A: Kline, B: Kline) -> bool:
    return (B.signature in A.nodes) and (A.signature in B.nodes)
```

The rationale for why literal nodes can't match signatures changed ("literal tokens use a 32-bit mask in the lower bits, which does not equal any signature value" instead of "bit 0 differs"), but the implementation is identical — it's a simple structural test on node values.

#### Signatures with bit 0 set
The Model must now handle klines whose signatures have bit 0 set (from literal-content flag). This affects:
- **Indexing:** Signatures with bit 0 set are valid keys in the signature index.
- **AND matching:** `k.signature & query.signature != 0` works correctly regardless of bit 0.
- **Equality:** Two klines with signature `1` but different nodes are different klines.

#### Tests
Same as previous plan, plus:
- Model correctly indexes and retrieves klines with signatures that have bit 0 set.
- `is_countersigned` works when signatures have bit 0 set (literal nodes still don't match because their full value includes `0xFFFFFFFF` lower bits).

---

### Phase 5: Significance Pipeline (depends on Model)
**Files:** `src/kalvin/significance.py` (complete rewrite), `src/kalvin/model.py`  
**Estimate:** 2 days  
**Spec:** `openspec/significance.md` (unchanged)

Same as previous plan. No spec changes affect this phase.

---

### Phase 6: Agent (depends on everything)
**Files:** `src/kalvin/agent.py`  
**Estimate:** 2 days  
**Spec:** `openspec/agent.md` (wording update: `1` not `0b10`)

Same 6-phase pipeline as before. All-literal kline signature is now `1`:

```
Canonical — all-literal: make_signature contributes bit 0 for each literal node,
Q's signature is 1 — a valid canonical signature.
```

#### Tests
- All-literal kline rationalisation → "frame" S1 (signature `1`).
- Countersignature cogitation.
- Cogitation max-pass limit.
- Mixed literal/non-literal klines.

---

## 3. Missing Details, Contradictions & Open Questions

### 3.1 ✅ Bit 1 Conflict — RESOLVED

**Previous plan flagged:** Bit 1 was reserved for literal-content, but Mod tokenizer maps the first alphabet character to bit 1.

**Resolution:** The spec changed to use bit 0 for literal-content. The Mod tokenizer's character bits start at bit 1. No conflict. Bit 1 is now free for character encoding as intended.

### 3.2 ✅ Bit 0 Ambiguity — RESOLVED

**Previous plans flagged:** Bit 0 was both the tokenizer's literal discriminator and the signature's clear-bit invariant.

**Resolution:** The new design gives bit 0 a single, consistent meaning:
- **In signatures:** bit 0 = literal-content flag (set by `make_signature`).
- **In packed nodes:** bit 0 = 0 (clear, not a literal).
- **In literal nodes:** lower 32 bits = `0xFFFFFFFF` (bit 0 set, plus 31 more bits).

The `is_literal` test changed from a single-bit test to a 32-bit mask test, so a signature with bit 0 set is NOT misidentified as a literal node.

### 3.3 ⚠️ All-Literal Klines All Have the Same Signature

All klines whose nodes are all literal produce `make_signature` = `1`. This means:
- Two completely different all-literal klines have the same signature.
- They are distinguishable only by their node sequences (kline equality uses both).
- The Model's signature index will have many klines under key `1`.
- AND matching: `signifies(1, 1) == True` — every all-literal kline is a candidate for every other all-literal kline.
- This is a potential **performance concern** — a model with many all-literal klines will produce large candidate sets for any all-literal query.

**Mitigation options:**
- Accept it. All-literal klines are fast-tracked in Assess (Phase 3) and never reach candidate retrieval.
- If they do reach retrieval (e.g., during cogitation), the significance pipeline will quickly reject irrelevant candidates.
- Document that all-literal klines should be promoted to base to avoid repeated processing.

### 3.4 ⚠️ Model Significance API Semantics (Still TBD)

Same as previous plans. `is_s1`, `s2_distance`, `s3_distance` remain TBD. **Highest-risk open question.**

**Recommendation:** Define initial simple semantics:
- `is_s1(node, candidate)`: `node == candidate.signature` (exact match).
- `s2_distance`: `(1 - s1_ratio) * D_boundary`.
- `s3_distance`: Based on bit overlap ratio, mapped to [D_boundary, D_MAX).

### 3.5 ⚠️ Countersignature with Signatures That Have Bit 0 Set

A signature with bit 0 set (from literal-content flag) could theoretically equal a non-literal node value. For example, a packed node `0b110` (chars A+B) used as a node in another kline would match a signature `0b111` (literal-content + chars A+B) via `in` operator (no, wait — `0b110 != 0b111`).

Actually, the `in` operator in `is_countersigned` does exact equality: `B.signature in A.nodes`. Since `B.signature` is the full `make_signature` output (including bit 0 if literals present), and `A.nodes` contains individual node values (not signatures), the match only happens when a node value exactly equals `B.signature`. This is a legitimate structural edge — the kline's node literally references the other kline's signature. The countersignature test is correct regardless of bit 0.

### 3.6 ⚠️ `is_literal` for BPE Tokens

BPE tokens are never literal: `is_literal(node) → False`. But what does the 32-bit mask test return for a BPE token?

```python
# BPE token: raw vocabulary index, e.g., 15496
# 0xFFFFFFFF == 4294967295
# 15496 & 0xFFFFFFFF == 15496 != 0xFFFFFFFF → False ✓
```

A BPE token with a type prefix (e.g., `POS_DET | 257 = 4194337`):
```python
# 4194337 & 0xFFFFFFFF == 4194337 != 0xFFFFFFFF → False ✓
```

The 32-bit mask test works correctly for BPE tokens. Even large combined values (type prefix | token ID) won't accidentally have all lower 32 bits set.

### 3.7 ⚠️ Literal Encoding: 32-Bit Code Point Limit

The new literal encoding uses `(codepoint << 32)`. Unicode code points currently fit in 21 bits (up to U+10FFFF). The upper 32 bits provide ample room. However, the raw integer encoding `encode(42, literal=True)` → `(42 << 32) | 0xFFFFFFFF` means values above `0xFFFFFFFF` (2^32 - 1) cannot be literally encoded without truncation. This limits raw integer literal encoding to 32-bit values. For the Mod tokenizer's use case (ASCII/Unicode characters), this is not a constraint.

### 3.8 ⚠️ STM Nodes-Signature Indexing: All-Literal Collision

With the new signature, all all-literal klines produce `nodes_sig = 1`. STM's dual-keyed index will store many klines under key `1`. The `find_by_nodes(1)` method returns the most recently added. This is correct behavior (same as multiple klines sharing any signature), but worth documenting as a degenerate case.

### 3.9 ⚠️ Tokenizer Default Encode Mode (Unchanged)

Same ambiguity as before. `encode(text)` default not specified. Recommend literal for round-tripping.

### 3.10 ⚠️ `__init__.py` and Torch Dependencies (Unchanged)

Remove device imports to unblock testing.

---

## 4. Dependency Graph & Build Order

```
Phase 1: Kline          ← no deps (leaf)
Phase 2: Signature      ← depends on tokenizer.is_literal (injected)
Phase 3: Tokenizer      ← depends on Kline, Signature (produces nodes)
Phase 4: Model          ← depends on Kline, Signature
Phase 5: Significance   ← depends on Model
Phase 6: Agent          ← depends on everything

Phase 2 and 3 can proceed in parallel.
Phase 3 has increased scope due to literal encoding rework.
```

## 5. Recommended Execution Order

| Step | Phase | Deliverable | Key Risk |
|------|-------|-------------|----------|
| 1 | Cleanup | Remove torch deps from `__init__.py`; fix test runner | Blocks everything |
| 2 | Phase 1 | New `Kline` with `literal` flag, normalized `nodes` | Subtype removal cascades |
| 3 | Phase 2 | `signature.py` with bit-0 literal-content flag | All signature values change |
| 4 | Phase 3 | Refactored tokenizers with **new literal encoding** | Biggest change in this revision |
| 5 | Phase 4 | Three-tier `Model` + `is_countersigned` | Largest phase |
| 6 | Phase 5 | Significance pipeline (rewrite) | TBD semantics |
| 7 | Phase 6 | Agent with 6-phase pipeline + countersignature cogitation | Integration risk |
| 8 | Integration | End-to-end tests, kscript compat | Data format changes |

## 6. Risk Assessment

| Risk | Severity | Change | Mitigation |
|------|----------|--------|------------|
| Model significance API semantics TBD (§3.4) | **High** | Unchanged | Define initial simple semantics; iterate. |
| Literal encoding rework affects all downstream | **High** | **New** | Phase 3 tests must be comprehensive; verify round-trips. |
| All-literal signature collision (§3.3) | **Medium** | **New** | All-literal klines fast-tracked in Assess; accept degenerate indexing. |
| Signature change invalidates all stored data | **High** | Unchanged | Clean break; rebuild knowledge graphs. |
| Significance rewrite breaks all existing tests | **High** | Unchanged | Write spec-conformant tests first. |
| Kline `nodes` simplification cascades | **Medium** | Unchanged | Phase 1 first; grep for subtype helpers. |
| `is_literal` 32-bit mask accidentally matches non-literal | **Low** | **New** | Only `0xFFFFFFFF` matches; packed nodes and BPE tokens don't have all lower 32 bits set. Verified (§3.6). |

## 7. Files to Create / Modify / Delete

### New files
- `src/kalvin/signature.py` — standalone signature functions (bit-0 literal-content flag)

### Major rewrites
- `src/kalvin/kline.py` — literal flag, normalized nodes
- `src/kalvin/mod_tokenizer.py` — **new literal encoding format** (32-bit mask)
- `src/kalvin/significance.py` — complete rewrite (pipeline, not class hierarchy)
- `src/kalvin/model.py` — three-tier architecture + `is_countersigned`
- `src/kalvin/agent.py` — 6-phase rationalisation + countersignature cogitation

### Moderate modifications
- `src/kalvin/tokenizer.py` — remove `make_signature`, BPE unchanged
- `src/kalvin/abstract.py` — remove `KSignificance`, update `KTokenizer.is_literal` semantics
- `src/kalvin/__init__.py` — remove device imports
- `src/kalvin/stm.py` — add bound/eviction, updated nodes-signature indexing

### Potential removals
- `src/kalvin/device.py` — torch dependency, unused by spec
- `src/kalvin/utils.py` — torch dependency, unused by spec
- `src/kalvin/graph.py` — functionality absorbed into Model

### Test files
- `tests/test_kline.py` (new)
- `tests/test_signature.py` (new — **must test bit-0 literal-content flag**)
- `tests/test_tokenizer.py` (update — **new literal encoding format**)
- `tests/test_model.py` (major update — add countersignature, signatures with bit 0)
- `tests/test_significance.py` (complete rewrite)
- `tests/test_agent.py` (major update)
- `tests/test_stm.py` (update for bound/eviction + new signature/tokenizer)

## 8. Bit Layout Summary (Post-Revision)

This is the definitive bit layout after all spec updates. Use this as a reference during implementation:

```
┌──────────────────────────────────────────────────────────────────┐
│                     NODE VALUE TYPES                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PACKED NODE (Mod tokenizer, non-literal)                        │
│  ┌────┬────┬────┬───────────────────────────────────────────┐    │
│  │  0 │  0-1  │ character bits (1–31 for Mod32, 1–63 Mod64) │    │
│  └────┴────┴────┴───────────────────────────────────────────┘    │
│  bit 0 = 0, is_literal = False                                   │
│                                                                  │
│  LITERAL NODE (Mod tokenizer)                                     │
│  ┌─────────────────┬────────────────────────────────────────┐    │
│  │ code point (32b) │ 0xFFFFFFFF (literal mask, lower 32b)  │    │
│  └─────────────────┴────────────────────────────────────────┘    │
│  lower 32 bits all set, is_literal = True                         │
│                                                                  │
│  BPE NODE (with type prefix)                                      │
│  ┌──────────────────────┬───────────────────────────────────┐    │
│  │ type prefix bits     │ vocabulary index (lower bits)      │    │
│  └──────────────────────┴───────────────────────────────────┘    │
│  is_literal = False (never literal)                               │
│                                                                  │
│  SIGNATURE (make_signature output)                                │
│  ┌────┬──────────────────────────────────────────────────────┐   │
│  │ LC │ OR-reduction of non-literal nodes                    │   │
│  └────┴──────────────────────────────────────────────────────┘   │
│  bit 0 = literal-content flag (LC), 1 if any literal nodes       │
│  bits 1-63 = OR of non-literal node values                       │
│  NO bit-pattern test — identified by role, not by pattern        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

DISCRIMINATORS:
  is_literal(node)  = (node & 0xFFFFFFFF) == 0xFFFFFFFF  (tokenizer)
  is_packed(node)   = (node & 0xFFFFFFFF) != 0xFFFFFFFF AND bit 0 == 0
  is_signature(val) = NO TEST — any uint64 by role
```
