# Sub-Plan: Build Phases — Design Decisions and Execution Plan

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 0–9
**Purpose:** Resolved design decisions that apply across components, and the
full execution plan with per-phase deliverables.

---

## 1. Resolved Design Decisions

### 1.1 Candidate Retrieval Efficiency

**Decision:** Implement `model.where(signature)` using linear scan initially.
Add a `candidates_for(signature)` method later if profiling shows it's needed.

**Rationale:** For models with <10K KLines, linear scan is fast enough.
The inverted bit-index optimization is a performance concern, not a
correctness concern. Ship first, optimize later.

**Future optimization:** Maintain an inverted index
`bit_position → set[signature]`. `candidates_for(sig)` would union the sets
for all set bits in `sig`.

### 1.2 Default Encode Mode

**Decision:** `tokenizer.encode(text)` NLP-tokenizes all input uniformly —
there is no packed/literal mode and no `pack` parameter.

**Rationale:** All text goes through the NLP tokenizer (BPE subword base) on
a single encoding path; there is nothing to branch on, so a mode parameter
is unnecessary. This simplifies all callers.

### 1.3 Persistence Format

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
- Per kline: `uint64` signature, `uint32` node count, `uint64` × N nodes

### 1.4 Project Dependencies

**Core (required):**

- Python ≥ 3.10
- pytest (dev)

**Optional:**

- `rustbpe` — BPE tokenizer training
- `tiktoken` — BPE tokenizer inference
- `pyarrow` — parquet data loading for BPE training

**Not required:** torch, numpy, matplotlib, spacy, textual.

---

## 2. Execution Plan

### Phase 0: Project Scaffold → **foundations.md §0**

**Estimate:** 0.5 day
**Deliverable:** Empty project with passing test runner.

### Phase 1: KLine → **foundations.md §2**

**Estimate:** 0.5 day
**Deliverables:** `src/kalvin/kline.py`, `tests/test_kline.py`

### Phase 2: Signature → **foundations.md §3**

**Estimate:** 0.5 day
**Deliverables:** `src/kalvin/signature.py`, `tests/test_signature.py`

### Phase 3: Tokenizer → **foundations.md §4**

**Estimate:** 1.5 days
**Deliverables:** `src/kalvin/tokenizer.py`, `src/kalvin/nlp_tokenizer.py`, `tests/test_tokenizer.py`

### Phase 4: STM → **foundations.md §5**, **specs/stm.md**

**Estimate:** 1 day
**Deliverables:** `src/kalvin/stm.py`, `tests/test_stm.py`

### Phase 5: Model → **model.md**

**Estimate:** 2–3 days (largest component)
**Deliverables:** `src/kalvin/model.py`, `tests/test_model.py`

### Phase 6: Significance Constants → **agent.md §1**

**Estimate:** 0.5 day
**Deliverables:** Constants (`D_MAX`, `MASK64`) inlined in `src/kalvin/agent.py`

### Phase 7: Events → **agent.md §2**

**Estimate:** 0.5 day
**Deliverables:** `src/kalvin/events.py`, `tests/test_events.py`

### Phase 8: Agent + Cogitator → **agent.md §3**

**Estimate:** 2 days
**Deliverables:** `src/kalvin/agent.py`, `tests/test_agent.py`

### Phase 9: Persistence & Polish

**Estimate:** 1 day

**Tasks:**

1. Binary serialization (`Agent.to_bytes` / `from_bytes`).
2. JSON serialization (`Agent.to_dict` / `from_dict`).
3. File save/load with auto-format detection.
4. Remove any unused dependencies.
5. Integration test: full pipeline from text input to persisted model.

---

## 3. Dependency Graph

```
Phase 1 (KLine) ──────┐
Phase 2 (Signature) ──┤
Phase 3 (Tokenizer) ──┼── Phase 4 (STM) ── Phase 5 (Model) ──┐
Phase 6 (Constants) ──┤                                        ├── Phase 8 (Agent)
                       ├── Phase 7 (Events) ──────────────────┘
Phase 0 (Scaffold) ────┘
```

**Parallelizable:** Phases 1, 2, 3, 6 have no interdependencies.
Phase 7 depends only on Phase 1.
Phase 4 depends on 1, 2, 3.
Phase 5 depends on 1, 2, 4.
Phase 8 depends on all.

---

## 4. Acceptance Criteria

Each phase is complete when:

1. **Implementation** matches the spec in the relevant sub-plan.
2. **All test cases** from the sub-plan pass.
3. **No regressions** — previously passing tests still pass.
4. **No external dependencies** beyond what's specified in §1.4.
