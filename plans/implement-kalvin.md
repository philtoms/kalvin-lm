# Kalvin: Build-From-Scratch Implementation Plan - Coordinator

**Purpose:** Master index for implementing Kalvin from zero. Each sub-plan is
self-contained (implementation + algorithm + test mapping). This file provides
the overview, build order, and cross-cutting concerns.

**Specs:** All component specs are in `specs/`. Sub-plans reference specs by ID
(e.g., KL-1, MOD-5) rather than duplicating spec content.

**Date:** 2026-04-29

---

## 0. System Architecture

See `docs/kalvin-vision.md` for the vision and conceptual model of what Kalvin
is. Component specs in `specs/` define the behavioral contracts.

---

## 1. System Architecture

### 1.1 Component Dependency Graph

```
┌──────────────────────────────────────────────────────────────────┐
│                    Agent (orchestrator)                            │
│  Depends on: Tokenizer, Model, Signature, Events                  │
│  Contains: Cogitator (background thread)                          │
│  Routing is inline (_route: node-membership, no model call)       │
├──────────────────────────────────────────────────────────────────┤
│  Cogitator (work-item processor)                                  │
│  Receives WorkItem(Q, C)                                          │
│  Expands: model.expand → QueryCandidate stream (significance)     │
│  Processes: countersignature check per QueryCandidate              │
├──────────────────────────────────────────────────────────────────┤
│  Model (STM → Frame → Base)                                      │
│  Depends on: Kline, Signature, STM (@stm spec)                   │
│  Computes significance internally via expand()                    │
├──────────┬──────────┬──────────────┬──────────────────────────────┤
│ Kline    │ Signature│   Tokenizer  │   Events                     │
│          │          │    (NLP)     │   (EventBus)                 │
├──────────┴──────────┴──────────────┴──────────────────────────────┤
│  Significance constants (D_MAX, MASK64) - in model.py           │
├──────────────────────────────────────────────────────────────────┤
│  Nodes (uint64) - the universal atom                              │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Build Order (Bottom-Up)

Components must be built and tested from the leaves up:

```
Phase 0: Project scaffold       - directories, dependencies, test runner
Phase 1: Kline                  - fundamental data unit
Phase 2: Signature              - OR-reduction identity computation
Phase 3: Tokenizer (NLP)      - text ↔ node conversion
Phase 4: STM                    - bounded dual-keyed index (@stm spec)
Phase 5: Model                  - three-tier memory
Phase 6: Significance Constants - D_MAX, MASK64 (in model.py)
Phase 7: Events                 - pub/sub for rationalisation
Phase 8: Agent                  - fast/slow split + Cogitator + WorkItem
Phase 9: Persistence            - serialisation, save/load
```

Phases 1, 2, 3, and 6 can proceed in parallel (all are leaf components).
Phase 9 is additive (extends Phase 8).

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
│   │   ├── stm.py               # Short-Term Memory
│   │   ├── model.py             # Three-tier Model (includes D_MAX, MASK64)
│   │   ├── significance.py      # (removed - constants in model.py)
│   │   ├── events.py            # EventBus + RationaliseEvent
│   │   └── agent.py             # Agent orchestrator (imports D_MAX from model)
├── tests/
│   ├── test_kline.py
│   ├── test_signature.py
│   ├── test_tokenizer.py
│   ├── test_stm.py
│   ├── test_model.py
│   ├── test_significance.py    # (removed - constants tested in test_model.py)
│   ├── test_events.py
│   └── test_agent.py           # Routing, short-circuit, Cogitator, work items
├── pyproject.toml
└── README.md
```

---

## 2. Sub-Plan Index

Each sub-plan is self-contained: spec, algorithm, implementation skeleton,
and test cases. An agent can pick up any sub-plan and implement it given
its dependencies are satisfied.

| Sub-plan                                           | Scope                                        | Source Phases | Depends On |
| -------------------------------------------------- | -------------------------------------------- | ------------- | ---------- |
| [`plans/impl/foundations.md`](impl/foundations.md) | Bit layout, KLine, Signature, Tokenizer, STM | 0-4           | Nothing    |

> **STM spec:** The full STM specification is in `specs/stm.md`. The
> foundations plan provides the implementation skeleton and test cases.
> | [`plans/impl/model.md`](impl/model.md) | Model + distance algorithm | 5 | Foundations |
> | [`plans/impl/agent.md`](impl/agent.md) | Significance constants, Events, Agent | 6-8 | Foundations, Model |
> | [`plans/impl/cogitator.md`](impl/cogitator.md) | Cogitator, CogitationHandler, WorkItem (slow-path dispatcher) | 8 | Model, Agent |
> | [`plans/impl/structural-grounding.md`](impl/structural-grounding.md) | Structural grounding + extended cogitation | A, A+ | Model, Agent |
> | [`plans/impl/build-phases.md`](impl/build-phases.md) | Resolved design decisions, phased build, test cases | 0-9 | All (execution plan) |
> | [`plans/nlp-pipeline.md`](nlp-pipeline.md) | NLP-BPE data preparation + node_to_sig integration | NLP | Foundations, Model, Agent |

### How to Use This Plan

1. **Read** this coordinator for the big picture.
2. **Implement** `plans/impl/foundations.md` (Phases 0-4). ✅ Complete
3. **Implement** `plans/impl/model.md` (Phase 5). ✅ Complete
4. **Implement** `plans/impl/agent.md` (Phases 6-8). ✅ Complete
5. **Implement** `plans/impl/structural-grounding.md` (Phase A + A+). ✅ Complete
6. **Reference** `plans/impl/build-phases.md` for per-phase test cases and
   the resolved design decisions that apply across components.

---

## 3. Node Encoding & Signatures

Every node is a `uint64`. Signatures are plain bitwise OR-reductions of node
values via `make_signature` (see **@signature spec**) — there is **no**
literal mechanism, no literal predicate, and no bit-0 literal-content flag.
NLP nodes (`(nlp_type32 << 32) | bpe_token_id`) participate in signature
construction directly, with no masking or special-casing. The Agent remains
the composition root for the tokenizer; nothing literal-related flows to
downstream components. (The literal-predicate / literal-mask design was never
implemented and is out of scope.)

---

## 4. Constants Reference

```python
# Bit layout
MASK64 = 0xFFFF_FFFF_FFFF_FFFF       # Full 64-bit mask

# Well-known signatures
IDENTITY = 0                      # No nodes

# Significance
_S3_BIAS = 1                         # Tier bias for S3 connotation hops (linear)
_S2_S3_DISTANCE = 100                 # S2|S3 boundary distance threshold
D_MAX = 0xFFFF_FFFF_FFFF_FFFF        # Maximum distance
S1_VALUE = 0xFFFF_FFFF_FFFF_FFFF     # Maximum significance
S4_VALUE = 0x0000_0000_0000_0000     # Minimum significance

# Model
STM_BOUND_DEFAULT = 256              # STM capacity
MAX_HOP = 100                        # S2 edge hop chain depth (edge_hops() traversal bound);
                                     # also the per-node penalty for a non-resolving mismatched node
COGITATE_TIMEOUT = 2.0               # Seconds before "done" event
```

---

## 5. Risk Assessment

| Risk                                                    | Severity | Mitigation                                                                                             |
| ------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| Model significance API semantics may need iteration     | High     | S2 distance uses per-node hop-distance algorithm; evolve based on real data                            |
| Candidate retrieval O(N) scan too slow for large models | Medium   | Profile first; add inverted bit index if needed                                                        |
| Cogitation thread safety bugs                           | Medium   | Thorough concurrent testing; keep thread logic simple                                                  |
| Cogitator MVP too simple (no graph expansion)           | Medium   | Now uses model.expand() for connotation discovery; see specs                                           |
| BPE tokenizer optional deps fragile                     | Low      | The NLP tokenizer is the production default; `rustbpe`/`tiktoken` remain optional subword dependencies |

---

## 6. Summary of What Gets Built When

| Phase | Component         | Files                                            | Est.       | Depends On | Sub-plan             | Status |
| ----- | ----------------- | ------------------------------------------------ | ---------- | ---------- | -------------------- | ------ |
| 0     | Scaffold          | `pyproject.toml`, dirs                           | 0.5d       | -          | foundations          | ✅     |
| 1     | KLine             | `kline.py`                                       | 0.5d       | -          | foundations          | ✅     |
| 2     | Signature         | `signature.py`                                   | 0.5d       | -          | foundations          | ✅     |
| 3     | Tokenizer         | `tokenizer.py`, `nlp_tokenizer.py`               | 1.5d       | -          | foundations          | ✅     |
| 4     | STM               | `stm.py`                                         | 1d         | 1, 2, 3    | foundations          | ✅     |
| 5     | Model             | `model.py`                                       | 2-3d       | 1, 2, 4    | model                | ✅     |
| 6     | Constants         | `model.py` (D_MAX, MASK64)                       | 0.5d       | —          | model                | ✅     |
| 7     | Events            | `events.py`                                      | 0.5d       | 1          | agent                | ✅     |
| 8     | Agent             | `agent.py` (routing), `cogitator.py` (slow path) | 2d         | 1-7        | agent, cogitator     | ✅     |
| 9     | Persistence       | `agent.py` (extend)                              | 1d         | 8          | agent                | ✅     |
| —     | KScript           | `kscript/`                                       | 5d         | 1-3        | kscript              | ✅     |
| A     | Struct. Grounding | `model.py`, `agent.py`                           | 1-2d       | 1-8        | structural-grounding | ✅     |
| A+    | Ext. Cogitation   | `model.py`, `agent.py`                           | 3-5d       | A          | structural-grounding | ✅     |
|       | **Total**         |                                                  | **22-27d** |            |                      |

---

## 7. Questions for Clarification

1. **BPE tokenizer priority:** _Resolved._ The NLP tokenizer is the production tokenizer; BPE is its subword base (`rustbpe`/`tiktoken` remain optional subword dependencies).

2. **Persistence format:** Is JSON sufficient, or is binary serialization required from day one?
   - _Recommendation:_ JSON for development; binary for production.

3. **is_s1 semantics:** Resolved: `is_s1(kline)` returns whether a kline is structurally grounded — canonical (`make_signature(nodes) == signature`) or countersigned.

4. **Thread model for cogitation:** Background thread or async/await?
   - _Recommendation:_ Background thread for simplicity.

5. **`dbg_text` field:** Include in KLine or keep separate?
   - _Recommendation:_ Optional implementation-level field. Not part of equality/hash.

6. **`abstract.py`:** Formal ABC classes or duck-typed protocols?
   - _Recommendation:_ `ABC` for Tokenizer (NLP tokenizer + BPE subword base). Duck typing for Model/Agent.

7. **Cogitator evolution:** Should future iterations add pass tracking or significance-based re-routing?
   - _Recommendation:_ Incremental evolution. Cogitation now expands via `model.expand()` yielding connotations. Next: top-k/top-p selection of connotation results.
