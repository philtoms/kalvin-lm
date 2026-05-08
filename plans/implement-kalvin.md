# Kalvin: Build-From-Scratch Implementation Plan - Coordinator

**Purpose:** Master index for implementing Kalvin from zero. Each sub-plan is
self-contained (spec + algorithm + test cases). This file provides the overview,
build order, and cross-cutting concerns.

**Date:** 2026-04-29
**Updated:** 2026-05-01 - significance internalized in model (distance→significance inversion moved from Cogitator to Model.expand)

---

## 0. What is Kalvin?

Kalvin is an **agent** - a system that receives new information, evaluates how it
relates to existing knowledge, and autonomously decides what to do next. This
capacity for choice is called **Agency**, and it arises from a mechanism called
**Significance**.

The system operates on two fundamental concepts:

- **Nodes** - opaque 64-bit unsigned integers (the atoms).
- **KLines** - identified, ordered sequences of nodes (the structures).

### The Core Pipeline

```
Input Text → Tokenizer → Nodes → KLine → Agent → Model (Knowledge Graph)
                                         ↓
                              Route (node membership, no model call)
                              S1 → fast response (promote)
                              S4 → fast response (novel)
                              S2/S3 → Cogitator (background)
                                         ↓
                              Cogitator: model.expand → connotations + distance
                                         ↓
                              Each QueryCandidate: check countersignature
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
│          │          │  (Mod / BPE) │   (EventBus)                 │
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
Phase 3: Tokenizer (Mod + BPE)  - text ↔ node conversion
Phase 4: STM                    - bounded dual-keyed index (@stm spec)
Phase 5: Model                  - three-tier knowledge graph
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
│   │   ├── mod_tokenizer.py     # Mod tokenizer (Mod32, Mod64)
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

| Sub-plan | Scope | Source Phases | Depends On |
|----------|-------|---------------|------------|
| [`plans/impl/foundations.md`](impl/foundations.md) | Bit layout, KLine, Signature, Tokenizer, STM | 0-4 | Nothing |

> **STM spec:** The full STM specification is in `specs/stm.md`. The
> foundations plan provides the implementation skeleton and test cases.
| [`plans/impl/model.md`](impl/model.md) | Model + distance algorithm | 5 | Foundations |
| [`plans/impl/agent.md`](impl/agent.md) | Significance constants, Events, Agent, Cogitator | 6-8 | Foundations, Model |
| [`plans/impl/structural-grounding.md`](impl/structural-grounding.md) | Structural grounding + extended cogitation | A, A+ | Model, Agent |
| [`plans/impl/build-phases.md`](impl/build-phases.md) | Resolved design decisions, phased build, test cases | 0-9 | All (execution plan) |

### How to Use This Plan

1. **Read** this coordinator for the big picture.
2. **Implement** `plans/impl/foundations.md` (Phases 0-4). ✅ Complete
3. **Implement** `plans/impl/model.md` (Phase 5). ✅ Complete
4. **Implement** `plans/impl/agent.md` (Phases 6-8). ✅ Complete
5. **Implement** `plans/impl/structural-grounding.md` (Phase A + A+). ✅ Complete
6. **Reference** `plans/impl/build-phases.md` for per-phase test cases and
   the resolved design decisions that apply across components.

---

## 3. `is_literal` — Standalone Function

`is_literal(node)` is a standalone function defined by the node encoding
layer (bit layout). It is **not** a tokenizer method. All components import
it directly:

```python
def is_literal(node: int) -> bool:
    return (node & 0xFFFF_FFFF) == 0xFFFF_FFFF
```

This eliminates the need for `is_literal_fn` injection:

```
# Before (injection pattern):
Kline.is_literal(is_literal_fn)     # Passed at call time
make_signature(nodes, is_literal_fn) # Passed at call time
Model(is_literal_fn=fn)              # Stored at construction
STM(is_literal_fn=fn)                # Stored at construction
Agent(tokenizer=tok)                 # Uses tok.is_literal

# After (direct import):
from kalvin.kline import is_literal  # or wherever defined
Kline.is_literal()                   # Uses is_literal internally
make_signature(nodes)                # Uses is_literal internally
Model()                              # No injection needed
STM()                                # No injection needed
Agent(tokenizer=tok)                 # No is_literal concern
```

The Agent is still the composition root for the tokenizer, but `is_literal`
no longer flows through it to downstream components.

---

## 4. Constants Reference

```python
# Bit layout
LITERAL_MASK = 0xFFFF_FFFF           # Lower 32 bits all set
MASK64 = 0xFFFF_FFFF_FFFF_FFFF       # Full 64-bit mask

# Well-known signatures
UNSIGNED = 0                         # No nodes
LITERAL_ONLY = 1                     # All-literal content

# Significance
_S3_BIAS = 9                         # Tier bias for S3 connotation hops
_S2_S3_DISTANCE = 100                 # S2|S3 boundary packed distance threshold
_TEMP_SCALE = 100                     # Significance units shifted per unit τ
D_MAX = 0xFFFF_FFFF_FFFF_FFFF        # Maximum distance
S1_VALUE = 0xFFFF_FFFF_FFFF_FFFF     # Maximum significance
S4_VALUE = 0x0000_0000_0000_0000     # Minimum significance

# Model
STM_BOUND_DEFAULT = 256              # STM capacity
MAX_HOP = 100                        # S2 edge hop chain depth / unresolvable penalty
COGITATE_TIMEOUT = 2.0               # Seconds before "done" event

# Tokenizer
MOD32_BITS = 31                      # Character bit positions
MOD64_BITS = 63
```

---

## 5. Risk Assessment

| Risk                                                    | Severity | Mitigation                                                                             |
| ------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------- |
| Model significance API semantics may need iteration     | High     | S2 distance uses per-node hop-distance algorithm; evolve based on real data            |
| All-literal signature collision (sig=1 for all)         | Medium   | Fast-path in Assess phase; accept degenerate indexing                                  |
| Candidate retrieval O(N) scan too slow for large models | Medium   | Profile first; add inverted bit index if needed                                        |
| Cogitation thread safety bugs                           | Medium   | Thorough concurrent testing; keep thread logic simple                                  |
| Cogitator MVP too simple (no graph expansion)           | Medium   | Now uses model.expand() for connotation discovery; see specs   |
| Literal mask accidental collision                       | Low      | Only `0xFFFFFFFF` lower 32 bits triggers; verified for all node types                  |
| BPE tokenizer optional deps fragile                     | Low      | Make BPE entirely optional; core system works with Mod only                            |

---

## 6. Summary of What Gets Built When

| Phase | Component    | Files                              | Est.       | Depends On | Sub-plan | Status |
| ----- | ------------ | ---------------------------------- | ---------- | ---------- | -------- | ------ |
| 0     | Scaffold     | `pyproject.toml`, dirs             | 0.5d       | -          | foundations | ✅ |
| 1     | KLine        | `kline.py`                         | 0.5d       | -          | foundations | ✅ |
| 2     | Signature    | `signature.py`                     | 0.5d       | -          | foundations | ✅ |
| 3     | Tokenizer    | `mod_tokenizer.py`, `tokenizer.py` | 1.5d       | -          | foundations | ✅ |
| 4     | STM          | `stm.py`                           | 1d         | 1, 2, 3    | foundations | ✅ |
| 5     | Model        | `model.py`                         | 2-3d       | 1, 2, 4    | model | ✅ |
| 6     | Constants    | `model.py` (D_MAX, MASK64)          | 0.5d       | —          | model | ✅ |
| 7     | Events       | `events.py`                        | 0.5d       | 1          | agent | ✅ |
| 8     | Agent        | `agent.py` (routing + Cogitator)   | 2d         | 1-7        | agent | ✅ |
| 9     | Persistence  | `agent.py` (extend)                | 1d         | 8          | agent | ✅ |
| —     | KScript      | `kscript/`                         | 5d         | 1-3        | kscript | ✅ |
| A     | Struct. Grounding | `model.py`, `agent.py`        | 1-2d       | 1-8        | structural-grounding | ✅ |
| A+    | Ext. Cogitation    | `model.py`, `agent.py`        | 3-5d       | A          | structural-grounding | ✅ |
|       | **Total**    |                                    | **22-27d** |            |          |

---

## 7. Questions for Clarification

1. **BPE tokenizer priority:** Is BPE support needed for MVP, or can we ship with Mod tokenizer only?
   - _Recommendation:_ Mod only for MVP. BPE is a separate concern.

2. **Persistence format:** Is JSON sufficient, or is binary serialization required from day one?
   - _Recommendation:_ JSON for development; binary for production.

3. **is_s1 semantics:** Resolved: `is_s1(kline)` returns whether a kline is structurally grounded — canonical (`make_signature(nodes) == signature`) or countersigned.

4. **Thread model for cogitation:** Background thread or async/await?
   - _Recommendation:_ Background thread for simplicity.

5. **`dbg_text` field:** Include in KLine or keep separate?
   - _Recommendation:_ Optional implementation-level field. Not part of equality/hash.

6. **`abstract.py`:** Formal ABC classes or duck-typed protocols?
   - _Recommendation:_ `ABC` for Tokenizer (multiple implementations). Duck typing for Model/Agent.

7. **Cogitator evolution:** Should future iterations add pass tracking or significance-based re-routing?
   - _Recommendation:_ Incremental evolution. Cogitation now expands via `model.expand()` yielding connotations. Next: top-k/top-p selection of connotation results.
