# Kalvin Development Roadmap

**Date:** 2026-05-26  
**Status:** Active  
**Scope:** Full build-from-scratch implementation of the Kalvin system.

**Source documents (three-layer model):**

| Layer | Purpose | Location |
|-------|---------|----------|
| **Origin** | WHY — purpose, philosophy, conceptual model | `docs/kalvin-origin.md` |
| **Specs** | WHAT — testable behavioural contracts | `specs/*.md` |
| **Plans** | HOW — implementation strategy, phasing, test mapping | `plans/**/*.md` |

**See also:** `docs/spec-plan-proposal.md` — the rules governing the three-layer model.

---

## Summary

Kalvin is a rationalising system that accepts, thinks, and talks in klines. A kline is a node-like structure — either an identity (no links) or a relationship (links to other klines). Kalvin's entire world is built from these two structures. Meaning for Kalvin is found in shape, not in semantics.

The system is taught via a training loop: an agent compiles KScript into kline structures, submits them to Kalvin via `rationalise()`, evaluates the responses, and decides what to teach next. Kalvin rationalises every kline identically regardless of context — the harness determines whether a session is training or operational.

The implementation consists of nine components built bottom-up from leaf nodes to the agent orchestrator, followed by structural grounding and extended cogitation enhancements, and finally the external training agent. No code has been implemented yet.

---

## Current State

| Capability | Status | Spec | Plan |
|---|---|---|---|
| KLine data model | ❌ Not started | `specs/kline.md` (KL-1..KL-14) | `plans/impl/foundations.md` §2 |
| Signature computation | ❌ Not started | `specs/signature.md` (SIG-1..SIG-10) | `plans/impl/foundations.md` §3 |
| Tokenizer (Mod32/Mod64, BPE) | ❌ Not started | `specs/tokenizer.md` (TOK-1..TOK-12) | `plans/impl/foundations.md` §4 |
| STM (bounded rolling index) | ❌ Not started | `specs/stm.md` (STM-1..STM-10) | `plans/impl/foundations.md` §5 |
| Model (3-tier: STM → Frame → Base) | ❌ Not started | `specs/model.md` (MOD-1..MOD-51) | `plans/impl/model.md` |
| Significance constants | ❌ Not started | `specs/model.md` §Significance Semantics | `plans/impl/agent.md` §1 |
| Events (pub/sub) | ❌ Not started | `specs/agent.md` §Events (AGT-23..AGT-28) | `plans/impl/agent.md` §2 |
| Agent (routing + rationalisation pipeline) | ❌ Not started | `specs/agent.md` (AGT-1..AGT-40) | `plans/impl/agent.md` §3 |
| Persistence (JSON + binary) | ❌ Not started | `specs/agent.md` (AGT-38..AGT-40) | `plans/impl/agent.md` §3 |
| KScript DSL (lexer → parser → compiler) | ❌ Not started | `specs/kscript.md` (KS-1..KS-33) | `plans/implement-kscript.md` |
| Structural grounding | ❌ Not started | `specs/model.md` (MOD-26..MOD-43) | `plans/impl/structural-grounding.md` §1 |
| Extended cogitation (S2 expansion) | ❌ Not started | `specs/model.md` (MOD-44..MOD-51) | `plans/impl/structural-grounding.md` §2 |
| Test harness (repurposed KScript TUI) | ❌ Not started | `specs/harness.md` (HRN-1..HRN-18) | `plans/impl/harness.md` |
| Model quality evaluation | ❌ Not started | — | — |

---

## Build Phases

The system is built bottom-up in strict dependency order. Each phase is complete when all its test cases pass and no regressions are introduced.

### Phase 0: Project Scaffold
**Estimate:** 0.5 day  
**Deliverable:** Empty project with passing test runner.  
**Plan:** `plans/impl/foundations.md` §0

- Directory structure: `src/kalvin/`, `tests/`, `kscript/`
- `pyproject.toml` with minimal dependencies (Python ≥ 3.10, pytest)
- Verify `pytest tests/ -v` runs (empty).

### Phase 1: KLine
**Estimate:** 0.5 day  
**Spec:** `specs/kline.md` — KL-1 through KL-14  
**Plan:** `plans/impl/foundations.md` §2  
**Depends on:** Nothing

The fundamental data unit. An identified, ordered sequence of uint64 nodes. Equality requires same signature AND same node sequence. Includes the standalone `is_literal(node)` function — the single authority for the literal/non-literal distinction.

### Phase 2: Signature
**Estimate:** 0.5 day  
**Spec:** `specs/signature.md` — SIG-1 through SIG-10  
**Plan:** `plans/impl/foundations.md` §3  
**Depends on:** KLine (for `is_literal`)

OR-reduction of nodes into a single uint64 identity key with a literal-content flag (bit 0). Includes `signifies(a, b)` for bitwise AND matching — the basis for candidate retrieval.

### Phase 3: Tokenizer
**Estimate:** 1.5 days  
**Spec:** `specs/tokenizer.md` — TOK-1 through TOK-12  
**Plan:** `plans/impl/foundations.md` §4  
**Depends on:** KLine (for node type)

Text ↔ node conversion. Mod tokenizer with packed encoding (uppercase alpha → single OR-reduced node) and literal encoding (everything else → one node per character with literal mask). BPE tokenizer is optional.

### Phase 4: STM
**Estimate:** 1 day  
**Spec:** `specs/stm.md` — STM-1 through STM-10  
**Plan:** `plans/impl/foundations.md` §5  
**Depends on:** KLine, Signature

Bounded, dual-keyed index over recently added klines. Each kline indexed by both its signature and its nodes signature. FIFO eviction at configurable bound (default 256). Enables transitive grounding.

### Phase 5: Model
**Estimate:** 2–3 days  
**Spec:** `specs/model.md` — MOD-1 through MOD-51  
**Plan:** `plans/impl/model.md`  
**Depends on:** KLine, Signature, STM

Three-tier knowledge graph (STM → Frame → Base). Storage, deduplication, lookup, graph traversal, promotion, and the `expand()` generator for significance computation. This is the largest component. The expand algorithm implements the full distance-significance pipeline: per-node hop chains, S2 signifies short-circuits, S3 connotation bridging, quadratic packing, and significance inversion.

### Phase 6: Significance Constants
**Estimate:** 0.5 day  
**Spec:** `specs/model.md` §Significance Semantics  
**Plan:** `plans/impl/agent.md` §1  
**Depends on:** Model

Constants `D_MAX`, `MASK64`, boundary values, and packing functions. Defined in `model.py` alongside the expand algorithm.

### Phase 7: Events
**Estimate:** 0.5 day  
**Spec:** `specs/agent.md` §Events — AGT-23 through AGT-28  
**Plan:** `plans/impl/agent.md` §2  
**Depends on:** KLine

Pub/sub event bus with `RationaliseEvent` (kind, query, proposal, significance). Thread-safe. Events: `ground`, `frame`, `done`.

### Phase 8: Agent + Cogitator
**Estimate:** 2 days  
**Spec:** `specs/agent.md` — AGT-1 through AGT-40  
**Plan:** `plans/impl/agent.md` §3  
**Depends on:** All above

The orchestrator. The `rationalise()` pipeline has six phases (Prepare → Ground Check → Assess → Retrieve Candidates → Route Each Candidate → background Cogitation). Fast path returns `True` (event already emitted). Slow path returns `False` (Cogitator processes work items on background thread). Routing is a pure node-membership test. Cogitation expands via `model.expand()` and emits proposals as frame events.

Includes persistence: JSON and binary serialization for model save/load.

### Phase 9: KScript DSL
**Estimate:** 2 days  
**Spec:** `specs/kscript.md` — KS-1 through KS-33  
**Plan:** `plans/implement-kscript.md`  
**Depends on:** KLine, Signature, Tokenizer (Phases 1–3)

Domain-specific language for compiling scripts into kline structures. Pipeline: Lexer → Parser → Compiler → `[CompiledEntry]`. Includes decompiler for best-effort reconstruction. Operators: COUNTERSIGN (`==`), UNDERSIGN (`=`), CONNOTATE (`>`), CANONIZE (`=>`). MCS expansion for multi-character signatures. JSON/JSONL/binary output formats. Public API class `KScript` and CLI.

### Phase A: Structural Grounding
**Estimate:** 1–2 days  
**Spec:** `specs/model.md` — MOD-26 through MOD-43, `specs/agent.md` — AGT-29, AGT-36, AGT-37  
**Plan:** `plans/impl/structural-grounding.md` §1  
**Depends on:** Phases 0–8 complete  
**Risk:** Low

S1 is determined by structure, not by boundary classification. A kline is grounded if its signature fully describes its nodes (`make_signature(nodes) == signature`) or it is countersigned. After ratification, all participating STM klines are promoted to frame — not just the ratified kline, but supporting identity and partial klines as well.

### Phase A+: Extended Cogitation
**Estimate:** 3–5 days  
**Spec:** `specs/model.md` — MOD-44 through MOD-51, `specs/agent.md` §S2 Expansion  
**Plan:** `plans/impl/structural-grounding.md` §2  
**Depends on:** Phase A  
**Risk:** Medium

When countersignature fails for an S2 result, the Cogitator attempts to reshape the kline toward canonical status. Misfit classification: underfitting (signature promises more than nodes deliver), overfitting (nodes carry more than signature captures), dual (both). Expansion proposals are emitted for agent ratification. No invention — every signature used must already exist in the model.

### Phase B: Test Harness
**Estimate:** 3–5 days  
**Spec:** `specs/harness.md` — HRN-1 through HRN-18  
**Plan:** `plans/impl/harness.md`  
**Depends on:** Phase A+  
**Risk:** Low

Repurpose the existing KScript TUI (`ui/kscript/`) into a training loop supervisor. The harness compiles KScript entries, submits only new entries to the Agent, tracks submission/satisfaction state monotonically, displays proposals with significance (raw hex + normalised 0.0–1.0), and provides ratification controls (auto in Run mode, manual in Step mode). Adds `Agent.countersign(kline)` for reciprocal kline generation. State persists through hot-reload cycles.

### Phase D: Multi-Agent Harness Server
**Estimate:** 12–17 days  
**Spec:** `specs/harness-server.md` — HRNS-1 through HRNS-24  
**Plan:** `plans/implement-harness-server.md`  
**Depends on:** Phase B  
**Risk:** Medium (GLM-5.1 integration quality unknown)

Refactor the harness from a monolithic TUI into a persistent multi-agent server. The harness becomes a message broker: participants (Kalvin, Trainer, Slack agent, TUI) communicate through addressed messages routed by the harness. Key changes: rename Agent → KAgent, remove internal EventBus in favour of direct adapter callbacks, add thread-safe message bus with single-dispatch event loop, add WebSocket protocol for client participants, build Trainer participant (curriculum execution, reactive scaffolding via GLM-5.1, ratification, escalation), build Slack participant (human↔Trainer communication), and thin the TUI into a rendering-only client.

### Phase C: Model Quality and Evaluation
**Estimate:** 2–3 weeks (ongoing)  
**Risk:** Medium  
**Depends on:** Phase B

Quality metrics (coverage, precision, learning rate), boundary calibration, performance profiling, tokenizer capacity analysis. Uses the test harness from Phase B to run Mary's World curriculum and measure Kalvin's learning behaviour. This is empirical work — the system may work well with current boundaries, or it may need significant tuning.

---

## Dependency Graph

```
Phase 0: Scaffold
    │
    ├── Phase 1: KLine ──────────┐
    ├── Phase 2: Signature ──────┤
    └── Phase 3: Tokenizer ──────┼── Phase 4: STM ── Phase 5: Model ──┐
                                 │                                     │
                                 ├── Phase 7: Events ─────────────────┤
                                 │                                     │
                                 └── Phase 9: KScript ────────────────┘
                                                                       │
                                                  Phase 6: Constants ──┤
                                                                       │
                                                  Phase 8: Agent + Cogitator
                                                                       │
                                                  Phase A: Structural Grounding
                                                                       │
                                                  Phase A+: Extended Cogitation
                                                                       │
                                                  Phase B: Test Harness
                                                                       │
                                                  Phase D: Multi-Agent Harness Server
                                                                       │
                                                  Phase C: Model Quality (ongoing)
```

**Parallelizable:** Phases 1, 2, 3 have no interdependencies. Phase 7 depends only on Phase 1. Phase 9 depends on Phases 1–3 and can proceed in parallel with Phases 4–8.

---

## Component Map

| Component | Spec | Plan | File | Test IDs | Estimate |
|-----------|------|------|------|----------|----------|
| KLine | `specs/kline.md` | `plans/impl/foundations.md` §2 | `src/kalvin/kline.py` | KL-1..KL-14 | 0.5d |
| Signature | `specs/signature.md` | `plans/impl/foundations.md` §3 | `src/kalvin/signature.py` | SIG-1..SIG-10 | 0.5d |
| Tokenizer | `specs/tokenizer.md` | `plans/impl/foundations.md` §4 | `src/kalvin/mod_tokenizer.py`, `src/kalvin/tokenizer.py` | TOK-1..TOK-12 | 1.5d |
| STM | `specs/stm.md` | `plans/impl/foundations.md` §5 | `src/kalvin/stm.py` | STM-1..STM-10 | 1d |
| Model | `specs/model.md` | `plans/impl/model.md` | `src/kalvin/model.py` | MOD-1..MOD-51 | 2–3d |
| Events | `specs/agent.md` §Events | `plans/impl/agent.md` §2 | `src/kalvin/events.py` | AGT-23..AGT-28 | 0.5d |
| Agent | `specs/agent.md` | `plans/impl/agent.md` §3 | `src/kalvin/agent.py` | AGT-1..AGT-40 | 2d |
| KScript | `specs/kscript.md` | `plans/implement-kscript.md` | `kscript/` (9 files) | KS-1..KS-33 | 2d |
| Test Harness | `specs/harness.md` | `plans/impl/harness.md` | `ui/kscript/` (7 files), `src/kalvin/agent.py` | HRN-1..HRN-18 | 3–5d |
| Harness Server | `specs/harness-server.md` | `plans/implement-harness-server.md` | `src/harness/`, `src/trainer/`, `src/participants/` | HRNS-1..HRNS-24 | 12–17d |
| **Subtotal** | | | | **~262 test criteria** | **~26–33d** |

---

## Spec ID Coverage

Total spec IDs across all components:

| Spec | ID Range | Count |
|------|----------|-------|
| `specs/kline.md` | KL-1..KL-14 | 14 |
| `specs/signature.md` | SIG-1..SIG-10 | 10 |
| `specs/tokenizer.md` | TOK-1..TOK-12 | 12 |
| `specs/stm.md` | STM-1..STM-10 | 10 |
| `specs/model.md` | MOD-1..MOD-51 | 51 |
| `specs/agent.md` | AGT-1..AGT-40 | 40 |
| `specs/kscript.md` | KS-1..KS-33 | 33 |
| `specs/harness.md` | HRN-1..HRN-18 | 18 |
| `specs/harness-server.md` | HRNS-1..HRNS-24 | 24 |
| **Total** | | **212** |

---

## Risk Assessment

| Risk | Severity | Phase | Mitigation |
|------|----------|-------|------------|
| `expand()` distance algorithm may need iteration | High | 5 | S2 distance uses per-node hop-distance; evolve based on real data from Phase C |
| All-literal signature collision (sig=1 for all) | Medium | 5 | Fast-path in Assess phase; accept degenerate indexing |
| Candidate retrieval O(N) scan too slow for large models | Medium | 5, C | Profile first; add inverted bit index if needed |
| GLM-5.1 scaffolding quality unpredictable | High | D | Start with simple reactive scaffolding; iterate on prompt engineering |
| Trainer reactive loop may not converge | Medium | D | Budget-based escalation; human always in the loop |
| WebSocket client reconnection reliability | Low | D | Silent drop on disconnect; reconnect re-registers |
| Cogitation thread safety bugs | Medium | 8 | Thorough concurrent testing; keep thread logic simple |
| Expansion proposal quality (too many / too few) | Medium | A+ | Top-k limit or depth controls in future iteration |
| BPE tokenizer optional deps fragile | Low | 3 | BPE entirely optional; core system works with Mod only |
| Literal mask accidental collision | Low | 1 | Only `0xFFFFFFFF` lower 32 bits triggers; verified for all node types |

---

## Open Questions

| # | Question | Affects | Priority |
|---|----------|---------|----------|
| 1 | Are current boundary constants well-calibrated for training? | Phase C | High |
| 2 | How many klines before candidate retrieval O(N) becomes a bottleneck? | Phase C | Medium |
| 3 | What does a non-trivial training curriculum (Mary's World) look like in practice? | Phase B | Medium |
| 4 | When does Mod(N) vocabulary saturation require BPE transition? | Phase C | Low |
| 5 | Should expansion proposals have depth/k limits? | Phase A+ | Low |
| 6 | BPE tokenizer priority: needed for MVP or Mod-only sufficient? | Phase 3 | Low |
| 7 | Persistence: JSON sufficient from day one, or binary required? | Phase 8 | Low |
| 8 | How well does GLM-5.1 generate scaffolding given misfit context? | Phase D | High |
| 9 | Should the Trainer's reactive budget be configurable per curriculum? | Phase D | Medium |

---

## Future Work

These are not required for the training loop to work.

- **KScript metadata** — annotations for script-level properties
- **Temporal events** — trajectory analysis for convergence rate, scaffolding effectiveness
- **Kalvin-as-tutor** — a Kalvin instance that acts as agent for another, using reentrant rationalisation for structural equivalence instead of exact kline matching
- **Inverted bit-index** — for sub-linear candidate retrieval in large models
- **BPE tokenizer** — for non-toy domains requiring larger vocabularies
- **Dynamic participant registration** — join/leave at runtime, not just at startup
- **Multiple concurrent training sessions** — interleaved curricula on a shared Kalvin instance
- **Trainer-as-curriculum-designer** — GLM-5.1 generates full curricula from natural language goals
