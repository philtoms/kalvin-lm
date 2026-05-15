# Proposal: Separating Concerns Between Origin, Specs, and Plans

**Date:** 2026-05-15  
**Status:** ✅ Complete  
**Scope:** Restructure `docs/`, `specs/`, and `plans/` to create a clean three-layer information hierarchy.

---

## The Problem

The current documents suffer from three kinds of confusion:

### 1. specs/ mixes specification with design narrative

- **`overview.md`** (400 lines) is a design document — it explains *why* the system works, the philosophy of agency, the four levels of significance, the rationalisation pipeline, and cogitation. It overlaps heavily with `docs/kalvin-origin.md`. It is not a testable specification.
- **`significance.md`** (201 lines) opens with *"Significance is a conceptual specification"* and lists code locations (`model.py`, `agent.py`). It is half architecture guide, half spec.
- **`model.md`** (696 lines) and **`agent.md`** (596 lines) include full algorithm pseudocode for `expand()` — this is implementation-level detail, not behavioral specification.

### 2. plans/ duplicate and rehost spec content

- Every sub-plan in `plans/impl/` starts with a "Spec" section that summarizes what's already in `specs/`. For example, `plans/impl/model.md` §1 repeats tier roles, construction, and deduplication rules from `specs/model.md`.
- `plans/implement-kscript.md` (978 lines) contains a complete spec inline rather than referencing `specs/kscript.md`.
- The coordinator `plans/implement-kalvin.md` includes architecture diagrams and constants tables that overlap with `specs/overview.md` and individual specs.

### 3. Test matrices are scattered and inconsistent

| Location | Has test matrices? | Format |
|---|---|---|
| `specs/kscript.md` | ✅ Yes (§13) | Summary table by category |
| `specs/kline.md` | ❌ No | — |
| `specs/signature.md` | ❌ No | — |
| `specs/tokenizer.md` | ❌ No | — |
| `specs/stm.md` | ❌ No | — |
| `specs/model.md` | ❌ No | — |
| `specs/agent.md` | ❌ No | — |
| `plans/impl/foundations.md` | ✅ Yes | Per-component tables |
| `plans/impl/model.md` | ✅ Yes | Per-section tables |
| `plans/impl/agent.md` | ✅ Yes | Per-phase tables |
| `plans/implement-kscript.md` | ✅ Yes | Per-task acceptance checkboxes |

The specs that define the behavior don't contain acceptance criteria. The plans that are supposed to be implementation guides contain both spec summaries AND the tests. An agent implementing from a plan doesn't know whether the plan's spec summary is current or the spec itself has been updated.

---

## The Proposal: Three Layers

```
docs/kalvin-origin.md    ←  WHY the system exists, WHAT the concepts mean
        ↓
specs/                   ←  WHAT each component must do (testable, no code)
        ↓
plans/                   ←  HOW to build it (files, phases, test cases)
```

Each layer has a single responsibility, a standard structure, and clear references to the layer above it. An agent can:
1. Read the **origin** to understand what the system is and keep specs aligned with the design vision.
2. Read the **specs** to understand what must be true and keep plans aligned with requirements.
3. Read the **plans** to implement code, mapping each test case to a spec acceptance criterion.

---

## Layer 1: Origin (`docs/kalvin-origin.md`)

**Responsibility:** The authoritative description of the system's purpose, philosophy, and conceptual model.

**Contains:**
- What Kalvin is (a rationalising agent)
- The conceptual model (klines, identities, relationships, significance)
- The four levels (S1–S4) and what they mean conceptually
- How teaching works (KScript → compilation → training loop)
- KScript language semantics (operators, syntax, compilation rules)
- The study model (S2 expansion, cogitation as self-directed study)

**Does NOT contain:**
- Component APIs, data field tables, function signatures
- Algorithm pseudocode
- Code locations (file names, module names)
- Test cases or acceptance criteria
- Build order, phased estimates

**Current state:** Already good. `docs/kalvin-origin.md` is well-written and clearly authoritative.

**Changes needed:** Minimal. Keep as-is. Specs reference it for conceptual grounding.

---

## Layer 2: Specifications (`specs/`)

**Responsibility:** Precise, testable specifications of each component. Define WHAT must be true, not HOW it is achieved.

### Standard structure (every spec follows this template):

```markdown
# [Component] Specification

## Overview
(1–2 paragraphs. Brief. No philosophy — that's in the origin.)

## Dependencies
(Cross-references to other specs this one depends on.)

## Definition
(Data structures, fields, types, invariants.)

## API
(Function signatures, preconditions, postconditions.)

## Behavioral Rules
(Equality, ordering, deduplication, routing — precise rules
 stated as testable assertions.)

## Test Matrix
(Numbered acceptance criteria. Every criterion is testable.
 Reference: "Origin §X" for the conceptual basis.)

## Out of Scope
(Explicit boundaries — what this spec does NOT define.)
```

### Key change: Every spec gets a Test Matrix

The test matrix is the contract between specs and plans. It defines the complete set of behaviors that any implementation must satisfy. Plans reference these by number.

Example for `specs/kline.md`:
```markdown
## Test Matrix

| ID   | Criterion                                          | Origin ref     |
| ---- | -------------------------------------------------- | -------------- |
| KL-1 | Construction with empty nodes produces empty list   | Origin §Klines |
| KL-2 | Construction with single int wraps into list        | Origin §Klines |
| KL-3 | Equality requires same signature AND same nodes     | Origin §Klines |
| KL-4 | Inequality if signatures differ                     | Origin §Klines |
| KL-5 | Inequality if node sequences differ                 | Origin §Klines |
| KL-6 | Equal KLines have equal hashes                      | —              |
| KL-7 | `is_literal` returns true for literal mask nodes    | Origin §Nodes  |
| KL-8 | `is_literal` returns false for packed nodes         | Origin §Nodes  |
| KL-9 | Empty kline is non-literal                          | —              |
```

### What moves out of specs/:

| From | Move to | Reason |
|---|---|---|
| `overview.md` conceptual narrative (§1–§6) | Origin doc or delete | Design philosophy, not spec |
| `overview.md` component summaries (§2–§9) | Individual specs (already there) | Redundant with component specs |
| `significance.md` code locations table | `plans/` or delete | Implementation detail |
| `significance.md` "conceptual" framing | Origin doc | Philosophy, not spec |
| `model.md` expand() algorithm pseudocode | `plans/` | Implementation strategy |
| `agent.md` cogitation pseudocode | `plans/` | Implementation strategy |
| `stm.md` code location table | `plans/` or delete | Implementation detail |

### What moves into specs/:

| From | Move to | Reason |
|---|---|---|
| Plans' "Spec" summary sections | Delete (reference specs/ instead) | Eliminate duplication |
| Plans' test cases | `specs/` test matrices | Acceptance criteria belong with the spec |
| `docs/roadmap.md` component descriptions | Keep in docs/ | Roadmap is a planning doc, not a spec |

### Revised specs/ contents:

| File | Purpose | Test Matrix? |
|---|---|---|
| `specs/kline.md` | KLine data structure | ✅ KL-1..KL-N |
| `specs/signature.md` | Signature creation and properties | ✅ SIG-1..SIG-N |
| `specs/tokenizer.md` | Text ↔ node conversion | ✅ TOK-1..TOK-N |
| `specs/stm.md` | Bounded dual-keyed index | ✅ STM-1..STM-N |
| `specs/model.md` | Three-tier knowledge graph + API contracts | ✅ MOD-1..MOD-N |
| `specs/agent.md` | Rationalisation pipeline + cogitation contracts | ✅ AGT-1..AGT-N |
| `specs/significance.md` | Significance semantics (merged into model.md or kept thin) | ✅ SIGF-1..SIGF-N |
| `specs/kscript.md` | KScript language spec | ✅ KS-1..KS-N |

`overview.md` is **removed** — its conceptual content lives in the origin doc, its component summaries live in individual specs.

---

## Layer 3: Implementation Plans (`plans/`)

**Responsibility:** HOW to implement the specs in code. Phased, actionable, buildable.

### Standard structure (every plan follows this template):

```markdown
# [Plan Name]

**Parent:** [link to parent plan or "none"]
**Phases:** [phase numbers]
**Estimate:** [time]
**Status:** [not started / in progress / complete]
**Spec refs:** [links to relevant specs]

## Spec References
(Links to specs. NOT copies. "See @model spec §Storage Operations" not a
 restatement of the API.)

## Implementation Tasks
(Per-file, per-function implementation guidance with pseudocode.)

### Task N: [Component] ([File])
- **Spec ref:** @model spec §Expand
- **Test mapping:** MOD-15..MOD-22
- **Pseudocode:** [algorithm details that go beyond spec into implementation]

## Test Mapping
(Which test files implement which spec acceptance criteria.)

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| MOD-1   | test_model.py | test_add_literal | ✅ |
| MOD-2   | test_model.py | test_add_nonliteral | ✅ |

## Design Decisions
(Resolved questions with rationale — things not dictated by specs.)

## Status
(What's done, what's not, blockers.)
```

### Key changes to plans/:

1. **Remove "Spec" sections** — replace with links to specs. Plans should say "See @model spec" not re-explain what the model does.

2. **Move algorithm pseudocode here** — the `expand()` algorithm currently in `specs/model.md` belongs in the plan. The spec should say `expand(Q, C) → Iterator[QueryCandidate]` with behavioral rules; the plan should say "here's the step-by-step implementation algorithm."

3. **Test cases reference spec IDs** — each test case in a plan maps to a specific spec acceptance criterion by ID. This creates a traceable chain: origin → spec criterion → plan test → code.

4. **Remove spec summaries from coordinator** — `plans/implement-kalvin.md` should reference specs for component descriptions, not duplicate them.

### Revised plans/ contents:

| File | Purpose |
|---|---|
| `plans/implement-kalvin.md` | Coordinator: build order, dependency graph, cross-cutting concerns |
| `plans/implement-kscript.md` | KScript implementation (references `specs/kscript.md`, doesn't re-spec) |
| `plans/impl/foundations.md` | Phases 0–4: bit layout, KLine, Signature, Tokenizer, STM |
| `plans/impl/model.md` | Phase 5: Model + expand algorithm |
| `plans/impl/agent.md` | Phases 6–8: Agent + Cogitator |
| `plans/impl/build-phases.md` | Design decisions, execution plan |
| `plans/impl/structural-grounding.md` | Challenges 6 + 6b |

---

## The Traceability Chain

```
Origin: "A kline is a node-like structure..."
         ↓ referenced by
Spec:   KL-3 "Two Klines are equal iff same signature AND same nodes"
         ↓ mapped by
Plan:   test_kline.py::test_equality → MOD-3 → ✅ passing
         ↓ implemented in
Code:   kline.py::def __eq__(self, other)
```

An agent at any layer can work in one direction:
- **Updating specs:** Read origin → check each spec aligns with origin concepts → update spec wording and test matrix.
- **Updating plans:** Read specs → check plan references current spec IDs → update implementation tasks and test mapping.
- **Updating code:** Read plan → implement task → write test mapped to spec ID → verify all acceptance criteria pass.

---

## Migration Steps

1. **Add test matrices to all specs.** ✅ Extracted test cases from plans into the corresponding spec's test matrix with numbered IDs (KL-*, SIG-*, TOK-*, STM-*, MOD-*, AGT-*, SGF-*, KS-*).
2. **Remove `specs/overview.md`.** ✅ Verified all content covered by origin doc + individual specs. Deleted.
3. **Merge `specs/significance.md` into `specs/model.md`.** ✅ Added §Significance Semantics to model.md. Thinned significance.md to a redirect page.
4. **Move algorithm pseudocode from specs to plans.** ✅ Replaced `expand()` algorithm in model.md with behavioral contract + pointer to plan. Removed §12 (Build Plan) from kscript.md (already in plans/).
5. **Remove spec duplication from plans.** ✅ Replaced all "### Spec" sections in plans with "### Spec Reference" links. Coordinator plan references origin doc instead of restating system description.
6. **Add spec-ID references to plan test cases.** ✅ Every test table in every plan now maps to a spec ID.
7. **Verify end-to-end traceability.** ✅ All specs have test matrices, all plans reference specs, all test cases have spec IDs.

---

## Migration Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add test matrices to all specs | ✅ Done |
| 2 | Remove `specs/overview.md` | ✅ Done |
| 3 | Split `specs/significance.md` into model.md + origin | ✅ Done |
| 4 | Move algorithm pseudocode from specs to plans | ✅ Done |
| 5 | Remove spec duplication from plans | ✅ Done |
| 6 | Add spec-ID references to plan test cases | ✅ Done |
| 7 | Verify end-to-end traceability | ✅ Done |

---

## Summary: What Goes Where

| Information type | Origin | Spec | Plan |
|---|---|---|---|
| System purpose and philosophy | ✅ | — | — |
| Conceptual model (nodes, klines, significance) | ✅ | — | — |
| Teaching model (KScript, training loop) | ✅ | — | — |
| Data structure definitions (fields, types) | — | ✅ | — |
| API contracts (signatures, pre/post conditions) | — | ✅ | — |
| Behavioral rules (equality, ordering, routing) | — | ✅ | — |
| Test matrix (acceptance criteria) | — | ✅ | — |
| Cross-component dependencies | — | ✅ | — |
| Explicit scope boundaries | — | ✅ | — |
| Algorithm pseudocode (implementation strategy) | — | — | ✅ |
| File structure and code locations | — | — | ✅ |
| Build order and phases | — | — | ✅ |
| Test mapping (spec ID → test function) | — | — | ✅ |
| Design decisions and rationale | — | — | ✅ |
| Implementation skeletons | — | — | ✅ |
| Status tracking | — | — | ✅ |
