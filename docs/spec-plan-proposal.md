# Keeping Documentation Aligned, Consistent, and Up to Date

**Date:** 2026-05-15  
**Status:** Active  
**Audience:** Agents and humans editing `docs/`, `specs/`, or `plans/`

---

## The Three-Layer Model

The project documentation is organised into three layers, each with a single responsibility:

```
docs/kalvin-origin.md    ←  WHY — purpose, philosophy, conceptual model
        ↓
specs/                   ←  WHAT — testable behavioural contracts
        ↓
plans/                   ←  HOW — implementation strategy, phasing, test mapping
```

Information belongs in exactly one layer. When it bleeds across layers, the
documentation drifts out of sync and agents can no longer trust what they read.

### Quick reference: what goes where

| Information type | Origin | Spec | Plan |
|---|:---:|:---:|:---:|
| System purpose and philosophy | ✅ | — | — |
| Conceptual model (nodes, klines, significance) | ✅ | — | — |
| Teaching model (KScript, training loop) | ✅ | — | — |
| Data structure definitions (fields, types, invariants) | — | ✅ | — |
| API contracts (signatures, pre/postconditions) | — | ✅ | — |
| Behavioural rules (equality, ordering, routing) | — | ✅ | — |
| Acceptance criteria (test matrix) | — | ✅ | — |
| Cross-component dependency declarations | — | ✅ | — |
| Explicit scope boundaries ("out of scope") | — | ✅ | — |
| Algorithm pseudocode / implementation strategy | — | — | ✅ |
| File structure and code locations | — | — | ✅ |
| Build order, phases, estimates | — | — | ✅ |
| Test mapping (spec ID → test function) | — | — | ✅ |
| Design decisions and rationale | — | — | ✅ |
| Status tracking | — | — | ✅ |

---

## Principles

### 1. Single source of truth

Every fact lives in exactly one document. All other documents **reference**
it — they do not paraphrase, summarise, or duplicate it.

**Bad:** A plan opens with a "Spec" section restating what
`specs/model.md` already says.  
**Good:** The plan says `See @specs/model §Expand` and links to the spec.

When a spec changes, no plan text needs updating because the plan never
copied the spec in the first place.

### 2. Downward references only

- **Origin** is self-contained — it never references specs or plans.
- **Specs** may reference the origin for conceptual grounding (`Origin §Klines`).
- **Plans** reference specs and the origin but never redefine their content.

Never create an upward dependency (e.g. a spec that says "as implemented in
`src/model.py`").

### 3. Testable specifications

Every spec must contain a **Test Matrix** — a numbered table of acceptance
criteria that fully describes the component's required behaviour.

```markdown
## Test Matrix

| ID   | Criterion                                    | Origin ref     |
| ---- | -------------------------------------------- | -------------- |
| KL-1 | Construction with empty nodes → empty list   | Origin §Klines |
| KL-2 | Construction with single int wraps into list | Origin §Klines |
| KL-3 | Equality requires same signature AND nodes   | Origin §Klines |
```

Plans map their test cases to these IDs. This creates the traceability chain:

```
Origin concept  →  Spec criterion (KL-3)  →  Plan test mapping  →  Code + test
```

### 4. Spec IDs are stable identifiers

Once assigned, a spec ID (e.g. `KL-3`, `MOD-12`) must not be renumbered or
reused. If a criterion is removed, mark it `[removed]` rather than shifting
IDs. New criteria are appended. This prevents plan test-mapping tables from
silently going out of date.

---

## Standard Templates

### Spec template

Every file in `specs/` follows this structure:

```markdown
# [Component] Specification

## Overview
(1–2 paragraphs. No philosophy — just what this component is.)

## Dependencies
(Cross-references to other specs.)

## Definition
(Data structures, fields, types, invariants.)

## API
(Function signatures, preconditions, postconditions.)

## Behavioural Rules
(Precise, testable assertions.)

## Test Matrix
(Numbered acceptance criteria with origin references.)

## Out of Scope
(Explicit boundaries.)
```

### Plan template

Every file in `plans/` follows this structure:

```markdown
# [Plan Name]

**Parent:** [link or "none"]  
**Status:** not started / in progress / complete  
**Spec refs:** [links]

## Spec References
(Links. NOT copies.)

## Implementation Tasks
### Task N: [Component] ([File])
- **Spec ref:** @model spec §Expand
- **Test mapping:** MOD-15..MOD-22
- **Pseudocode:** [implementation details beyond the spec]

## Test Mapping
| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| MOD-1   | test_model.py | test_add_literal | ✅ |

## Design Decisions
(Resolved questions with rationale.)

## Status
(Progress, blockers.)
```

---

## Consistency Checks

Run these checks whenever you edit documentation:

### After editing the origin (`docs/kalvin-origin.md`)

1. **Scan all specs** for origin references (`Origin §…`). If you renamed,
   removed, or renumbered a section, update every spec that references it.
2. **Check conceptual coverage.** Does every concept in the origin appear in
   at least one spec's test matrix? If not, the concept is unspecified and
   either the spec needs a new criterion or the origin content is aspirational
   and should be marked as such.
3. **No code.** If you added file names, function names, or algorithm details,
   move them to the relevant plan.

### After editing a spec (`specs/*.md`)

1. **Verify test matrix completeness.** Every behavioural rule in the spec
   must have at least one test-matrix entry. If you added a new rule, add a
   new criterion (append the next ID).
2. **Check plan test mappings.** Every spec ID that appears in a plan's test
   mapping must still exist in the spec. If you removed a criterion, update
   the plan's mapping table (mark the test as `[removed]` or delete the row).
3. **Check for plan-spec duplication.** Re-read the relevant plan sections.
   If the plan contains a paragraph that says the same thing as the spec, replace
   it with a reference link.
4. **No pseudocode.** If you wrote algorithm steps or implementation strategy,
   move that content to the plan and replace it with a behavioural contract.

### After editing a plan (`plans/**/*.md`)

1. **Verify spec references resolve.** Every `@specs/…` or spec ID mention
   must point to an existing section or criterion. Broken references mean the
   plan is describing behaviour that no longer matches the spec.
2. **Verify test mapping accuracy.** Each row in the test mapping table should
   name a real spec ID and a real test function. If the test was renamed or
   moved, update the row.
3. **No spec content.** If the plan contains a paragraph that restates what
   the spec says (data structures, behavioural rules, API contracts), delete it
   and replace with a reference.
4. **No origin content.** If the plan contains philosophical or conceptual
   explanation, move it to the origin doc or delete it.

### When adding a new component

1. Write the origin concept first (or verify it's already there).
2. Create the spec from the template. Fill in the test matrix with IDs using
   an appropriate prefix.
3. Create or update the plan. Add implementation tasks and a test mapping
   table referencing the new spec IDs.
4. Verify the chain: origin section → spec criterion → plan task → test function.

---

## Common Drift Patterns and How to Fix Them

| Symptom | Cause | Fix |
|---|---|---|
| Plan and spec disagree on a data structure | Spec was updated, plan was not | Delete the plan's description; replace with `See @specs/…` |
| Spec contains algorithm pseudocode | Implementation detail leaked upward | Move to the plan; replace in spec with a behavioural contract |
| Origin and spec both explain the same concept | Spec duplicates origin narrative | Thin the spec to 1–2 sentence overview + origin reference |
| Test matrix has gaps (behavioural rules with no criterion) | Rules were added without matrix entries | Add new criteria (append IDs) for every uncovered rule |
| Plan test mapping references spec IDs that don't exist | Spec was restructured | Reconcile: update mapping or restore missing criteria as `[removed]` |
| Two specs define the same thing differently | Scope creep / unclear boundaries | Check each spec's "Dependencies" and "Out of Scope" sections; merge or split as needed |

---

## Structural Invariants

These are always true. If you find a violation, fix it before making any other
changes.

1. **No `specs/overview.md` exists.** Overview content belongs in the origin
   doc or in individual specs.
2. **Every spec has a Test Matrix section.** If a spec is missing one, add it.
3. **No plan contains a "Spec" section that restates spec content.** Plans
   contain "Spec Reference" sections with links only.
4. **No spec mentions a file name or code location.** That belongs in plans.
5. **No origin doc mentions a spec ID, file name, or test.** The origin is
   purely conceptual.
6. **Spec IDs are never renumbered.** Removed criteria are marked `[removed]`;
   new criteria are appended.

---

## Filing Checklist

Before committing any documentation change, confirm:

- [ ] The change touches only one layer's content (origin, spec, or plan).
- [ ] If it touches the origin: all spec origin-references are still valid.
- [ ] If it touches a spec: the test matrix still covers every behavioural rule.
- [ ] If it touches a spec: all plan test mappings referencing this spec are
      still accurate.
- [ ] If it touches a plan: all spec references and IDs resolve correctly.
- [ ] No content has been duplicated across layers — only references added.
