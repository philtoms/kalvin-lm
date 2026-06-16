# Cascade Development Model

**Purpose:** Agent instructions for producing specs, plans, and implementation tasks using the cascade development model.

---

## The Cascade

```
docs/kalvin-vision.md    ←  WHY — purpose, philosophy, conceptual model
        ↓
specs/                   ←  WHAT — testable behavioural contracts
        ↓
plans/                   ←  HOW — implementation strategy, phasing, test mapping
        ↓
.kb/tasks                ←  INSTRUCT - task creation for kb triage
```

Every fact lives in exactly one layer. Other layers **reference** it — never paraphrase, duplicate, or restate.

### Content ownership

| Vision only                | Spec only                              | Plan only                                     |
| -------------------------- | -------------------------------------- | --------------------------------------------- |
| System purpose, philosophy | Data structure definitions, invariants | Algorithm pseudocode, implementation strategy |
| Conceptual model           | API contracts, pre/postconditions      | File structure, code locations                |
| Teaching model             | Behavioural rules                      | Build order, phases, estimates                |
|                            | Acceptance criteria (test matrix)      | Test mapping (spec ID → test function)        |
|                            | Cross-component dependencies           | Design decisions and rationale                |
|                            | Explicit scope boundaries              | Status tracking                               |

---

## Structural Rules

1. **Downward references only.** Vision never references specs/plans. Specs may reference vision. Plans reference specs and vision.
2. **Spec IDs are stable.** Never renumber. Removed → `[removed]`. New → appended.
3. **Every spec has a Test Matrix.** Every behavioural rule → at least one matrix entry.
4. **Plans link, never copy.** Use `See @specs/…` — never restate spec content.
5. **No file names in specs.** No spec IDs in vision. Code locations belong in plans only.

---

## Workflow

### Step 1 — Spec

Create or update specs in `specs/`:

- **Overview** — 1–2 sentences, no philosophy.
- **Dependencies** — cross-references to other specs.
- **Definition** — data structures, fields, types, invariants.
- **API** — signatures, pre/postconditions.
- **Behavioural Rules** — precise, testable assertions.
- **Test Matrix** — numbered criteria with vision references (`KL-1`, `MOD-12`).
- **Out of Scope** — explicit boundaries.

Verify: every behavioural rule has at least one test matrix entry.

### Step 2 — Plan

Create or update plans in `plans/`:

- **Spec References** — links only.
- **Implementation Tasks** — per component: spec ref, test mapping, pseudocode.
- **Test Mapping Table** — spec ID → test file → test function → status.
- **Design Decisions** — resolved questions with rationale.
- **Status** — progress, blockers.

Verify: every spec reference resolves, every mapped spec ID exists, no spec content duplicated.

### Step 3 — Consistency Check

Verify the full traceability chain:

- vision section → spec criterion → plan task → test function.
- No content duplicated across layers.
- No broken references.

### Step 4 — Commit

Stage and commit all grilling documentation updates (specs, plans, vision changes). **Ask the user for explicit confirmation before committing.**

### Step 5 — Create Implementation Tasks

For each implementation task in the plan, create a kb task:

```
kb_task_create({
  description: "<task summary — what to implement, spec refs, acceptance criteria>",
  depends: ["<IDs of prerequisite tasks>"]
})
```

Each description must include: what needs to be implemented, spec reference links, and acceptance criteria (test matrix IDs).
