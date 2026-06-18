# Cascade Development Model

**Purpose:** Agent instructions for maintaining specs, plans, and implementation tasks using the cascade development model.

---

## The Cascade

```
docs/kalvin-vision.md    ←  WHY — purpose, philosophy, conceptual model
        ↓
specs/                   ←  WHAT — testable behavioural contracts
        ↓
plans/                   ←  HOW — implementation strategy, phasing, test mapping
```

Every fact lives in exactly one layer. Other layers **reference** it — never paraphrase, duplicate, or restate.

### Content ownership

| Vision only                | Spec only                              | Plan only                                     |
| -------------------------- | -------------------------------------- | --------------------------------------------- |
| System purpose, philosophy | Data structure definitions, invariants | Algorithm pseudocode, implementation strategy |
| Conceptual model           | API contracts, pre/postconditions      | File structure, code locations                |
| Teaching model             | Behavioural rules                      | Build order, phases, estimates                |
|                            | Acceptance criteria (test matrix)      | Test mapping (spec ID)                        |
|                            | Cross-component dependencies           | Design decisions and rationale                |
|                            | Explicit scope boundaries              | Status tracking                               |

---

## Structural Rules

1. **Downward references only.** Vision never references specs/plans. Specs may reference vision. Plans reference specs and vision.
2. **Spec IDs are stable.** Never renumber. Removed → `[removed]`. New → appended.
3. **Every spec has a Test Matrix.** Every behavioural rule → at least one matrix entry.
4. **Plans link, never copy.** Use `See @specs/…` — never restate spec content.
5. **No file names in specs.** No spec IDs in vision. Code locations belong in plans only.

### Rule #5 — what counts as a "file name" (clarification)

A **file name** under Rule #5 is any literal token that names a concrete file or
path the production code creates, reads, or writes. It covers two classes:

- **Project/config filenames** (e.g. `training.harness.yaml`) — a code-location
  concern. KB-322 purged this class from `specs/` entirely; specs reference it by
  concept ("the project harness config" / "the per-session harness config"). That
  ruling stands unchanged.
- **Runtime data-format filenames and path templates** (e.g. `config.json`,
  `cmd.json`, `status.json`, `events.jsonl`, `meta.json`, `state.json`,
  `model.bin`, `runs/<n>/`, `curricula/<slug>.md`) — the files a session writes
  at runtime. KB-331 extended KB-322's principle here: specs describe these by
  concept; the concrete names and the on-disk layout live in the plan layer,
  which owns file structure and code locations.

The WHAT layer describes a persisted artefact by its **concept** — the data
structure and what it represents ("the session configuration", "the event
stream", "the run directory", "the command file"). The concept→file mapping
(which concept names which real file) is documented **once** in the plan layer;
production code and tests use the real filenames, and the spec's concept terms
map 1:1 to them via the plan.

A token is **not** a file name (it stays in specs) when it names:

- a **data-structure concept** — the field tables, types, and invariants that
  define a format (the *Session Configuration*, *Event Frame*, *Command Frame*,
  *Status Object*, *Snapshot Metadata* tables). The structure is spec-owned; only
  the file that persists it is plan-owned.
- a **wire/message payload** — the JSON keys and action names inside a message
  (e.g. `{"action": "scaffold"}`) are data-structure definitions, not file names.

KB-331's per-category ruling — applied to the auto-tune session files. It is
deliberately **uniform** (context-free): every listed filename token is purged
from `specs/` regardless of where it appears, which keeps the policy
mechanically enforceable by a flat, line-oriented guard with no context filter.

- **Data-format definition headers** — *reword*. A header like
  `### Event Frame (events.jsonl)` names both a concept (the Event Frame) and a
  file. Drop the file: `### Event Frame`. The structure table stays verbatim; the
  canonical filename moves to the plan. The concept unambiguously identifies the
  format; which file holds it is a code location.
- **Directory / file-structure trees** — *move to plans*. A literal file tree is
  "file structure," which the content-ownership table assigns to Plan only. The
  spec cross-references the plan; it does not copy the tree.
- **Filenames in behavioural rules, API/table description columns, and
  test-matrix rows** — *reword to concept*. The rule or contract describes *what
  happens*; the filename is incidental to it (this holds even in the CLI/API
  subcommand tables, where the description is about behaviour, not file
  identity). Replace the filename with the concept term; the concept→file mapping
  lives in the plan. Spec IDs, origin-ref columns, and test-matrix row identity
  are never altered.

The regression guard in `tests/test_config_name_consistency.py` enforces this:
`specs/` must contain none of the purged filename tokens (a flat `_offenders()`
scan), while plans legitimately hold them.

---

## Workflow

### Step 1 — Spec

Update or create specs in `specs/`:

- **Overview** — 1–2 sentences, no philosophy.
- **Dependencies** — cross-references to other specs.
- **Definition** — data structures, fields, types, invariants.
- **API** — signatures, pre/postconditions.
- **Behavioural Rules** — precise, testable assertions.
- **Test Matrix** — numbered criteria with vision references (`KL-1`, `MOD-12`).
- **Out of Scope** — explicit boundaries.

Verify: every behavioural rule has at least one test matrix entry.

### Step 2 — Plan

Update or create plans in `plans/`:

- **Spec References** — links only.
- **Implementation Tasks** — per component: spec ref, test mapping, pseudocode.
- **Test Mapping Table** — spec ID → test file → status.
- **Design Decisions** — resolved questions with rationale.
- **Status** — progress, blockers.

Verify: every spec reference resolves, every mapped spec ID exists, no spec content duplicated.

### Step 3 — Consistency Check

Verify the full traceability chain:

- vision section → spec criterion → plan task.
- No content duplicated across layers.
- No broken references.
