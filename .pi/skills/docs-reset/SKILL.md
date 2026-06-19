---
name: docs-reset
description: Executes a documentation reset — consolidates the cascade docs (docs/, specs/, plans/, CONTEXT.md) to WHAT-IS by deleting spent ADRs and fully-implemented plans, stripping spec history (tombstones, "previously/legacy" prose), and archiving everything under a git tag with a recovery manifest. Use when the user says "/docs-reset" or asks to reset, consolidate, freshen, or lean-down the documentation trace.
---

# Docs Reset

> This is a multi-step workflow. Work through steps 1–8 **in order** — each step's output feeds the next, and several are sync points that require explicit user confirmation before you continue. Steps 1–4 are read/plan only: do not edit any cascade doc until step 5. **Pause for user confirmation** after the classification (step 3) and before each commit (steps 5 and 7). Do not batch or skip steps.

## Convention

A reset consolidates the development trace to WHAT-IS: every surviving doc reads as a description of how things are **now**, not a record of how they got there. The convention is settled; a reset executes it, it does not re-open it. To change the convention, edit this section directly (ideally via a fresh grilling session).

**Scope.** Cascade docs only: `docs/`, `specs/`, `plans/`, `CONTEXT.md`. Out of scope (separate concerns with their own lifecycles): `auto-tune/`, `.kb/tasks/`, `curricula/`, `dev/`.

**WHAT-IS by layer.**
- *Specs* describe only the current contract. Tombstone rows (`X removed — replaced by Y`) and `[removed]` markers are deleted. A load-bearing absence is encoded as a positive rule, not a tombstone.
- *Plans* that are fully-implemented are deleted — their conclusion lives in the code or spec, and design rationale survives only as a code comment where code-alone-would-mislead. An active plan (remaining work, or an open kb task depends on it) is kept but stripped to remaining work.
- *ADRs* are temporary scaffolding: once their decision is absorbed into code/spec/plan, the ADR file is deleted.
- *CONTEXT.md* and *docs/kalvin-vision.md* are already WHAT-IS by construction; a reset touches them only for renamed/retired terms or mechanism leakage.

**Mechanics (two commits).** (1) Migrate each artifact's surviving conclusion into its proper layer and commit; (2) tag that commit `docs-archive-YYYY-MM-DD` — it holds every to-be-deleted file in pre-deletion form; (3) delete the spent artifacts, append to `docs/ARCHIVE.md`, commit.

**`docs/ARCHIVE.md`** is a persistent, append-only, one-line index per archived artifact — artifact → where its conclusion migrated → recovery tag. An index entry is a finding aid, never a narrative. Git is the archive; the repo is lean.

The reference files expand these rules with worked examples and the exact git sequence:
- `references/per-layer-rules.md` — WHAT-IS by layer with classification examples (step 2).
- `references/archive-mechanics.md` — two-commit git sequence, `ARCHIVE.md` format, verification (steps 5–8).

## Entry Points

### Fresh Reset

No un-archived runbook exists at `plans/docs-reset-<YYYY-MM-DD>.md`, and no `docs-archive-<YYYY-MM-DD>` tag points at a pre-deletion commit from today. → Complete step 1 (Inventory), then proceed through the workflow.

### Resume Reset

A runbook exists at `plans/docs-reset-<date>.md` but its tag `docs-archive-<date>` does not yet exist (commit 1 done, commit 2 pending), OR the runbook is only partially complete. → Read the runbook, confirm with the user where it stopped, jump to that step.

---

## 1. Inventory (always first)

Establish the current state. Do not decide anything yet — just gather.

- **List cascade docs:** `docs/` (incl. `docs/adr/`), `specs/`, `plans/` (incl. `plans/impl/`), `CONTEXT.md`.
- **Find history smell:** `grep -rniE "previously|used to|legacy|\[removed\]|removed — replaced|before \(|after \(|formerly|renamed from|superseded|archival note" specs/ plans/ docs/` — these are the WHAT-IS violations to resolve.
- **Find broken refs:** `grep -rniE "kalvin-origin|@origin" specs/ plans/ docs/` (and any other known renamed artifacts).
- **ADR statuses:** read the `Status:` line of every `docs/adr/*.md`. Superseded and accepted-but-absorbed ADRs are delete candidates.
- **Plan completion signals:** grep each plan for "all implementation complete / all tests pass / estimate: done / complete — merged". These are spent candidates.
- **SAFETY — in-flight work:** list the kb board's `in-progress` and `todo` columns. **Every open task blocks deletion of any plan/ADR touching its area.** Record the blocking task IDs in the runbook.

Output of this step: a raw inventory. Decisions happen in step 2.

## 2. Classify per layer (apply the WHAT-IS rules)

For each artifact in the inventory, classify it using `references/per-layer-rules.md`. The classification is one of:

- **KEEP-CLEAN** — active doc; strip its history but retain it (specs; active plans).
- **MIGRATE-THEN-DELETE** — its surviving conclusion must land in code/spec/plan first, then the file is deleted (spent plans; accepted ADRs whose conclusion isn't yet absorbed).
- **DELETE** — pure history with no surviving conclusion, or already superseded (superseded ADRs; tombstone rows; "previously" prose).
- **CODE-TRUTH-CHECK** — can't classify without reading the code (e.g., a "legacy format" loader — live or dead?).

Record the classification per artifact. This becomes the runbook's body.

## 3. Surface judgement calls to the user

Present the classification to the user before executing. Specifically call out:

- Every **CODE-TRUTH-CHECK** item with your finding (is the code live or dead?).
- Every **MIGRATE-THEN-DELETE** item with *where* the conclusion is migrating, and whether the migration target already contains it (no-op) or needs an edit.
- Every plan you believe is **spent**, with the evidence (code exists, tests green, no in-flight task).
- Any **code-comment-worthy rationale** you propose to add at an implementation site (only where code-alone-would-mislead).

Do not execute until the user confirms the classification. Adjust on feedback.

## 4. Generate the runbook

Write `plans/docs-reset-<YYYY-MM-DD>.md` (use today's date) capturing: scope, safety constraints (blocking task IDs), the per-artifact classification table, the migration targets, and the delete list. Model it on the cascade plan format but keep it operational — it is a checklist, not a spec.

This runbook is itself a spent artifact: after the reset lands it gets one `ARCHIVE.md` row and is deleted under its own tag (see §Final).

## 5. Execute — Commit 1 (Migrate)

Edit cascade docs **in place** to absorb conclusions and strip history. **Do not delete spent files yet.** Concretely:

- Specs: delete tombstone rows / `[removed]` markers; rewrite "previously/legacy" prose as current rules; encode load-bearing absences as positive rules; fix broken refs.
- Plans: for kept-active plans, strip completed phases, status tables, green test matrices, before/after refactor code, "estimate: done" lines. (Spent plans are NOT touched here — they're deleted in commit 2.)
- ADRs: if a conclusion isn't yet absorbed, migrate it to its target layer now.
- Code comments: add any agreed code-comment-worthy rationale at the implementation site.

Commit with a migrate message. **This commit's tree still contains every to-be-deleted file** — that is the archive guarantee.

## 6. Tag

```bash
git tag -a docs-archive-<YYYY-MM-DD> -m "Documentation reset <YYYY-MM-DD>: pre-deletion archive. See docs/ARCHIVE.md." <commit-1-sha>
```

Verify the guarantee before proceeding:

```bash
git show docs-archive-<YYYY-MM-DD>:docs/adr/<some-deleted-file>.md   # must return the file
```

## 7. Execute — Commit 2 (Delete + Index)

- Delete every MIGRATE-THEN-DELETE and DELETE artifact (spent plans, superseded/absorbed ADRs).
- Delete the runbook itself (it is now spent).
- Append one row per deleted artifact to `docs/ARCHIVE.md` (create if absent) using the format in `references/archive-mechanics.md`.
- Commit with a delete+index message.

## 8. Verify

Run the verification commands in `references/archive-mechanics.md`:
- No stale `@origin` / renamed-artifact refs remain.
- No history smell remains in specs (except legitimately-live legacy like a still-shipped loader).
- Every surviving spec behavioural rule still has a test-matrix entry.
- The tag exists and recovers a spot-checked file.

Report the before/after doc count to the user.

---

## Rules (apply throughout)

1. **Safety first.** Never delete a plan or ADR whose area has an open kb task. Re-check the board at steps 1, 3, and 7.
2. **Migrate before delete.** No deletion until the conclusion has landed in its proper layer.
3. **Two commits, exactly.** Commit 1 migrates (no deletions). Tag. Commit 2 deletes + indexes. Do not collapse them — the tagged commit must provably contain the to-be-deleted files.
4. **Index, never narrative.** `docs/ARCHIVE.md` rows are one-line finding aids (artifact → migrated-to → tag). No rationale, no "previously/now", no considered-alternatives. Git is the archive.
5. **Do not renumber spec IDs.** Removed IDs are deleted, not tombstoned; the archive manifest preserves traceability. New IDs append.
6. **Scope discipline.** Only `docs/`, `specs/`, `plans/`, `CONTEXT.md`. `auto-tune/`, `.kb/tasks/`, `curricula/`, `dev/` are out of scope — separate concerns.
7. **Ask before committing** unless this is executing a kb task. Ad-hoc reset work needs explicit user confirmation for each commit.

## Reference files (load on demand)

- `references/per-layer-rules.md` — WHAT-IS by layer, with worked examples of each classification. Load when classifying (step 2).
- `references/archive-mechanics.md` — `ARCHIVE.md` row format, exact two-commit git sequence, verification commands. Load when executing (steps 5–8).
