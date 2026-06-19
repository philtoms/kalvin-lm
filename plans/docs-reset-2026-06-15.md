# Documentation Reset — 2026-06-15

**Type:** one-off operational runbook (not a Kalvin implementation plan).
**Scope:** cascade docs only — `docs/`, `specs/`, `plans/`, `CONTEXT.md`.
**Out of scope:** `auto-tune/`, `.kb/tasks/`, `curricula/`, `dev/`.
**Authority:** `.pi/skills/docs-reset/SKILL.md` → Convention.

This plan executes the reset convention for the current development trace. It
is itself a spent artifact once the reset lands; it is archived under its own
tag per the convention (see §Final).

The goal is WHAT-IS: every surviving doc reads as a description of how things
are now. History (how it got here, considered-and-rejected alternatives,
removed APIs, completed builds, green test tables) is deleted and recoverable
via git.

---

## Safety constraints (hard)

1. **Do not delete an active plan.** A plan is _spent_ only if ALL of: the code
   it describes exists and matches, its tests are green, AND no open kb task
   depends on it. Before deleting any plan, confirm the in-progress/todo kb
   columns are empty for the area it covers.
2. **Migrate before delete.** An artifact is deleted only after its surviving
   conclusion has landed in its proper layer (code, spec, or a kept plan).
3. **Archive guarantee.** The tagged commit must physically contain every
   to-be-deleted file. Use the two-commit sequence in §Mechanics exactly.

---

## Mechanics (two commits)

**Commit 1 — Migrate.** Edit cascade docs in place: strip specs to current
contract, migrate plan/ADR conclusions to code/spec/plan. Do NOT delete spent
files yet. Commit. This commit's tree still contains every pre-deletion file.

**Tag.** `git tag -a docs-archive-2026-06-15 -m "…" <commit-1>`.

**Commit 2 — Delete + index.** Delete spent ADRs/plans. Fix dangling
references. Append one-line-per-artifact rows to `docs/ARCHIVE.md`. Commit.

Recovery (any future point): `git show docs-archive-2026-06-15:<path>`.

---

## Pass 1 — Specs (edit in place, commit 1)

**Rule applied:** a spec describes only the current contract. Delete tombstone
rows and `[removed]` markers; encode a load-bearing absence as a positive rule.

### Tombstone rows to delete

- `specs/model.md` — rows `MOD-R1` (`model.add()` removed — replaced by cascade
  API) and `MOD-R2` (`model.promote()` removed — replaced by `add_ltm()`).
  **Positive-rule check:** confirm `model.md` API section states the write API
  is exactly `add_stm()` / `add_frame()` / `add_ltm()`. If absent, add it as a
  positive rule, then delete the tombstone rows.

### "Previously / used to / legacy / Before→After" prose to strip

- `specs/stm.md:32` — "Used to derive the nodes signature…". Rewrite to the
  current fact without the "used to".
- `specs/cascade-control.md:34` — "previously escalated on EVERY event… Now:".
  Rewrite as the current behavioural rule; drop the contrastive framing.
- `specs/s3-auto-countersign.md:7,23` — two "previously…" sentences. Rewrite as
  current behaviour.

### Legacy-format handling — CODE-TRUTH CHECK required

- `specs/curriculum.md` describes a **legacy flat format** alongside the current
  format, with rules `CRS-31` (`load()` handles legacy format) and backward-compat
  fields. **Before editing:** confirm against `src/` whether the legacy loader is
  still live code.
  - If **dead**: delete the legacy-format rows/fields and the backward-compat rules.
  - If **live**: keep — it is a current contract, not history.

### Consistency fixes (broken refs from ADR-0009's incomplete cutover)

- `specs/agent.md:391`, `specs/significance.md:13`, `specs/kline.md:31` — still
  reference `docs/kalvin-origin.md` / `@origin`. Retarget to
  `docs/kalvin-vision.md` / `@vision`.

---

## Pass 2 — ADRs (migrate, then delete in commit 2)

**Rule applied:** an ADR is temporary scaffolding. Once its decision is absorbed
into code/spec/plan, the file is deleted; git is the archive.

| ADR                                   | Status                     | Action                                                                                  |
| ------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------- |
| 0001 addressed message bus            | Accepted                   | Verify conclusion is in `specs/` (message bus contract). If yes → delete.               |
| 0002 four-tier memory                 | (decision)                 | Verify write-API table is in `specs/model.md`. If yes → delete.                         |
| 0003 file-mediated auto-tune protocol | Accepted                   | Verify in `specs/auto-tune.md`. If yes → delete.                                        |
| 0004 nlp bindings from comments       | **Superseded** (spec v2.0) | Conclusion already absorbed → delete.                                                   |
| 0005 undersign is S3                  | **Superseded by 0006**     | Redundant → delete.                                                                     |
| 0006 op is structural state           | Accepted, supersedes 0005  | Verify glossary (Structural State, Identity) + specs carry it. If yes → delete.         |
| 0007 band-anchored significance       | Accepted, supersedes DD-1  | Verify in `specs/significance-normalization.md`. If yes → delete.                       |
| 0008 node taxonomy bound/unbound      | Accepted                   | Verify in `specs/kline.md`, `signature.md`, `tokenizer.md` + glossary. If yes → delete. |
| 0009 vision document merge            | Accepted                   | Conclusion = the clean `docs/kalvin-vision.md`, which already exists. → delete.         |

**Migration step per ADR:** confirm the conclusion is materially present in the
named spec/code/glossary. Only when present does the ADR get a delete + an
ARCHIVE.md row. If any conclusion is NOT yet absorbed, migrate it first (that is
the whole point of commit 1), then delete.

For each deleted ADR, also surface any **code-comment-worthy rationale** — only
where code-alone-would-mislead a future editor into reverting the decision. That
rationale becomes a comment at the implementation site, not a doc.

---

## Pass 3 — Plans (delete spent, strip active; commit 2)

**Rule applied:** fully-implemented plans are deleted; active plans are stripped
to remaining work.

### Spent-vs-active determination (per plan)

A plan is **spent** iff: (a) the code it describes exists and matches, (b) its
tests are green, (c) no open kb task depends on it. Otherwise it is **active**.

### Likely-spent candidates (verify each against code + tests + kb board)

`plans/impl/`: `expand-robustness.md`, `training-log.md`, `structural-grounding.md`,
`reactive-scaffolding.md`, `countersign-resolution.md`, `rename-unsigned-to-identity.md`,
`cogitator-drain.md`, `cascade-control.md`, `nlp-first-curriculum-annotations.md`,
`s3-auto-countersign.md`, `node-taxonomy.md`, `foundations.md`.

`plans/`: `implement-harness-server.md`, `implement-unpack.md`, `implement-kscript.md`,
`implement-kalvin.md`, `role-based-routing.md`.

**For `plans/impl/foundations.md`** specifically: delete the entire **"Archival
Note (added 2026-06-13)"** block (UNSIGNED→IDENTITY mapping). If the plan is kept
as active, that note is history regardless and must go; the current terminology
is authoritative in `CONTEXT.md`.

**For any plan kept as active**, strip: completed phases, status tables, green
test matrices (the tests themselves are the contract), before/after code
illustrating already-done refactors, and any "estimate: done" lines. Keep only
remaining work + unresolved design decisions.

### Safety re-check before any plan deletion

Re-read the kb in-progress and todo columns. Anything referencing the plan's
area blocks its deletion.

---

## Pass 4 — docs/ and CONTEXT.md (commit 1 / 2)

- `docs/cascade-development.md:10` — the cascade diagram still shows
  `docs/kalvin-origin.md`. Retarget to `docs/kalvin-vision.md`.
- `docs/kalvin-vision.md` — already clean (verified: no mechanism/history
  leakage). No action.
- `CONTEXT.md` — glossary already lean (43 terms, no duplicates). No action
  beyond the Operating Note already added this session.

---

## Final — Archive index (commit 2)

Create `docs/ARCHIVE.md` if absent; append rows. Format:

```
| Artifact | Migrated to | Recover via |
|----------|-------------|-------------|
| docs/adr/0005-undersign-is-s3.md | docs/adr/0006 (→ specs; glossary Structural State) | docs-archive-2026-06-15 |
| …      | …           | …           |
```

One row per deleted artifact. No narrative. After commit 2, this runbook itself
(`plans/docs-reset-2026-06-15.md`) is a spent plan → it gets one final row and is
deleted in the same commit, archived under the same tag.

---

## Verification (post-reset)

- `grep -rn "kalvin-origin\|@origin" docs/ specs/ plans/` → empty.
- `grep -rniE "previously|used to|legacy|\[removed\]|removed — replaced" specs/` →
  only live-code legacy (curriculum loader, if still live).
- Every surviving spec behavioural rule still has a test-matrix entry.
- `git tag -l "docs-archive-*"` lists `docs-archive-2026-06-15`.
- `git show docs-archive-2026-06-15:docs/adr/0002-four-tier-memory.md` returns the
  file (spot-check the archive guarantee on one artifact).

---

## What this reset does NOT do

- Does not touch `auto-tune/`, `.kb/tasks/`, `curricula/`, or `dev/`.
- Does not renumber spec IDs (cascade rule: stable; removed → now deleted, not
  tombstoned — the archive manifest preserves traceability).
- Does not edit code, except adding a code comment where a deleted ADR's
  decision would mislead a future editor.
