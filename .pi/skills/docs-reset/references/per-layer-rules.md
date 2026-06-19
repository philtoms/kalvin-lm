# Per-Layer WHAT-IS Rules

Expansion of the **Convention** in `SKILL.md`. These are worked examples for
classification during step 2.

The goal: every surviving doc reads as a description of how things are **now**,
not a record of how they got there.

---

## Specs — KEEP-CLEAN (edit in place, never delete)

A spec is the active behavioural contract. It is never deleted in a reset; it
is stripped to its current contract.

### DELETE (history, no current contract)

- **Tombstone rows.** Test-matrix or API rows whose only content is "X removed
  — replaced by Y." Example: `| MOD-R1 | model.add() removed — replaced by
  cascade API |`. The row describes an absence, not a behaviour.
- **`[removed]` markers.** The cascade's old "Removed → `[removed]`" convention
  is retired. A tombstone is exactly the history a reset kills.
- **"Previously / used to / before→after" prose.** Rewrite as the current rule.
  Example: "`_handle_reactive()` previously escalated on EVERY event… Now: X"
  → just state X as the rule, drop the contrastive framing.

### KEEP but rewrite (current contract expressed as history)

- Any rule stated as "we used to do A, now we do B" — keep B, delete A and the
  transition. The spec must read as if B was always the rule.

### ENCODE-AS-POSITIVE-RULE (the one exception to deleting tombstones)

If an API's **absence is itself a constraint** an implementer could violate
(reintroducing `model.add()`), do not leave a tombstone — encode it as a
positive rule in the API section:

> The write API is exactly `add_stm()` / `add_frame()` / `add_ltm()`. There is
> no general `add()`.

A positive rule orients; a tombstone narrates.

### CODE-TRUTH-CHECK (defer to code)

A spec describing a "legacy format" alongside the current format is only
history if the legacy code is dead. **Read the code first.**

- If the legacy loader is **dead** → delete the legacy rows, fields, and
  backward-compat rules.
- If the legacy loader is **live** (still shipped) → keep it. It is a current
  contract, not history. A reset never deletes a true description of live code.

### Consistency fixes (broken refs)

Renamed artifacts leave dangling references. The vision-document merge
(ADR-0009) renamed `kalvin-origin.md` → `kalvin-vision.md` and `@origin` →
`@vision`, but active specs still pointed at the old names. A reset sweeps
these:

```bash
grep -rniE "kalvin-origin|@origin" specs/ plans/ docs/
```

Retarget every hit to the current name.

---

## Plans — DELETE (spent) or KEEP-CLEAN (active)

### A plan is SPENT iff ALL of:

1. The code it describes exists and matches the plan.
2. Its tests are green.
3. **No open kb task depends on it.** (Hard safety constraint — re-check the
   board at steps 1, 3, and 7.)

A spent plan is deleted. Its conclusion already lives in the code and spec;
design rationale survives only as a **code comment where code-alone-would-
mislead**. Everything else (phases, estimates, status checklists, pseudocode,
green test matrices, before/after refactor illustrations) is history of *how
it got built* — git is the archive.

### Why spent plans are deleted, not compressed

A "Design Decisions remnant" (strip phases/status, keep rationale as a standing
doc) is the tempting middle path. Reject it: that remnant is a second place
that rots. Exhibit: `plans/impl/foundations.md` carried an "Archival Note
(added 2026-06-13)" mapping old op-string names to current structural-state
names. It oriented nobody — the code already used the new names and the
glossary already defined them. It existed only to narrate a transition. Compress
it to nothing and nothing is lost.

### A plan is ACTIVE (KEEP-CLEAN) if any of:

- It has remaining work.
- An open kb task depends on it.
- Its code doesn't yet exist or doesn't match.

Strip an active plan to **remaining work only**: delete completed phases, status
tables, green test matrices (the tests themselves are the contract), before/
after code illustrating already-done refactors, and "estimate: done" lines.
Keep unresolved design decisions and the test-mapping for not-yet-written tests.

### The loop this closes

"ADR migrates to plan" is a **during-build** rule. Once the build is done, the
plan is spent and code+spec is truth. So rationale migrated ADR→plan has no
permanent doc home — it either is embodied in code/spec (needs no separate
statement) or becomes a code comment at the implementation site. There is no
third resting place.

---

## ADRs — MIGRATE-THEN-DELETE

An ADR is temporary scaffolding. By construction it is a *decision record*
(Context / Considered Options / rejected-because / Consequences). Stripped to
"what-is" it stops being an ADR — so it is deleted, not rewritten.

### Deletion is valid once the conclusion is absorbed

For each ADR, confirm its decision is materially present in its target layer:

| ADR kind | Target layer |
|----------|--------------|
| API/contract decision | the spec's API section |
| Terminology rename | `CONTEXT.md` glossary + specs using the term |
| Architecture decision | the relevant spec + a code comment if surprising |
| Doc-structure decision (e.g. vision merge) | the resulting doc itself |

If absorbed → delete the ADR + add an `ARCHIVE.md` row.
If NOT absorbed → migrate the conclusion first (that is commit 1's job), then delete.

### Superseded ADRs

An ADR marked "Superseded by X" is redundant — X (and its absorption target)
already carries the decision. Delete outright. Example: ADR-0005 superseded by
ADR-0006 → 0005 is pure history.

### The code-comment carve-out

Most ADR rationale does NOT need to survive anywhere — the code embodies it.
The narrow exception: a decision that, if unseen, would mislead a future editor
into **reverting** it. That rationale becomes a comment at the implementation
site, not a doc. This is a judgement made at the code, not a doc-curation
decision made far from it.

---

## CONTEXT.md — KEEP (already lean)

The glossary is term definitions; the Operating Notes are process rules. By
construction neither carries implementation history. A reset does not edit
`CONTEXT.md` except to:

- Add/remove glossary terms that have been renamed or retired (and sweep refs).
- Update Operating Notes if a process rule has changed.

---

## docs/ — mixed

- **`docs/kalvin-vision.md`** (origin layer): purpose, philosophy, conceptual
  model. Already registers as WHAT-IS. A reset checks only for mechanism leakage
  (operators, loop steps, file names) and ejects any found.
- **`docs/cascade-development.md`**: the model definition. Update only if the
  cascade itself changed (e.g., a renamed origin filename in its diagram).
- **`docs/adr/`**: emptied by the reset as ADRs are deleted.
- **`docs/ARCHIVE.md`**: created/maintained by the reset. Index, never narrative.
