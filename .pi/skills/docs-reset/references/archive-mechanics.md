# Archive Mechanics

The exact git sequence for steps 5–8 of a reset, the `ARCHIVE.md` row format,
and the same-day collision rule. Expansion of the **Mechanics** convention in
`SKILL.md`.

---

## Two commits, exactly

The archive guarantee rests on a tagged commit whose tree **still contains**
every to-be-deleted file. That requires two distinct commits, never collapsed:

1. **Commit 1 — Migrate.** Edit cascade docs in place to absorb conclusions
   and strip history. Do **not** delete any spent file yet. The tree at this
   commit holds every to-be-deleted artifact in its pre-deletion form.

2. **Tag.** Annotate commit 1 so its tree is recoverable by name.

3. **Commit 2 — Delete + Index.** Delete the spent artifacts; append one row
   per deleted artifact to `docs/ARCHIVE.md`.

Because commit 1 precedes any deletion, `git show <tag>:<path>` recovers the
pre-deletion form of any deleted file. Collapsing the two commits destroys
this — the spend would have no tagged, file-complete ancestor.

### Why not `git mv` or a single "tidy" commit

A reset is a documentation event, not a refactor. The deletion is the point
(it removes cruft), and the recoverability is the point (git stays the
archive). A move or rename implies continuity; a reset asserts that the
artifact's conclusion has migrated and the shell is dead.

---

## Tagging

```bash
git tag -a docs-archive-<date>[-N] -m "Documentation reset <date>: pre-deletion archive. See docs/ARCHIVE.md." <commit-1-sha>
```

- `<date>` is `YYYY-MM-DD`.
- `[-N]` is absent for the first reset of a day; `-2`, `-3`, … for subsequent
  same-day resets (see §Same-day collisions below).
- **Never pass `-f` / `--force` to move an existing tag.** Moving a tag
  rewrites the archive pointer and silently destroys the prior run's guarantee.

Verify before proceeding to commit 2:

```bash
git show docs-archive-<date>[-N]:<some-to-be-deleted-file>   # must return the file
```

If this returns empty or errors, do **not** proceed — the guarantee is broken.

---

## Same-day collisions

A day may need more than one reset (a second pass after the first surfaces
more cruft; a partial resume completing same-day). The bare `<date>` name is
taken by the first run, and `git tag` will not overwrite it (correctly).
Rather than force it, sequence the follow-on runs:

1. Before writing the runbook, probe for an existing tag family for today:

   ```bash
   git tag -l "docs-archive-$(date +%Y-%m-%d)*"
   ```

2. **First run of the day** (no hits, or only hits you already created this
   session): bare name — `docs-archive-<date>`, runbook `plans/docs-reset-<date>.md`.

3. **Subsequent same-day run** (a tag for today already exists): choose the
   smallest N≥2 such that `docs-archive-<date>-N` does not yet exist. Both
   the runbook path and the tag use the suffix: `plans/docs-reset-<date>-N.md`,
   `docs-archive-<date>-N`.

The sequence is monotonic and append-only, mirroring the `ARCHIVE.md` index
itself: each run gets its own tag and its own index rows. An `ARCHIVE.md` row
records the exact suffixed tag that recovers its artifact.

### What "today" means across a midnight boundary

`<date>` is fixed at the moment the runbook is written (step 4). If a run
straddles local midnight, keep the original `<date>` for both commits and the
tag — the runbook's date stamps the run, not the wall clock at commit time.

---

## `ARCHIVE.md` format

`docs/ARCHIVE.md` is an **append-only, one-line finding aid** — never a
narrative. One row per deleted artifact:

| Artifact | Conclusion migrated to | Recovery tag |
|----------|-----------------------|-------------|
| `plans/impl/<name>.md` | Implemented — conclusion absorbed into `specs/<x>.md` (rules Rn); rationale retained in code comments where code-alone-would-mislead | `docs-archive-<date>[-N]` |

Rules:

- **Artifact** — the deleted path, as it existed at the tagged commit.
- **Conclusion migrated to** — where the surviving truth now lives. For a
  fully-absorbed artifact, name the target layer + IDs/sections. For a
  superseded artifact, name what superseded it. No "previously/now", no
  considered-alternatives, no rationale. Git is the archive.
- **Recovery tag** — the exact tag name (with `-N` if applicable) whose tree
  holds the pre-deletion file.

The header preamble (the `git show <tag>:<artifact>` recovery recipe) is
written once when the file is created; subsequent resets append rows only.

---

## Verification (step 8)

After commit 2 lands, confirm the reset is clean and the archive is sound:

```bash
# No stale renamed-artifact refs remain
grep -rniE "kalvin-origin|@origin" specs/ plans/ docs/   # expect: no hits

# No history smell remains in specs (except legitimately-live legacy code)
grep -rniE "previously|used to|legacy|\[removed\]|removed — replaced|formerly|superseded|archival note" specs/

# Every surviving spec behavioural rule still has a test-matrix entry
# (manual: confirm no matrix row was orphaned by a deletion)

# The tag exists and recovers a spot-checked file
git tag -l "docs-archive-<date>[-N]"
git show docs-archive-<date>[-N]:<a-deleted-file> | head
```

Report the before/after doc count to the user (e.g., "specs: 19→19, plans:
20→5, ADRs: 0→0").
