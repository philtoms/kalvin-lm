---
type: source
title: "Observation: Stale trs1 stash still in list; dev/probes gitignored"
tags:
  - git
  - recovery
  - stash
  - footgun
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-stale-trs1-stash-still-in-list-dev-probes-gitignored
relevance: medium
observed_at: 2026-10-31T22:54:48.122Z
source_context: Recovering the working tree after an accidental stash pop during harness verification
---

# 🔍 Observation: Stale trs1 stash still in list; dev/probes gitignored

Two footguns hit in one command: (1) `git stash push -- dev/dialogue/harness.py dev/probes/` fails silently-to-null because dev/probes is gitignored (.gitignore:83) — pathspec matches no tracked files. The `&&` short-circuit then let a `;`-chained `git stash pop` apply a PRE-EXISTING stale stash (stash@{0} "WIP on trs1: afbd6d8", containing old src/trs/model.py + dev/dialogue/trs_harness.py), conflicting with the staged model.py WIP. Recovered via `git checkout --ours` + `git add` for model.py and `git rm` for trs_harness.py; the stale trs1 stash is still in the list (pop on conflict keeps it). The stale stash remains — decide whether to drop it. (2) tests/test_rationalisation_certificate.py has 2 failures that reproduce on clean HEAD in a temp worktree — pre-existing, unrelated to the harness. Also: dev/probes is intentionally ignored — probe scripts are local-only and never in git status.

*Relevance: medium*
*Context: Recovering the working tree after an accidental stash pop during harness verification*
*Tags: git recovery stash footgun*

---
*Observed: 2026-10-31T22:54:48.122Z*
