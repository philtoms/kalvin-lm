# Sub-Plan: Auto-Tune Session Layout — File Structure & Concept→File Mapping

**Spec refs:** [`@specs/auto-tune.md`](../../specs/auto-tune.md), [`@specs/curriculum.md`](../../specs/curriculum.md)
**Status:** Reference document (no implementation tasks)

## Purpose

The single authoritative reference for the auto-tune session on-disk file
structure and the mapping from spec concept terms to the real filenames the
production code uses. Per `docs/cascade-development.md` Structural Rule #5
("No file names in specs. Code locations belong in plans only."),
`specs/auto-tune.md` references its persisted artefacts **by concept**; the
concrete file names and the directory layout live here, because file structure
and code locations are Plan-owned. This plan exists to preserve format identity
without carrying file names in the spec.

## Concept → File Mapping

The auto-tune spec describes each persisted artefact by concept. The production
code (`src/training/participants/auto_tune/session.py`, `supervisor.py`,
`snapshots.py`) reads and writes these concrete files:

| Concept (spec term)                 | Real file                 | Defined in spec                                   |
| ----------------------------------- | ------------------------- | ------------------------------------------------- |
| Session configuration               | `config.json`             | `@specs/auto-tune.md` §Session Configuration      |
| Command frame (the command file)    | `cmd.json`                | `@specs/auto-tune.md` §Command Frame              |
| Status object                       | `status.json`             | `@specs/auto-tune.md` §Status Object              |
| Event stream / event log            | `events.jsonl`            | `@specs/auto-tune.md` §Event Frame                |
| Run directory                       | `runs/<n>/`               | `@specs/auto-tune.md` §Snapshot and Restore       |
| Run state snapshot                  | `runs/<n>/state.json`     | `@specs/auto-tune.md` §Snapshot and Restore       |
| Run event log                       | `runs/<n>/events.jsonl`   | `@specs/auto-tune.md` §Snapshot and Restore       |
| Run model snapshot                  | `runs/<n>/model.bin`      | `@specs/auto-tune.md` §Snapshot and Restore       |
| Run metadata (snapshot metadata)    | `runs/<n>/meta.json`      | `@specs/auto-tune.md` §Snapshot Metadata          |
| Curriculum state file               | `curricula/<slug>.json`   | `@specs/auto-tune.md` §Reset, `@specs/curriculum.md` |
| Curriculum file (generated)         | `curricula/<slug>.md`     | `@specs/curriculum.md` §Curriculum Generation     |

## Session Directory Tree

Sessions live inside a git worktree at `.worktrees/auto-tune/<session>/`. The
session directory is nested within:

```
.worktrees/auto-tune/<session>/          # Git worktree (full source checkout on auto-tune/<session> branch)
├── src/                                  # Source code (isolated from main repo)
├── auto-tune/
│   └── <session>/
│       ├── config.json           # Session configuration
│       ├── cmd.json              # Single command (pi writes, supervisor consumes)
│       ├── status.json           # Supervisor heartbeat
│       ├── events.jsonl          # Append-only event stream
│       └── runs/
│       │   └── 001/
│       │       ├── state.json    # Curriculum tracking state snapshot
│       │       ├── events.jsonl  # Full event log up to snapshot
│       │       ├── model.bin     # Kalvin model snapshot (if exists)
│       │       └── meta.json     # Git metadata, timestamp
└── ...                                  # Other project files
```
