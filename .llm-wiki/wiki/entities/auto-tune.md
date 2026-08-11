---
type: entity
title: Auto-Tune
description: The project's experimental loop for tuning Kalvin's rationalisation behaviour — an LLM coding agent runs repeated sessions, observes the reactor/cogitator/rationaliser, edits the significance-model code, and re-runs.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Auto-Tune

The project's experimental loop for tuning Kalvin's rationalisation behaviour.

## Overview

An LLM coding agent (pi) runs repeated sessions against a curriculum, observes
how the reactor / [[concepts/cogitation|cogitator]] /
[[entities/rationalise|rationaliser]] actually behave, edits the
significance-model code (`expand()`, `significance.py`, the rationaliser), and
re-runs to confirm. It is _not_ training — Kalvin is not learning during
auto-tune; the agent is tuning the engine that does the learning.

Implemented in `src/training/auto_tune/`. The CLI supervisor
(`supervisors/cli_supervisor.py`) is the headless, file-based bridge that lets
the agent observe and control sessions via `events.jsonl` and `cmd.json`.
Launched via `python -m training.auto_tune`.

Auto-tune follows [[concepts/engine-first]] discipline: suspect the engine
before retreating to .ks authoring.

_Avoid_: tuning session (ambiguous with training session), auto-train (it's not
training), auto-tune supervisor (the CLI includes the full auto-tune tool, not
just the supervisor).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — engine-first discipline
- [[concepts/engine-first]] — the operating posture
- [[concepts/non-judging-harness]] — provides the trace auto-tune reads
- [[entities/harness]] — the runtime auto-tune drives
- [[entities/expand]], [[entities/similarfit]] — the strategies auto-tune tunes
