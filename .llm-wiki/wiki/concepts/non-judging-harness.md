---
type: concept
title: Non-judging harness
description: Harness principle — it compiles, feeds, retrieves, and presents with no verdict and no band-matching. Judgement belongs to the trainer (a pi agent outside the loop).
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Non-judging harness

The defining principle of the dialogue harness.

## Definition

The harness compiles a [[entities/kscript|KScript]] source, feeds each compiled
entry to the engine one at a time, retrieves the engine's
`(batch, observations)` response, and presents the trace. It never judges — no
verdict, no band-matching, no significance gate. Judgement belongs to the trainer
(a pi agent, outside the loop), which reads the trace, makes decisions, edits the
engine and/or the source, and re-runs.

This is what makes the harness suitable for [[entities/auto-tune|auto-tune]]: it
is pure mechanism, leaving all interpretation to the agent that consumes its
output.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — the principle
- [[concepts/harness]] — the general runtime concept
- [[entities/harness]] — the dialogue harness implementation
- [[concepts/engine-first]] — the companion process discipline
- [[entities/auto-tune]] — the agent that consumes the non-judging trace
