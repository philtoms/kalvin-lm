---
type: entity
title: Harness (implementation)
description: The training harness runtime — the multi-agent WebSocket server and message bus, plus the synchronous dialogue harness. Both drive the K engine.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Harness (implementation)

The runtime implementations of the [[concepts/harness]] concept.

## Overview

Two implementations exist in the codebase:

- **`src/training/harness/`** — the multi-agent runtime. A WebSocket server
  (`server.py`) + addressed message bus (`bus.py`) that loads participants
  (Rationaliser adapter, Trainer, supervisors) and routes role-addressed
  messages between them. Launched via `python -m training.harness`.
- **`src/dialogue/harness.py`** — the dialogue harness. A minimal,
  synchronous, [[concepts/non-judging-harness|non-judging]] loop: compile a
  KScript source, feed each entry to the [[entities/k-engine|engine]] one at a
  time, present the `(batch, observations)` trace. Launched via
  `python -m dialogue.harness`.

Both are pure mechanism — judgement belongs to a trainer agent outside the loop
(in auto-tune, the pi agent). The factory functions that assemble the engine
(signifier, state, strategy) live in the dialogue harness.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — the Harness glossary entry
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — the dialogue harness's non-judging principle
- [[concepts/harness]] — the general concept
- [[concepts/non-judging-harness]] — the dialogue harness's defining principle
- [[entities/k-engine]] — what both harnesses drive
- [[entities/auto-tune]] — the agent that consumes the trace
