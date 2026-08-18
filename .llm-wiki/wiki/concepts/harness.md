---
type: concept
title: Harness
description: The multi-agent runtime that loads agents as participants and runs a dialogue loop between them — a message broker routing role-addressed messages.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-003
    resource: /sources/SRC-2026-08-11-003.md
---

# Harness

The multi-agent runtime that loads agents as participants and runs a dialogue
loop between them.

## Definition

The harness is a message broker. Agents send role-addressed messages through it
and it routes each message to all subscribers of that role. Participants never
communicate directly. Three roles are registered on the bus:
[[concepts/trainee]], [[concepts/trainer]], and [[concepts/supervisor]].

There is no training mode: Kalvin rationalises all klines identically. The
harness code distinguishes training from operational use, not Kalvin itself —
training harnesses use [[concepts/significance]] to decide what to teach next.

Two harnesses exist in the codebase: the multi-agent
[[entities/harness|training harness]] (WebSocket broker, asynchronous) and the
lean [[concepts/non-judging-harness|dialogue harness]] (synchronous, compile →
feed → present). Both share the principle that the harness itself does not
judge.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-003](/sources/SRC-2026-08-11-003.md) — harness manages the flow; significance drives teaching
- [[concepts/agent]] — what the harness loads
- [[concepts/trainee]], [[concepts/trainer]], [[concepts/supervisor]] — the roles
- [[concepts/non-judging-harness]] — the harness's defining principle
- [[entities/harness]] — the training-harness implementation
