---
type: concept
title: Trainer
description: A rationaliser — the trainer-side peer of the trainee, sharing the same engine but keeping S1 ratifications and S2 proposals. Cogitates and escalates.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Trainer

A rationaliser — the trainer-side peer of the trainee.

## Definition

The Trainer shares the same rationalising engine as the [[concepts/trainee]]
and differs only in the significance bands it keeps (S1 ratifications and S2
proposals). It cogitates over incoming [[concepts/proposal|proposals]] and emits
its own; it [[concepts/escalation|escalates]] to the [[concepts/supervisor]]
only when its [[concepts/cogitation]] yields no reply. Registered on the
[[concepts/harness]] bus with role `trainer`.

The trainer's mechanical S2/S3 handling (auto-countersign of structurally
matching proposals, within-lesson recurrence dedup) lives in the Reactor; it
never cogitates, never submits reactive scaffolding, and never escalates. Those
are decider concerns owned by the supervisor.

_Avoid_: auto-agent, training bot, the deterministic ratifier of the earlier
path (it now rationalises; see `src/dialogue/`).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/trainee]], [[concepts/supervisor]] — the other roles
- [[concepts/cogitation]] — shared with the trainee
- [[concepts/escalation]] — when the trainer surfaces to the supervisor
- [[concepts/ratify]] — the trainer ratifies proposals
- [[entities/k-engine]] — the shared rationalising engine
