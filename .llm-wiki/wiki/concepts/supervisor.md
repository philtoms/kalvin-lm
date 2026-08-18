---
type: concept
title: Supervisor
description: "An agent that resolves proposals the Trainer escalates — deciding ratify, scaffold, or continue. Medium-independent: TUI, Slack, CLI, or LLMSupervisor."
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Supervisor

An Agent that resolves the [[concepts/proposal|proposals]] the
[[concepts/trainer]] [[concepts/escalation|escalates]].

## Definition

The supervisor decides — [[concepts/ratify|ratify]],
[[concepts/scaffolding|scaffold]], or continue — and is independent of medium.
TUI, Slack, CLI, and an LLMSupervisor all share the same capabilities; a
judgement may be a human decision or an LLM's internal assessment. Registered on
the [[concepts/harness]] bus with role `supervisor`.

The supervisor owns the decider concerns the Reactor and Trainer do not:
cogitation over what scaffolding to write, submitting reactive scaffolding, and
choosing to escalate to a human.

_Avoid_: UI (too narrow), human (a supervisor may be an LLMSupervisor).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/escalation]] — how proposals reach the supervisor
- [[concepts/ratify]], [[concepts/scaffolding]] — the decisions available
- [[concepts/trainer]] — who escalates
- [[entities/auto-tune]] — auto-tune uses an LLMSupervisor via a CLI supervisor
