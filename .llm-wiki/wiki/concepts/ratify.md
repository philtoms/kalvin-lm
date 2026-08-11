---
type: concept
title: Ratify
description: The action of countersigning a selected proposal; usually performed by the Trainer during curriculum execution. Ratification promotes a proposal to S1 and records provenance.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-003
    resource: /sources/SRC-2026-08-11-003.md
---

# Ratify

The action of countersigning a selected proposal.

## Definition

To ratify is to countersign a proposal another agent has offered, making it
mutually held. Usually performed by the Trainer during curriculum execution, and
more generally by any agent providing external confirmation. Ratification is the
only path to **S1**: Kalvin can autonomously reach S2 at most, so full knowing
requires another agent to ratify. The countersign records provenance — *who*
vouched for the kline — establishing its authority.

This is the operative sense of "countersign" and "ratification": the same act
viewed from different angles. "Countersign" emphasises the structural mechanism
(the reciprocal `{A:[B]}`, `{B:[A]}` pair); "ratification" emphasises the
significance-level consequence (proposal → S1); "ratify" is the verb.

_Avoid_: treating ratification as something Kalvin can do alone — it is
inherently an inter-agent act.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — glossary definition (CONTEXT.md)
- [SRC-2026-08-11-003](/sources/SRC-2026-08-11-003.md) — ratification as the only path to S1; provenance via countersigning
- [[concepts/relational-tokens]] — COUNTERSIGNS (`==`) emits the reciprocal pair
- [[concepts/grounding]] — ratification is how a kline takes up LTM residency and becomes grounded
- [[concepts/escalation]] — the boundary between what the Trainer resolves itself and what it surfaces for ratification
