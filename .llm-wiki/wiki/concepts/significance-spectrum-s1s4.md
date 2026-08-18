---
type: concept
title: Significance Spectrum (S1–S4)
description: The four significance levels — S1 recognised, S2 contested, S3 suggested, S4 unrecognised — that every kline claim and every rationalisation is measured against.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
  - id: SRC-2026-08-11-003
    resource: /sources/SRC-2026-08-11-003.md
---

# Significance Spectrum (S1–S4)

The four levels at which a kline can be held — claimed by its structure, derived
by rationalisation, and authored as a target.

## Definition

The same S1–S4 labels run through three distinct uses; the spectrum itself is
the shared scale.

| Level | Name        | Meaning                          | Structural claim      | Engine emission (cogitate) |
| ----- | ----------- | -------------------------------- | --------------------- | -------------------------- |
| **S1** | recognised  | fully grounded; known            | Identity, Canon       | grounds                    |
| **S2** | contested   | partial; diverges — a proposal   | Misfit (no/under/over)| proposes                   |
| **S3** | suggested   | indirect; associative connection | Misfit (connote/denote)| countersigns              |
| **S4** | unrecognised| complete novelty — an ask        | Unknown               | asks                       |

- **Structural significance** — the S-level a kline's *structure* claims, with
  no model and no observer (see [[concepts/structural-significance]]).
- **Rational significance** — a participant's own S-level for the kline after
  traversing the model (see [[concepts/rational-significance]]).
- **Target significance** — the S-level a KScript *authors* as the expected
  answer (the training target).

S2 and S3 are active reasoning states — Kalvin cogitates, retracing paths and
generating proposals — not failed S1s. Kalvin can autonomously reach S2 at most;
S1 requires [[concepts/ratify|ratification]] by another agent, which records provenance.

In the engine, `route()` dispatches on structural significance (the `sig_level`)
rather than the producer's compiled stamp, so the same four levels also drive
the fast/slow routing decision (see [[concepts/fast-route-vs-slow-route]]).

_Avoid_: treating the cogitate "emission kinds" (asks/countersigns/proposes/
grounds) as a separate taxonomy — they are what the engine *does* at each level
of the same spectrum. "sig_level" is the code symbol for structural
significance, not a different concept.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — Structural / Rational Significance definitions
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — `sig_level` dispatch and cogitate emission kinds
- [SRC-2026-08-11-003](/sources/SRC-2026-08-11-003.md) — spectrum as the visibility of degree-of-fit
- [[concepts/structural-significance]] — the structural claim
- [[concepts/rational-significance]] — the rationalised measurement
- [[concepts/target-significance]] — the authored answer
- [[concepts/ratify]] — the only path to S1
- [[concepts/fast-route-vs-slow-route]] — routing dispatches on sig_level
