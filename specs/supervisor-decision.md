# Supervisor Decision — Specification

## Overview

The reactive decision in a training session — what to do when Kalvin produces a proposal the Trainer cannot auto-ratify — is owned by a **supervisor participant**. The Trainer surfaces the decision, gates the run until it is answered, and applies the answer; it never decides. There is one model: the decider is always a supervisor participant, and the LLMSupervisor is one such participant (a peer of the TUI, Slack, and CLI participants).

## Dependencies

- `@specs/harness-server.md` — the three roles, the message bus, the shared command protocol, the supervisor message tables
- `@specs/trainer-satisfaction.md` — the Trainer's mechanical work (auto-countersign, recurrence dedup, lesson progression); unchanged by this spec
- `@specs/cogitator.md` — Kalvin's slow-path rationalisation thread (model expansion, S2 misfit expansion); unrelated to the LLMSupervisor, which owns reactive-scaffolding generation as a supervisor participant
- `@specs/auto-tune.md` — the CLI supervisor as a peer decider; the KLine Display Object and Significance Object payload sub-shapes referenced below
- `@kalvin-vision.md` §The Dialog — the vision basis: the other agent "can decide what to do next — ratify, scaffold, correct, or submit new information"

## Definitions

### Reactive Decision

The choice made when Kalvin produces a proposal the Trainer cannot auto-ratify — a proposal not resolved by auto-countersign (structural match to a loaded expectation) or recurrence dedup. One of three answers: **ratify**, **scaffold**, or **continue** (see §Decision Answers).

There is no distinction between request types: every Kalvin request carries a proposal, and a proposal the Trainer cannot auto-ratify is escalated regardless of significance band. An S4 identity request for a signature the Trainer holds no kline for is the same decision as an S2/S3 that did not auto-countersign — the proposal (here `{X: []}`) is carried in the decision request, and the significance band is context, not a discriminator.

### Decider

The supervisor participant that owns reactive decisions for a session. A session launches exactly one decider (convention — see §One Decider). Implementations are medium-independent and share one contract: the TUI participant (human), the Slack participant (human), the CLI supervisor (pi, via the file protocol), and the LLMSupervisor (LLM). They differ only in the process that produces an answer; on the bus they are indistinguishable.

### Decision Request

The message the Trainer emits to the `supervisor` role when a reactive decision is needed. Always enriched with the context a decider requires. Contract in §API.

### Decision Gate

The Trainer's hold-and-replay mechanism. While a decision request is pending, the Trainer holds subsequent trainee events in an internal queue; the bus never blocks (the Trainer's handler returns immediately). On an answer the Trainer applies it and replays the held events. **Unconditional** — it applies in every session, regardless of which decider is launched. The gate is what makes an asynchronous decider (any bus-mediated participant) work: without it, a burst of trainee events would overtake the pending decision.

### Decision Answer

The message a decider sends to the `trainer` role to resolve a pending decision request. Contract in §API.

### LLMSupervisor

The LLMSupervisor is a supervisor participant — a peer of the TUI, Slack, and CLI participants — that resolves decision requests via an LLM. It replaces the previous inline `cogitate_fn` attachment inside the Trainer. Its reasoning internals (prompt construction, tool schema, response extraction, LLM client) are the LLMSupervisor pipeline (§LLMSupervisor Pipeline). It registers as role `supervisor` and answers via the decision-answer message like any decider.

## API

### Decision Request message

| Field | Value |
|-------|-------|
| `role` | `supervisor` |
| `action` | `ratify_request` |
| `message.proposal` | the proposal KValue (the request Kalvin made) |
| `message.query` | the expectation kline |
| `message.significance` | the proposal's significance |
| `message.misfit` | `{underfit, overfit, underfit_gap, overfit_mask}` — always present |
| `message.curriculum_context` | `{objective, approach, lesson_prose}` when available, else a legacy string — always present |

Precondition: the proposal did not auto-countersign and is not a within-lesson recurrence (`@specs/trainer-satisfaction.md`). The `misfit` and `curriculum_context` fields are no longer conditional on a delegation flag — they are emitted on every decision request so every decider receives the same context. The KLine Display Object and Significance Object sub-shapes are defined in `@specs/auto-tune.md`.

### Decision Answer message

| Field | Value |
|-------|-------|
| `role` | `trainer` |
| `action` | `supervisor_decision` |
| `message.decision` | `ratify` \| `scaffold` \| `continue` |
| `message.proposal` | the pending proposal (the one the decision request carried) |
| `message.text` | KScript source — present only when `decision` is `scaffold` |

Precondition: a decision request is pending. Postcondition: the Trainer applies the decision (see §Decision Answers), clears the pending marker, and replays held events.

### Trainer event handler (reactive branch)

On an S2/S3 event that does not auto-countersign and is not a recurrence: emit the decision request, arm the decision gate (set the pending marker; subsequent trainee events are held), and return. The Trainer performs no ratify, submit, or escalate action itself.

## Behavioural Rules

### Decision ownership

1. On a proposal the Trainer cannot auto-ratify, the Trainer emits exactly one decision request to the `supervisor` role, enriched with `misfit` and `curriculum_context`. The Trainer performs no `countersign`, `submit`, or escalation action itself for that proposal.
2. The LLMSupervisor, when launched, registers as a supervisor participant and resolves decision requests via the decision-answer message — identical in contract to the TUI, Slack, and CLI deciders. It is not wired inside the Trainer.
3. There is no Trainer-side reactive-round budget and no automatic `budget_exhaustion` or `low_confidence` escalation. A decider that cannot help answers `continue` or `ratify` as it sees fit; escalation as a Trainer-side mechanism is removed.

### Decision gate

4. While a decision request is pending, the Trainer holds subsequent trainee events (`ground`, `frame`, `error`, `drained`) in an internal queue; the bus does not block. The handler returns immediately after stashing a held event.
5. The decision-answer message is never held; it is processed immediately even while events are queued.
6. On a decision answer the Trainer applies it, clears the pending marker, and replays held events through its normal handler.
7. A replayed event that raises a new decision request re-arms the gate; remaining events stay held. This yields the multi-turn decision loop — one decision per held-stream segment.
8. The gate is unconditional: it arms on every decision request in every session, regardless of which decider (or whether a decider) is launched.

### Decision answers

9. `ratify` → the Trainer sends `{role: trainee, action: countersign, message: proposal}` (S1 ratification).
10. `scaffold` → the Trainer sends `{role: trainee, action: submit, message: text}` (the carried KScript), identical to a lesson submission.
11. `continue` → the Trainer skips the pending proposal; no bus effect.
12. A `scaffold` whose KScript fails to compile is reported back to the decider as an `error` event; the decider may retry with corrected source.

### Proposals the Trainer resolves (no escalation)

13. A proposal that structurally matches a loaded expectation is auto-countersigned by the Trainer; it never reaches a decider (`@specs/trainer-satisfaction.md`).
14. A within-lesson recurrence (the same proposal kline seen twice) is deduped structurally; it never reaches a decider (`@specs/trainer-satisfaction.md`).

### One decider

15. A session is expected to launch exactly one decider. The harness does not enforce this: a decision request is routed to all `supervisor` subscribers, and any of them may answer.
16. Behaviour with multiple deciders answering is undefined and out of scope. The one-decider convention matches today's usage (a session runs the TUI, or Slack, or the CLI supervisor, or the LLMSupervisor — not several).

### LLMSupervisor Pipeline

When the LLMSupervisor receives a decision request, it builds a prompt from the request's `misfit` and `curriculum_context`, calls its LLM client, extracts a scaffold answer, sanitises it, and emits a decision answer. The sanitisation and decompilation mechanics are specified here; the prompt wording and LLM client are implementation details.

17. The LLMSupervisor's system prompt MUST document only the KScript syntax the lexer supports: identifiers are uppercase letters only (A–Z) with no hex literals; the only operators are `==` (countersign), `=>` (canonize), `=` (undersign), `>` (relationship); comments are parenthesised `(...)` only.
18. The misfit summaries passed to the LLM MUST be decompiled to human-readable KScript (e.g. `M > H`), not hex repr; on decompilation failure they fall back to repr.
19. The LLMSupervisor MUST sanitise scaffolding source by removing `#`-prefixed comment lines before attempting compilation. If sanitisation removes all content, it MUST produce no scaffolding (answering `continue` or `ratify`) without attempting compilation.

## Test Matrix

| ID | Criterion | Origin ref |
|----|-----------|------------|
| SD-1 | A proposal the Trainer cannot auto-ratify emits exactly one enriched `ratify_request` (`misfit` + `curriculum_context` always present); the Trainer emits no `countersign`/`submit`/escalation for it | §Decision ownership |
| SD-2 | The LLMSupervisor is not wired inside the Trainer; when launched it registers as a `supervisor` participant and answers via `supervisor_decision` | §Decision ownership |
| SD-3 | No reactive-round budget is tracked; no `budget_exhaustion` or `low_confidence` escalation is ever emitted by the Trainer | §Decision ownership |
| SD-4 | While a decision request is pending, trainee events (`ground`/`frame`/`error`/`drained`) are held and the bus does not block | §Decision gate |
| SD-5 | A `supervisor_decision` is processed immediately even while events are held | §Decision gate |
| SD-6 | On a decision answer the Trainer applies it, clears the pending marker, and replays held events | §Decision gate |
| SD-7 | A replayed event that raises a new decision request re-arms the gate, yielding a multi-turn loop | §Decision gate |
| SD-8 | The gate arms on every decision request regardless of which decider is launched (no delegation flag branches the behaviour) | §Decision gate |
| SD-9 | `ratify` → `{trainee, countersign, proposal}` | §Decision answers |
| SD-10 | `scaffold` → `{trainee, submit, text}` | §Decision answers |
| SD-11 | `continue` → no bus effect (skip) | §Decision answers |
| SD-12 | A `scaffold` with invalid KScript yields an `error` event back to the decider | §Decision answers |
| SD-13 | A structurally matching proposal is auto-countersigned and never reaches a decider | §Proposals the Trainer resolves |
| SD-14 | A within-lesson recurrence is deduped structurally and never reaches a decider | §Proposals the Trainer resolves |
| SD-15 | A decision request is routed to all `supervisor` subscribers; the harness does not enforce a single decider | §One decider |
| SD-16 | The LLMSupervisor system prompt contains no hex literal syntax (rule 17) | §LLMSupervisor Pipeline |
| SD-17 | The LLMSupervisor system prompt contains no invalid operators `~>`, `<-`, `->` (rule 17) | §LLMSupervisor Pipeline |
| SD-18 | LLMSupervisor misfit summaries are decompiled to KScript (rule 18) | §LLMSupervisor Pipeline |
| SD-19 | LLMSupervisor misfit summaries fall back to repr on decompilation failure (rule 18) | §LLMSupervisor Pipeline |
| SD-20 | The LLMSupervisor strips `#`-prefixed comment lines before compilation (rule 19) | §LLMSupervisor Pipeline |
| SD-21 | Scaffolding that is entirely comments produces no scaffolding without compilation (rule 19) | §LLMSupervisor Pipeline |

## Out of Scope

- Multiple concurrent deciders and arbitration between them — the one-decider convention (rule 15) is a usage rule, not a machine-enforced invariant; multi-decider behaviour is undefined.
- A decider-initiated "stuck" broadcast signal — a decider that cannot help answers `continue` or `ratify`; there is no separate escalation channel. Reintroducing one is a future concept.
- The LLMSupervisor's prompt wording, model choice, and LLM client internals beyond the sanitisation/decompilation contract in §LLMSupervisor Pipeline.
- Participant file and package locations, process architecture, and build phasing — plan-owned.
- The auto-tune codebase-tuning loop — unaffected; `@specs/auto-tune.md`.
- Changes to Kalvin, the harness, the message bus, or the WebSocket protocol.
- Goal-based curriculum generation — unaffected.
