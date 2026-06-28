# Reactive Decision Delegation — Specification

## Overview

The Trainer's LLM agent (the Cogitator) owns one decision during a training session: what reactive scaffolding to write when Kalvin hits the slow path (an S2/S3 proposal that did not auto-countersign). This spec defines a flag that disables that decision and delegates it to a supervisor participant instead — so an auto-tune session can let pi act as the reactive agent without changing Kalvin, the harness, or the bus protocol.

When the flag is on (default), behaviour is unchanged: the Cogitator cogitates, submits scaffolding, and escalates on budget exhaustion or low confidence. When the flag is off, the Reactor does not cogitate, does not self-submit, and does not escalate — every unhandled S2/S3 is surfaced to the supervisor as a decision the supervisor must answer.

## Dependencies

- `@specs/harness-server.md` — Trainer participant, reactive mode, shared command protocol, supervisor message tables
- `@specs/cogitator.md` §Reactive Scaffolding Submission — Cogitator sanitisation and submission pipeline
- `@specs/auto-tune.md` — CLI supervisor event/command frames, per-session harness config generation

## Definitions

### The LLM Agent Flag

A Trainer configuration field that gates reactive cogitation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `trainer.llm.enabled` | `bool` | `true` | When `true`, the Cogitator is wired for reactive scaffolding (today's behaviour). When `false`, the Reactor enters **delegated mode** and defers every reactive decision to the supervisor. |

The flag is an additional gate on top of the existing `KALVIN_LLM_API_KEY` environment variable. When the flag is `false`, the Cogitator is not wired regardless of the API key, and the Reactor does not fall back to the no-client path (which today escalates `low_confidence` on every round).

### Delegated Mode

The Reactor's behaviour when `trainer.llm.enabled` is `false`. In delegated mode the Reactor:

- Still attempts auto-countersign on every S2/S3 proposal (the fast-path match is independent of the flag).
- Does **not** increment the reactive-round counter, does **not** call `cogitate_fn`, does **not** submit reactive scaffolding, and does **not** escalate.
- Returns `False` from `process_s2_s3` for any proposal that does not auto-countersign, so the Trainer emits a decision request (see below) and waits for the supervisor.

Delegated mode has no automatic budget escalation. The supervisor is the sole decision-maker; progress is bounded only by the supervisor's responses.

### Reactive-Round Budget (Default Mode)

When `trainer.llm.enabled` is `true` (the default), the Reactor tracks a
reactive-round counter per lesson with a budget of `max_reactive_rounds`
(default 5). Each S2/S3 proposal that reaches the Reactor increments the
counter:

- The first event to reach the budget escalates `budget_exhaustion` to the
  supervisor (one escalation per lesson).
- Every subsequent event while the counter is past the budget is **silently
  dropped** — no escalation, no logging, no bus messages. This prevents the
  Reactor from spinning on the event stream while the cogitator drains.

The counter resets at the start of each lesson.

### Decision Request

An enriched `ratify_request` carrying the context the supervisor needs to make the reactive decision — the same context the Cogitator's prompt consumes. The enrichment is added when the flag is `off`; it is optional and ignored by supervisor participants that do not use it.

| Field | Type | Description |
|-------|------|-------------|
| `proposal` | KLine Display Object | Kalvin's S2/S3 proposal (existing) |
| `query` | KLine Display Object | The expectation kline (existing) |
| `significance` | Significance Object | Significance of the proposal (existing) |
| `misfit` | `{underfit, overfit, underfit_gap, overfit_mask}` | Misfit diagnosis between query and proposal; `underfit`/`overfit` are `bool`, the gap/mask are the bit differences |
| `curriculum_context` | `{objective, approach, lesson_prose}` \| `str` | Pedagogical context for the current lesson; structured fields when available, else a legacy string |

The KLine Display Object and Significance Object formats are defined in `@specs/auto-tune.md` (§Significance Object, §KLine Display Object).

### Scaffold Command

A supervisor command that submits reactive scaffolding to Kalvin — the supervisor-side equivalent of the Cogitator's successful output. It maps to the same bus message the Reactor sends internally for reactive scaffolding.

| Layer | Form | Bus message |
|-------|------|-------------|
| Shared command parser (`src/training/participants/commands.py`) | free-text `scaffold:<kscript source>` | `{role: "trainee", action: "submit", message: <kscript>}` |
| Auto-tune command file | `{"action": "scaffold", "text": <kscript>}` | supervisor dispatches via the shared command parser → same bus message |

The `submit` action is interpreted by Kalvin's adapter exactly as for any lesson submission: the KScript is compiled and each entry rationalised. Compilation failures come back to the supervisor as `error` events (`@specs/harness-server.md` §Compilation Errors), so the supervisor receives feedback on invalid scaffolding without a separate validation path.

## Behavioural Rules

### Flag

1. `trainer.llm.enabled` defaults to `true`. When unset, behaviour is identical to today.
2. When `true` and `KALVIN_LLM_API_KEY` is set, the Cogitator is wired for reactive scaffolding (today's behaviour).
3. When `true` and `KALVIN_LLM_API_KEY` is unset, the Cogitator is not wired (today's behaviour — the no-client path).
4. When `false`, the Cogitator is not wired regardless of the API key, and the Reactor enters delegated mode.

### Delegated Mode

5. In delegated mode, the Reactor still auto-countersigns structurally matching S2/S3 proposals; the flag does not affect the fast path.
6. In delegated mode, an S2/S3 proposal that does not auto-countersign produces no cogitation, no scaffolding submission, and no escalation from the Reactor.
7. In delegated mode, the reactive-round budget is not consulted and budget-exhaustion escalation never fires. The supervisor is the sole decision-maker.
8. The Trainer emits a decision request (enriched `ratify_request`) for every S2/S3 proposal that does not auto-countersign when the flag is `off`, carrying the misfit diagnosis and curriculum context.

### Default-Mode Budget

14. The first S2/S3 event to reach `max_reactive_rounds` (default 5) in a lesson escalates `budget_exhaustion` to the supervisor.
15. Every subsequent S2/S3 event while past the budget is silently dropped (no escalation, no logging, no bus message).
16. The reactive-round counter resets at the start of each lesson.

### Supervisor Answers

9. On a decision request the supervisor answers with one of: `ratify` (accept the proposal), `scaffold` (write reactive scaffolding), or `continue` (skip and proceed).
10. The `scaffold` command submits KScript to Kalvin via the `trainee` `submit` action, identical to a lesson submission. The Trainer does not interpret or re-route scaffolded source.
11. A `scaffold` whose KScript fails to compile is reported back to the supervisor as an `error` event; the supervisor may retry with corrected source.

### Decision Gate (Delegated Mode)

The supervisor is the **gating** decision-maker in delegated mode — "progress is bounded only by the supervisor's responses" is a runtime guarantee, not an aspiration. When the Trainer emits a `ratify_request` it sets a pending-decision marker and **holds** subsequent KAgent events (`ground`/`frame`/`error`/`drained`) in an internal queue until the supervisor replies. The bus never blocks: the Trainer's message handler stashes held events and returns immediately, so the harness event loop (a single-threaded dispatch) cannot deadlock.

The supervisor's answer arrives as a `supervisor_decision` message addressed to the `trainer` role, carrying `{decision, proposal, text?}`:

| decision | Trainer action | Bus effect |
|----------|----------------|-----------|
| `ratify` | countersign the pending proposal | `{trainee, countersign, proposal}` → S1 ratification (KP-2) |
| `scaffold` | submit the carried KScript | `{trainee, submit, text}` |
| `continue` | skip the pending proposal | none |

On receipt the Trainer applies the decision, clears the pending marker, and **replays** the held events through its normal handler. A replayed event that raises a new `ratify_request` re-arms the gate (remaining events stay held), yielding the multi-turn delegated loop: one decision = one held-stream segment, repeated until the hold drains and the curriculum can advance. This is why a delegated run yields multiple sequential `ratify_request`s for a single lesson rather than fanning them out as a burst the supervisor cannot keep up with.

The CLI supervisor routes `ratify`/`scaffold`/`continue` to the `trainer` role as `supervisor_decision` when a proposal is pending (the Trainer applies the countersign itself, avoiding double-routing via the trainee). The shared `commands.py` parser is unchanged, so the Slack/TUI human flow keeps its direct trainee-countersign path for non-gated (default-mode) use. A `continue` with no pending proposal is a no-op acknowledge, as before.

### Auto-Tune Integration

12. Auto-tune writes `trainer.llm.enabled: false` into the per-session harness config it generates, so an auto-tune session always runs in delegated mode with pi as the reactive decision-maker.
13. Auto-tune's command file accepts the `scaffold` action; the CLI supervisor dispatches it through the shared command parser. The `ratify`/`scaffold`/`continue` actions are routed to the `trainer` role as `supervisor_decision` when a proposal is pending (§Decision Gate).
14. Auto-tune launches the harness and supervisor subprocesses with `KALVIN_DATA_DIR` pointing at the main checkout's `data/` when the session lives in a git worktree (`<main-repo>/.worktrees/...`), since the worktree's `data/` (tokenizer BPE engine, model binary) is gitignored build artifacts. `kalvin.paths` documents this env var as the worktree escape hatch.

## Test Matrix

| ID | Criterion | Origin ref |
|----|-----------|------------|
| RD-1 | `trainer.llm.enabled` defaults to `true`; unset config behaves as today | §Flag |
| RD-2 | Flag `true` + API key set wires the Cogitator (unchanged behaviour) | §Flag |
| RD-3 | Flag `false` does not wire the Cogitator even when the API key is set | §Flag |
| RD-4 | Flag `false`: Reactor auto-countersigns structurally matching S2/S3 proposals (fast path unaffected) | §Delegated Mode |
| RD-5 | Flag `false`: a non-matching S2/S3 proposal produces no cogitation, no scaffolding submission, no escalation | §Delegated Mode |
| RD-6 | Flag `false`: the reactive-round counter is not incremented and budget-exhaustion escalation never fires | §Delegated Mode |
| RD-7 | Flag `false`: every non-matching S2/S3 emits a decision request carrying `misfit` and `curriculum_context` | §Delegated Mode |
| RD-7a | Flag `false`: a decision request whose query entry is already satisfied still reaches the supervisor (the Trainer's `is_satisfied(query)` skip is scoped to non-delegated mode) | §Delegated Mode |
| RD-7b | The `misfit` and `curriculum_context` fields survive supervisor-side event enrichment (written to `events.jsonl` verbatim when present) | §Delegated Mode |
| RD-8 | Flag `true` + no API key: existing no-client behaviour unchanged (escalates `low_confidence`) | §Flag |
| RD-8a | Default mode: first event reaching `max_reactive_rounds` escalates `budget_exhaustion` | §Reactive-Round Budget (Default Mode) |
| RD-8b | Default mode: events past the budget are silently dropped (no escalation/log/bus message) | §Reactive-Round Budget (Default Mode) |
| RD-9 | `scaffold:<kscript>` parses to a command that sends `{trainee, submit, <kscript>}` | §Scaffold Command |
| RD-10 | `scaffold` command compiles and submits via Kalvin's adapter like any `submit` | §Supervisor Answers |
| RD-11 | A `scaffold` with invalid KScript yields an `error` event back to the supervisor | §Supervisor Answers |
| RD-12 | Auto-tune's per-session harness config sets `trainer.llm.enabled: false` | §Auto-Tune Integration |
| RD-13 | Auto-tune's command file accepts `{"action": "scaffold", "text": <kscript>}` and dispatches via the shared parser | §Auto-Tune Integration |
| RD-14 | Delegated mode: while a `ratify_request` is pending, the Trainer holds KAgent events and the bus does not block; a `supervisor_decision` reply resolves it and replays held events | §Decision Gate |
| RD-15 | A replayed held event that raises a new `ratify_request` re-arms the gate, yielding a multi-turn delegated loop | §Decision Gate |
| RD-16 | `supervisor_decision` `{ratify}` emits `{trainee, countersign, proposal}`; `{scaffold}` emits `{trainee, submit, text}`; `{continue}` is a skip | §Decision Gate |
| RD-17 | The CLI supervisor routes `ratify`/`scaffold`/`continue` to the `trainer` role as `supervisor_decision` when a proposal is pending | §Auto-Tune Integration |

## Out of Scope

- Goal-based curriculum generation — unaffected by this flag; out of scope for auto-tune (`@specs/auto-tune.md` §Out of Scope).
- A new bus action for scaffolding — `scaffold` reuses the existing `trainee` `submit` action.
- A new event type — delegation enriches the existing `ratify_request`; no parallel event stream.
- Automatic timeout/budget escalation in delegated mode — the supervisor is the sole decision-maker.
- Changes to Kalvin, the Cogitator prompt, or the reactive-scaffolding sanitisation pipeline (`@specs/cogitator.md` §Reactive Scaffolding Submission).
- A decision gate for default (non-delegated) mode — the Cogitator's reactive round is the default-mode equivalent; only delegated mode uses the explicit hold/replay gate.
- Reduction of the multi-turn decision count (e.g. auto-satisfying a query's remaining proposals after one ratification) — a future tuning lever, not part of the delegation contract.
