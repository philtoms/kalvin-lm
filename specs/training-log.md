# Training Log Specification

## Overview

The training log provides a structured, human-readable server-side trace of training operations. Every significant event in the training pipeline — session lifecycle, lesson submission, rationalisation, satisfaction — is logged at an appropriate level with decompiled KScript source for readability. The log enables a developer to reconstruct what happened during a training run without needing to observe the session in real time.

## Dependencies

- `specs/harness-server.md` — Trainer, Reactor, KAgentAdapter, harness server
- `specs/agent.md` — rationalise events, significance levels
- `specs/kline.md` — `kline_display` for KLine → source rendering

## Definition

### Log Event Categories

| Category | Logger | Level | When |
|----------|--------|-------|------|
| Session start | `trainer.trainer` | INFO | Training session begins |
| Session end | `trainer.trainer` | INFO | Training session ends (not yet implemented) |
| Lesson submit | `trainer.trainer` | INFO | Lesson submitted to KAgent |
| Lesson compile | `trainer.trainer` | INFO | Entries compiled from lesson KScript |
| Lesson complete | `trainer.trainer` | INFO | All entries in a lesson satisfied |
| Curriculum complete | `trainer.trainer` | INFO | All lessons in curriculum submitted |
| Rationalise event | `trainer.trainer` | INFO | KAgent ground or frame event received |
| Supervisor decision applied | `trainer.trainer` | INFO | A pending decision resolved: ratify / scaffold / continue (SD-9/10/11) |
| Entry submit | `harness.adapter` | INFO | Compiled entries submitted to KAgent |
| Countersign | `harness.adapter` | INFO | Countersign sent to KAgent |
| Auto-countersign match | `trainer.reactor` | INFO | Proposal structurally matched expectation |
| Auto-countersign miss | `trainer.reactor` | DEBUG | No structural match found |
| Auto-countersign dup | `trainer.reactor` | DEBUG | Already-satisfied entry re-matched |
| Recurring proposal dropped | `trainer.reactor` | INFO | Within-lesson recurrence re-submitted at declared S4 (drop signal) (SD-14) |
| [removed] — Reactive scaffolding | — | — | no inline cogitation; the LLMSupervisor decides (SD-3) |
| [removed] — Cogitation failure | — | — | no inline cogitation (SD-3) |
| [removed] — Budget exhaustion | — | — | no reactive budget (SD-3) |
| [removed] — Escalation | — | — | no escalation mechanism (SD-3) |
| Compilation error | `harness.adapter` | ERROR | KScript compilation failed |

### Log Format

All log messages follow the Python logging standard format configured in `src/training/harness/__main__.py`:

```
<timestamp> [<logger_name>] <LEVEL>: <message>
```

### Rationalise Event Log Format

S1 fast-path events:
```
<KIND> <decompiled_query> → S1 (fast path)
<KIND> <decompiled_query> → S1 (fast path) ← <decompiled_proposal>
```

S2/S3 slow-path events:
```
<KIND> <decompiled_query> → <normalised_significance>
<KIND> <decompiled_query> → <normalised_significance> | proposal: <decompiled_proposal>
```

Where:
- `KIND` is `GROUND` or `FRAME` (uppercased)
- `decompiled_query` and `decompiled_proposal` are produced by `kalvin.kline.kline_display`
- `normalised_significance` is `significance / D_MAX`, formatted to 2 decimal places

### Lesson Submit Log Format

```
Submitting lesson <label> (<progress>/<total>)
```

### Lesson Complete Log Format

```
Lesson <label> complete — entries: <satisfied>/<submitted> satisfied, <lessons_done>/<lessons_total> lessons done
```

### Auto-Tune Harness Log Capture

When running under auto-tune, the harness server's stderr is redirected to `<session_dir>/harness.log`. This captures all Python logging output from the harness process.

## Behavioural Rules

### Trainer Logging

1. On session start, log the total lesson count and curriculum file path at INFO level.
2. On lesson submission, log the lesson label and progress (n/total) at INFO level.
3. On lesson submission, log the raw KScript source at DEBUG level.
4. After compilation, log the number of compiled entries at INFO level.
5. On each KAgent event, decompile the query and (if present) the proposal for log readability. If decompilation fails, fall back to `repr()`.
6. S1 events log with `→ S1 (fast path)` and the proposal (if any).
7. S2/S3 events log with the normalised significance and the proposal (if any).
8. On lesson completion, log the satisfaction counts at INFO level.
9. On curriculum completion, log at INFO level.
10. On applying a supervisor decision (ratify / scaffold / continue) to a pending proposal, log the applied action at INFO (`@specs/supervisor-decision.md` SD-9/10/11).

### Reactor Logging

11. On auto-countersign match, log at INFO level.
12. On auto-countersign miss (no match), log at DEBUG level.
13. On auto-countersign of an already-satisfied entry, log at DEBUG level.
14. On dropping a within-lesson recurrence (re-submitting at declared S4), log at INFO level (`@specs/supervisor-decision.md` SD-14).
15. [removed] — the Trainer no longer produces reactive scaffolding; the LLMSupervisor owns its own logging (`@specs/supervisor-decision.md`).
16. [removed] — no inline cogitation; the LLMSupervisor decides.
17. [removed] — no reactive budget (`@specs/supervisor-decision.md` SD-3).
18. [removed] — no escalation mechanism (`@specs/supervisor-decision.md` SD-3).

### Adapter Logging

19. On entry submission, log the count of compiled entries at INFO level.
20. On compilation error, log the error at ERROR level.
21. On countersign, log the KLine at INFO level.

### Auto-Tune Log Capture

22. `start_harness` redirects the harness subprocess stderr to `<session_dir>/harness.log`.
23. The log file is overwritten on each harness start (not appended across runs).

## Test Matrix

| ID | Criterion | Origin ref |
|----|-----------|------------|
| TL-1 | Session start logs lesson count and curriculum path | §Trainer Logging |
| TL-2 | Lesson submit logs label and progress | §Trainer Logging |
| TL-3 | Lesson submit logs KScript source at DEBUG level | §Trainer Logging |
| TL-4 | Compiled entry count logged after compilation | §Trainer Logging |
| TL-5 | S1 events log with decompiled query and "fast path" | §Trainer Logging |
| TL-6 | S2/S3 events log with normalised significance and decompiled proposal | §Trainer Logging |
| TL-7 | Decompilation failure falls back to repr() | §Trainer Logging |
| TL-8 | Lesson complete logs satisfaction counts | §Trainer Logging |
| TL-9 | Curriculum complete logged at INFO level | §Trainer Logging |
| TL-10 | Auto-countersign match logged at INFO | §Reactor Logging |
| TL-11 | Auto-countersign miss logged at DEBUG | §Reactor Logging |
| TL-12 | [removed] — reactive scaffolding logging; no inline cogitation (`@specs/supervisor-decision.md` SD-3) | §Reactor Logging |
| TL-13 | [removed] — cogitation failure logging; no inline cogitation (`@specs/supervisor-decision.md` SD-3) | §Reactor Logging |
| TL-14 | [removed] — budget exhaustion logging; no reactive budget (`@specs/supervisor-decision.md` SD-3) | §Reactor Logging |
| TL-15 | [removed] — escalation logging; no escalation mechanism (`@specs/supervisor-decision.md` SD-3) | §Reactor Logging |
| TL-16 | Entry submission logged with count at INFO | §Adapter Logging |
| TL-17 | Compilation error logged at ERROR | §Adapter Logging |
| TL-18 | Countersign logged with KLine at INFO | §Adapter Logging |
| TL-19 | Auto-tune start_harness captures stderr to harness.log | §Auto-Tune Log Capture |
| TL-20 | harness.log is overwritten (not appended) on start | §Auto-Tune Log Capture |
| TL-21 | Supervisor decision application (ratify/scaffold/continue) logged at INFO | §Trainer Logging |
| TL-22 | Within-lesson recurrence drop logged at INFO | §Reactor Logging |
| TL-23 | Auto-countersign of an already-satisfied entry logged at DEBUG | §Reactor Logging |

## Out of Scope

- Structured JSON logging — plain text is sufficient for developer tracing
- Log rotation or size management
- Configurable log levels via the harness config
- Logging from the supervisor, TUI, or Slack participant
- Session end log event (future work — the session end path has no INFO log yet)
