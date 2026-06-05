# Training Log Specification

## Overview

The training log provides a structured, human-readable server-side trace of training operations. Every significant event in the training pipeline — session lifecycle, lesson submission, rationalisation, satisfaction, escalation — is logged at an appropriate level with decompiled KScript source for readability. The log enables a developer to reconstruct what happened during a training run without needing to observe the session in real time.

## Dependencies

- `specs/harness-server.md` — Trainer, Reactor, KAgentAdapter, harness server
- `specs/agent.md` — rationalise events, significance levels
- `specs/kscript.md` — decompiler for KLine → source

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
| Entry submit | `harness.adapter` | INFO | Compiled entries submitted to KAgent |
| Countersign | `harness.adapter` | INFO | Countersign sent to KAgent |
| Auto-countersign match | `trainer.reactor` | INFO | Proposal structurally matched expectation |
| Auto-countersign miss | `trainer.reactor` | DEBUG | No structural match found |
| Auto-countersign dup | `trainer.reactor` | DEBUG | Already-satisfied entry re-matched |
| Reactive scaffolding | `trainer.reactor` | INFO | Scaffolding generated from cogitation |
| Cogitation failure | `trainer.reactor` | WARNING | Cogitation produced no scaffolding |
| Budget exhaustion | `trainer.reactor` | WARNING | Reactive rounds exhausted |
| Escalation | `trainer.reactor` | ERROR | Escalation sent to supervisor |
| Compilation error | `harness.adapter` | ERROR | KScript compilation failed |

### Log Format

All log messages follow the Python logging standard format configured in `src/harness/__main__.py`:

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
- `decompiled_query` and `decompiled_proposal` are produced by `kscript.decompiler.Decompiler`
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

### Reactor Logging

10. On auto-countersign match, log at INFO level.
11. On auto-countersign miss (no match), log at DEBUG level.
12. On auto-countersign of an already-satisfied entry, log at DEBUG level.
13. On reactive scaffolding, log the round number, confidence, and first 100 chars of scaffolding source at INFO level.
14. On cogitation failure (no scaffolding produced), log at WARNING level.
15. On reactive budget exhaustion, log the round count at WARNING level.
16. On escalation, log the reason and detail at ERROR level.

### Adapter Logging

17. On entry submission, log the count of compiled entries at INFO level.
18. On compilation error, log the error at ERROR level.
19. On countersign, log the KLine at INFO level.

### Auto-Tune Log Capture

20. `start_harness` redirects the harness subprocess stderr to `<session_dir>/harness.log`.
21. The log file is overwritten on each harness start (not appended across runs).

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
| TL-12 | Reactive scaffolding logged with round, confidence, source | §Reactor Logging |
| TL-13 | Cogitation failure logged at WARNING | §Reactor Logging |
| TL-14 | Budget exhaustion logged at WARNING with round count | §Reactor Logging |
| TL-15 | Escalation logged at ERROR with reason | §Reactor Logging |
| TL-16 | Entry submission logged with count at INFO | §Adapter Logging |
| TL-17 | Compilation error logged at ERROR | §Adapter Logging |
| TL-18 | Countersign logged with KLine at INFO | §Adapter Logging |
| TL-19 | Auto-tune start_harness captures stderr to harness.log | §Auto-Tune Log Capture |
| TL-20 | harness.log is overwritten (not appended) on start | §Auto-Tune Log Capture |

## Out of Scope

- Structured JSON logging — plain text is sufficient for developer tracing
- Log rotation or size management
- Configurable log levels via harness.yaml
- Logging from the supervisor, TUI, or Slack participant
- Session end log event (future work — the session end path has no INFO log yet)
