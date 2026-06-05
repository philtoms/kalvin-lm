# Training Log Implementation Plan

**Status:** implemented (auto-tune session: training-log, run 1)
**Spec refs:** `specs/training-log.md`

## Spec References

- `@specs/training-log.md` — TL-1 through TL-20
- `@specs/harness-server.md` — Trainer, Reactor, KAgentAdapter

## Implementation Tasks

### Task 1: Trainer logging (src/trainer/trainer.py)

- **Spec ref:** @specs/training-log §Trainer Logging, TL-1 through TL-9
- **Details:**
  - Import `Decompiler` from `kscript.decompiler`
  - `_start_session()`: log lesson count and curriculum path (TL-1)
  - `_submit_next_lesson()`: log lesson label and progress (TL-2), log KScript at DEBUG (TL-3), log compiled entry count (TL-4)
  - `_handle_kagent_event()`: decompile query and proposal for log output; log S1 as "fast path" (TL-5), log S2/S3 with normalised significance (TL-6); fallback to repr on decompilation failure (TL-7)
  - `_check_lesson_complete()`: log satisfaction counts (TL-8)
  - `_submit_next_lesson()` curriculum-complete branch: log at INFO (TL-9)

### Task 2: Reactor logging (src/trainer/reactor.py)

- **Spec ref:** @specs/training-log §Reactor Logging, TL-10 through TL-15
- **Details:**
  - `_auto_countersign()`: log match at INFO (TL-10), log miss at DEBUG (TL-11), log duplicate at DEBUG (TL-11)
  - `_handle_reactive()`: log scaffolding with round/confidence/source at INFO (TL-12), log cogitation failure at WARNING (TL-13), log budget exhaustion at WARNING (TL-14)
  - `_escalate()`: log reason at ERROR (TL-15)

### Task 3: Adapter logging (src/harness/adapter.py)

- **Spec ref:** @specs/training-log §Adapter Logging, TL-16 through TL-18
- **Details:**
  - `_handle_submit()`: log entry count at INFO (TL-16), log compilation error at ERROR (TL-17)
  - `_handle_countersign()`: log KLine at INFO (TL-18)

### Task 4: Auto-tune harness log capture (src/participants/auto_tune/lifecycle.py)

- **Spec ref:** @specs/training-log §Auto-Tune Log Capture, TL-19, TL-20
- **Details:**
  - `start_harness()`: redirect stderr to `harness.log` instead of DEVNULL (TL-19)
  - Open file in write mode (overwrites on each start) (TL-20)

## Test Mapping

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| TL-1 | test_training_log.py | test_session_start_log | ☐ |
| TL-2 | test_training_log.py | test_lesson_submit_log | ☐ |
| TL-3 | test_training_log.py | test_lesson_submit_debug_kscript | ☐ |
| TL-4 | test_training_log.py | test_compiled_entry_count_log | ☐ |
| TL-5 | test_training_log.py | test_s1_fast_path_log | ☐ |
| TL-6 | test_training_log.py | test_s2_s3_significance_log | ☐ |
| TL-7 | test_training_log.py | test_decompile_fallback_repr | ☐ |
| TL-8 | test_training_log.py | test_lesson_complete_log | ☐ |
| TL-9 | test_training_log.py | test_curriculum_complete_log | ☐ |
| TL-10 | test_training_log.py | test_auto_countersign_match_log | ☐ |
| TL-11 | test_training_log.py | test_auto_countersign_miss_debug_log | ☐ |
| TL-12 | test_training_log.py | test_reactive_scaffolding_log | ☐ |
| TL-13 | test_training_log.py | test_cogitation_failure_warning | ☐ |
| TL-14 | test_training_log.py | test_budget_exhaustion_warning | ☐ |
| TL-15 | test_training_log.py | test_escalation_error_log | ☐ |
| TL-16 | test_training_log.py | test_entry_submit_count_log | ☐ |
| TL-17 | test_training_log.py | test_compilation_error_log | ☐ |
| TL-18 | test_training_log.py | test_countersign_log | ☐ |
| TL-19 | test_training_log.py | test_harness_log_capture | ☐ |
| TL-20 | test_training_log.py | test_harness_log_overwrite | ☐ |

## Design Decisions

1. **Decompile in the Trainer, not the Reactor** — the Trainer owns the event pipeline and already imports the Decompiler. The Reactor deals with match/action logic and shouldn't need decompilation.
2. **Fallback to repr() on decompilation failure** — decompilation can fail for novel or edge-case KLines. The log must never throw; repr() is always safe.
3. **Proposal shown on S1 when present** — some S1 events carry a proposal (e.g., countersign results). Logging it makes the trace complete.
4. **Scaffolding source truncated to 100 chars** — reactive scaffolding can be long. First 100 chars gives context without flooding the log.
5. **harness.log as stderr redirect** — Python logging goes to stderr by default. Redirecting stderr captures everything without changing the logging configuration.

## Auto-Tune Session Evidence

- Session: `auto-tune/training-log`
- Baseline: `runs/001/` (before logging changes)
- Observed output: `auto-tune/training-log/harness.log` (after logging changes)
- Curriculum: `curricula/first-steps.md` (3 lessons, all fast-path S1 + one S3 escalation)

## Status

Implementation complete. Tests not yet written.
