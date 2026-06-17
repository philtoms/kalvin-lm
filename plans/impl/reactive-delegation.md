# Reactive Decision Delegation — Implementation Plan

**Status:** not started
**Spec refs:** `specs/reactive-delegation.md`

## Spec References

- `@specs/reactive-delegation.md` — RD-1 through RD-13 (owns this feature)
- `@specs/harness-server.md` — Trainer participant, shared command protocol, supervisor message tables (table-row additions only)
- `@specs/auto-tune.md` — CLI supervisor event/command frames, per-session harness config (rule 7a, table-row additions)
- `@specs/reactive-scaffolding.md` — Cogitator sanitisation pipeline (unchanged)

## Implementation Tasks

### Task 1: Flag wiring in the harness entry point (`src/training/harness/__main__.py`)

- **Spec ref:** @specs/reactive-delegation §Flag (RD-1, RD-2, RD-3)
- **Test mapping:** `tests/test_harness_main.py` → `TestBuildLLMClient`, `TestTrainerFactoryLLMWiring`
- **Details:**
  - Read `trainer.llm.enabled` from `trainer_cfg`, default `True`.
  - In `trainer_factory`: if `enabled` is `False`, do not call `_build_llm_client` (pass `llm_client=None`) and pass a `delegate_reactive=True` flag to `Trainer`.
  - If `enabled` is `True`, today's path: build the client from the env var (may still be `None`), pass `delegate_reactive=False`.
  - Pseudocode:
    ```
    llm_enabled = bool(trainer_cfg.get("llm", {}).get("enabled", True))
    llm_client = _build_llm_client(trainer_cfg) if llm_enabled else None
    Trainer(..., llm_client=llm_client, delegate_reactive=not llm_enabled)
    ```

### Task 2: Reactor delegated mode (`src/training/trainer/reactor.py`)

- **Spec ref:** @specs/reactive-delegation §Delegated Mode (RD-4, RD-5, RD-6)
- **Test mapping:** `tests/test_reactor.py` → new `TestDelegatedMode`
- **Details:**
  - Add `delegate_reactive: bool = False` to `Reactor.__init__`.
  - In `process_s2_s3`: after `_auto_countersign` returns `False`, if `delegate_reactive` is `True`, return `False` immediately **without** calling `_handle_reactive` (no round increment, no cogitate, no escalation).
  - `cogitate_fn` may be `None` in delegated mode; `_handle_reactive` is never reached so the `None` path is irrelevant.
  - Pseudocode:
    ```
    def process_s2_s3(self, event):
        if self._auto_countersign(event.proposal):
            return True
        if self._delegate_reactive:
            return False          # defer to supervisor; no side effects
        self._handle_reactive(event)
        return False
    ```

### Task 3: Trainer delegates + emits enriched decision request (`src/training/trainer/trainer.py`)

- **Spec ref:** @specs/reactive-delegation §Delegated Mode (RD-7), §Flag (RD-8)
- **Test mapping:** `tests/test_trainer.py` → new `TestDelegatedReactiveDecisions`
- **Details:**
  - Add `delegate_reactive: bool = False` to `Trainer.__init__`; forward to `Reactor`.
  - When `delegate_reactive` is `True`, skip the Cogitator auto-wiring block entirely (do not construct `_cogitate_adapter`), regardless of `llm_client`. The Reactor will not cogitate.
  - In `_handle_kagent_event`, the `ratify_request` send already exists for non-auto-matched S2/S3. Extend its payload with `misfit` and `curriculum_context` when `delegate_reactive` is `True`:
    - Reuse the misfit computation from `_cogitate_adapter` (`classify_misfit`, `make_signature`, gap/mask).
    - `curriculum_context`: derive `{objective, approach, lesson_prose}` from the current lesson if available, else a legacy string. (Mirror what `CogitationRequest` would carry.)
  - RD-8 (no-client path unchanged when flag on): ensure `delegate_reactive=False` + `llm_client=None` still wires the `None` cogitate_fn and escalates `low_confidence` as today — i.e. only `delegate_reactive=True` suppresses escalation.

### Task 4: Scaffold command (`src/training/participants/commands.py`)

- **Spec ref:** @specs/reactive-delegation §Scaffold Command (RD-9, RD-10, RD-11)
- **Test mapping:** `tests/test_commands.py` → new `TestScaffoldCommand`
- **Details:**
  - Add `ScaffoldCommand(Command)` with `text: str`; `to_messages` returns `[(TRAINEE_ROLE, "submit", self.text)]`.
  - In `parse_command`: recognise `scaffold:` / `scaffold ` prefix (case-insensitive); text after the prefix is the KScript source (may be multi-line). Place the rule before the file-path heuristic so multi-line KScript is not misclassified.
  - RD-10/RD-11: the produced `submit` is handled by Kalvin's adapter exactly as any lesson submit (HRNS-7); compile failures return as `error` events (HRNS-8) — no new validation path. Assert the bus message in `to_messages`; the compile-error round-trip is already covered by existing adapter tests.

### Task 5: CLI supervisor scaffold dispatch (`src/training/participants/auto_tune/supervisor.py`)

- **Spec ref:** @specs/reactive-delegation §Auto-Tune Integration (RD-13)
- **Test mapping:** `tests/test_auto_tune_supervisor.py` (extend command-dispatch tests)
- **Details:**
  - In `_process_command`, handle `action == "scaffold"`: reconstruct `f"scaffold:{cmd['text']}"` and pass through `parse_command`, mirroring the existing `goal` action handler. Dispatch the resulting messages.

### Task 6: Auto-tune sets the flag off (`src/training/participants/auto_tune/lifecycle.py`)

- **Spec ref:** @specs/reactive-delegation §Auto-Tune Integration (RD-12), @specs/auto-tune rule 7a
- **Test mapping:** `tests/test_auto_tune_lifecycle.py` → `TestSessionHarnessConfig`
- **Details:**
  - In `_generate_session_harness_config`: after loading the project `harness.yaml` into `data`, set `data.setdefault("trainer", {})["llm"] = {"enabled": False}` (merge without clobbering existing `llm.base_url`/`model` overrides).
  - Pseudocode:
    ```
    trainer = data.setdefault("trainer", {})
    llm = trainer.setdefault("llm", {})
    llm["enabled"] = False
    ```

### Task 7: Document the flag in project config (`harness.yaml`)

- **Spec ref:** @specs/reactive-delegation §Flag
- **Details:**
  - Add a commented `enabled: true` line under `trainer.llm` in `harness.yaml` noting that auto-tune overrides it to `false` per session.

## Test Mapping

| Spec ID | Test file | Test function / class | Status |
|---------|-----------|-----------------------|--------|
| RD-1 | `tests/test_harness_main.py` | `TestBuildLLMClient` (default-true case) | ☐ |
| RD-2 | `tests/test_harness_main.py` | `TestTrainerFactoryLLMWiring` (flag on + key wires Cogitator) | ☐ |
| RD-3 | `tests/test_harness_main.py` | `TestTrainerFactoryLLMWiring` (flag off does not wire, even with key) | ☐ |
| RD-4 | `tests/test_reactor.py` | `TestDelegatedMode.test_auto_countersign_still_runs` | ☐ |
| RD-5 | `tests/test_reactor.py` | `TestDelegatedMode.test_no_cogitate_no_submit_no_escalate` | ☐ |
| RD-6 | `tests/test_reactor.py` | `TestDelegatedMode.test_no_round_increment_no_budget_escalation` | ☐ |
| RD-7 | `tests/test_trainer.py` | `TestDelegatedReactiveDecisions.test_enriched_ratify_request` | ☐ |
| RD-8 | `tests/test_trainer.py` | existing no-client path (assert `delegate_reactive=False` still escalates `low_confidence`) | ☐ |
| RD-9 | `tests/test_commands.py` | `TestScaffoldCommand.test_parses_and_maps_to_submit` | ☐ |
| RD-10 | `tests/test_commands.py` | `TestScaffoldCommand.test_to_messages_trainee_submit` | ☐ |
| RD-11 | `tests/test_commands.py` | `TestScaffoldCommand` (note: compile-error round-trip covered by adapter HRNS-8 tests) | ☐ |
| RD-12 | `tests/test_auto_tune_lifecycle.py` | `TestSessionHarnessConfig.test_sets_llm_enabled_false` | ☐ |
| RD-13 | `tests/test_auto_tune_supervisor.py` | scaffold action dispatch test | ☐ |

## Design Decisions

1. **Flag is additive over the env-var gate, not a replacement.** `trainer.llm.enabled=false` suppresses the Cogitator even when `KALVIN_LLM_API_KEY` is set, and — critically — suppresses the no-client `low_confidence` escalation that today misrepresents "cogitation disabled." The env var continues to gate client construction when the flag is on.

2. **Delegated mode is an explicit Reactor state, not "cogitate_fn is None."** The existing `None` path escalates `low_confidence` every round; delegated mode must do nothing. A dedicated `delegate_reactive` flag keeps the two semantics distinct and avoids overloading the absence of a client.

3. **Reuse `ratify_request`, do not add a new event type.** The Trainer already emits `ratify_request` for every non-auto-countersigned S2/S3. Enriching its payload (misfit, curriculum context) in delegated mode avoids a parallel event stream and keeps the supervisor's per-event blocking loop unchanged.

4. **`scaffold` reuses the `trainee` `submit` action.** It is the exact bus message the Reactor sends for reactive scaffolding today, so Kalvin, the adapter, and the compile-error feedback path are reused verbatim — no new bus action, no new validation.

5. **No automatic budget/timeout in delegated mode.** The supervisor (pi) is the sole decision-maker. A hang guard is unnecessary for auto-tune, where pi is always attentive; pi abandons a decision by sending `continue`.

6. **Auto-tune owns the flag flip, not the user.** `_generate_session_harness_config` always writes `trainer.llm.enabled: false`, so an auto-tune session is always delegated. The project `harness.yaml` default stays `true` so normal (human-supervised) sessions are unaffected.

## Status

- Specs written: `specs/reactive-delegation.md`; table-row additions to `specs/harness-server.md` and `specs/auto-tune.md` (rule 7a).
- Implementation: not started.
