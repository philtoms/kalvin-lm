# Realign Training Roles Plan

**Status:** documentation cascade complete; implementation (T1–T6) pending.
**Spec refs:** `@specs/supervisor-decision.md` — SD-1 through SD-21
**Affects:** `@specs/cogitator.md`, `@specs/auto-tune.md`, `@specs/harness-server.md`, `@specs/trainer-satisfaction.md`, `@specs/training-log.md`, `CONTEXT.md`

## What Changes

The reactive decision moves out of the Trainer and into the supervisor role, unconditionally. Four coupled changes follow from one thesis ("the decider is always a supervisor participant"):

1. **The LLMSupervisor becomes a supervisor participant** — a separate-process WebSocket client, a peer of the TUI, Slack, and CLI participants. It replaces the inline `cogitate_fn` auto-wired inside the Trainer constructor. (The relocating component was previously called "the Cogitator" in the trainer code; renamed to **LLMSupervisor** to disambiguate it from Kalvin's Cogitator — `src/kalvin/cogitator.py`, the slow-path rationalisation thread, which is untouched.)
2. **The decision gate becomes unconditional.** Today it exists only in "delegated mode" (the `delegate_reactive` / `trainer.llm.enabled` path). With the decider always bus-mediated and therefore asynchronous, the hold/replay gate is always correct — and the `if delegate_reactive` branches around it are removed.
3. **The Trainer stops deciding reactively.** It surfaces decisions (always-enriched `ratify_request`), gates while pending, and applies answers. The Reactor sheds its reactive-round budget, cogitation call, and escalation paths — leaving auto-countersign + recurrence + return-false-to-surface. Trainer-side escalation is removed entirely.
4. **Auto-tune's supervisor splits out.** The CLI supervisor (`supervisor.py`, `events.py`, and the file protocol) moves out of `auto_tune/` to sit beside the TUI and Slack participants. `auto_tune/` retains only the codebase-tuning loop. The per-session harness-config generation that forced `trainer.llm.enabled: false` becomes vestigial — auto-tune simply does not launch an LLMSupervisor participant.

Side effect fixed for free: the LLMSupervisor no longer runs on the bus dispatch thread, so an LLM call no longer blocks all harness traffic (a latent bug in the inline model).

The flag `trainer.llm.enabled` and the `delegate_reactive` parameter are removed. There is one model, not two.

## Spec References

- `@specs/supervisor-decision.md` §Decision ownership — SD-1, SD-2, SD-3
- `@specs/supervisor-decision.md` §Decision gate — SD-4 through SD-8
- `@specs/supervisor-decision.md` §Decision answers — SD-9 through SD-12
- `@specs/supervisor-decision.md` §Proposals the Trainer resolves — SD-13, SD-14
- `@specs/supervisor-decision.md` §One decider — SD-15
- `@specs/supervisor-decision.md` §LLMSupervisor Pipeline — SD-16 through SD-21

## Implementation Tasks

### Task T1: LLMSupervisor as a supervisor participant (SD-2)

- **Spec ref:** @specs/supervisor-decision.md §Decision ownership (LLMSupervisor), §LLMSupervisor Pipeline
- **New files:** `src/training/supervisors/llm_supervisor.py` (class `LLMSupervisor`)
- **Relocated (complete):** the prompt/extraction/sanitisation pipeline now lives in `src/training/supervisors/llm_supervisor.py` (the `LLMSupervisor` class plus `Cogitator`, `build_prompt`, `build_tool_definitions`, `extract_result`, `_strip_hash_comments`, `CogitationRequest`/`MisfitInfo`/`CogitationResult`) — folded into the supervisor that owns it, not imported from the trainer. The shared LLM transport (`LLMClient`/`LLMResponse`/`OpenAICompatibleClient`, the GLM-5.1 client) is extracted to `src/training/harness/llm.py` so the LLMSupervisor pipeline and the Trainer's `CurriculumGenerator` share it without either importing from the other. The old `src/training/trainer/cogitation.py` is deleted. The SD-16…21 sanitisation contract is owned by `@specs/supervisor-decision.md`.
- **Details:**
  - A WebSocket client participant mirroring `src/training/supervisors/auto_tune/supervisor.py` (pre-T6 location) / the relocated CLI supervisor structure: connect, register as `supervisor`, receive frames, on `ratify_request` build a prompt from the frame's `misfit` + `curriculum_context`, call the LLM, extract + sanitise the scaffold, emit a `supervisor_decision` answer (`ratify`/`scaffold`/`continue` derived from the result).
  - Runs as a separate process (D3.a). Lifecycle helper added peer to the CLI supervisor's `start_supervisor`/`stop_supervisor`.
  - The large `_cogitate_adapter` closure currently inside `Trainer.__init__` (misfit display, `CogitationRequest` assembly) moves here as participant logic. Note: misfit *computation* (`_compute_misfit`) and curriculum-context derivation stay on the Trainer (it owns curriculum knowledge); the LLMSupervisor *consumes* the enriched fields the decision request carries.

### Task T2: Remove the inline LLMSupervisor wiring and the flag (SD-1, SD-3, SD-8)

- **Spec ref:** @specs/supervisor-decision.md §Decision ownership
- **Files:** `src/training/trainer/trainer.py`, `src/training/harness/__main__.py`
- **Details:**
  - Delete the `cogitate_fn` and `delegate_reactive` parameters from `Trainer.__init__`; delete the `_cogitate_adapter` closure and the auto-wire block. **Keep `llm_client`** — it is still used by `CurriculumGenerator` for goal resolution (`trainer.py:663`), which is a one-shot startup transform, not an in-the-loop reactive decision. Goal resolution is unaffected by D1.a.
  - Delete `_resolve_llm_wiring` (and its `trainer.llm.enabled` reading) from `src/training/harness/__main__.py`; `_build_llm_client` remains and is wired directly to `llm_client` (no delegation mode). The `trainer.llm.enabled` config key is no longer read. The LLMSupervisor is launched (or not) as a participant, configured like any other client participant.
  - `training.harness.yaml` gains an optional participant entry for the LLMSupervisor (client class), replacing the `trainer.llm` section.

### Task T3: Generalise the decision gate; always enrich the request (SD-1, SD-4 through SD-8)

- **Spec ref:** @specs/supervisor-decision.md §Decision gate, §API
- **Files:** `src/training/trainer/trainer.py`, `src/training/trainer/reactor.py`
- **Details:**
  - In `Trainer._handle_kagent_event`, remove the `if self._delegate_reactive` guard around misfit/context computation and around arming `_pending_decision`: the enrichment and the gate now run on every proposal the Trainer cannot auto-ratify (per G3 — there is no request/proposal distinction; an ungrounded S4 identity request is escalated like any other unresolvable proposal).
  - The held-event queue, `_handle_supervisor_decision`, and the replay loop already implement SD-4/5/6/7; they become the only path (no parallel inline path).

### Task T4: Strip the Reactor to surface-and-gate support (SD-1, SD-3, SD-13, SD-14)

- **Spec ref:** @specs/supervisor-decision.md §Proposals the Trainer resolves; ownership rules
- **Files:** `src/training/trainer/reactor.py`
- **Details:**
  - `Reactor.process_s2_s3` becomes: auto-countersign? → `True`; recurrence? → dedup and `True`; else → `False` (let the Trainer surface the decision request and arm the gate).
  - Delete `_handle_reactive`, `_cogitate`, `_check_budget`, `_escalate`, the `cogitate_fn`/`max_reactive_rounds`/`delegate_reactive` parameters, and the `Action` dataclass (already unused). The `notify`/escalation emission path is removed.
  - Auto-countersign and recurrence logic are unchanged (SD-13, SD-14). This removes the reactive-round budget and silent-drop-past-budget behaviour that previously guarded the inline cogitation loop.

### Task T5: Unify decision-answer routing (SD-9, SD-10, SD-11, SD-12)

- **Spec ref:** @specs/supervisor-decision.md §Decision answers
- **Files:** `src/training/supervisors/commands.py`, `src/training/supervisors/cli_supervisor.py` (post-T6), `src/training/supervisors/slack_agent.py`, `src/training/supervisors/tui_client.py`
- **Details:**
  - `ratify`/`scaffold`/`continue` always route to the `trainer` role as `supervisor_decision` when a proposal is pending — the CLI supervisor's `_route_delegated_decision` becomes the universal path, not a delegated-mode branch.
  - The non-gated direct-to-trainee `countersign` path for `ratify` (used by the Slack/TUI human flow in the old default mode) is removed; the Trainer applies every decision. `commands.py` `parse_command` is simplified accordingly. This unifies human and non-human deciders (SD-8 requires it: bypassing the gate would allow double-countersign if two supervisors answered).
  - The CLI supervisor's `continue`-with-no-pending-proposal no-op acknowledge is retained (SD-11 edge).

### Task T6: Move the CLI supervisor out of auto_tune

- **Spec ref:** @specs/auto-tune.md §CLI Supervisor (peer decider); @specs/supervisor-decision.md §Definitions (Decider)
- **Files:** move `src/training/supervisors/auto_tune/supervisor.py` and `src/training/supervisors/auto_tune/events.py` → `src/training/supervisors/cli_supervisor.py` and `src/training/supervisors/cli_events.py` (names tentative); update `src/training/supervisors/__init__.py` exports and `src/training/supervisors/auto_tune/lifecycle.py` import paths
- **Details:**
  - The CLI supervisor is a peer of `tui_client.py` and `slack_agent.py`; it is not part of the codebase-tuning loop and leaves `auto_tune/`.
  - `auto_tune/` retains `session.py`, `lifecycle.py`, `orchestrate.py`, `snapshots.py`, `cli.py`, `__main__.py`.
  - In `lifecycle._generate_session_harness_config`, delete the block that forces `trainer.llm.enabled: false` (vestigial after T2). Auto-tune's "no LLMSupervisor" property is now expressed by not launching one, not by a config flag.

## Test Mapping

| Spec ID | Test file | Status |
|---------|-----------|--------|
| SD-1 | `tests/test_trainer.py` | ☐ (extend; assert enrichment always present, no self-decision) |
| SD-2 | `tests/test_cogitation.py` + new `tests/test_llm_supervisor.py` | ☐ |
| SD-3 | `tests/test_reactor.py` | ☐ (remove budget/escalation tests; assert no escalation emitted) |
| SD-4 | `tests/test_trainer.py` (gate tests, formerly delegated-mode) | ☐ (de-flag existing tests) |
| SD-5 | `tests/test_trainer.py` | ☐ |
| SD-6 | `tests/test_trainer.py` | ☐ |
| SD-7 | `tests/test_trainer.py` | ☐ |
| SD-8 | `tests/test_trainer.py` | ☐ (gate fires regardless of decider) |
| SD-9 | `tests/test_trainer.py` + `tests/test_reactive_scaffolding.py` | ☐ |
| SD-10 | `tests/test_trainer.py` + `tests/test_reactive_scaffolding.py` | ☐ |
| SD-11 | `tests/test_trainer.py` | ☐ |
| SD-12 | `tests/test_reactive_scaffolding.py` | ☐ |
| SD-13 | `tests/test_reactor.py` (existing auto-countersign tests) | ✅ (unchanged) |
| SD-14 | `tests/test_reactor.py` (existing recurrence tests) | ✅ (unchanged) |
| SD-15 | `tests/test_trainer.py` / harness routing test | ☐ |
| SD-16–21 | `tests/test_cogitation.py` (sanitisation/decompilation; relocate with the LLMSupervisor) | ☐ |

Existing tests asserting the `delegate_reactive` flag, the two-mode budget, or Trainer-side escalation are removed or rewritten against the unified model. `tests/test_cascade_control.py` is removed (the reactive-round budget is gone). `tests/test_cogitator_drain.py` may be obsoleted entirely (the drain was an inline-Cogitator concern; Kalvin's Cogitator drain is unaffected and owned by `@specs/cogitator.md`).

## Design Decisions

**D1.a — Reactive scaffolding is a supervisor responsibility (resolved).** The decider is always a supervisor participant; the Trainer surfaces and gates, never decides. Rationale: honours the harness's "all participants are equal" principle, makes the LLMSupervisor and human/pi deciders true peers, and dissolves the `trainer.llm.enabled` flag and its dual code paths. Accepted by the user.

**D2.a — One decider per session by convention (resolved).** The harness does not enforce single-decider; a session is expected to launch exactly one. Rationale: matches today's usage exactly and avoids primary-decider arbitration machinery. Multi-decider is a follow-on (out of scope, SD-15). Accepted by the user.

**D3.a — The LLMSupervisor runs as a separate process (resolved).** A WebSocket client participant, launched like the CLI supervisor. Rationale: cleanest isolation and reuses the existing subprocess lifecycle pattern; as a side effect it removes the latent bus-blocking bug (an inline LLM call today runs on the single-threaded bus dispatch and stalls all traffic). Accepted by the user.

**G1 — Two-Cogitator collision resolved by rename (resolved).** "Cogitator" properly belongs to Kalvin's slow-path rationalisation thread (`src/kalvin/cogitator.py`, `@specs/cogitator.md`); the LLM scaffolding cogitator (`src/training/trainer/cogitation.py`) was the interloper and is renamed **LLMSupervisor**; it relocated into `src/training/supervisors/llm_supervisor.py` (its pipeline folded in), the old `src/training/trainer/cogitation.py` is deleted, and the shared LLM transport moved to `src/training/harness/llm.py`. The RS-1…4 sanitisation rules move out of `cogitator.md` into `@specs/supervisor-decision.md` §LLMSupervisor Pipeline (G1b). `cogitator.md` is now solely about Kalvin's Cogitator.

**G2 — harness-server.md keeps bus-message facts only (resolved).** The Trainer section keeps what actions it emits and to which roles; reactive-mode and escalation bullets are removed; HRNS-13/14 are `[removed]`-tombstoned. The decision contract is owned by `@specs/supervisor-decision.md`; the loop model by `@specs/trainer-satisfaction.md`. Ratify routing unified through the Trainer (`supervisor_decision`) for all deciders — HRNS-27/34 reworded accordingly.

**G3 — Request = proposal (resolved).** There is no distinction between request types: every Kalvin request carries a proposal, and a proposal the Trainer cannot auto-ratify is escalated regardless of significance band. The S4 ungroundable-identity stall (`trainer-satisfaction.md` rules 16–17, `ungroundable_request`) folds into SD-1; those rules are `[removed]`. The significance band is context in the payload, not a discriminator.

**Misfit/context computation stays on the Trainer (T3 note).** The Trainer emits the decision request, so it computes the enrichment; the LLMSupervisor consumes the enriched fields rather than recomputing them. Single source of truth for misfit diagnosis.

## Documentation Cascade — COMPLETE

All steps executed:

1. **`CONTEXT.md`** — glossary revised via a focused grill. Net −3 entries: deleted **Escalation** (old stuck-signal noun), **Trainer Modes**, **Fast Path**, **Slow Path** (the latter three duplicated the significance bands with different words). Added **Escalation** (redefined: the Trainer deferring a proposal it cannot auto-ratify). Revised **Ratify** (folds in auto-ratify + supervisor path), **Scaffolding** and **Reactive Scaffolding** ("by the supervisor"). **Trainer**, **Pre-compiled Scaffolding**, **Goal** confirmed accurate, untouched.
2. **`@specs/supervisor-decision.md`** — new spec written (SD-1…21), then broadened per G3 (request=proposal) and given §LLMSupervisor Pipeline per G1b.
3. **`@specs/cogitator.md`** — §Reactive Scaffolding Submission removed (RS rules relocated to supervisor-decision.md); AGT-49…57 `[removed]`-tombstoned pointing to SD-16…21; now solely about Kalvin's Cogitator.
4. **`@specs/auto-tune.md`** — rule 7a (flag-forcing) retired; CLI supervisor noted as peer decider; event/command-frame rows updated.
5. **`@specs/harness-server.md`** — reactive-mode/escalation bullets removed; §S2/S3 Auto-Countersign Suppression → §Auto-Ratify and Escalation; SAC budget clauses removed; ratify routing unified (`supervisor_decision` to trainer); HRNS-13/14/46/49 tombstoned, HRNS-18/27/31/34/37 reworded.
6. **`@specs/trainer-satisfaction.md`** — stall rules 16–17 `[removed]` (folded into SD-1); TS-13 tombstoned; out-of-scope updated.
7. **`@specs/training-log.md`** — budget/escalation logging rules 13–16 `[removed]`.

## Status

- **Phase:** documentation cascade complete. Implementation in progress: the LLMSupervisor pipeline relocation (T1, file move) is complete; T2–T6 pending.
- **Build order:** T2+T3+T4 as one Trainer-refactor change; then T1 (LLMSupervisor relocation) and T6 (CLI supervisor relocation) as isolated moves; T5 (routing unification) last, touching all supervisors.
- **Blockers:** none.
