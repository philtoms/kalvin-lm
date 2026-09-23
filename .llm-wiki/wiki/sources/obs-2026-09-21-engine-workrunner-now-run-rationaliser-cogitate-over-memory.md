---
type: source
title: "Observation: Engine+WorkRunner now run rationaliser/cogitate over Memory"
tags:
  - architecture
  - engine
  - work-runner
  - memory
  - convergence
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-engine-workrunner-now-run-rationaliser-cogitate-over-memory
relevance: high
observed_at: 2026-09-21T17:54:15.010Z
source_context: Engine+work_runner migration onto rationaliser/cogitator/Memory
---

# ⭐ Observation: Engine+WorkRunner now run rationaliser/cogitate over Memory

Engine and WorkRunner converged onto the dialogue architecture (uncommitted on `dialogue`): Engine.rationalise is now the Rationaliser's code — fast path grounds S1-stamped receipts into Memory (silent: no "ground"/fast-path "frame" events anymore) and refuses S4; everything else queues on the Memory work list, and each queued kline is submitted to the WorkRunner (WorkItem query|candidate|level tuple is gone — backlog is plain KLines). WorkRunner._run_work_item now runs ONE cogitate pass (kalvin.cogitator.cogitate) over the shared Memory per item and routes each emission to WorkHandler.on_emission(query_kline, emission); Engine publishes each as a "frame" RationaliseEvent with the query voice at structurally derived significance. Both modules hold Memory (state) instead of Model; Engine serialization is now the Memory JSON snapshot (save/load/from_dict/to_dict via Memory; AgentCodec/bin format dropped; old kalvin.bin saves incompatible). Engine gained rebind(state) (swaps state under rationaliser+runner); model property replaced by state. Adapter: _handle_load rebuilds via Memory.load + engine.rebind; STM pre-registration block dropped (Model.is_countersigned mechanism gone); _EngineLike protocol updated (codec gone). encode_text: --format dropped. Verified: 95 tests, mhall byte-identical, training imports ok, live smoke — 7 queued items → 7 cogitate passes, forced emission → frame event, save/load/rebind/countersign round-trips. Orphaned from the runtime path now: Model, agent_codec, expand, proposals. Nuance: runner runs N passes for N queued items (not until-stalled) — settle semantics slightly weaker than the callers' reentry loop.

*Relevance: high*
*Context: Engine+work_runner migration onto rationaliser/cogitator/Memory*
*Tags: architecture engine work-runner memory convergence*

---
*Observed: 2026-09-21T17:54:15.010Z*
