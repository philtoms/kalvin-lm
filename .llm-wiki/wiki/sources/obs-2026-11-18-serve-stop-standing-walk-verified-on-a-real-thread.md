---
type: source
title: "Observation: serve()/stop() standing walk verified on a real thread"
tags:
  - cogitator
  - threading
  - serve
  - condition
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-serve-stop-standing-walk-verified-on-a-real-thread
relevance: high
observed_at: 2026-11-18T16:30:06.783Z
source_context: Implementing while True + condition-wait serve loop on the Cogitator
---

# ⭐ Observation: serve()/stop() standing walk verified on a real thread

Cogitator now has a standing server: serve() is the forever-loop the threaded harness wants — while True, waiting on threading.Condition(threading.RLock()) when the stack is empty, woken by rationalise()'s append+notify_all (both under the condition's lock, so no lost wakeup). stop() sets the flag and notifies; the serve drains what remains, then returns — verified serve-alive=False after join. The turn body was extracted to _turn(); cogitate() remains the synchronous drain (with cond, while stack: _turn()); one driver per stack at a time (cogitate or serve, never both). RLock is load-bearing: sinks run on the server thread mid-turn while the lock is held — a sink that pushes a new run (rationalise) must not deadlock; verified by the probe's serve preempt-sink case (run2 pushed from run1's claim sink ON the server thread, both streams identical to solo). Also verified serve preempt-wait (main-thread push lands mid-walk via the condition). Probe: dev/probes/interleave_stack.py; tests 159+16 xfail; mhall/wdmh byte-identical to HEAD. Not yet committed. NEXT FORK: converting the dialogue harness itself to serve — the blocker is step attribution of sinks: _claim/_stall attribute to Runner._step_index which the harness thread mutates while the server fires sinks for earlier steps (claims carry the ask signature so they attribute by lookup; stalls are bare ⟨σ,[]⟩ and cannot). present() also needs steps sorted by index since end_step fires out of order, and _step must stop using len(steps) as the index.

*Relevance: high*
*Context: Implementing while True + condition-wait serve loop on the Cogitator*
*Tags: cogitator threading serve condition*

---
*Observed: 2026-11-18T16:30:06.783Z*
