---
type: source
title: "Real-stack overlay: position-free turns enable interleaving"
status: insight
category: architecture
created: 2026-11-18
updated: 2026-11-18
slug: real-stack-overlay-position-free-turns
---

# Real-stack overlay: position-free turns enable interleaving

Rewriting Kalvin's Cogitator from a parent-chain walk (the running entry held in a register, `pop()` walking `entry.parent`) onto a real LIFO stack surfaced the property that makes interleaving sound: **a turn must read only the stack's top entry and that entry's run state — never the entry's position**.

The working discipline (see [[sources/obs-2026-11-18-real-stack-cogitator-verified-behaviour-identical]]):
- Peek, don't pop-to-register. The top entry runs; `push` appends the child and leaves the parent suspended *beneath* it; `pop` backtracks to whatever resumes on top. Resume needs no saved continuation — the resumed entry simply re-evaluates its current candidate, because freshness (`at` map vs frame length) decides re-descent.
- Push-without-descent is fatal, not a style choice: if the parent keeps running after a push it will consume the candidate before the child's frames arrive, and the re-evaluation loop (the heart of the machine) can never fire. This is the single defect that made the sketched version produce zero claims.
- All per-run state (frame, counter, stalls, events, ask/goal) rides on the entries via a per-run _State; ancestry bars stay per-entry immutable snapshots. Then runs are position-free by construction and can share one stack — demonstrated by dev/probes/interleave_stack.py: 3 scenarios (2 domains) round-robined by segment rotation, byte-identical per-run streams, claims grounding mid-interleave at the same within-run event indices.

Generalization: any deterministic machine whose step function is (state-on-node × neighborhood-free) can have its implicit recursion overlaid onto an explicit shared stack. The verification recipe that made this cheap: byte-diff the full harness trace tables (mhall.ks, wdmh.ks) against the pre-rewrite HEAD — behavior preservation is a diff, not an argument. Beware baseline swaps poisoning diffs: copy the baseline in, run, restore, and md5-verify which version produced each capture.

*Category: architecture*

---
*Captured: 2026-11-18*

## Related

_Add links to related pages._
