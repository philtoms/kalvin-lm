---
type: source
title: Reentry redesigns break on departure paths
status: insight
category: architecture
created: 2026-11-19
updated: 2026-11-19
slug: cogitator-reentry-redesign-departure-paths
---

# Reentry redesigns break on departure paths

When the Cogitator's reentry was redesigned from fresh()/frame-length gating to seen-map candidate rotation, every *departure path* had to be re-derived — and the redesign only re-derived the happy one. Checklist for reviewing such a redesign:

1. **Every child outcome needs a parent continuation**: framed, stalled, exhausted — each must resume or rotate the parent. A missed outcome means the run's stack drains with no `run.finish`, and the harness silently records no step (symptom: "script N only completes step N−1"). The uniform rule that emerged: every child departure resumes the parent *in place*; rotation is just the parent's next turn meeting the next unseen candidate via `seen`. No separate backtrack primitive needed.
2. **A stall probe (C=None child) is a candidate** — it must mark `seen` (KNone sentinel) or re-push loops forever; a served stall must re-push *through the parent* (recomputing candidates from the grown domain), never re-run the bare probe.
3. **The root is special in every branch**: framed-at-root, tail-at-root — each needs the finish, not `stack.append(None)`.
4. **Component priority must fall through, not elif**: when U's push fails, O and F must still be tried (old code's independent ifs).
5. **Search-power tradeoff**: consuming candidates once (seen) vs fresh-gated re-pairing changes grounding outcomes — HEAD grounded wdmh via a late S3 exchange re-opening WDH×what; strict seen-consumption cannot (the pairing is burned). Diagnose by diffing the claim/stall lines against HEAD, not just exit status.

Debug technique: wrap `Cogitator._turn`/`_push` with a logging shim and run the harness Runner programmatically — the event stream is only flushed at `run.finish`, so vanished runs need live-tap instrumentation to see.

Related: [[sources/obs-2026-11-03-wdmh-step-4-trace-5-hlal-rows-one-reentry-11-pairings-elisio]], [[sources/obs-2026-10-30-cogitator-rationalise-rewritten-as-frame-propagating-reentry]]

*Category: architecture*

---
*Captured: 2026-11-19*

## Related

_Add links to related pages._
