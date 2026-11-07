---
type: source
title: "Pseudocode audit: three false lines, the re-run invisible; corrected in docstring"
status: insight
category: architecture
created: 2026-11-03
updated: 2026-11-03
slug: pseudocode-audit-three-false-lines-recheck-invisible
---

# Pseudocode audit: three false lines, the re-run invisible; corrected in docstring

Audited the original six-term pseudocode against the shipped Cogitator (src/rationalisation/cogitator.py). Verdict: the skeleton was right, but three lines had become FALSE and one load-bearing mechanism was INVISIBLE — the truth lived in the C1–C7 footnotes, not the code. (1) `if not cands: frame(E)` contradicted both the implementation and its own comment — a stall waits, and only a cross-parent second stall mints the joint frame (the exchange). (2) The unconditional `break` after ground is the exact measured bug that starved the wdmh exchange — only the claim stops the run. (3) `ground` frames the grounded region (resolutions are frames) — absent. (4) The deepest gap: as literally written, the for-loop runs once, so `U_resolved` can never flip after a push — resolution actually happens by the RE-RUN: an entry that pushed children stays on the stack and re-executes its for-loop when they drain, re-reading framed(U)/framed(O); plus the exhaust-pop (grounds nothing, pushes nothing → pops) which is also what makes the loop terminate. Corrected pseudocode now in the module docstring — every line maps 1:1 to the loop (stall-waits/exchange, ground-frames, claim-only stop, exhaust-pop, and the re-check stated as a trailing comment); the footnotes shrank to true fine print (candidate rules, goal-alone opening, MAX_DEPTH, boundary rules). Lesson for spec-pseudocode: any behaviour that only exists as a footnote to a code block will drift — the pseudocode must either show the loop or lie.

*Category: architecture*

---
*Captured: 2026-11-03*

## Related

_Add links to related pages._
