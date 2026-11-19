---
type: source
title: "Observation: Bare `==` RHS now stamps ask goal — mhall-ag.ks runs"
tags:
  - kscript
  - compiler
  - fix
  - goal
  - harness
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-bare-rhs-now-stamps-ask-goal-mhall-ag-ks-runs
relevance: high
observed_at: 2026-11-18T17:06:08.117Z
source_context: Compiler-side fix for bare `== SVO` RHS goal stamping
---

# ⭐ Observation: Bare `==` RHS now stamps ask goal — mhall-ag.ks runs

Compiler-side fix landed in src/ks/ast_emitter.py (_process_scope_body): a COUNTERSIGNS ask now falls back to the first bare Signature item in scope.items when no OperatorScope RHS exists, stamping dbg.goal from it (item.id / item.inline_annotation — note a bare Signature IS the sig, no .sig wrapper; first attempt used item.sig.id and AttributeError'd). `(Mary had a little lamb) == SVO` now compiles ask MHALL with goal='SVO'; the goal kline is the compound canon SVO:[S,V,O] (auto-derived), vs the authored SVO:[Subject,Verb,Object] from the `=>` body form. dev/dialogue/harness.py now runs mhall-ag.ks (exit 0, one scenario, both sides stall at S4 — empty domain beyond the word identities). Regression test added: test_countersigns_bare_rhs_stamps_the_goal in tests/test_ask_marker.py. Full suite: 160 passed, 16 xfailed.

*Relevance: high*
*Context: Compiler-side fix for bare `== SVO` RHS goal stamping*
*Tags: kscript compiler fix goal harness*

---
*Observed: 2026-11-18T17:06:08.117Z*
