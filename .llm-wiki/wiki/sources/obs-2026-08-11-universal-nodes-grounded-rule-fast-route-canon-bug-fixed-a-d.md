---
type: source
title: "Observation: Universal nodes-grounded rule; fast-route canon bug fixed; a:[Det] unblocks"
tags:
  - engine
  - grounding
  - fast-route
  - ltm
status: observation
created: 2026-08-11
updated: 2026-08-11
slug: obs-2026-08-11-universal-nodes-grounded-rule-fast-route-canon-bug-fixed-a-d
relevance: high
observed_at: 2026-08-11T17:16:34.723Z
source_context: "Lean harness: new universal grounding rule (all nodes grounded before signature)"
---

# ⭐ Observation: Universal nodes-grounded rule; fast-route canon bug fixed; a:[Det] unblocks

New grounding rules in the lean engine (engine.py fast_route + engine_state.py _is_groundable), per the user's spec:

1. UNIVERSAL: a signature grounds only once every one of its nodes is in LTM.
2. Fast-route S1 canon grounds only if all nodes grounded (else slow-routes to await them) — closes the latent fast-route bug where canons grounded without their nodes.
3. Fast-route S1 identity grounds unconditionally (self-referential {S:[S]}, chicken-and-egg resolved by self-reference).

`_is_groundable` collapsed to: `is_identity -> True; else all(node in ltm for node in kline.nodes)`. The old branches (single-node-relationship reciprocal check, general misfit signature-in-ltm check) are gone — subsumed by the universal rule. Critically this means a single-node CONNOTES like `a:[Det]` now grounds as soon as `Det` (its node) is in LTM — K acquires the new word `a` by its class. This was the key unblock: `a:[Det]` -> `a` in LTM -> `ALL:[a,little,lamb]` groundable -> `ALL:[Object]` resolves -> `MHALL:[SVO]`/`SVO:[MHALL]` reciprocal countersign pair completes.

Result on mhall: 20 groundings (same raw count as baseline), BUT strictly better quality. Baseline had 2 SPURIOUS groundings — `DH:[did,have]` and `WDMH:[what,did,Mary,have]` (canons grounded via the fast-route bug without their nodes `did`/`have`/`what` in LTM) — and was MISSING `a:[Det]`. New rules: `a:[Det]` grounds (+1 legitimate), the two spurious canons correctly refuse to ground (-2), net 20 but all legitimate. Remaining work_list residue (`L:[Mod]`, `DH:[did,have]`, `WDMH:[what,did,Mary,have]`) is genuine: `Mod`/`did`/`have`/`what` are MTS-expanded leaf words never offered as identities.

Cogitate loop also hardened: converted from `for idx in range(...)` to a `while idx >= 0` with an in-bounds re-check, because the broader _is_groundable triggers more _promote cascades that remove arbitrary work-list entries mid-pass (was throwing IndexError). _incoming list removed from engine.py (dead state — only rationalise.py reads it).

Files: src/dialogue/engine.py, src/dialogue/engine_state.py, CONTEXT.md (Grounding glossary: removed "canon self-grounds" claim), docs/behaviour-notes.md (routing + grounding rules rewritten). All 1220 tests pass.

*Relevance: high*
*Context: Lean harness: new universal grounding rule (all nodes grounded before signature)*
*Tags: engine grounding fast-route ltm*

---
*Observed: 2026-08-11T17:16:34.723Z*
