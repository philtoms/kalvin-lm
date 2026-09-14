---
type: source
title: "Observation: Derivation engine port committed (73e4eba)"
tags:
  - derivation
  - engine
  - commit
  - def17
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-derivation-engine-port-committed-73e4eba
relevance: medium
observed_at: 2026-09-14T22:34:09.051Z
source_context: Committing the derivation engine port with review fixes
---

# 🔍 Observation: Derivation engine port committed (73e4eba)

Committed 73e4eba "kalvin: derivation port into engine; anchor refinement + composed dedup; wedge regression test" on branch algebra (working tree clean). Contents: user's port (src/dialogue/derivation.py new, engine.py wired to Derivation via _select, abstract.py/signifier.py KSig typing) + my review fixes (anchor refinement _refine_anchor with occurrence requirement in _walk_b, _ground_composed dedup in both dialogue and kalvin cores, KSig label-drop fix, derive() docstring placement) + tests/test_derivation.py regression test_b_walk_arrival_without_resolution_does_not_ground. All 44 tests green. Open items on record: kalvin/dialogue derivation near-duplication (consolidation decision), Engine._select reusing Def 16 occurrence for goal enumeration (doc/engine vocabulary collision, needs CONTEXT.md or doc note eventually).

*Relevance: medium*
*Context: Committing the derivation engine port with review fixes*
*Tags: derivation engine commit def17*

---
*Observed: 2026-09-14T22:34:09.051Z*
