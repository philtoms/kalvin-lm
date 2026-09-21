---
type: source
title: "Observation: Def 1 honored in engine: measure/units/same_content on KSignifier; derivation.py fully seam-routed"
tags:
  - engine
  - refactoring
  - values
  - seam
status: observation
created: 2026-09-20
updated: 2026-09-20
slug: obs-2026-09-20-def-1-honored-in-engine-measure-units-same-content-on-ksigni
relevance: critical
observed_at: 2026-09-20T19:06:14.525Z
source_context: Routing WORD_BITS/_atom_bits through the KSignifier seam
---

# 🔴 Observation: Def 1 honored in engine: measure/units/same_content on KSignifier; derivation.py fully seam-routed

Engine now honors kalvin-algebra Def 1 (opaque value space): KSignifier gained measure (μ, abstract), units (ledger granularity for Defs 18–19, abstract — disjoint unit values composing to value, one per word bit in NLPSignifier), and same_content (concrete default: residual(a,b)==0 and residual(b,a)==0 — content equality). NLPSignifier implements measure = (v & _TYPE_MASK).bit_count() and units = per-word-bit KNode generator. significance.py: word_atom_count is now a thin seam delegate (_SIGNIFIER = NLPSignifier() singleton, the one chosen-realisation point); misfit_mass reimplemented as the Def 1 derived form μ(a∨b) − μ(a∧b); WORD_BITS retained only as the documented reference-realisation constant (hop.py's scope trawl reads it directly); atom-weighted → content-weighted vocabulary synced with the doc. derivation.py is now fully seam-routed: _atom_bits deleted, acq ledger keyed by signifier.units values, masked containment (excess & v & WORD_BITS == excess) → residual emptiness, _jaccard via signifier.measure, ASK_SIG import gone. SEMANTIC TIGHTENING (flagged): usable() self-witness guard and the _walk "never bridge to s" filter previously compared & ~ASK_SIG exact values (BPE cargo participated); both now use same_content (content equality, BPE cargo inert) — the Def 1-honoring reading: a kline carrying s's words is carrying s. All 70 tests pass unchanged. Remaining leaks (follow-ups): hop.py WORD_BITS reach-composition in the scope trawl; dialogue/derivation.py's own _atom_bits copy; dialogue/harness & engine_state & ~ASK_SIG sites.

*Relevance: critical*
*Context: Routing WORD_BITS/_atom_bits through the KSignifier seam*
*Tags: engine refactoring values seam*

---
*Observed: 2026-09-20T19:06:14.525Z*
