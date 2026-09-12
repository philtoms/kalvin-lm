---
type: source
title: "Observation: ks2.md vs harness-path code: 8 deviations found, spec-faithful code dormant"
tags:
  - ks2
  - conformance
  - review
  - dialogue
  - harness
  - formalisation
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-ks2-md-vs-harness-path-code-8-deviations-found-spec-faithful
relevance: high
observed_at: 2026-09-12T10:08:35.808Z
source_context: Deviation review of ks2.md normative spec vs source code via the harness entry app
---

# ⭐ Observation: ks2.md vs harness-path code: 8 deviations found, spec-faithful code dormant

Reviewed the kalvin system through the harness entry (dialogue/harness.py → engine.py → Cogitator → engine_state/kline/significance/signifier/ks-compiler) against docs/ks2.md (4th pass). Top contradictions: (1) ask is a decree bit ASK_BPE_TOKEN=1<<63 (signifier.py:45) read operationally (engine.py:143), while ks2 §4/§13 forbids marks — plain s:[] unknowns are refused on receipt; (2) coverage implemented as containment node_in (n&s)==n (signifier.py:105, sig_level kline.py:371) not overlap, so single-node Overfit a:[ab] and Under+over ab:[bc] band S3 not S2; (3) selection is signature overlap (Cogitator._candidates cogitator.py:289, signifies) — exactly what Def 16 excludes; (4) cogitator.py:50-51 passes raw hop count as the significance byte (inverted semantics: exact fill grades 0x00/S4), while the faithful γ (gamma_aggregate, acq_depth=D̄+Ĥ, delta knob) exists only in dormant kalvin/expand.py / pivot_fill.py — Ĥ recorded+persisted but never priced on the entry path; (5) is_canon (kline.py:280) lacks Def 13's well-foundedness n∉ν_K. Also: no derivation states/licence table/T1-T2 bounds in the active strategy; scoping clause (misfit_mass) only in dormant pivot_fill; node-identity matching instead of atom-level misfit; compiler stamps CANONIZES/COUNTERSIGNS→S2 vs §13's S1-once-solved, MTS→S1 has no §13 basis; classifier runs on two domains (signature_of full 64-bit vs residual word-masked) breaking Def 9's g=e=∅ iff s=signature_of(ν). Engine currently constructs only Cogitator (ExpandFit/Reentry/PivotFill commented out); dormant Reentry changes the signature mid-derivation, violating Def 12. Conformances affirmed: KValue significance-blind equality, countersign protocol, harness queue freedom, word-bit space = a₀..a₃₀ with ask bit outside A.

*Relevance: high*
*Context: Deviation review of ks2.md normative spec vs source code via the harness entry app*
*Tags: ks2 conformance review dialogue harness formalisation*

---
*Observed: 2026-09-12T10:08:35.808Z*
