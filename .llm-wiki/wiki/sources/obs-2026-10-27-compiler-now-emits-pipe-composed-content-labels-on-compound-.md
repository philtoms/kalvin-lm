---
type: source
title: "Observation: Compiler now emits pipe-composed content labels on compound sigs"
tags:
  - trs
  - labels
  - compiler
  - meet
status: observation
created: 2026-10-27
updated: 2026-10-27
slug: obs-2026-10-27-compiler-now-emits-pipe-composed-content-labels-on-compound-
relevance: high
observed_at: 2026-10-27T17:38:18.656Z
source_context: "Fixing meet label extraction (Option A: compiler-emitted content labels)"
---

# ⭐ Observation: Compiler now emits pipe-composed content labels on compound sigs

Chose Option A: compiler now emits content labels. NLPSignifier.signature_of composes labels via KNode.merge ('|'-union) instead of "".join concatenation (src/kalvin/signifier.py). TokenEncoder: compound-sig branch and _compose_concat dropped .with_label(identifier) — composed label rides through; compound-ref nodes carry the referenced compound's composed label (recursive, e.g. ALLQuery → 'a|little|lamb|Query'); the ASK marker branch carries the composed label instead of clobbering with entry.sig. Authored identifiers survive in _compound_sigs/_compound_labels and KDbg.label, so display is unchanged. Verified: meet(KNode(0x4200000000,'had|Verb'), KNode(0xE000000000,'Subject|Verb|Object')) → KNode(0x4000000000,'Verb'); residue → 'had' / 'Subject|Object'. 134 tests pass; display via both harnesses unchanged.

*Relevance: high*
*Context: Fixing meet label extraction (Option A: compiler-emitted content labels)*
*Tags: trs labels compiler meet*

---
*Observed: 2026-10-27T17:38:18.656Z*
