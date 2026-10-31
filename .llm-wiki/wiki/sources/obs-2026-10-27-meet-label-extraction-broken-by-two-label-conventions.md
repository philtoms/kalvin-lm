---
type: source
title: "Observation: meet label extraction broken by two label conventions"
tags:
  - trs
  - labels
  - meet
  - compiler
status: observation
created: 2026-10-27
updated: 2026-10-27
slug: obs-2026-10-27-meet-label-extraction-broken-by-two-label-conventions
relevance: high
observed_at: 2026-10-27T17:30:05.074Z
source_context: Diagnosing trs.model.meet label extraction on compiled signatures
---

# ⭐ Observation: meet label extraction broken by two label conventions

meet(KNode(0x4200000000,'hadVerb'), KNode(0xE000000000,'SVO')) yields label 'hadVerb∧SVO' instead of 'Verb'. Root cause: two label conventions. Compiled labels are opaque — TokenEncoder labels compound sigs with the authored identifier (.with_label(entry.sig) → 'SVO', 'hadVerb'; _compose_concat likewise) and NLPSignifier.signature_of joins node labels by plain concatenation ("".join, signifier.py:80). Only runtime composition (KNode.merge, trs can) uses '|'. trs.model._tokens only understands '|'-compositions, so intersection of ['hadVerb'] × ['SVO'] is empty and meet falls back to the provenance expression. Verified: compiling mhall.ks gives hadVerb=had(bit1)|Verb(bit2), SVO=Subject(5)|Verb(6)|Object(7), meet=0x4000000000=word 'Verb'. Same defect affects residue's U/O labels. trs tests assert no labels, so any fix direction is test-neutral.

*Relevance: high*
*Context: Diagnosing trs.model.meet label extraction on compiled signatures*
*Tags: trs labels meet compiler*

---
*Observed: 2026-10-27T17:30:05.074Z*
