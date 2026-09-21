---
type: source
title: "Observation: MTS retired (85083f5): compound/expansion/canon triple aligned everywhere"
tags:
  - domain-modeling
  - glossary
  - kscript
  - ks
  - terminology
  - committed
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-mts-retired-85083f5-compound-expansion-canon-triple-aligned-
relevance: high
observed_at: 2026-09-21T12:13:25.656Z
source_context: Committing the MTS retirement and vocabulary alignment
---

# ⭐ Observation: MTS retired (85083f5): compound/expansion/canon triple aligned everywhere

Committed 85083f5: MTS retired as a domain term, vocabulary aligned across docs and code. Investigation finding: term INVERSION — 'compound' was the domain concept (7 CONTEXT.md uses, 4 algebra uses, all concept-level) but had NO glossary entry, while MTS's entry body was compiler mechanics; docs used MTS only as parenthetical aliases and the Canon avoid-list already disowned it ('an example, not the concept'). Resolution: CONTEXT.md MTS entry → Compound entry (multi-word signature, OR-reduction, no bit of its own, introduced by expansion or CONNOTES concatenation, initials-synthesis capability); tombstone in avoid-list ('MTS — superseded name for the expansion'); kalvin-algebra.md:1396 alias scrub (semantic-free, normative content untouched); code rename ~130 lines/7 files — is_mts→is_expansion, _emit_mts→_expand_compound, mts_idx→canon_idx, _mts_canonicalise_seen→_canon_seen, synthetic 'MTS' op key (pseudo-op for provenance, live via token_encoder)→'EXPANSION'. Canonical triple now: compound (the thing) / expansion (the mechanism) / canon (the kline shape). 101 tests green, fixtures byte-identical. INCIDENT during task: five src/dialogue/ files (cogitator, derivation, expand_fit, pivot_fill, reentry) found deleted from working tree by something outside the session — restored via git checkout from HEAD; watch for recurrence. Session commits: f653367, 4d1bbb9, a083c47, 85083f5.

*Relevance: high*
*Context: Committing the MTS retirement and vocabulary alignment*
*Tags: domain-modeling glossary kscript ks terminology committed*

---
*Observed: 2026-09-21T12:13:25.656Z*
