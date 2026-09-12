---
type: source
title: "Observation: User wants 2-doc structure; citation graph already favours ks2 as authority"
tags:
  - docs
  - formalisation
  - ks2
  - ks3
  - kalvin-symbolic
  - authority
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-user-wants-2-doc-structure-citation-graph-already-favours-ks
relevance: high
observed_at: 2026-09-12T10:55:34.618Z
source_context: Deciding the authoritative/accessible split across the three formal docs
---

# ⭐ Observation: User wants 2-doc structure; citation graph already favours ks2 as authority

User revealed doc-landscape intent: originally planned kalvin-symbolic (mathematical prose) as authoritative, with ks2 (readable) and ks3 (programmer-facing) derived from it; ideally wants 2 docs — one authoritative, one accessible. My review + grep evidence shows authority has already migrated to ks2: engine code cites ks2 Def numbers (significance.py:357 'ks2 Def 14', kline.py:302 'Def 13', cogitator.py:315 'Def 16'), tests reference ks2, ks3 blockquotes are verbatim ks2 text, and nothing references kalvin-symbolic except historical status-line mentions. kalvin-symbolic was updated in place (2026-09-12) to match ks2, making it a strict subset with incompatible Def numbering (its Def 6 = ks2's Def 10). Recommended consolidation: ks2 authoritative + ks3 accessible (the quote-wins pair), retire kalvin-symbolic — deletion or archive, both near-zero cost since no inbound refs need fixing. Re-pointing authority at kalvin-symbolic would require re-quoting every ks3 blockquote and re-pointing in-code citations. User decision pending.

*Relevance: high*
*Context: Deciding the authoritative/accessible split across the three formal docs*
*Tags: docs formalisation ks2 ks3 kalvin-symbolic authority*

---
*Observed: 2026-09-12T10:55:34.618Z*
