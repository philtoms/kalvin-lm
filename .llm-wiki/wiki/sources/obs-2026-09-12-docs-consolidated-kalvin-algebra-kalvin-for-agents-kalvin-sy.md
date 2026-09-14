---
type: source
title: "Observation: Docs consolidated: kalvin-algebra + kalvin-for-agents; kalvin-symbolic deleted"
tags:
  - docs
  - rename
  - kalvin-algebra
  - kalvin-for-agents
  - consolidation
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-docs-consolidated-kalvin-algebra-kalvin-for-agents-kalvin-sy
relevance: critical
observed_at: 2026-09-12T11:03:19.372Z
source_context: Consolidating the formal docs to the 2-doc structure
---

# 🔴 Observation: Docs consolidated: kalvin-algebra + kalvin-for-agents; kalvin-symbolic deleted

Doc consolidation executed per user decision: docs/kalvin-symbolic.md deleted (git rm), docs/ks2.md renamed to docs/kalvin-algebra.md (the authority), docs/ks3.md renamed to docs/kalvin-for-agents.md (the accessible face). All inbound references updated: CONTEXT.md:51, tests/test_gamma.py, tests/test_relationship.py, src/kalvin/significance.py (2), src/kalvin/kline.py (3), src/dialogue/cogitator.py — all now cite 'kalvin-algebra(.md) Def/§ N'. kalvin-for-agents.md H1 and status line changed from 'written/restated for programmers' to 'for agents' to match the new filename (internal 'In programmer terms' framing device kept — it names the technique, not the audience); vetoable if unwanted. Three historical kalvin-symbolic mentions remain by design: kalvin-algebra.md:3 status line now says '(that first pass is deleted)', and the §13 provenance line 'Absorbed from kalvin-symbolic.md §5' left identical in kalvin-algebra.md:236 and its verbatim quote in kalvin-for-agents.md:501 to preserve quote-wins byte-identity. 35 tests pass. Changes staged (git mv/rm) + unstaged edits, not committed — awaiting user confirmation.

*Relevance: critical*
*Context: Consolidating the formal docs to the 2-doc structure*
*Tags: docs rename kalvin-algebra kalvin-for-agents consolidation*

---
*Observed: 2026-09-12T11:03:19.372Z*
