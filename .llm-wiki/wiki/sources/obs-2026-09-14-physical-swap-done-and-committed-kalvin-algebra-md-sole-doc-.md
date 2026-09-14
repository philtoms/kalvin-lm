---
type: source
title: "Observation: Physical swap done and committed: kalvin-algebra.md sole doc, probe in dev/algebra"
tags:
  - docs
  - consolidation
  - commit
  - kalvin-algebra
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-physical-swap-done-and-committed-kalvin-algebra-md-sole-doc-
relevance: high
observed_at: 2026-09-14T08:18:19.325Z
source_context: Committing the doc consolidation and probe relocation
---

# ⭐ Observation: Physical swap done and committed: kalvin-algebra.md sole doc, probe in dev/algebra

Phil completed the physical swap: deleted docs/kalvin-for-agents.md and docs/behaviour-notes.md, renamed docs/kalvin-simplified.md → docs/kalvin-algebra.md (status line updated to "This document is the single normative definition of Kalvin" — supersedes clause removed since it named itself), and moved the probe to dev/algebra/worked-example-wdmh.py (docstring already repointed to docs/kalvin-algebra.md, run path updated). All inbound references (CONTEXT.md, src/kalvin/kline.py, src/kalvin/significance.py, tests) cite kalvin-algebra.md §N with identical numbering — the earlier prediction that retargeting would be mechanical proved moot since the name was inherited by the renamed doc. Verified: 35/35 tests pass, probe 7/7 PASS from new location. Committed as 9a32fd8 (docs+probe, 19 files) + 7844064 (wiki observations). Working tree clean. The pending physical-swap follow-up from obs-2026-09-13 is now closed.

*Relevance: high*
*Context: Committing the doc consolidation and probe relocation*
*Tags: docs consolidation commit kalvin-algebra*

---
*Observed: 2026-09-14T08:18:19.325Z*
