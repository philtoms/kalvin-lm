---
type: source
title: "Observation: Def 13 mirror clause: only canonicalisation exercised; S3 instance contradicted"
tags:
  - kalvin-algebra
  - def13
  - mirror
  - canonicalisation
  - targeting
  - traversal
status: observation
created: 2026-09-18
updated: 2026-09-18
slug: obs-2026-09-18-def-13-mirror-clause-only-canonicalisation-exercised-s3-inst
relevance: high
observed_at: 2026-09-18T08:21:22.476Z
source_context: Auditing docs/kalvin-algebra.md Def 13 mirror clause usage across doc and code
---

# ⭐ Observation: Def 13 mirror clause: only canonicalisation exercised; S3 instance contradicted

In docs/kalvin-algebra.md Def 13, the band-general Mirror derivations clause has three instances with very different status: (1) Canon instance = canonicalisation, fully live in both engines (canonicalisations() in src/kalvin/derivation.py:165 and src/dialogue/derivation.py:156); (2) covered-misfit instance = reverse targeting, licensed in the letter of Def 14 (l.616) and enumerated by both engines' targetings() (kalvin derivation.py:225-236, dialogue derivation.py:195-208) but never observed to fire — the §13 fragment analysis (l.1349) rejects the only reverse candidate o:[m] for worsening mismatch, and no test exercises it; (3) uncovered instance = reverse traversal, whose witness is stale: l.579 claims §9's worked example crosses all:[o] reverse ("whichever direction arrival supplies") but the current example crosses it forward licensed by heading, and §9's closing line (l.905) says direction is "fixed by heading, not by arrival" — a direct contradiction. Core descend() is heading-only; only the dialogue line still uses the S3 mirror instance (slot_walk occurrence-licensed reverse moves, src/dialogue/derivation.py:241-243; dev/dialogue/probe_wdmh_s3_walk.py MODE C explores reverse-only walks). slot_walk's docstring cites "Def 17" for the occurrence licence, which is now Jaccard — stale citation.

*Relevance: high*
*Context: Auditing docs/kalvin-algebra.md Def 13 mirror clause usage across doc and code*
*Tags: kalvin-algebra def13 mirror canonicalisation targeting traversal*

---
*Observed: 2026-09-18T08:21:22.476Z*
