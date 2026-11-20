---
type: source
title: "Observation: Simplified finish: root S2 departure only — claim precedes report by construction"
tags:
  - cogitator
  - finish
  - claims
  - simplification
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-simplified-finish-root-s2-departure-only-claim-precedes-repo
relevance: high
observed_at: 2026-11-20T08:06:14.586Z
source_context: User simplified the mechanism after removing live and the claim filter
---

# ⭐ Observation: Simplified finish: root S2 departure only — claim precedes report by construction

User ruled: the live counter and first-past-the-post claim filter masked the real problem — finish firing before the wdmh claim lands. They removed both AND the root's tail finish: the ONLY finish is now the root's S2 departure (rotation exhausted) plus the entry cap; the root's tail departure pushes a None the walk discards, and the run lives on through residue. Verified: mhall and wdmh claim exactly once each in both modes and both candidates orders; in all four runs the claim lands with the stack CLEAN (no residue at or after the claim) — because the exchange's witness notify lands near the top at exchange time, everything later stacks above it, LIFO drains those first, and the witness's departure appends the root last, so the claiming turn sees an empty stack. Root re-runs with residue mid-run are common (several per run) but transient. Caveats recorded, not blocking per user ("not unduly worried if it is not exactly clean"): (1) once-ness of claim/finish is emergent from exploration order, not structural — residue queued below an exchange witness, or a root-child draining after the claim, would produce a second broadcast and a second finish (duplicated step); (2) a root that never aligns never finishes — ungrounded runs leave no report (no end_step, no Step recorded; stalls still broadcast live through the sink). docs/rationalisation.md updated: finish rides the root's S2 departure only; WALK documents the ungrounded-run behavior; ALIGN/§2.4 restored to broadcast-at-every-alignment (no filter).

*Relevance: high*
*Context: User simplified the mechanism after removing live and the claim filter*
*Tags: cogitator finish claims simplification*

---
*Observed: 2026-11-20T08:06:14.586Z*
