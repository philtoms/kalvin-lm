---
type: source
title: "Observation: Proposal-pops-STM + frame removal: dedup lost, residue cleaner"
tags:
  - dialogue
  - engine
  - stm
  - frame
status: observation
created: 2026-08-17
updated: 2026-08-17
slug: obs-2026-08-17-proposal-pops-stm-frame-removal-dedup-lost-residue-cleaner
relevance: high
observed_at: 2026-08-17T14:09:44.918Z
source_context: Lean harness tuning session on src/dialogue/engine.py
---

# ⭐ Observation: Proposal-pops-STM + frame removal: dedup lost, residue cleaner

Uncommitted engine changes (remove-from-STM-on-proposal; frame memory call sites removed: in_frame, signature_seen deleted) were run on mhall. Grounded set unchanged (8 identities; MHALL/ALL/WDMH/DH still stall in STM). S4 emissions rose 9→20 — frame was the dedup owner, now no owner exists. STM residue improved: MHALL keeps authored nodes instead of the strategy's rewritten [SVO]. Removal-on-proposal forecloses the S3 countersign path: S2 misfits pop before they can become countersignable. Oddity: WDMH:[Mary,DH] at step 9 unpacks unseen nodes (what/did/have) but emits no S4 asks; the 4-node form at step 23 does. Behaviour-notes 'expand emits 1 S2 at step 10' no longer reproduces (0 S2s).

*Relevance: high*
*Context: Lean harness tuning session on src/dialogue/engine.py*
*Tags: dialogue engine stm frame*

---
*Observed: 2026-08-17T14:09:44.918Z*
