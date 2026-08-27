---
type: source
title: "Observation: ASK dedup-hit clobbered authored compound definition (DH ['DH'] bug)"
tags:
  - kscript
  - compiler
  - bugfix
  - asks
status: observation
created: 2026-08-27
updated: 2026-08-27
slug: obs-2026-08-27-ask-dedup-hit-clobbered-authored-compound-definition-dh-dh-b
relevance: high
observed_at: 2026-08-27T12:05:51.775Z
---

# ⭐ Observation: ASK dedup-hit clobbered authored compound definition (DH ['DH'] bug)

Bug found & fixed in ASK compiler change: a bare compound whose MTS canon was already emitted by an earlier authored scope got a dedup HIT in `_emit_mts`, and the ask-branch replaced that SHARED canon entry with ASK — destroying the authored compound's defining registration. Symptom: `dev/ks/compile.py` printed `DH ['DH'] CANONIZES` — the encoder, missing `WDMH`'s registered signature, computed the block canon `WDMH:[DH]` as `signature_of([DH]) == DH`, so sig collapsed onto the DH compound value with DH's label. Fix in `src/ks/ast_emitter.py`: track `mts_created` (entry-count delta around `_emit_mts`); replace-in-place only when this scope created the canon, else emit a fresh ASK entry with the canon's nodes and leave the authored def intact. WDMH-underfit output now shows both `WDMH:[what,did,Mary,have]` (authored def) and `ASK:[what,did,Mary,have]` (bare ask). Tests back at baseline.

*Relevance: high*
*Tags: kscript compiler bugfix asks*

---
*Observed: 2026-08-27T12:05:51.775Z*
