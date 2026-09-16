---
type: source
title: "Observation: γ(WDMH,MHALL)=S3 via Mary+have; MHALL value drifts per compile"
tags:
  - gamma
  - jaccard
  - word-binding
  - values
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-wdmh-mhall-s3-via-mary-have-mhall-value-drifts-per-compile
relevance: high
observed_at: 2026-09-16T14:22:21.805Z
source_context: γ(WDMH, MHALL) verification question
---

# ⭐ Observation: γ(WDMH,MHALL)=S3 via Mary+have; MHALL value drifts per compile

γ(WDMH, MHALL) arithmetic confirmed empirically (probe_gamma_atoms.py): WDMH atoms {what, did, Mary, have}, wdmh-compile MHALL atoms {Mary, have, A, L}; shared {Mary, have} = 2 of 6 union → J = 0.333, γ = J·δ^0 → byte 0x55 → S3 (> S4). Coverage over BOTH M and H because the wdmh annotation binds H→'have' and MHALL's MTS expansion resolves through the same binding. Byte-scale note: S2's range starts at J ≥ 0.5 — coverage lifts off the S4 floor but 2/6 lands S3 even though C(A,B) is structurally a covered misfit (S2); graded and structural are two readings, the byte carries the graded one. LATENT FACT for the derivation work: the written signature MHALL compiles to DIFFERENT values per script — mhall.ks → {Mary, had, a, little, lamb} (H→had, 5 atoms), wdmh compile → {Mary, have, A, L} (H→have, 4 atoms) — because Word Binding is per-script and word_bits only pins individual words. The graded measurement uses the current compile's values; the state's framed mhall-MHALL and the wdmh MHALL are different values under one name. Value-equality done-checking between the engine's held goal and the harness answer key will diverge across scripts.

*Relevance: high*
*Context: γ(WDMH, MHALL) verification question*
*Tags: gamma jaccard word-binding values*

---
*Observed: 2026-09-16T14:22:21.805Z*
