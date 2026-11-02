---
type: source
title: "Observation: Verifier found exhaustion-grounds-without-evidence defect in _in_path"
tags:
  - rationalisation
  - soundness
  - certificate
  - verifier
  - bug-found
  - determinism
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-verifier-found-exhaustion-grounds-without-evidence-defect-in
relevance: critical
observed_at: 2026-10-31T16:11:44.181Z
source_context: "Phase 2: certificate verifier + property tests over the rationalisation model"
---

# 🔴 Observation: Verifier found exhaustion-grounds-without-evidence defect in _in_path

Phase 2 (certificate verifier + property tests) found a genuine soundness defect in the model. src/rationalisation/verify.py re-derives a grounded claim's justification from run records (leaves: empty, witness, witness-held frame, goal-given frame, exchange, proven). Corpus result: mhall grounds with certificate {empty 7, witness-held 3, proven 2, witness 1}; wdmh grounds WITH an exchange leaf — the mutual-witness rule is load-bearing in a canonical script. THE DEFECT: the exhausted-candidates known (_reenter's `not batches` branch) can resolve a reentry on policy bars alone. _in_path compares SIGNATURE VALUES ONLY, so a held kline sharing only a sig with an ancestor entry (e.g. domain 9:[1] vs entry 9:[32,1]) is barred as a "re-derivation"; when it was the only non-identity candidate, the reentry resolves _known with no frame/witness/pairing/exchange behind it and the claim grounds unbacked. Minimal repro: Domain = identities 1..32 + Kline(9,[1]); entry Kline(9,{32,1}); goal Kline(1,[1]) → claim ⟨9,{1}⟩ grounds with region 40 (bits 8|32) backed by nothing — and without the bug the u-chain would stall at identity 32 (no honest grounding exists). Fix fork presented to user: (A) make _in_path compare klines (sig+content), not sigs; (B) exhaustion-is-not-evidence — the exhausted known only when provable, else stall; possibly both. Test marked xfail(strict=False) as regression tripwire; docs/rationalisation.md T3 amended with the fifth leaf (proven) and a Known defect paragraph. Also this session: src/trs renamed to src/rationalisation (tests renamed test_rationalisation_*), CONTEXT.md glossary fully resynced (engine strategy terms retired — Hop/Derivation/Replace/Licence/Scope/Slot/STM/LTM/etc; duplicate Frame/Reentry entries my earlier edit created are resolved; Certificate term added), T2 determinism machine-checked (digest identical across PYTHONHASHSEED 0/1/random via subprocess), T1 bounds checked on corpus. Verifier subtleties learned: exchange leaves survive frame-stripping by design (stall pair is the evidence); wdmh/mhall certificates differ in kind (proven-heavy vs exchange-carrying).

*Relevance: critical*
*Context: Phase 2: certificate verifier + property tests over the rationalisation model*
*Tags: rationalisation soundness certificate verifier bug-found determinism*

---
*Observed: 2026-10-31T16:11:44.181Z*
