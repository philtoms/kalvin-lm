---
type: source
title: "Observation: Spec R3 synced: substitute and re-enter in-chain, no canonicalisation"
tags:
  - trs
  - spec
  - r3
  - substitution
  - sync
status: observation
created: 2026-10-07
updated: 2026-10-07
slug: obs-2026-10-07-spec-r3-synced-substitute-and-re-enter-in-chain-no-canonical
relevance: medium
observed_at: 2026-10-07T14:07:21.036Z
source_context: Syncing the spec R3 rule with the substitute-only code
---

# 🔍 Observation: Spec R3 synced: substitute and re-enter in-chain, no canonicalisation

SPEC R3 SYNCED with code (user ruling: 'R3 rules now out of sync with code — it's still supporting canonicalising and re-entering, whereas R3 in code simply substitutes'). Docs/trs-significance.md changes: (1) rule header 'Rule R3 — misfit (S3): substitute, canonicalise, re-enter' → 'substitute, re-enter'; (2) R3's product now 'keeps the candidate's signature and carries the substituted adjacency: C′ = ⟨σ, N′⟩' — the old 'E′ = ⟨can(N′), N′⟩' entry-kline and 'the prior signature is discarded' language removed; 'Substitution is the whole of the rule: the signature makes its claim, and the substituted adjacency is the re-pairing'; re-entry is IN-CHAIN (overlap on σ, not can(N′)); (3) §4 entropy commentary: old 'R3 discards the surprised claim... product is self-canonical H=0' → 'R3 substitutes and re-enters on the candidate's own signature — correspondence may emerge from the re-pairing (S3 → S2 or S1), or not (S3 → S4)'; (4) Definition 8 entries sentence: products of R2 arrive as entries, R3's does not; (5) trace definition: R3 substitution continues a trace without a new entry; (6) §5 preamble and the §5 diagram line 'N′ ──► ⟨can(N′), N′⟩' → '⟨σ, N′⟩ (in-chain)'. Zero ⟨can(N′),N′⟩ references remain. NOT yet touched (pending the redesign's later steps): R2's three-products-re-enter-as-entries language (step 1.2's single-response consumption + U recursion + F accumulation is still code-only in rationalise2), and §7's old-runner instantiation. 159 tests passing.

*Relevance: medium*
*Context: Syncing the spec R3 rule with the substitute-only code*
*Tags: trs spec r3 substitution sync*

---
*Observed: 2026-10-07T14:07:21.036Z*
