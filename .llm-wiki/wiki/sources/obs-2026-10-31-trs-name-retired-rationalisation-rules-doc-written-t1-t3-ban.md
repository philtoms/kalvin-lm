---
type: source
title: "Observation: TRS name retired — rationalisation rules doc written, T1–T3 banked"
tags:
  - trs
  - rationalisation
  - spec
  - rulings
  - terminology
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-trs-name-retired-rationalisation-rules-doc-written-t1-t3-ban
relevance: critical
observed_at: 2026-10-31T15:26:03.232Z
source_context: Answering whether the implementation is still a term rewriting system; writing docs/rationalisation.md
---

# 🔴 Observation: TRS name retired — rationalisation rules doc written, T1–T3 banked

Ruling: the src/trs mechanism is NOT a term rewriting system — "TRS" retired as the model's name. The mechanism is a deterministic, bounded, depth-first coverage search over a bit-vector content algebra with a monotone store: rows are obligations, reentries are subgoals (LIFO, u-side first), frames are earned lemmas, exchange is a mutual-witness rule, claim grounding is QED. Evidence: state is a machine configuration not a term; the spec's R2/R3 product rules don't exist (F/U/O now computed on sides sig|can(adj), R3 substitution replaced by the o-side witness settle); rules are global (frame cascades, exchange, first-stored witness); termination is by bounds (MAX_DEPTH=10, MAX_ROWS=4096), determinism by control regime. New rules document: docs/rationalisation.md — code-anchored rule inventory (POP/KNOWN/CANDIDATES/PAIR/SETTLE/FRAME/RESOLVE/CLAIM/STALL/EXCHANGE/HALT), invariants, and banked guarantees T1 termination (measure+bounds argument), T2 determinism (all choice points ordered; KNode=int hashing; labels display-only), T3 soundness (grounding ⇒ covering certificate: justification tree over empty regions, domain canons, earned frames, exchange pairs; induction on the event clock). T4 completeness explicitly NOT claimed — the CANDIDATES bars each need their own argument. docs/trs-significance.md retired to docs/attic/ with banner. Terminology pinned in CONTEXT.md: Rationalisation/Cogitator/Side/Row/Reentry/Frame/Stall/Exchange/Claim/Domain rewritten or added; Cogitation entry replaced (its "_Avoid_: the cogitator" contradicted the code's class name). Doc name rationale: "rationalisation" is the code's own operation name (Cogitator.rationalise), claims no theory, no collision in the carried-forward path. Deferred: src/trs package rename (import churn), full CONTEXT.md glossary resync (still engine-era entries: Hop, Derivation, Replace, Licence, Scope, Slot), phase-2 certificate verifier + property tests, Maude (dropped for now per user ruling). Notable code facts recorded: candidates iterate D insertion order (the closers-first sort in _reenter is commented out — history, not active); a non-canon goal rationalises goal-less (full search); ask runs store exactly the claim and only on grounding; KNOWN (candidates existed but barred) is distinct from STALL (no candidates at all).

*Relevance: critical*
*Context: Answering whether the implementation is still a term rewriting system; writing docs/rationalisation.md*
*Tags: trs rationalisation spec rulings terminology*

---
*Observed: 2026-10-31T15:26:03.232Z*
