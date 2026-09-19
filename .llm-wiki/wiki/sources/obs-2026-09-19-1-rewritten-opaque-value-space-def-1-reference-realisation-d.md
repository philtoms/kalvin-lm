---
type: source
title: "Observation: §1 rewritten: opaque value space (Def 1) + reference realisation (Def 2)"
tags:
  - algebra
  - docs
  - values
  - scaling
  - decision
status: observation
created: 2026-09-19
updated: 2026-09-19
slug: obs-2026-09-19-1-rewritten-opaque-value-space-def-1-reference-realisation-d
relevance: critical
observed_at: 2026-09-19T06:22:16.720Z
source_context: Rewriting kalvin-algebra.md §1 to opaque value space for vocabulary scaling
---

# 🔴 Observation: §1 rewritten: opaque value space (Def 1) + reference realisation (Def 2)

kalvin-algebra.md §1 rewritten per user decision: the algebra is now parameterised over an opaque value space. New Definition 1 — Value space: (V, ∅, ∨, ∧, ∖, μ) with five capabilities (composition, overlap, residue, content measure μ written |v|, decidable equality); misfit mass |x Δ y| = |x ∨ y| − |x ∧ y| and J(x,y) = |x ∧ y|/|x ∨ y| derived; μ exact by requirement (licensing T1 and §10 invariants read strict inequalities — realisations may scale the universe but not approximate μ). Definition 2 — Reference realisation: the word-bit space (A = {a₀…a₃₀}, V = 2^A, Boolean laws by construction); its two engine facts called out as realisation-only: value-is-a-machine-word (31-word vocabulary bound) and value-is-its-own-address. Scaling recipe in the doc: each distinct word gets an integer id (first-encountered, unbounded — user specified int, not u32), a value is the exact set of its ids, signifier interns values under a content key so equal content shares one address; nesting-by-reference unchanged. Definition numbering preserved (Def 1/Def 2) so no downstream citations broke. Consequent edits: §0.1 notation table reframed (Implementation column → Reference realisation; ¬v row removed to Def 2; |x| = μ(x)); Def 8 coverage "shares at least one atom" → overlap/share content; Def 9 u = s ∧ ¬σ(ν) → u = s ∖ σ(ν); "misfit atom"/"gap atoms" → misfit content (Def 15, appendix); "atom-weighted" → content-weighted (Defs 17-19, invariants); decidability "the atom set is finite" → "the value space may be taken finite"; ASK marker phrasing "atom space" → content measure. CONTEXT.md synced: Atom entry now cites Def 2 as reference-realisation gloss; Value entry now the opaque five-op interface citing Def 1; Coverage/Shape/Misfit Mass/Slot/Scope/ASK entries atom→content wording. Worked examples intentionally left in atom vocabulary (they run in the reference realisation). NOTE: engine (significance.py, derivation.py) still bypasses the KSignifier seam — imports WORD_BITS, iterates _atom_bits directly; the missing abstract method is measure/μ.

*Relevance: critical*
*Context: Rewriting kalvin-algebra.md §1 to opaque value space for vocabulary scaling*
*Tags: algebra docs values scaling decision*

---
*Observed: 2026-09-19T06:22:16.720Z*
