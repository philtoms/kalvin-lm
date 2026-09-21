---
type: source
title: "Observation: Tetrad completed: single char is the pure attractor (environmental reading)"
tags:
  - kscript
  - ks
  - word-binding
  - binding-scope
  - annotations
  - syntax
  - design
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-tetrad-completed-single-char-is-the-pure-attractor-environme
relevance: high
observed_at: 2026-09-21T09:34:41.187Z
source_context: "Extending the case-triad to a tetrad: single-char attraction as the fourth rule case"
---

# ⭐ Observation: Tetrad completed: single char is the pure attractor (environmental reading)

Completed the case rule to a tetrad per user's point that bare single chars are a rule case: ALL-UPPER multi-char = compound of attractors (MTS), Capitalized = self-carried word that binds its initial (Mood ≡ M(ood)), lowercase-first = self-carried literal, single char = pure attractor — the only context-DEPENDENT reading (environmental: pulls its word from the ambient word list). All four verified live: `M = X` under `(mary had a little lamb)` → mary:[X] (case-insensitive attraction, word case preserved); two Ls → little/lamb via occurrence counter; node position attracts too (A = M → a:[Mary]); unattracted stays raw (M:[X], no error). Framing: carried words vs attracted letters; Mood is the bridge (self-annotating). Tier order now motivated: self-carried bind_override outranks word-list attraction. Single-char case is typographic — not an 'uppercase' rule (h witness binds compound H per CONTEXT.md:226). Implementation already exists (BindingScope.resolve, src/ks/binding_scope.py — word-list tier with `word[0].lower() == char.lower()` and occurrence counters); only codification in CONTEXT.md's Word Binding entry was missing.

*Relevance: high*
*Context: Extending the case-triad to a tetrad: single-char attraction as the fourth rule case*
*Tags: kscript ks word-binding binding-scope annotations syntax design*

---
*Observed: 2026-09-21T09:34:41.187Z*
