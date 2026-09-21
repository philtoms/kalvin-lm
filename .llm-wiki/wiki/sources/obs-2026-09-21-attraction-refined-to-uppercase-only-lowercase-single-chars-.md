---
type: source
title: "Observation: Attraction refined to uppercase-only: lowercase single chars are literal words"
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
slug: obs-2026-09-21-attraction-refined-to-uppercase-only-lowercase-single-chars-
relevance: high
observed_at: 2026-09-21T09:40:30.980Z
source_context: Refining single-char attraction after user pushback on lowercase article miss-binding
---

# ⭐ Observation: Attraction refined to uppercase-only: lowercase single chars are literal words

User pushback accepted: single-char ambient attraction refined to UPPERCASE-ONLY (option 1), rejecting the stopword opt-out (option 2 — language-specific, brittle, case already does the job). My prior claim 'single-char case gates nothing' was wrong — it left lowercase single chars straddling literal-word and attractor rows; the article 'a' is the collision case. Verified miss-binding exists TODAY: `(a little lamb)` then `(axe)` then `L = a` → little:[axe] (most-recent list wins, case-insensitive). Refined rule — first character's case decides the reading uniformly: upper-single=attractor, upper-all-upper-multi=compound(MTS), upper-mixed-multi=self-carried expansion, lower-ANY-length=literal word (single chars included), brackets=explicit case-blind (witness doctrine h(ad) binds H). New stability guarantee: lowercase vocabulary is inert/immune to environment; only sig-case chars are environment-sensitive. Impact: one behavior change (bare lowercase single chars → literal; _resolve_char raw fallback already handles it, gate goes in BindingScope.resolve for lowercase); MTS unaffected (compound chars always uppercase); 3 assertions in worktree test_ks_binding_scope.py:255-263 flip to None (resolve('m')=='mary' etc. — deliberate spec change); binding_scope.py:80 docstring; CONTEXT.md:226 'letter binds not its case' must be scoped to authored bindings. Residual: uppercase single-char words (I, O) attract themselves when listed — benign identity.

*Relevance: high*
*Context: Refining single-char attraction after user pushback on lowercase article miss-binding*
*Tags: kscript ks word-binding binding-scope annotations syntax design*

---
*Observed: 2026-09-21T09:40:30.980Z*
