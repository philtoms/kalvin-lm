---
type: source
title: 'Observation: O→Object resolution in mhall stands; "unresolved" is a knowledge criterion'
tags:
  - ks
  - binding
  - inline-override
  - mhall
  - object
  - terminal
status: observation
created: 2026-10-26
updated: 2026-10-26
slug: obs-2026-10-26-o-object-resolution-in-mhall-stands-unresolved-is-a-knowledg
relevance: high
observed_at: 2026-10-26T15:05:48.755Z
source_context: Clarifying the terminal-unknown rule against mhall.ks L = O
---

# ⭐ Observation: O→Object resolution in mhall stands; "unresolved" is a knowledge criterion

In mhall.ks the statement `L = O` resolves O→Object (giving DENOTES lamb:[Object]) — O is NOT unresolved at binding level. Mechanism: the parser decomposes mixed-case words as initial+tail (Mood ≡ M(ood)), so `Object < Query` parses as sig O with inline annotation (bject); the inline override bind_override('O','Object') lands in the enclosing SVO-block scope and remains visible from inside ALL's subscript scope via resolve()'s innermost-first scope walk. User ruled (2026-10-26): the resolution stands — "Object has no identity/canon of its own so it generates S4. There is no difference between O and Object in this context. The raw symbol is an int." I.e. "unresolved" for the terminal-unknown rule is a knowledge criterion (terminal's word lacks its own identity/canon), not a binding failure; the minted unknown is Object:[] (the resolved form).

*Relevance: high*
*Context: Clarifying the terminal-unknown rule against mhall.ks L = O*
*Tags: ks binding inline-override mhall object terminal*

---
*Observed: 2026-10-26T15:05:48.755Z*
