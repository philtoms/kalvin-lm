---
type: source
title: "Observation: One ask structure: sig|ASK_SIG:[nodes] everywhere; annotation utterances are asks"
tags:
  - ask
  - compiler
  - parser
  - unification
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-one-ask-structure-sig-ask-sig-nodes-everywhere-annotation-ut
relevance: high
observed_at: 2026-09-16T16:19:26.699Z
source_context: One ask structure — no empty form, annotation utterances always asks
---

# ⭐ Observation: One ask structure: sig|ASK_SIG:[nodes] everywhere; annotation utterances are asks

Unified the ask to ONE structure per user ruling: every ask is `sig|ASK_SIG:[nodes]` — no empty ask form. Changes: (1) unbound single-char/bare-token asks emit `[self]` nodes (`a` → `a|ASK:[a]`) instead of `[]` — both ast_emitter sites (bare scope, subscript gap); (2) `(a)` — a sigless annotation alone — was compiling to IDENTITY `a:[a]`, NOT the ask: the parser synthesizes the annotation's initials into an OperatorScope (parser.py `_parse_construct` annotation branch, lookahead: SIGNATURE/INDENT ⇒ prefix annotation, else synthesized scope) and the annotation's own words bind the char, flipping the single-char ASK path's binding discriminator to IDENTITY. Fix: OperatorScope.synthetic flag (ast.py, set at synthesis); a synthetic scope is always the ask, its binding resolving nodes — symmetric with the multi-word `(a big cat)` → `ABC|ASK:[a,big,cat]` path. Discovered en route: a bare token whose resolved text equals itself (e.g. `cat` after `(a big cat)` heads it) compiles as the ask — the discriminator is textual difference (c→cat differs ⇒ identity; cat→cat equal ⇒ ask), pre-existing, unchanged. (3) Fixed a latent bug my _answer port introduced: identity comparisons used the MARKED signature (`e.kline.nodes == [kline.signature]`) — never equal to unmarked node values; now compare against the masked base. (4) Docs: algebra §13 (`a|ASK:[a]`, one-structure prose), §4 (Unknown = S4's no-content shape, no compiled ask takes it), CONTEXT.md ASK entry, script-reading.md; is_unknown docstring no longer calls the empty form "the structural form of an ask". Tests: test_one_ask_structure..., test_word_bound_bare_token_is_identity_synthetic_is_ask (58 pass). wdmh/mhall end-to-end unchanged (2/5 steps, 0 escalations). Note: `(x)\ny` parses as prefix annotation heading y — annotations only synthesize their own ask scope at EOF/non-signature next.

*Relevance: high*
*Context: One ask structure — no empty form, annotation utterances always asks*
*Tags: ask compiler parser unification*

---
*Observed: 2026-09-16T16:19:26.699Z*
