---
type: source
title: "Observation: Three threads closed (a083c47): reserved doctrine, Word(x) error, one-word tails"
tags:
  - kscript
  - ks
  - parser
  - syntax
  - design
  - committed
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-three-threads-closed-a083c47-reserved-doctrine-word-x-error-
relevance: high
observed_at: 2026-09-21T10:48:04.903Z
source_context: Committing the three-thread closure
---

# ⭐ Observation: Three threads closed (a083c47): reserved doctrine, Word(x) error, one-word tails

Committed a083c47, closing all three open threads from the case-rule work. (1) `? ! ,` held indefinitely reserved — CONTEXT.md doctrine records them as reserved-syntax candidates with the bracketed witness as escape hatch (w(hat?) → what?); attraction already matches punctuated prose words so nothing is unreachable; each symbol would foreclose its natural syntax (postfix ask `?`, negation `!`, item separator `,`) because glued forms are lexically ambiguous. (2) Capitalized-literal gap closed by DOCTRINE not syntax: a Capitalized word declares its binding — Word(x) is now a ParseError at both positions (was silently inert sig-side, mangled Subjectx node-side via _extract_inline_word concatenation). (3) Inline tails witness one word: whitespace/newline-bearing tails (M(ary had)) raise ParseError — a spaced word can never match any list word (word lists are whitespace-split); phrasal content belongs in prefix annotations. Implementation: parser._check_inline_annotation (multi-char + whitespace checks, both attachment sites); test_explicit_annotation_suppresses_expansion replaced by error assertions; w(hat?) regression locked in test_ks_symbol_range. 101 tests green, fixtures unaffected. Session principle: the surface refuses what it cannot mean — reserved chars reachable via brackets, incoherent annotations fail at compile time. Session commits: f653367 (case rule) → 4d1bbb9 (symbol range) → a083c47 (surface honesty).

*Relevance: high*
*Context: Committing the three-thread closure*
*Tags: kscript ks parser syntax design committed*

---
*Observed: 2026-09-21T10:48:04.903Z*
