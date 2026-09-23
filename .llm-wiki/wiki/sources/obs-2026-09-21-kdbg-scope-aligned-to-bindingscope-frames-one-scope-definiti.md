---
type: source
title: "Observation: KDbg.scope aligned to BindingScope frames — one scope definition in the compiler"
tags:
  - domain
  - kscript
  - scope
  - binding
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-kdbg-scope-aligned-to-bindingscope-frames-one-scope-definiti
relevance: high
observed_at: 2026-09-21T14:43:11.366Z
source_context: Aligning KDbg.scope with BindingScope frames after scope-terminology discussion
---

# ⭐ Observation: KDbg.scope aligned to BindingScope frames — one scope definition in the compiler

Domain decision: KDbg.scope now reports the BindingScope frame an entry resides in (BindingScope.depth property; root/global frame = 0, each pushed CANONICALISES subscript one deeper), replacing the parallel _scope_depth counter over AST nesting. Key correction: AST nesting ≠ scope — non-binding operator nesting (Subject < M) introduces no binding frame; a statement resides where its bindings resolve. The user's code-scope model maps exactly: scope-0 persists file-wide until a later same-frame annotation replaces it (most-recent-first word lists), inner frames shadow outer (innermost-first resolution), stack depth answers level-of-residence. mhall depths now 0/1/2 matching visual indent (was 0/2/5). Expansions remain depth+1 (never scope-0) — a convention, not residence; a KDbg expansion flag would be the fully honest form. Engine "Scope" (CONTEXT.md Def 23, hop trawl) reframed by user as project scope — a visibility window over memory, not a binding frame; rename would clear the last term collision but is algebra-docs territory. Probe: dev/dialogue/probe_scope0_groups.py; 95 tests pass; mhall 1 step, wdmh 2 steps unchanged.

*Relevance: high*
*Context: Aligning KDbg.scope with BindingScope frames after scope-terminology discussion*
*Tags: domain kscript scope binding*

---
*Observed: 2026-09-21T14:43:11.366Z*
