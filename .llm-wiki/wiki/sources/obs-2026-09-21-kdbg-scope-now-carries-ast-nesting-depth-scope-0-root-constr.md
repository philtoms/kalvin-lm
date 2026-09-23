---
type: source
title: "Observation: KDbg.scope now carries AST nesting depth; scope-0 = root constructs"
tags:
  - kscript
  - compiler
  - harness
  - scope
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-kdbg-scope-now-carries-ast-nesting-depth-scope-0-root-constr
relevance: high
observed_at: 2026-09-21T14:11:42.273Z
source_context: Making KDbg.scope carry nesting depth for harness step delineation
---

# ⭐ Observation: KDbg.scope now carries AST nesting depth; scope-0 = root constructs

Root cause of the 8-step mhall run: KDbg.scope was a boolean authored(0)/expansion(1) flag, not nesting depth — every authored entry at any depth was scope-0. Fixed in src/ks/ast_emitter.py: added _scope_depth counter; _compile_children walks nested constructs at depth+1; _emit_entry writes scope=depth for authored entries and depth+1 for expansions (never 0); the ASK-in-place _replace writes scope=self._scope_depth instead of hardcoded 0. AST nesting is deeper than visual indentation (the parser gives `==`/goal scopes and operator items their own scope levels) but only 0 vs non-zero matters: exactly the root construct's own entries are scope-0. mhall.ks → 1 step; wdmh.ks → 2 steps (scaffold root W=O before ask root); an isolated annotation emits its shorthand ask at scope 0 → own step; annotations consumed by a following scope emit no entry. Probe at dev/dialogue/probe_scope0_groups.py; 95 tests pass.

*Relevance: high*
*Context: Making KDbg.scope carry nesting depth for harness step delineation*
*Tags: kscript compiler harness scope*

---
*Observed: 2026-09-21T14:11:42.273Z*
