---
type: source
title: "Observation: kscript import statement implemented (shared-emitter walk)"
tags:
  - kscript
  - import
  - compiler
  - parser
  - harness
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-kscript-import-statement-implemented-shared-emitter-walk
relevance: critical
observed_at: 2026-09-23T16:09:08.025Z
source_context: Extending kscript syntax with an import statement
---

# 🔴 Observation: kscript import statement implemented (shared-emitter walk)

Implemented `import <module>` in kscript end-to-end. Grammar: `script ::= import* construct*` — `import` reserved at construct position only (stays a word in items/annotations), must precede all other constructs (ParseError otherwise). Semantics: the Compiler walks imports depth-first emitting each module through ONE shared ASTEmitter and root BindingScope — module entries prepend in import order, its word lists seed the importing script like known_words (script's own lists win, most-recent-first), compound canons/dedup/resolution_cache span the boundary (MHALL reuses mhall's decomposition+signature: no per-compile drift), file boundary resets occurrence counters (BindingScope.reset_counters — without it mhall's own L resolutions consumed the counter so the script's L fell off the list). Guardrails: CompileError on cycles (chain-tracked), diamond imports collapse (imported-set), unresolvable/missing-resolver errors carry the import's line/col. Resolution via injected resolver: path_resolver(dir) searches <dir>/<name>.ks; compile_source/Compiler/KScript gained `resolver=` param; dev/dialogue/harness.py gained Harness.resolver (main() sets path_resolver(script.parent)) threaded through _sig_to_label/_single_token_label/present; dev/ks/compile.py + build_state.py wired. Files: src/ks/{ast,parser,ast_emitter,binding_scope,compiler,__init__}.py, dev/dialogue/harness.py, tests/test_ks_import.py (17 tests), data/scripts/wdmh_import.ks (the user's script = wdmh.ks + `import mhall`), CONTEXT.md Import entry. Also fixed harness goal lookup to skip ask-marked entries (`not is_ask(g.kline.signature)`) — with imports the module's MHALL|ASK precedes the goal canon, and §13 says an ask never heads a goal list. 107/107 tests green; import run reaches `WDMH → SVO done j1=1.000`.

*Relevance: critical*
*Context: Extending kscript syntax with an import statement*
*Tags: kscript import compiler parser harness*

---
*Observed: 2026-09-23T16:09:08.025Z*
