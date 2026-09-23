---
type: source
title: "kscript import: shared-emitter semantics and the ask-never-heads-a-goal fix"
status: insight
category: architecture
created: 2026-09-23
updated: 2026-09-23
slug: kscript-import-shared-emitter-semantics
---

# kscript import: shared-emitter semantics and the ask-never-heads-a-goal fix

Two semantics decisions made while implementing `import` ([[sources/obs-2026-09-23-kscript-import-statement-implemented-shared-emitter-walk]]) that generalise:

1. **Shared-emitter over chained compiles.** Instead of compiling the imported module with a fresh Compiler and reassembling entries + known_words (the -p state flow), the module emits through the SAME ASTEmitter + root BindingScope. This gives for free what the seeded flow reconstructs by hand: compound `resolution_cache` reuse (the importing script cannot drift a module's decomposition), `_canon_seen` dedup (one MHALL canon, not a temporal twin), and word-list seeding in acquisition order. The cost was two file-boundary duties the emitter didn't have: reset occurrence counters (else the module's own resolutions consume the counter its words owe the script) and clear `_pending_annotation` (else the module's trailing sigless ask leaks as the importer's first scope's prefix annotation).

2. **Goal lookup must skip ask-marked entries.** dev/dialogue/harness.py's goal lookup took the first entry matching `dbg.label == goal` with nodes — fine when entries held only the current script, but an import puts the module's `MHALL == SVO` ask (label MHALL, nodes riding, ASK_SIG set) ahead of the goal canon. §13: an ask never heads a goal list. Grading against an ask-marked goal silently deflates γ (the ASK atom joins the union). Related invariant worth remembering: a compound's *block canon* (e.g. `MHALL:[had, ALL]` from `WDMH == MHALL =>`'s nested scope) is the goal the -p flow actually picks — the first non-ask label match — not the full expansion canon.

Related: [[sources/obs-2026-09-16-wdmh-mhall-s3-via-mary-have-mhall-value-drifts-per-compile]] (the drift import eliminates), [[sources/obs-2026-09-22-mhall-reaches-its-goal-mhall-subject-verb-object-proposed-at]].

*Category: architecture*

---
*Captured: 2026-09-23*

## Related

_Add links to related pages._
