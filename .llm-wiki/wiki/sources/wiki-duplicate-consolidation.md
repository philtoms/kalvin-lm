---
type: source
title: Consolidated wiki duplicate concept pages
status: insight
category: wiki-maintenance
created: 2026-08-11
updated: 2026-08-11
slug: wiki-duplicate-consolidation
---

# Consolidated wiki duplicate concept pages

When the wiki synthesizer produced duplicate concept pages from multiple sources, the right fix was to pick one canonical page per concept (by authority: CONTEXT.md > vision > behaviour-notes), write the consolidated definition there citing all sources, **delete** the duplicate files, patch inbound links in source-page mention-lists, and `wiki_rebuild_meta`. The wiki has no redirect mechanism, and redirects only make sense for legitimate alternate names — not for correcting synthesis errors.

Three clusters consolidated (2026-08-11):
- **MTS**: deleted `mts-module-type-signatures` (a synthesis misnomer — "Module Type Signatures"; the M is Multi-token per CONTEXT.md). Canonical: [[concepts/mts-multi-token-signature]].
- **Ratification**: deleted `ratification` and `countersign`. Canonical: [[concepts/ratify]] (the CONTEXT.md glossary verb). "Countersign" = the structural mechanism; "ratification" = the S1 consequence; "ratify" = the verb — same act.
- **Significance**: deleted `siglevel` (a code symbol, not a concept) and `s1s2s3s4` (cogitate emission kinds are what the engine *does* at each level, not a separate taxonomy). Canonical: [[concepts/significance-spectrum-s1s4]]. Kept [[concepts/structural-significance]] and [[concepts/rational-significance]] distinct — they are the two flavours of significance in CONTEXT.md, not duplicates of the spectrum.

Concept count: 53 → 49. Registry clean, no dangling entries.

Lesson for future ingestions: when a new source yields a concept whose name collides with an existing page, check CONTEXT.md first for the canonical term, and prefer merging over creating a parallel page.

*Category: wiki-maintenance*

---
*Captured: 2026-08-11*

## Related

_Add links to related pages._
