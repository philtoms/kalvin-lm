---
type: source
title: "Observation: Thin WDMH scope: exact-value trawl test, late scaffold feed, ALL drift"
tags:
  - trawl
  - scope
  - hop
  - engine
  - ask
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-thin-wdmh-scope-exact-value-trawl-test-late-scaffold-feed-al
relevance: high
observed_at: 2026-09-16T16:05:17.109Z
source_context: Diagnosing the thin WDMH→MHALL Def 23 scope
---

# ⭐ Observation: Thin WDMH scope: exact-value trawl test, late scaffold feed, ALL drift

Diagnosed the thin Def 23 scope for the WDMH ask with goal MHALL (user saw 3 klines: 'had, what, Mary' = MHALL:[had,ALL] via node had, the ask itself, MHALL:[Mary,had,A,L,L] via node Mary). Three independent causes, reproduced live (dev/dialogue/probe_live_scope.py, probe_scope_values.py): (1) **The trawl's touch test is exact-value membership** — `int(k.signature) in reach` / `int(n) in reach` — while klines touch at the WORD-BIT level: MarySubject's sig contains Mary's word bit, hadVerb's contains had's, yet neither scopes (round 2 = none). Def 22's coverage already tests value intersection (`any(int(n) & kc ...)`); the trawl is inconsistent with selection's semantics. Simulated word-bit-masked touch (mask 0x7FFFFFFF00000000) on the post-run reservoir: 16/18 klines over 4 structured rounds (round 1: connotes + ask + both MHALLs + had:[did,have] + ALL:[A,L,L]; round 2: SVO via Subject; round 3: QueryObject/lamb via Object; round 4: ALLQuery + ALL:[a,little,lamb]) — exactly the 'many useful klines' expected, with genuine depth structure. NOTE: naive full-value AND over-collapses (18/18 in one round — token-id bits leak); the word-bit mask is the honest atom level. (2) **Feed ordering starves the reservoir**: the ask (group 1 T03) hops before the scaffold group (group 2) is fed; `did have => had` is the LAST feed at script EOF and had:[did,have] is never cogitated — the verb bridge is absent from every WDMH hop. (3) The mhall corpus chain is severed at ALL by value drift: the goal's ALL node = char compound A|L (0xc0000000004d) vs corpus canon ALL:[a,little,lamb] = word compound 0x1c00001efd — exact-value trawl can't cross; word-bit masking does (round 4). Identities stay terminal-filtered by design. wdmh feed order: g1 T01 (MHALL:[had,ALL]+identities+MHALL canon), T02 MHALL:[MHALL] S4, T03 the ask; g2 T01 (ALLO/whatO/ALL:[A,L,L]), T02 had:[did,have] then EOF.

*Relevance: high*
*Context: Diagnosing the thin WDMH→MHALL Def 23 scope*
*Tags: trawl scope hop engine ask*

---
*Observed: 2026-09-16T16:05:17.109Z*
