---
type: source
title: "Observation: wdmh step-4 trace: 5 HLAL rows = one reentry, 11 pairings, elision"
tags:
  - trs
  - trace-table
  - wdmh
  - cogitator
  - events
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-wdmh-step-4-trace-5-hlal-rows-one-reentry-11-pairings-elisio
relevance: medium
observed_at: 2026-11-03T12:54:22.845Z
source_context: "Explaining wdmh step-4 trace: HLAL entered 5 times, resolved twice"
---

# 🔍 Observation: wdmh step-4 trace: 5 HLAL rows = one reentry, 11 pairings, elision

Reading wdmh step-4 trace: the event table's row unit is a PAIRING EVENT, not an entry. One reentry of HLAL (o-side of the opening WDMH×MHALL pairing, value 0x1e00000000 = had|a|little|lamb) paired 11 held candidates in one for-loop pass but only 5 rows are visible because _render_events elides plain pair events (shown iff frame minted, kind!=pair, or depth==1): 3 frame-minting pair events (the three MHALL-sig klines: [had,ALL], canon, [SVO]) + 2 closes. "resolved" = a close (both sides settled at pairing time). The first close (vs MHALL:[had,ALL], F==E mints frame HALL, U=O=∅) pops the entry mid-loop but the candidate loop keeps pairing — hence rows continue after a resolve. Second close = ALL canon + had-residual now covered by the row-2 mint. After close, remaining candidates MHALL canon / MHALL:[SVO] push the Mary and SVO depth-3 children (Mary pushed last → top → next rows). The depth-1 opening row appears twice: blank at start, 'grounded' at end — entries stay on the stack below their children and re-run their candidate loop when they re-surface, closing once descendant frames cover U and O (no back-propagation primitive; re-pairing + memo is the mechanism). Display quirk: same value renders HLAL (E/O columns, via KNode compiler label 'had|lamb|a|little') and HALL (frame column, via int→alloc word order 'had a little lamb').

*Relevance: medium*
*Context: Explaining wdmh step-4 trace: HLAL entered 5 times, resolved twice*
*Tags: trs trace-table wdmh cogitator events*

---
*Observed: 2026-11-03T12:54:22.845Z*
