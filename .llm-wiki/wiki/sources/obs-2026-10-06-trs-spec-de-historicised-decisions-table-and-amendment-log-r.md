---
type: source
title: "Observation: TRS spec de-historicised: decisions table and amendment log removed"
tags:
  - trs
  - spec
  - documentation
  - history
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-trs-spec-de-historicised-decisions-table-and-amendment-log-r
relevance: medium
observed_at: 2026-10-06T11:56:25.718Z
source_context: Removing decisions table and amendment history from the TRS spec
---

# 🔍 Observation: TRS spec de-historicised: decisions table and amendment log removed

AMENDMENT-HISTORY PURGE (user ruling): 'This document is not historical — that is the wiki's responsibility.' Removed from docs/trs-significance.md: (a) the entire ## Decisions section (intro + D1–D16 table), (b) the 'Amended twenty times on review' changelog paragraph. The spec now flows title → status → shared-centre prose → §0 directly. All dangling body references cleaned: '(D3)'/'(D2)'/'(D4)'/'(D10)'/'(D11)'/'(Definition 8, D16)' stripped or replaced with definition references; nine D-references in code comments and test docstrings also rewritten (model.py, trs_harness.py, test files now cite Definitions, never decision IDs). BONUS FIX: the trs_harness.py module docstring still claimed 'Word identities are seeded as the presentation substrate' — stale since amendment 19; rewritten to 'Nothing is seeded: unrecognised nodes route through S4 as acquire asks (Definition 8)'. grep-clean (zero D-refs), 156 tests passing. The decision/amendment HISTORY now lives only in the wiki (this observation and its predecessors). Uncommitted.

*Relevance: medium*
*Context: Removing decisions table and amendment history from the TRS spec*
*Tags: trs spec documentation history*

---
*Observed: 2026-10-06T11:56:25.718Z*
