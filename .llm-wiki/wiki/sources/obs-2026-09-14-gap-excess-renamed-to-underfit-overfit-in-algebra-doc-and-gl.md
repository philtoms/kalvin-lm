---
type: source
title: "Observation: gap/excess renamed to underfit/overfit in algebra doc and glossary"
tags:
  - terminology
  - kalvin-algebra
  - glossary
  - context
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-gap-excess-renamed-to-underfit-overfit-in-algebra-doc-and-gl
relevance: high
observed_at: 2026-09-14T12:20:00.112Z
source_context: Terminology rename in the normative algebra doc and domain glossary
---

# ⭐ Observation: gap/excess renamed to underfit/overfit in algebra doc and glossary

Renamed the load-bearing terms gap→underfit and excess→overfit in docs/kalvin-algebra.md (Definition 9 retitled "Underfit and overfit"; variables g/e renamed u/o; Def 10 table, classifier pseudocode, §6 covered-misfit replacements, §9 worked example, §10 slot derivation, §11 Jaccard derivation all updated; one informal "content gap" reworded to "content mismatch"). CONTEXT.md: deleted the Gap and Excess glossary entries (63→61 terms); the quantity definitions folded into the existing Shape entry; Exact/Replace/Slot entries updated. Note: src/ and dev/algebra still use gap/excess as local variable names and in docstrings citing Def 9 (e.g. src/kalvin/kline.py:282, dev/algebra/worked-example-wdmh.py prints gap=/excess= labels). Not committed.

*Relevance: high*
*Context: Terminology rename in the normative algebra doc and domain glossary*
*Tags: terminology kalvin-algebra glossary context*

---
*Observed: 2026-09-14T12:20:00.112Z*
