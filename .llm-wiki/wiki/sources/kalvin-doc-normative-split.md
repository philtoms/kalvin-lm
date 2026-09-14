---
type: source
title: "Kalvin doc normative split: algebra vs CONTEXT.md"
status: insight
category: domain-modeling
created: 2026-09-12
updated: 2026-09-12
slug: kalvin-doc-normative-split
---

# Kalvin doc normative split: algebra vs CONTEXT.md

The two formal docs have a deliberate normative split, now stated in CONTEXT.md's intro: docs/kalvin-algebra.md is normative for the formal system and its terminology (§12 explicitly fixes terms); CONTEXT.md is normative for role names, tier mechanics (STM/Frame/LTM), the multi-agent training loop, and KScript compilation mechanics (algebra §14 defers these to CONTEXT.md). When aligning the **Domain Glossary** (CONTEXT.md): algebra terms enter as domain objects (atom, value, kline, witness, correspondence) or terms of operation (signature_of, fit, band, replace, mode, done/stuck/abandoned, ask, slot, misfit mass); each entry cites the algebra Def/§ number. Code-vocabulary check before pruning is essential — 'signature behaviour', 'space()/time() reentry' and 'Structural Significance' existed only in CONTEXT.md and were safe to replace, while signature_of (71 uses), misfit_mass, band, gap, hop are load-bearing code terms that lacked definitions.

*Category: domain-modeling*

---
*Captured: 2026-09-12*

## Related

_Add links to related pages._
