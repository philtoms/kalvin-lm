---
type: source
title: "KNode labels: one composition convention, identifiers in dbg"
status: insight
category: architecture
created: 2026-10-27
updated: 2026-10-27
slug: knode-label-conventions-content-vs-identifier
---

# KNode labels: one composition convention, identifiers in dbg

Kalvin had two KNode label conventions: compiled labels (TokenEncoder identifiers like 'SVO', 'hadVerb'; signature_of ""-concatenation like 'SubjectVerbObject') vs runtime composition (KNode.merge, trs can: 'A|B'). trs.model._tokens only reads the '|' form, so meet/residue on compiled sigs degraded to provenance expressions ('hadVerb∧SVO').

Resolution (2026-10-27): one convention — content labels everywhere. [[sources/obs-2026-10-27-meet-label-extraction-broken-by-two-label-conventions]] diagnosed it; the fix made signature_of compose via KNode.merge and stopped TokenEncoder clobbering with identifiers.

Pattern to reuse: when a reference model derives names from operand structure ([[concepts/canon]] folds, meets, residuals), every producer feeding it must emit the compositional form — authored identifiers belong in provenance side-channels (dbg, registries), never on the value. Check both conventions before adding a third consumer.

*Category: architecture*

---
*Captured: 2026-10-27*

## Related

_Add links to related pages._
