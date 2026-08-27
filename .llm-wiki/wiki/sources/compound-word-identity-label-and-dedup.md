---
type: source
title: Compound-word identity labels and MTS dedup
status: insight
category: bugfix
created: 2026-08-27
updated: 2026-08-27
slug: compound-word-identity-label-and-dedup
---

# Compound-word identity labels and MTS dedup

Evaluating dev/ks/compile.py over wdmh-underfit.ks surfaced two TokenEncoder defects (fixed in 117f49e): (1) `_emit_mts_for_tokens` built its self-identity kline with `nodes=[compound]` — a bare int — so `_normalize_nodes` wrapped an *unlabelled* KNode and every compound-word identity rendered `Mary:['']`, reading like an unknown form in traces. Fix: pass `KNode(compound, dbg_label)`. (2) Every multi-token word emitted its identity twice: once as the scope-2 compound-word decomposition extra (fired while encoding the canon's nodes) and once as the scope-1 MTS identity entry. The existing dedup (`_compound_identity_emitted`) never fired for the latter because the entry took the `is_compound_ref` branch (the decomposition had registered the word in `_compound_sigs`) before reaching the IDENTITY branch. Fix: check emit-status before branch selection and drop the duplicate entry. Diagnostic lesson: to inspect symbolic entries, construct `ASTEmitter(scope=BindingScope() with pushed root)` — an unscoped emitter leaves chars raw and silently changes emission. [[concepts/mts-multi-token-signature]]

*Category: bugfix*

---
*Captured: 2026-08-27*

## Related

_Add links to related pages._
