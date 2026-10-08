---
type: source
title: "Spec resync: derive claims from code, not from the doc's own restatements"
status: insight
category: trs
created: 2026-10-08
updated: 2026-10-08
slug: trs-spec-resync-derive-claims-not-restatements
---

# Spec resync: derive claims from code, not from the doc's own restatements

When a spec (docs/trs-significance.md) is updated rule-by-rule across rounds, a rule change (R2: witness lookup → decomposition reconstruction) can leave a *different section's* restatement of the same rule stale (Definition 13's entropy-flow bullet). Both sections derived R2's behaviour in different vocabulary ("held head" vs "node-clean"), so the contradiction was invisible to a rule-by-rule diff. Grep the whole doc for each rule's name and every derived quantity (F/U/O, H, witness) after changing a rule, not just the rule's own section.

Code warts worth remembering (src/trs/model.py rationalise3, as of 2026-10-08): (1) candidate-formation like-for-like filter `[nb.sig] != nb.adj` is VACUOUS — list never equals frozenset, so identity-witness neighbours (⟨s,{s}⟩) DO form candidates (visible in dev/probes/probe_ask_step2_trace.py: `Mary:[Mary]` forms an S2 candidate); fix requires a design decision. (2) module-global `seen` memo is never cleared — cross-run contamination in one process. (3) tests test_r2_never_reduces_entropy / test_r2_products_well_formed pin the OLD witness-lookup r2_products; the entropy theorem holds for the reconstruction too (equality iff node-clean) but the tests don't exercise it. See [[sources/obs-2026-10-01-trs-spec-decode-total-rules-use-only-defined-kline-vocabular]].

*Category: trs*

---
*Captured: 2026-10-08*

## Related

_Add links to related pages._
