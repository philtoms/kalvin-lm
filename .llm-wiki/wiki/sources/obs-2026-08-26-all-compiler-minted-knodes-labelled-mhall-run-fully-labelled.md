---
type: source
title: "Observation: All compiler-minted KNodes labelled; mhall run fully labelled"
tags:
  - knode
  - labels
  - token-encoder
  - compiler
status: observation
created: 2026-08-26
updated: 2026-08-26
slug: obs-2026-08-26-all-compiler-minted-knodes-labelled-mhall-run-fully-labelled
relevance: high
observed_at: 2026-08-26T13:43:06.872Z
source_context: Labelling all KNodes including MTS/compound nodes in token_encoder
---

# ⭐ Observation: All compiler-minted KNodes labelled; mhall run fully labelled

Completed KNode labelling end-to-end in src/ks/token_encoder.py. Three mint sites now label every compiler-produced value: (1) _encode_node single-token case → KNode(tokens[0], word); (2) _emit_mts_for_tokens compound → KNode(compound, dbg_label); (3) block-canon registry lookups (node_str in _compound_sigs) → KNode(registry_value, node_str) — this was the SVO gap; (4) single-token sigs → KNode(sig_tokens[0], entry.sig) and multi-token sigs register into _compound_labels. Additionally KLine.signature is now always a KNode (constructor wraps plain ints; unlabelled signatures inherit dbg.label), and _normalize_nodes coerces all plain ints in node lists, so engine/harness-generated klines (countersigns, identities) are uniformly KNode. Verified on full mhall.ks Harness.run: 0 unlabelled nodes and 0 unlabelled signatures across replies/STM/LTM. Tests 993 passed (pre-existing WIP failures unchanged). Remaining unlabelled-in-principle: nodes minted by pure arithmetic (SIG_TAUGHT 0xC0 flags etc.) get KNode() with empty label via normalize — acceptable, no authored word exists.

*Relevance: high*
*Context: Labelling all KNodes including MTS/compound nodes in token_encoder*
*Tags: knode labels token-encoder compiler*

---
*Observed: 2026-08-26T13:43:06.872Z*
