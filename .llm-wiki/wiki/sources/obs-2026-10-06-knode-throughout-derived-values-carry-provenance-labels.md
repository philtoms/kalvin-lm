---
type: source
title: "Observation: KNode throughout: derived values carry provenance labels"
tags:
  - trs
  - knode
  - metadata
  - labels
  - opacity
  - realisation
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-knode-throughout-derived-values-carry-provenance-labels
relevance: high
observed_at: 2026-10-06T08:36:24.390Z
source_context: Reworking src/trs to carry KNode metadata throughout
---

# ⭐ Observation: KNode throughout: derived values carry provenance labels

KNode ADOPTED THROUGHOUT src/trs (user ruling: 'kline opacity is algebraic, not implementation' — the algebra's operations never look inside nodes; that is NOT permission for the implementation to coerce nodes to bare int and discard metadata). Changes: src/trs/model.py now imports KNode (kalvin.kline — KSig = KNode, int subclass with .label, hash/compare as plain values so labels never affect equality); Kline.sig: KNode, adj: frozenset[KNode]; can() folds via KNode.merge (OR-labels compose 'A|B'); residue() and new meet() return KNode with PROVENANCE labels ('A∖B', 'A∧B' when both operands labelled — derived values self-describe their derivation); decompose uses the labelled meet. dev/dialogue/trs_harness._to_kline masks at the boundary PRESERVING labels (KNode(int(sig) & WORD_MASK, sig.label)) instead of int-coercing. Effect on traces: halted entries read 'Mary|had|lamb|a|little∖Mary:[]' — the derivation names itself; display-side decomposition (alloc cover) demoted to a fallback for genuinely unlabelled values; pending proposals read 'Subject|Verb|Object∖Subject'. 158 tests green (plain ints remain valid inputs — KNode IS int; hypothesis suite unchanged). Known cosmetic: merge label order follows frozenset iteration (not authored order). REMINDER: illegal-store ruling still pending (vacuous proofs: R2 fit-branch fetches AND R3 no-op substitutions — 'Subject:[Mary]' now visible in traces). Uncommitted: harness work + content-space masking + KNode rework.

*Relevance: high*
*Context: Reworking src/trs to carry KNode metadata throughout*
*Tags: trs knode metadata labels opacity realisation*

---
*Observed: 2026-10-06T08:36:24.390Z*
