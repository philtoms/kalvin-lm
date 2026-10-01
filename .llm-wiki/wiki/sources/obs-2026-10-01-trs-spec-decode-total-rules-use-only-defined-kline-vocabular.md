---
type: source
title: "Observation: TRS spec: decode total; rules use only defined kline vocabulary"
tags:
  - trs
  - significance
  - terminology
  - decode-total
  - rules-follow-definitions
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-trs-spec-decode-total-rules-use-only-defined-kline-vocabular
relevance: high
observed_at: 2026-10-01T10:49:17.053Z
source_context: Withdrawing residual/empty-adjacency terms from rules; making decode total
---

# ⭐ Observation: TRS spec: decode total; rules use only defined kline vocabulary

Eleventh round on docs/trs-significance.md: terms 'residual' and 'empty adjacency' withdrawn from the rewrite rules (§5); rules now use only kline/band/decode/entry vocabulary. Definition 6 (decode) made TOTAL: decode_D : Node → P_fin(Node) — known adjacency for a held canonical signature, ∅ otherwise; law can(decode_D(s)) = s for held s; Def 6 notes the definitional chain itself (a signature not held decodes to no adjacency; a kline carrying no adjacency values S4 per Definition 11 case 1). Counterparts already covered it (Def 2: adjacency possibly empty; Def 4: can(∅) = ⊥; Def 11 case 1) — decode was the only gap, now closed. Can_D removed again as vestigial. R2 reduced to: compute F,U,O (D12), decode, produce three klines (non-empty heads), cross-wise pairing sentence, re-enter. The claims equations (can(N_F) = F etc.) dropped from R2 — derivable from Def 6's law; rules follow definitions and never restate them. §5 intro: 'A product with no adjacency routes to R4 — the misfit asks...'. R4: 'a re-entering product, or a candidate whose neighbour supplied none (Definition 11, case 1)'. D7/D8 reworded. New method sentence added to preamble: 'Method: the rules (§5) follow the definitions (§§1–4) and never lead them — a rule uses only terms the definitions supply.' Amendment note at seven rounds. Remaining 'residual' mentions only in the changelog and the §6 cross-doc mapping table label (outside the rules).

*Relevance: high*
*Context: Withdrawing residual/empty-adjacency terms from rules; making decode total*
*Tags: trs significance terminology decode-total rules-follow-definitions*

---
*Observed: 2026-10-01T10:49:17.053Z*
