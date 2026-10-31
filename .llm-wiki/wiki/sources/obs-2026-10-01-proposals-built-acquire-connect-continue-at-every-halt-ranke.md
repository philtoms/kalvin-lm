---
type: source
title: "Observation: Proposals built: acquire/connect/continue at every halt, ranked by effort"
tags:
  - trs
  - proposals
  - acquire
  - connect
  - continue
  - definition-15
  - halt
  - discharge
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-proposals-built-acquire-connect-continue-at-every-halt-ranke
relevance: high
observed_at: 2026-10-01T13:47:01.652Z
source_context: Building the proposals thin layer at rationalisation halts
---

# ⭐ Observation: Proposals built: acquire/connect/continue at every halt, ranked by effort

PROPOSALS built as the thin layer (amendment 14, Definition 15, docs/trs-significance.md): every halt generates a kline-shaped request — ACQUIRE (witness gap: unheld-witness arrival ⟨U,∅⟩, candidate whose neighbour supplied none, expansion that emptied its adjacency — request: present an entry witnessing the head), CONNECT (overlap gap: entry terminated with no neighbours — request: present content overlapping σ_E), CONTINUE (frontier halt: bound stopped the run — request names the minimal-effort pending kline: run on). DISCHARGE: presented entry E discharges a proposal with target t iff (σ_E ∨ can(N_E)) ∧ t ≠ ⊥ — the answer touches the gap, the answer's own rationalisation does the grounding; like the engine's asks, a proposal never grounds itself. Proposals do not enter D and are not entries; ordering among proposals is strategy (§7 extended). Code (src/trs/model.py): Proposal(kind, kline, effort) dataclass; proposals(r) maps every end to exactly one proposal in end order; RUNNER RECORDINGS closed the two null-kline gaps — the emptied-expansion end now records its candidate ⟨C,{AB}⟩ with effort, and the bound end records the frontier's minimal-effort pending entry (min by effort over the queue). All five end reasons now carry (kline, effort); proposals() has no None case in practice. Tests: unit fixtures for each kind (unheld-witness ⟨B,∅⟩ effort=1 → acquire; empty Domain → connect with the entry kline; overfit cycle bound=1 → continue naming cheapest frontier member; k(AB) terminal + k(ABC,{AB}) + entry ⟨C,{C}⟩ → emptied end records k(C,{AB})); property: every end yields exactly one proposal, kinds valid, effort == effort(kline) ∈ [0,1], and every end carries kline+effort. 152 tests passing. Uncommitted. Notable: the user's read was right — the End triple was already the raw material; the layer added the name, the mapping, the two recordings, and the spec definition.

*Relevance: high*
*Context: Building the proposals thin layer at rationalisation halts*
*Tags: trs proposals acquire connect continue definition-15 halt discharge*

---
*Observed: 2026-10-01T13:47:01.652Z*
