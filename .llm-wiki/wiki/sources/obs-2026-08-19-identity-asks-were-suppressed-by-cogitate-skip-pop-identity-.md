---
type: source
title: "Observation: Identity asks were suppressed by cogitate skip + pop_identity over-pop"
tags:
  - dialogue
  - engine
  - identity-asks
  - bugfix
status: observation
created: 2026-08-19
updated: 2026-08-19
slug: obs-2026-08-19-identity-asks-were-suppressed-by-cogitate-skip-pop-identity-
relevance: high
observed_at: 2026-08-19T13:24:59.267Z
source_context: Investigating why K never asks pig:[] / MHALL:[] identity asks in the lean harness
---

# ⭐ Observation: Identity asks were suppressed by cogitate skip + pop_identity over-pop

Two bugs suppressed identity (S4) asks in the lean harness: (1) cogitate's is_unknown arm removed the STM entry then fell through to idx += 1, skipping the next entry whenever the removal made idx equal the shrunk length — MHALL:[] was skipped the turn it entered STM; (2) EngineState.pop_identity's scan had no break and no is_unknown check, so grounding any kline under a signature (e.g. MHALL:[SVO] via promote cascade) popped the pending MHALL:[] unknown ask permanently, despite its docstring saying "pending Unknown ask". Fixed both (continue after S4 removal; pop first unknown only). Consequence: the engine now asks MHALL:[] early, the harness answers with the compiled word form MHALL:[Mary, had, a, little, lamb] (sig 254), and it grounds. Downstream both curricula now degrade differently: mhall.ks stops on unanswerable countersignature pairings (what:[did], duplicated asks) at step 6; wdmh-underfit.ks completes with S1=2 asks but leaves countersign residue (Subject:[SVO], Object:[SVO]) in STM. The countersignature arm (runs before the misfit arm in cogitate, no continue) is the next thing to examine.

*Relevance: high*
*Context: Investigating why K never asks pig:[] / MHALL:[] identity asks in the lean harness*
*Tags: dialogue engine identity-asks bugfix*

---
*Observed: 2026-08-19T13:24:59.267Z*
