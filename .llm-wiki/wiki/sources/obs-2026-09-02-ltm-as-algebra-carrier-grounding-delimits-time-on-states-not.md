---
type: source
title: "Observation: LTM as algebra carrier; grounding delimits; time on states not signatures"
tags:
  - formalisation
  - algebra
  - ltm
  - grounding
  - time
status: observation
created: 2026-09-02
updated: 2026-09-02
slug: obs-2026-09-02-ltm-as-algebra-carrier-grounding-delimits-time-on-states-not
relevance: high
observed_at: 2026-09-02T13:51:57.560Z
source_context: "Formalising Kalvin's algebra: grounding, memory tiers, and time"
---

# ⭐ Observation: LTM as algebra carrier; grounding delimits; time on states not signatures

User's tier/time thesis (docs/kalvin-symbolic.md Appendix A.3), adopted after challenge: LTM is the algebra's carrier; Frame and STM belong to the operational semantics. Refinements made against the user's original claims: (1) Frame is not just an input representation — it is the solver's frontier (monotonic, signature-keyed, source of LTM promotions), load-bearing for the future carrier; STM is the attention trace. (2) Grounding delimits in two grades: Frame grounding delimits search steps (commits nothing), LTM grounding delimits operation completion (extends the carrier); since LTM grounding arrives via ratification, ratification is the ultimate operation delimiter — kept outside the algebra. Only LTM-grounded operations admit formal denotational semantics. (3) Time must be represented, but placed on states not operation signatures: carrier as increasing chain M_t (clean under monotonicity). Two open representations: external time (trace over grounding events) vs internal time (arrival position encoded in structure, exploiting that kline node lists are ordered — makes memory-as-single-kline temporal by construction). Critical argument: the union abstraction is NOT safe because recency preference is load-bearing — identical kline sets with different arrival orders rationalise differently. Time is not forgettable.

*Relevance: high*
*Context: Formalising Kalvin's algebra: grounding, memory tiers, and time*
*Tags: formalisation algebra ltm grounding time*

---
*Observed: 2026-09-02T13:51:57.560Z*
