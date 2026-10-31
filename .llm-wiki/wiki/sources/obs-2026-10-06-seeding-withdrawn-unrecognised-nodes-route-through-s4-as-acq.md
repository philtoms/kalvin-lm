---
type: source
title: "Observation: Seeding withdrawn: unrecognised nodes route through S4 as acquire asks"
tags:
  - trs
  - i-1
  - seeding
  - s4-routing
  - amendment-19
  - d16
  - bootstrap
  - asks
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-seeding-withdrawn-unrecognised-nodes-route-through-s4-as-acq
relevance: high
observed_at: 2026-10-06T10:46:44.113Z
source_context: Withdrawing kline seeding; unrecognised nodes route through S4
---

# ⭐ Observation: Seeding withdrawn: unrecognised nodes route through S4 as acquire asks

AMENDMENT 19 (D16, user ruling): I-1 DOES NOT LICENCE SEEDING — the harness's identity seeding was a D10 bypass and is withdrawn. 'Any kline entering the system whose nodes are not already known will be routed through S4 for each unrecognised node': implemented in rationalise() uniformly per pop (products' nodes are held by construction, so only presentations trigger it) — each unrecognised node n yields End('R4: unrecognised node', ⟨n, ∅⟩, E=1), an acquire ask via proposals(); processing of the entry's recognised content CONTINUES. Spec: Definition 8 restated (presentation admits any kline; unrecognised nodes route S4; 'the system never seeds'), I-1's construction paragraph rewritten (closure over initial content; at the boundary an unrecognised node is 'an ask, not an admission; the system references only what it holds, and asks for what it does not'), D16 row, changelog (19). NOTE: I-1 stays inductively sound — stores copy HELD adjacencies, so D closes over its initial content. CONSEQUENCE (from-empty inertness, reported): a fresh Domain holds nothing and nothing can ever store (candidates need held neighbours) — script runs from empty are pure ask cascades (wdmh: every entry halts with unrecognised-node asks + no-candidates; pending proposals accumulate; nothing discharges because nothing is ever held). Bootstrap is now genuinely an open design question for continuous operation — knowledge must enter via some presentation sequence against a non-empty initial D, or acquisition must be able to CREATE held content (currently it cannot: S4 asks, R1 stores). ALSO: another NON-ATOMIC edit failure found and repaired this round (the amendment-18 count and changelog note had never landed despite my verification claim — verified by grep this time, both restored). Tests: connect-from-empty now expects the connect proposal PLUS per-node acquires; 158 passing. Uncommitted.

*Relevance: high*
*Context: Withdrawing kline seeding; unrecognised nodes route through S4*
*Tags: trs i-1 seeding s4-routing amendment-19 d16 bootstrap asks*

---
*Observed: 2026-10-06T10:46:44.113Z*
