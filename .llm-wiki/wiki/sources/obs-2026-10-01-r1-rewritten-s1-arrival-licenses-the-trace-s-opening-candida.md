---
type: source
title: "Observation: R1 rewritten: S1 arrival licenses the trace's opening candidate"
tags:
  - trs
  - r1
  - trace
  - rewrite-rule
  - proof-of-understanding
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-r1-rewritten-s1-arrival-licenses-the-trace-s-opening-candida
relevance: high
observed_at: 2026-10-01T13:11:27.673Z
source_context: "R1 rewrite rule: store the trace's opening candidate"
---

# ⭐ Observation: R1 rewritten: S1 arrival licenses the trace's opening candidate

R1 REWRITTEN as the proof-of-understanding rule (amendment 11, docs/trs-significance.md): reaching S1 is a proof of full entry understanding; what enters D is the candidate that OPENED the trace (formed on the presented entry), not the candidate formed on the final entry — the latter is constructable from the former by reapplying the rules. Supporting changes: 'trace' now DEFINED in Definition 9 (rules follow definitions); D10 restated ('a kline enters D at R1 iff a trace it opens reaches S1'); Definition 8's persistence paragraph updated; R1 retitled 'proof of full understanding (S1): store the opening candidate'; tree diagram updated. Implementation: rationalise() queue now carries (entry, origin) pairs — first-round candidates open their traces, products inherit the opener's origin, S1 arrival stores the opener (src/trs/model.py). Tests: model test now shows one entry yielding TWO stores — canonical first-round opener ⟨AB,{A,B}⟩ AND non-canonical opener ⟨AB,{A}⟩ licensed by its S2 trace grounding at S1 (this is the new semantics in miniature); the reentry test asserts k(C,{AB}) stored and k(AB,AB) NOT stored (constructable); property no-fabrication now asserts every store's head == can(entry.adj) (all stores are first-round openers). HONEST CORRECTION during the round: I initially claimed the I-2 seam closes — WRONG: first-round S1 openers are still canonical and can collide with a held canonical head (reachable: held k(AB,{AB}) canonical + k(ABD,{A,B}) + entry ⟨A,{A,B}⟩ mints k(AB,{A,B})). Reentry stores no longer mint canonicals; the seam persists for first-round stores; graded-significance investigation stays parked. Also: the property generator closed_domains() now enforces I-2-validity (skips klines that would put a second canonical adjacency under a held head) — I-2, like I-1, is a system-maintained invariant that arbitrary hand-built domains can violate. 136 tests passing. Uncommitted.

*Relevance: high*
*Context: R1 rewrite rule: store the trace's opening candidate*
*Tags: trs r1 trace rewrite-rule proof-of-understanding*

---
*Observed: 2026-10-01T13:11:27.673Z*
