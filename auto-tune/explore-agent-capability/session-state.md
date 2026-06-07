# Auto-Tune Session State

## Goal
Investigate LLM-agent capability using the default curriculum as a basis for exploratory lessons. Understand what triggers rationalisation, escalation, and reactive scaffolding, then exercise those paths.

## Done Criteria
Open session — learn and decide as we go.

## Session
- **Name:** explore-agent-capability
- **Curriculum:** curricula/conflict-drill.md
- **Branch:** auto-tune/explore-agent-capability
- **Started:** 2026-06-07

## Current Phase
observing

## Next Action
Analyze run 4 findings. The conflict-drill triggered budget exhaustion — the first escalation seen. Decide: (a) increase max_reactive_rounds and retry, (b) design curriculum to test escalation recovery, or (c) document findings and close session

## Run Log

### Run 4 (latest) — conflict-drill
- **Code changes:** wrote custom curriculum `curricula/conflict-drill.md`; switched config to use it
- **Curriculum:** 4 lessons — identities A,B,C,D → A==B countersign → AB=>D C canonize → E + ACE=>B D
- **Observation:**
  - **Lesson 1:** 4/4 auto-countersigned (identities)
  - **Lesson 2:** A==B countersign → S1 fast path, 6/6 satisfied
  - **Lesson 3:** AB=>D C compiled to 6 entries. AB=>A B canonize S1. D,C grounded S1. **AB>D C hit S3 misfit** (query sig=0x6, nodes=[D,C] vs candidate B identity sig=0x4). LLM round 1: confidence **0.65** (lowest seen!), scaffolding "(address underfitting: proposal B => A B does not cover expected relationship AB > D C)\nAB > D C\nAB". Lesson 3 complete: 7/8 satisfied.
  - **Lesson 4:** Compiled 7 entries (E, ACE=>A C E, ACE=>B D, + identities). **Cogitator still processing lesson 3's leftover work items → AB>D C events arrive during lesson 4.** Reactive rounds escalate:
    - Round 1 (0.75): "Scaffold missing identities... Composite identity AB"
    - Round 2 (0.75): "bridge underfit gap... AB == A B"
    - Round 3 (0.85): "AB == A B\nAB > D C" — best scaffolding so far
    - Round 4 (0.75): **DUAL misfit** (underfit=True, overfit=True, gap=0x4, mask=0x2) → "AB = B A\nAB > D C"
    - Round 5: **budget_exhaustion** → escalated. Round 6 also escalated.
  - **Total LLM calls:** 5 (1 lesson 3 + 4 lesson 4). Total LLM latency: ~122s across 5 calls.
  - Lesson 4 complete: 7/11 satisfied. Curriculum complete.
  - Post-completion: multiple promote_participating calls (AB sig=0x6, ACE sig=0x2a).
- **Verdict:** success — exercised S3, reactive scaffolding, dual misfit, budget exhaustion escalation, cross-lesson cogitation spillover. First escalation seen.

### Run 3 — mhall-svo-equivalence
- 5 lessons, all S1 fast path, 23/23 satisfied, zero S2/S3, LLM never invoked.
- **Verdict:** graduated curriculum eliminates reactive scaffolding entirely

### Run 2 — first-steps-s2
- 1 S3 event in lesson 5, LLM confidence 0.95, 6/7 satisfied.
- **Verdict:** LLM exercised, valid scaffolding

### Run 1 — first-steps baseline
- Clean pass, 3 lessons, all auto-countersigned.
- **Verdict:** baseline — too simple

## Patterns & Notes

### Key Finding: Conflict-Driven Reactive Cascade
- **Prior countersigns create populated klines** that conflict with later canonize entries
- Single conflict (AB=>D C with existing {A:[B]}) generated **6+ S3 candidates** in the cogitator
- The cogitator processes candidates sequentially — each S3 expansion triggers its own reactive round
- Result: cascading reactive rounds that exhaust the budget

### Key Finding: Cross-Lesson Cogitation Spillover
- The cogitator is a background thread — work items from lesson 3 arrived during lesson 4
- The reactor processes these with lesson 4's state (entries, round counter)
- This means lesson 4's reactive budget was consumed by lesson 3's leftover work items
- **Lesson 4's own entries (ACE=>B D) were never independently processed** — they got lost in the spillover

### Key Finding: Dual Misfit (Run 4, Round 4)
- First observed dual misfit: underfit=True AND overfit=True
- gap=0x4 (missing bits from B's sig), mask=0x2 (excess bits from A's presence)
- The dual misfit triggers all three expansion generators: underfit, overfit, and dual
- LLM diagnosed it correctly: "both underfitting and overfitting"

### Key Finding: LLM Confidence Range
- Run 2: 0.95 (single misfit, clean context)
- Run 4: 0.65–0.85 (degraded by accumulated model state, repeated AB>D C failures)
- Lower confidence correlates with accumulated failed scaffolding rounds

### Key Finding: Entry Satisfaction Degrades Across Runs
- Run 2: 6/7 (85%) — 1 entry unsatisfied
- Run 4 lesson 3: 7/8 (87.5%) — 1 entry unsatisfied
- Run 4 lesson 4: 7/11 (63.6%) — 4 entries unsatisfied
- Budget exhaustion leaves entries permanently unsatisfied

### Curriculum Design Spectrum
- **S1-only:** graduated curricula (mhall-svo-equivalence) — safe, no LLM needed
- **Mild S3:** first-steps-s2 — one conflict, one LLM call, high confidence
- **Cascade:** conflict-drill — multiple conflicts, cascading S3, budget exhaustion

## Files Modified
- `curricula/conflict-drill.md` (new)
- `auto-tune/explore-agent-capability/config.json` (curriculum path updates)
