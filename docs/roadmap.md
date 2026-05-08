# Kalvin Development Roadmap: From Current System to Trainable Knowledge Agent

**Date:** 2026-05-08  
**Status:** Strategic planning document (updated)  
**Scope:** Challenges and milestones for evolving Kalvin from its current rationalisation engine toward the trainable, feedback-driven learning system described in `docs/learning.md`

**Implementation plan:** [`plans/impl/structural-grounding.md`](../plans/impl/structural-grounding.md) — detailed implementation plan for the current phase (Challenges 6 + 6b).

**See also:**
- `docs/training-loop.md` — the detailed mechanism for the training loop, which resolved several challenges previously listed here.
- `docs/training-schedule.md` — the practical perspective on training: curriculum design, priming vs. querying, Mary's World (MW) reference scenario, and the operational harness.
- `docs/extended-cogitation.md` — the design for S2 expansion: how the Cogitator reshapes partial understanding into proposals for ratification.

---

## Executive Summary

Kalvin today is a **knowledge rationalisation engine**: it receives klines, evaluates their significance against a model, and integrates them via a fast (routing) and slow (cogitation) path. `docs/learning.md` describes a **trainable agent that learns through recursive rationalisation, structured feedback, and self-directed study**.

A detailed design pass on the training loop (`docs/training-loop.md`) resolved the central architectural questions. The gap between the current system and the learning vision is smaller than originally estimated:

1. **The Training Interface** requires no new APIs. The teacher uses existing `rationalise()`, the event bus, and countersignature. Ratification is countersignature — the teacher rationalises the reciprocal kline.
2. **Scaffolding** is not a new mechanism. Each scaffolding step is rationalised through the same agent. All queries within a script share a single frame (STM → Frame → Base), so promotions from scaffolding are immediately visible to ongoing cogitation.
3. **Agency & Study** is the existing cogitator, viewed from the learning perspective. No new mechanism required.
4. **Structural Grounding** replaces the sharpened S1 promotion. S1 is determined by structure (signature describes nodes or countersigned). After ratification, all STM klines involved in the ratification process are promoted to frame — including S4 identity klines. Frames hold S4–S1.
5. **Curriculum & Session Management** is handled by one agent per script for the MVP. All queries within a script share the same frame. Future work adds temporal events and trajectory analysis.

What remains are genuine implementation challenges: implementing structural grounding, building a teacher coordinator, calibrating boundaries, and evaluating model quality.

---

## Current State: What Exists Today

| Capability | Status | Notes |
|---|---|---|
| KLine data model | ✅ Complete | `kline.py`, literal/non-literal node types |
| Signature computation | ✅ Complete | `signature.py`, OR-reduction, `signifies()` |
| Mod tokenizer (Mod32/Mod64) | ✅ Complete | Packed + literal encoding |
| BPE tokenizer | ❌ Not started | Optional dependency; not needed for MVP |
| STM (bounded rolling index) | ✅ Complete | `stm.py`, dual-keyed, eviction |
| Model (3-tier: STM → Frame → Base) | ✅ Complete | `model.py`, ~600 lines |
| `model.expand()` generator | ✅ Complete | Connotation discovery, S2 signifies, S3 bridging |
| Significance pipeline | ✅ Complete | Distance → inversion → boundary classification |
| Routing (`_route`) | ✅ Complete | Node-membership test, agent-level |
| Agent rationalisation pipeline | ✅ Complete | 6-phase fast path |
| Cogitator (background thread) | ✅ Complete | Work items, countersignature, re-rationalisation |
| Response sampling (temperature, top-k, top-p) | ✅ Complete | Boundary shifting, streaming pipeline |
| KScript DSL | ✅ Complete | Lexer, parser, compiler, decompiler |
| Events (pub/sub) | ✅ Complete | `ground`, `frame`, `done` |
| Persistence (JSON/binary) | ✅ Complete | JSON + binary serialization, save/load |
| **All existing tests** | ✅ 329 passing | No regressions |
| **Structural grounding** | 🔄 Plan ready | [`plans/impl/structural-grounding.md`](../plans/impl/structural-grounding.md) Phase A |
| **Extended cogitation** | 🔄 Plan ready | [`plans/impl/structural-grounding.md`](../plans/impl/structural-grounding.md) Phase A+ |
| **Frame factory** | ❌ Not started | Method for creating new frames (kalvin instances) |
| **Teacher implementation** | ❌ Not started | External coordinator using existing APIs |
| **Model quality evaluation** | ❌ Not started | Test harness, metrics, calibration |

---

## Resolved Challenges

The following were originally listed as challenges. The training loop design pass resolved them.

### ~~Challenge 1: The Training Interface~~ — Resolved

**Original assumption:** Required new APIs (`confirm`, `correct`, `instruct`), new data structures (`Proposal`), and transactional model mutations (rollback on `correct`).

**Resolution:** The teacher uses existing APIs. No new protocol needed:

- **Submit query:** `kalvin.rationalise(query_kline)` — already exists, returns True/False.
- **Receive proposals:** Subscribe to the event bus, receive `frame` events — already exists.
- **Ratify (confirm):** `kalvin.rationalise(reciprocal_kline)` — creates countersignature, achieving structural S1. Already exists.
- **Correct:** No action. The proposal stays in STM. No rollback needed. The teacher simply ignores the proposal and moves on to scaffolding.
- **Instruct (scaffold):** `kalvin.rationalise(scaffold_kline)` on a new frame — already exists.

The teacher is an external coordinator that composes existing API calls. It does not need new agent methods, new data structures, or transactional semantics. See `docs/training-loop.md` §"The Teacher's Responsibilities".

### ~~Challenge 2: Scaffolding & Recursive Learning~~ — Resolved

**Original assumption:** Required a foreground interactive process where the agent pauses rationalisation, waits for feedback, and manages recursive depth limits. Also assumed monotonic significance tracking.

**Resolution:** Scaffolding uses the same agent. Each scaffolding kline goes through `rationalise()` on the same agent — there is one frame per script, not per rationalise call. All queries within a script share the same STM → Frame → Base, so promotions from scaffolding are immediately visible to ongoing cogitation without any cross-frame sharing mechanism.

No foreground pausing needed. The teacher submits queries and listens to events from the single agent's event bus. Cogitation runs on the background thread. Recursive depth is managed by the teacher (it decides how many scaffolding rounds to attempt).

Monotonicity is a guarantee about grounded information, not significance values. The frame is append-only. See `docs/training-loop.md` §"Monotonicity: Constructive Grounding".

### ~~Challenge 3: Agency & Self-Directed Study~~ — Resolved

**Original assumption:** Required a new study mechanism with prioritisation, re-visitation, readiness flagging, and study budgets.

**Resolution:** Study is the existing cogitator. The cogitator already processes S2/S3 work items in the background, checks countersignature, and re-rationalises. This *is* study. The distinction between "training" and "study" is teacher presence, not system behavior.

Structural grounding ensures that promotion occurs after ratification and applies to all participating STM klines — not just S1 klines. S4 identity klines and S2/S3 partial klines involved in ratification are promoted alongside S1 klines, enriching the frame for future cogitation.

Future enhancements (prioritisation, re-visitation, readiness thresholds) may improve study quality but are not required for the MVP.

### ~~Challenge 4: The S1 Ratification Boundary~~ — Resolved

**Original assumption:** Required redefining S1 from computed to confirmed, introducing a `RatificationStatus` enum, breaking the fast path, making countersignature provisional, and solving a bootstrapping problem.

**Resolution:** S1 remains computed. The change is structural:

1. **Structural S1** — a kline is S1 if its signature fully describes its nodes (`make_signature(nodes) == signature`) or it is countersigned. This is a structural interrogation, not a boundary check.
2. **Canonical S1** (all-literal, self-grounded) is self-ratifying. No change. These auto-promote immediately.
3. **Promotion after ratification** — all STM klines involved in the ratification process are promoted to frame, regardless of their individual significance level. Frames hold S4–S1.
4. **S4 identity klines** — ratified S4 means the system knows a kline exists with verified identity. Promoting S4 klines enriches the model for future cogitation.

Ratification is countersignature — the teacher rationalises the reciprocal kline, creating a mutual cross-reference that makes the proposal structurally S1. After ratification, promotion applies to all participating klines.

No new enum, no new state, no backward compatibility issue, no bootstrapping problem. The fast path is unchanged. The change is in the promotion logic: promotion follows ratification and is not restricted to S1.

### ~~Challenge 5: Convergence & Monotonicity~~ — Resolved

**Original assumption:** Required proving that significance monotonically increases during training, and enforcing this invariant.

**Resolution:** Monotonicity is a guarantee about **grounded information**, not significance values. Significance may fluctuate as the model grows (new candidates, altered distances). But grounded information is permanent: once a kline is promoted to a frame, it stays. The model can only grow richer — never poorer. The frame is append-only.

Convergence is an empirical question, not a formal guarantee. The teacher decides when understanding is sufficient. The system provides the mechanisms (rationalise, events, countersignature, agent frame); the teacher provides the judgment.

---

## Remaining Challenges

### Challenge 6: Structural Grounding

**Status:** Plan ready — [`plans/impl/structural-grounding.md`](../plans/impl/structural-grounding.md) Phase A  
**Effort:** Small–medium (changes to promotion logic and ratification path)

The current cogitator auto-promotes any result classified as S1. This needs to change to a model where:

1. **S1 is determined by structure** — a kline is S1 (grounded) if its signature fully describes its nodes or it is countersigned. This is a structural interrogation, not a boundary classification.
2. **Promotion occurs after ratification** — all STM klines involved in the ratification process are promoted to frame, not just S1 klines.
3. **Frames hold S4–S1** — S4 identity klines are promoted alongside S1 grounded klines. Ratified S4 means the system knows a kline exists with verified identity.

**Why:** Ensuring all klines involved in ratification are available to future rationalisation enhances Kalvin's ability to cogitate beyond simple pattern matching. A richer frame provides more graph topology for the cogitator to traverse.

**Implementation:**
- Modify the structural grounding check: `make_signature(nodes) == signature` or `is_countersigned(kline)`.
- Modify the promotion path so that ratification triggers promotion of all participating STM klines, not just the ratified kline.
- Ensure S4 identity klines are correctly promoted when involved in ratification.

**Future work:** Over-fitting and under-fitting in the context of frames holding mixed significance levels is now covered in `docs/extended-cogitation.md` (Challenge 6b).

### Challenge 6b: Extended Cogitation (S2 Expansion)

**Status:** Plan ready — [`plans/impl/structural-grounding.md`](../plans/impl/structural-grounding.md) Phase A+
**Depends on:** Challenge 6 (Structural Grounding)
**Effort:** Medium

When countersignature fails for an S2 result, the Cogitator currently moves on.
Extended cogitation adds a new phase: the Cogitator attempts to **expand** the
candidate kline toward canonical status by reshaping its nodes to match its
signature.

For a candidate with signature `S` and nodes signature `N`:

- **Underfitting** (`S & ~N != 0`): search the model for klines whose
  signatures contribute to the gap, add their nodes.
- **Overfitting** (`N & ~S != 0`): remove excess nodes, verify the removed
  group's signature exists in the model.
- **Dual misfit**: both operations may apply.

**Universal constraint:** every signature generated during expansion must
already exist in the model (no invention, no data loss, ratifiability).

**Why:** This is the mechanism for self-directed study. The Cogitator works
through partial understanding, generating proposals that a teacher can
ratify. Without it, S2 klines that fail countersignature are simply abandoned.

**Full design:** `docs/extended-cogitation.md`

**Deliverables:**
- Misfit classification in the Cogitator's `_process` method
- `generate_expansions()` — searches model for valid node additions/removals
- `validate_expansion()` — enforces universal constraint
- Proposal emission via existing `frame` event mechanism
- Tests: underfit expansion, overfit expansion, dual misfit, constraint violations

### Challenge 7: Teacher Infrastructure

**Status:** Not started (next after Challenges 6 + 6b)  
**Effort:** Small–medium

The training loop uses one agent per script — a single `Agent()` instance whose frame accumulates all ratified knowledge. No frame factory is needed; the teacher constructs agents directly:

```
agent = Agent(model=Model(base=existing_model))
```

Sampling parameters are agent-level properties, adjustable between rationalise calls:

```
agent.sampling = Sampling(temperature=1.5, top_k=40, top_p=0.95)
```

All queries for a script are rationalised through this single agent. There is no frame stack, no cross-frame sharing, and no multi-agent event subscription.

**The teacher** is an external coordinator (Python class, initially) that:
1. Compiles KScript into queries and expectations
2. Constructs an agent for each script
3. Subscribes to the agent's event bus
4. Compares proposals against expectations (kline equality for MVP)
5. Countersigns by rationalising the reciprocal
6. Constructs corrective scaffolding when proposals don't match
7. Adjusts sampling parameters as needed

**Risk:** Low. The teacher is mostly orchestration of existing APIs. No new agent methods needed.

### Challenge 8: Model Quality & Evaluation

**Status:** Not started  
**Effort:** Medium–ongoing

The training loop's effectiveness depends on model quality. No quality metrics exist today.

**Needs:**
1. **Test harness** — pairs of (query, model state, expected significance level). The current MHALL test is anecdotal.
2. **Quality metrics** — coverage (fraction of expected klines that can be grounded), precision (when the system claims S1, is it correct?), learning rate (scaffolding rounds to reach S1).
3. **Boundary calibration** — whether `_S3_BIAS`, `_pack`, `_S2_S3_DISTANCE`, and `_TEMP_SCALE` produce useful classifications for real learning.
4. **Performance profiling** — candidate retrieval is O(N). The inverted bit-index optimization may be needed sooner than expected as the model grows through training.
5. **Tokenizer capacity** — Mod(N) vocabulary saturation point for non-trivial domains.

**Risk:** Medium. This is empirical work. The system may work well with current boundaries, or it may need significant tuning. The only way to find out is to run training sessions and measure.

### Challenge 9: KScript Metadata (Future)

**Status:** Not started  
**Effort:** Small  
**Priority:** Low (agent-level sampling parameters work for MVP)

KScript currently has no mechanism for encoding sampling parameters alongside a script. For the MVP, the teacher sets parameters on the agent directly. Future work may add metadata annotations to KScript:

```
@temp=1.5 @top_k=20
MHALL => SVO
  ...
```

This is polish, not a blocker. The training loop works without it.

### Challenge 10: Temporal Events & Trajectory Analysis (Future)

**Status:** Not started  
**Effort:** Medium  
**Priority:** Low (MVP uses single-agent frame without temporal analysis)

The agent's frame captures the full learning trajectory for a script. Future work adds temporal metadata to events so the teacher (or a kalvin-as-tutor) can analyze the trajectory: rate of convergence, effectiveness of scaffolding, etc.

This is the foundation for kalvin-as-tutor: a kalvin instance that acts as teacher for another kalvin instance, using reentrant rationalisation instead of exact kline equality for proposal evaluation.

---

## Dependency Graph (Updated)

```
Challenge 6:  Structural Grounding
    ↓
Challenge 6b: Extended Cogitation (S2 Expansion)
    ↓
Challenge 7:  Teacher Infrastructure
    ↓
Challenge 8:  Model Quality & Evaluation (ongoing, parallel)

Challenge 9:  KScript Metadata          (future, low priority)
Challenge 10: Temporal Events & Tutor    (future, low priority)
```

The dependency chain is now linear and shallow: structural grounding → extended cogitation → build teacher → evaluate quality. The future challenges (9, 10) are independent enhancements.

---

## Revised Phasing

### Phase A: Structural Grounding (Challenge 6)
**Estimate:** 1–2 days  
**Risk:** Low  
**Implementation:** [`plans/impl/structural-grounding.md`](../plans/impl/structural-grounding.md) §1

Implement structural grounding: S1 determined by structure, promotion after ratification for all participating klines, frames holding S4–S1.

**Deliverables:**
- `model.is_s1()` — structural S1 check (canonical or countersigned)
- `model.promote_participating(query, candidate)` — promote all STM klines involved in ratification
- Modified agent promotion path: explicit promotion in rationalise() and cogitation
- Tests: structural S1 detection, S4 identity promotion, multi-significance frame contents
- No spec changes required (behavioral change within existing spec)

### Phase A+: Extended Cogitation (Challenge 6b)
**Estimate:** 3–5 days
**Risk:** Medium
**Depends on:** Phase A
**Implementation:** [`plans/impl/structural-grounding.md`](../plans/impl/structural-grounding.md) §2

Implement S2 expansion in the Cogitator: misfit classification, node addition/removal
with ratification constraint enforcement, proposal emission.

**Deliverables:**
- `model.classify_misfit()` — underfit/overfit/dual classification
- `model.generate_expansions()` — proposal generation with model search
- Cogitator `_process()` Phase 2 — expansion after failed countersignature
- Proposal emission via existing `frame` events
- Tests for all expansion types
- Spec updated: `specs/agent.md` §S2 Expansion, `specs/model.md` expansion API

### Phase B: Teacher (Challenge 7)
**Estimate:** 1–2 weeks  
**Risk:** Low

Implement a teacher class and test the three training scenarios from `docs/training-loop.md`.

**Deliverables:**
- `Teacher` class that orchestrates the training loop
- Tests for Scenarios A, B, C from `docs/training-loop.md`
- Single-agent event subscription

### Phase C: Model Quality & Evaluation (Challenge 8)
**Estimate:** 2–3 weeks (ongoing)  
**Risk:** Medium

Build the evaluation infrastructure. Run training sessions. Calibrate boundaries.

**Deliverables:**
- Significance test suite
- Quality metrics (coverage, precision, learning rate)
- Boundary calibration data
- Performance profiling results

### Phase D: Future Enhancements (Challenges 9, 10)
**Estimate:** TBD  
**Priority:** After MVP is proven

KScript metadata, temporal events, kalvin-as-tutor. These are not required for the training loop to work. They enhance the teacher's capability and the system's introspection.

---

## Summary of Open Technical Questions (Updated)

| # | Question | Affects | Priority | Status |
|---|---|---|---|---|
| 1 | How to determine structural S1 (signature describes nodes or countersigned)? | Challenge 6 | High | Design complete — structural interrogation |
| 2 | How to identify all STM klines involved in a ratification process? | Challenge 6 | High | TBD |
| 3 | Frame factory: method on Agent or standalone? | Challenge 7 | High | Resolved — no frame factory needed, one agent per script |
| 4 | How many klines before candidate retrieval O(N) becomes a bottleneck? | Challenge 8 | Medium | Empirical |
| 5 | Are current boundary constants well-calibrated for training? | Challenge 8 | High | Empirical |
| 6 | When does Mod(N) vocabulary saturation require BPE transition? | Challenge 8 | Low | Empirical |
| 7 | What does a non-trivial training curriculum look like? | MW (`training-schedule.md`) | Medium | Active research |

---

## Resolved Questions (Previously Open)

| # | Original Question | Resolution |
|---|---|---|
| 1 | How should `Proposal` represent partial understanding? | No `Proposal` type needed. Events carry the kline. The teacher compares using kline equality. |
| 2 | Should `correct` roll back model mutations or prevent them entirely? | Neither. Proposals that are not ratified are not promoted. The teacher simply moves on. |
| 3 | Can Kalvin articulate what it doesn't know (scaffold requests)? | Not in MVP. The teacher diagnoses gaps and constructs scaffolding. |
| 4 | Is the distance algorithm monotonic with respect to model growth? | Monotonicity is about grounded information (append-only frames), not significance values. No formal guarantee needed. |
| 5 | What significance threshold indicates "ready for ratification"? | For MVP: exact kline equality. For future: reentrant rationalisation (kalvin-as-tutor). |
| 6 | How to bootstrap S1 in an empty model under ratification? | Canonical S1 (all-literal, self-grounded) is self-ratifying. No bootstrapping problem. |
| 7 | Should study and training run concurrently, or must they be serialised? | They already run concurrently. Each frame's cogitator is independent. |
| 8 | What does a Kalvin curriculum look like for a non-trivial domain? | Deferred to Phase D. |

---

## Conclusion

The original roadmap overestimated the gap between the current system and the learning vision. The detailed training loop design pass revealed that the existing architecture — rationalise, events, cogitation, countersignature — already provides most of what's needed. The training loop is an *usage pattern* over existing APIs, not a new subsystem.

The two concrete implementation tasks are:
1. **Structural grounding** — S1 determined by structure, promotion after ratification for all participating klines.
2. **Teacher** — orchestration code that composes existing APIs into the training loop pattern.

Everything else (model quality evaluation, boundary calibration, future enhancements) is empirical work and polish that follows from a working training loop.

The central insight holds: **learning and rationalisation are the same process, applied recursively**. The training loop makes this concrete — each rationalise call processes a query through the same frame, each countersignature is a ratification, and the growing frame is the learning trajectory.
