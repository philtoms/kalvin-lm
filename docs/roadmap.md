# Kalvin Development Roadmap

**Date:** 2026-05-14  
**Status:** Active  
**Scope:** Implementation challenges for building the training system described in the origin documents.

**See also:**
- `docs/kscript-intro.md` — the training system from KScript's perspective.
- `docs/kalvin-intro.md` — the training system from Kalvin's perspective.
- `docs/learning-and-training.md` — learning, training, and the training loop mechanism.
- `docs/extended-cogitation.md` — S2 expansion: how the Cogitator reshapes partial understanding.
- `plans/impl/structural-grounding.md` — detailed implementation plan for the current phase.

---

## Summary

The origin documents describe a training system of three components: a KScript compiler, an agent, and a Kalvin instance, coordinated by a harness. The agent compiles scripts, presents klines to Kalvin, evaluates Kalvin's responses, and decides what to teach next. Kalvin rationalises every kline the same way regardless of context.

The existing implementation already provides most of what the training system needs:

- **`rationalise()`** — the single entry point for all kline processing. No new APIs required.
- **Events** — `frame`, `ground`, `done` events carry proposals and results to the agent.
- **Countersignature** — the ratification mechanism. The agent rationalises the reciprocal kline.
- **Cogitator** — background S2/S3 processing. This is Kalvin's study.
- **KScript** — compilation from scripts to klines.

What remains are four implementation challenges, addressed in dependency order.

---

## Current State

| Capability | Status | Notes |
|---|---|---|
| KLine data model | ✅ Complete | `kline.py`, literal/non-literal node types |
| Signature computation | ✅ Complete | `signature.py`, OR-reduction, `signifies()` |
| Mod tokenizer (Mod32/Mod64) | ✅ Complete | Packed + literal encoding |
| STM (bounded rolling index) | ✅ Complete | `stm.py`, dual-keyed, eviction |
| Model (3-tier: STM → Frame → Base) | ✅ Complete | `model.py` |
| `model.expand()` generator | ✅ Complete | Connotation discovery, S2 signifies, S3 bridging |
| Significance pipeline | ✅ Complete | Distance → inversion → boundary classification |
| Routing (`_route`) | ✅ Complete | Node-membership test |
| Agent rationalisation pipeline | ✅ Complete | 6-phase fast path |
| Cogitator (background thread) | ✅ Complete | Work items, countersignature, re-rationalisation |
| KScript DSL | ✅ Complete | Lexer, parser, compiler, decompiler |
| Events (pub/sub) | ✅ Complete | `ground`, `frame`, `done` |
| Persistence (JSON/binary) | ✅ Complete | JSON + binary serialization |
| **All existing tests** | ✅ 294 passing | No regressions |
| **Structural grounding** | ✅ Complete | `model.is_s1()`, `promote_participating()`, structural S1 check |
| **Extended cogitation** | ✅ Complete | Misfit classification, expansion proposals, S2 reshaping |
| **Agent infrastructure** | ❌ Not started | External coordinator using existing APIs |
| **Model quality evaluation** | ❌ Not started | Test harness, metrics, calibration |

---

## The Challenges

### Challenge 1: Structural Grounding

**Status:** ✅ Complete  
**Effort:** Small–medium

The current cogitator auto-promotes any result classified as S1 by boundary check. Under the training model, S1 is determined by structure, not by classification: a kline is grounded if its signature fully describes its nodes (`make_signature(nodes) == signature`) or it is countersigned.

After ratification, all STM klines involved in the ratification process are promoted to frame — not just the ratified kline, but supporting S4 identity klines and S2/S3 partial klines as well. Frames hold S4–S1. This enriches the model for future rationalisation and cogitation.

**Implementation:**
- `model.is_s1()` — structural S1 check (canonical or countersigned)
- `model.promote_participating()` — promote all STM klines involved in ratification
- Modify agent promotion path: explicit promotion at ratification points, not implicit in `_publish()`

### Challenge 2: Extended Cogitation (S2 Expansion)

**Status:** ✅ Complete  
**Depends on:** Challenge 1  
**Effort:** Medium

When countersignature fails for an S2 result, the Cogitator currently discards it. Extended cogitation adds the ability to reshape S2 klines toward canonical status by adding missing nodes or removing redundant ones, then emitting the result as a proposal for the agent to ratify.

For a candidate with signature `S` and nodes signature `N`:
- **Underfitting** (`S & ~N != 0`): search the model for klines whose signatures fill the gap, add their nodes.
- **Overfitting** (`N & ~S != 0`): remove excess nodes, verify the removed group's signature exists in the model.
- **Dual misfit**: both operations may apply.

Every signature generated during expansion must already exist in the model — no invention, no data loss.

**Full design:** `docs/extended-cogitation.md`

### Challenge 3: Agent Infrastructure

**Status:** 🔄 Next  
**Depends on:** Challenge 2 (✅)  
**Effort:** Small–medium

The agent is an external coordinator that compiles KScript, constructs one Kalvin instance per script, submits queries via `rationalise()`, subscribes to events, and ratifies proposals via countersignature. No new Kalvin APIs are needed — the agent orchestrates existing ones.

The agent:
1. Compiles KScript into queries and expectations.
2. Constructs a Kalvin instance for each script.
3. Subscribes to the instance's event bus.
4. Compares proposals against expectations (kline equality for MVP).
5. Countersigns acceptable proposals by rationalising the reciprocal.
6. Constructs corrective scaffolding when proposals don't match.
7. Re-submits queries when proposals are not acceptable within a timeout.

**Risk:** Low. The agent is orchestration of existing APIs.

### Challenge 4: Model Quality and Evaluation

**Status:** Not started  
**Depends on:** Challenge 3  
**Effort:** Medium–ongoing

The training system's effectiveness depends on model quality. No quality metrics exist today.

**Needs:**
1. **Test harness** — pairs of (query, model state, expected significance level).
2. **Quality metrics** — coverage (fraction of expected klines grounded), precision (when the system claims S1, is it correct?), learning rate (scaffolding rounds to reach S1).
3. **Boundary calibration** — whether current constants produce useful classifications for real learning.
4. **Performance profiling** — candidate retrieval is O(N). An inverted bit-index may be needed as the model grows.
5. **Tokenizer capacity** — Mod(N) vocabulary saturation point for non-trivial domains.

**Risk:** Medium. This is empirical work — the system may work well with current boundaries, or it may need significant tuning.

---

## Future Challenges

### KScript Metadata

KScript may benefit from metadata annotations for script-level properties. Low priority — the training loop works without it.

### Temporal Events and Trajectory Analysis

The agent's frame captures the full learning trajectory for a script. Temporal metadata on events would enable trajectory analysis: rate of convergence, effectiveness of scaffolding. This is also the foundation for Kalvin-as-tutor: a Kalvin instance that acts as agent for another, using reentrant rationalisation instead of exact kline equality for proposal evaluation.

---

## Dependency Graph

```
Challenge 1: Structural Grounding
    ↓
Challenge 2: Extended Cogitation
    ↓
Challenge 3: Agent Infrastructure
    ↓
Challenge 4: Model Quality (ongoing, parallel)
```

The dependency chain is linear and shallow. Challenges 1 and 2 are internal changes to Kalvin's model and cogitation. Challenge 3 is external orchestration. Challenge 4 is evaluation.

---

## Phasing

### Phase A: Structural Grounding (Challenge 1) ✅ Complete
**Estimate:** 1–2 days  
**Risk:** Low  
**Implementation:** `plans/impl/structural-grounding.md` §1

### Phase A+: Extended Cogitation (Challenge 2) ✅ Complete
**Estimate:** 3–5 days  
**Risk:** Medium  
**Depends on:** Phase A (✅)  
**Implementation:** `plans/impl/structural-grounding.md` §2

### Phase B: Agent Infrastructure (Challenge 3) — NEXT
**Estimate:** 1–2 weeks  
**Risk:** Low  
**Depends on:** Phase A+ (✅)

### Phase C: Model Quality (Challenge 4)
**Estimate:** 2–3 weeks (ongoing)  
**Risk:** Medium  
**Depends on:** Phase B

### Phase D: Future Enhancements
KScript metadata, temporal events, Kalvin-as-tutor. Not required for the training loop to work.

---

## Open Questions

| # | Question | Affects | Priority |
|---|---|---|---|
| 1 | ~~How to identify all STM klines involved in a ratification process?~~ Resolved: `promote_participating()` | Challenge 1 | ~~High~~ Done |
| 2 | Are current boundary constants well-calibrated for training? | Challenge 4 | High |
| 3 | How many klines before candidate retrieval O(N) becomes a bottleneck? | Challenge 4 | Medium |
| 4 | What does a non-trivial training curriculum look like? | MW | Medium |
| 5 | When does Mod(N) vocabulary saturation require BPE transition? | Challenge 4 | Low |
