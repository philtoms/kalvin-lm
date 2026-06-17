# Sub-Plan: Cogitator — Background Slow-Path Dispatcher

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phases:** 8 (split from the former Agent phase)
**Depends on:** Model (expand, generate_expansions, tiered writes), Events, Signature

## 1. Spec References

See **@cogitator spec** for the full definition (Cogitator, CogitationHandler,
WorkItem, QueryCandidate consumption, S1 fast-path, S2 expansion,
significance boundaries, lifecycle).
See **@cogitator-drain spec** for inter-lesson drain semantics.
Test matrix: AGT-29 through AGT-42 (relocated from @agent spec, IDs stable).

See **@agent spec** §Cogitation for the seam (the agent submits work items
and is the primary `CogitationHandler`).

## 2. Implementation

**Files:** `src/kalvin/cogitator.py` (canonical home), `src/kalvin/agent.py`
(consumer + handler), `tests/test_agent.py`, `tests/test_cogitator_drain.py`,
`tests/test_cascade_control.py`.

### Module responsibility

`src/kalvin/cogitator.py` owns:

- `CogitationHandler` — `@runtime_checkable` Protocol (`on_s1`, `on_expansion`).
- `WorkItem` — `NamedTuple(query, candidate, level)`.
- `Cogitator` — background daemon thread; backlog, `submit`, `join`, `drain`,
  `_run`, `_run_work_item` (S1 fast-path + expand/classify/propose loop).

All significance/expansion logic is delegated to `kalvin.expand`
(`boundaries`, `classify`, `expand`, `propose_expansions`).

### Agent wiring

`src/kalvin/agent.py` imports `Cogitator`, `CogitationHandler`, `WorkItem`
from `kalvin.cogitator`. It must **not** redefine them. `KAgent` constructs
its `Cogitator` passing `model`, `adapter`, and `self` (handler), and
implements `on_s1` / `on_expansion`. Re-exports from `kalvin.agent` are kept
for backward compatibility with existing tests, but the canonical import
location is `kalvin.cogitator`.

### Pseudocode — `_run_work_item`

```
run_work_item(WorkItem(query, candidate, level)):
  if level == "S1":
    handler.on_s1(query, candidate)   # fast-path, skip expand()
    return
  s12, s23, s34 = boundaries()
  for qc in model.expand(query, candidate):
    band = classify(qc.significance, s12, s23, s34)
    if band == "S4": continue
    if band == "S1":
      handler.on_s1(query, candidate); break
    else:
      for proposal, sig in propose_expansions(model, qc.candidate, qc.significance):
        handler.on_expansion(qc.query, proposal, sig, original_candidate=qc.candidate)
```

See `plans/impl/structural-grounding.md` for the expansion algorithm detail.

## 3. Test Mapping

| Spec ID | Test File | Test Function / Class | Status |
| ------- | --------- | --------------------- | ------ |
| AGT-29  | `tests/test_agent.py` | `TestCogitatorStructuralGrounding` (countersignature cascade) | ✅ |
| AGT-30  | `tests/test_agent.py` | `TestCogitator` (join stops thread) | ✅ |
| AGT-31  | `tests/test_agent.py` | `TestWorkItem` (fields) | ✅ |
| AGT-32  | `tests/test_agent.py` | `TestCogitator` (all yields processed) | ✅ |
| AGT-33  | `tests/test_agent.py` | `TestCogitatorWithFakeHandler` (on_s1) | ✅ |
| AGT-34  | `tests/test_agent.py` | `TestCogitatorWithFakeHandler` (expansion → Frame) | ✅ |
| AGT-35  | `tests/test_agent.py` | `TestCogitator` (proposals at any sig) | ✅ |
| AGT-36  | `tests/test_agent.py` | `TestCogitatorStructuralGrounding` (structural check) | ✅ |
| AGT-37  | `tests/test_agent.py` | `TestCogitatorStructuralGrounding` (structural S1 cascade) | ✅ |
| AGT-38  | `tests/test_cascade_control.py` | s2-before-s1 no submit | ✅ |
| AGT-39  | `tests/test_agent.py` | `test_cogitator_stops_on_s1` | ✅ |
| AGT-40–42 | `tests/test_agent.py` | satisfaction guard / completion | ✅ |

## 4. Design Decisions

1. **Canonical module is `kalvin.cogitator`.** The agent imports from it; no
   duplicate class bodies. Re-exports on `kalvin.agent` preserve existing
   test import paths without duplicating definitions.

2. **No renumbering of AGT- IDs.** Per cascade rule 2, relocated test-matrix
   rows keep their stable IDs (AGT-29..AGT-42). They now live under
   @cogitator spec but trace to the same tests.

## 5. Status

✅ Complete (the module split was started previously; this plan formalises the
canonical home and the spec/plan separation).
