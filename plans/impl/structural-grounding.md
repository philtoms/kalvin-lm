# Sub-Plan: Structural Grounding + Extended Cogitation

**Parent:** [`docs/roadmap.md`](../../docs/roadmap.md)
**Challenges:** 6 (Structural Grounding) + 6b (Extended Cogitation)
**Estimate:** 4–7 days
**Depends on:** Current system (Phases 0–8 complete, 329 tests passing)

---

## 0. What Changes and Why

The current system auto-promotes klines classified as S1 by the Cogitator's
boundary check. Under the training model, this is wrong: **S1 is determined
by structure, not by boundary classification.** A kline is grounded (S1) if
its signature fully describes its nodes or it is countersigned.

Additionally, the current Cogitator discards S2 results that fail
countersignature. Extended cogitation adds the ability to reshape these
klines toward canonical status, emitting proposals for teacher ratification.

These two changes enable the training loop described in
`docs/training-loop.md`.

### What Stays the Same

- All 329 existing tests continue to pass.
- The fast path (Phases 1–5 of rationalisation) is unchanged.
- Routing (`_route`) is unchanged — node-membership test, no model call.
- The `expand()` algorithm and `QueryCandidate` are unchanged.
- Sampling (temperature, top_k, top_p) is unchanged.
- Events, serialization, KScript are unchanged.
- All specs (kline, signature, tokenizer, stm, significance, kscript) are
  unchanged.

### What Changes

| Area                     | Current Behavior                           | New Behavior                                                                |
| ------------------------ | ------------------------------------------ | --------------------------------------------------------------------------- |
| S1 determination         | Boundary classification in Cogitator       | Structural: `make_signature(nodes) == signature` or countersigned           |
| Promotion trigger        | Cogitator classifies as S1 → promote query | After ratification (countersignature), promote all participating STM klines |
| Promotion scope          | Single kline promoted                      | All STM klines involved in the ratification event promoted                  |
| S2 post-countersignature | Nothing — candidate discarded              | Extended cogitation: attempt underfit/overfit expansion                     |
| Frame contents           | S1 and S4 only                             | S4–S1: all klines promoted after ratification                               |

---

## 1. Phase A: Structural Grounding (Challenge 6)

**Estimate:** 1–2 days  
**Risk:** Low

### 1.1 Structural S1 Check

Add a method to Model that determines whether a kline is structurally S1:

```python
def is_s1(self, kline: KLine) -> bool:
    """Determine if a kline is structurally grounded (S1).

    A kline is S1 if:
    1. Its signature fully describes its nodes (canonical), OR
    2. It is countersigned by another kline in the model.
    """
    if self._is_canon(kline):
        return True
    return self.is_countersigned(kline)
```

The countersigned check searches the model for a kline whose signature
equals the kline's nodes_signature and whose sole node equals the kline's
signature:

```python
def is_countersigned(self, kline: KLine) -> bool:
    """Check if kline is countersigned by any kline in the model."""
    nodes_signature = make_signature(kline.nodes)
    for countersigner in self.find_all(nodes_signature):
        if len(countersigner.nodes) == 1 and countersigner.nodes[0] == kline.signature:
            return True
    return False
```

**Location:** `src/kalvin/model.py`

### 1.2 Promotion After Ratification

Change the promotion path so that when a countersignature is discovered
(the ratification event), all STM klines that participated in the
ratification process are promoted — not just the ratified kline.

"Participating klines" are:

1. The query kline
2. The candidate kline
3. Any klines in STM whose signatures or nodes overlap with the
   query-candidate pair's node set

```python
def promote_participating(self, query: KLine, candidate: KLine) -> int:
    """Promote all STM klines involved in a ratification event.

    After countersignature is detected between query and candidate,
    promote both plus any STM klines whose signatures appear in the
    union of their nodes. This enriches the frame with S4 identity
    klines and S2/S3 partial klines involved in the ratification.

    Returns the number of klines promoted.
    """
    # Collect all signatures from the participating pair
    node_sigs = set()
    for n in query.nodes:
        if not is_literal_node(n):
            node_sigs.add(n)
    for n in candidate.nodes:
        if not is_literal_node(n):
            node_sigs.add(n)
    node_sigs.add(query.signature)
    node_sigs.add(candidate.signature)

    # Find all STM klines with matching signatures
    to_promote = []
    for kl in self._stm._order:
        if kl.signature in node_sigs:
            to_promote.append(kl)

    # Also promote the query and candidate if they're in STM
    for kl in [query, candidate]:
        if kl not in to_promote:
            to_promote.append(kl)

    count = 0
    for kl in to_promote:
        if self.promote(kl):
            count += 1
    return count
```

**Location:** `src/kalvin/model.py`

### 1.3 Agent: Use Structural Grounding

Modify `agent.py` to use structural grounding instead of boundary-based
promotion. The key design principle: **promotion follows ratification**.
Promotion happens at explicit ratification points, not implicitly inside
`_publish()`.

#### Fast Path (Phase 3: Assess)

No changes. The fast path already correctly identifies canonical S1
(self-grounded check at line ~440). This is structurally correct.

#### agent.py: rationalise() Phase 5

```python
# Before:
if level == "S1":
    self._model.promote(kline)
    ...

# After:
if level == "S1":
    self._model.promote_participating(kline, candidate)
    ...
```

#### agent.py: \_publish()

Remove the implicit promote from `_publish`. Promotion is now explicit
at ratification points, not bundled into event emission:

```python
# Before:
def _publish(self, kind, query, proposal, significance):
    if kind == "frame" and significance in (D_MAX - 1, 0):
        self._model.promote(proposal)
    self._event_bus.publish(...)

# After:
def _publish(self, kind, query, proposal, significance):
    self._event_bus.publish(RationaliseEvent(kind, query, proposal, significance))
```

#### agent.py: Cogitator.\_process() — S2 Expansion

The countersignature check has been moved to `rationalise()` Phase 3
(Assess). `_process` now handles only S2 expansion:

#### agent.py: Cogitator.\_run_work_item() — Boundary S1

When a QC is classified as S1 by boundary but the candidate is not
structurally S1 (e.g. temperature-promoted S2), don't promote. Still
call on_s1 for re-rationalisation — the re-rationalisation may discover
new structure:

```python
# Before:
if band == "S1":
    self._on_s1(query, candidate)

# After:
if band == "S1":
    if self._model.is_s1(candidate):
        self._model.promote_participating(query, candidate)
    self._on_s1(query, candidate)
```

The candidate is a model kline (from `model.where()`), so
`is_s1(candidate)` works correctly.

### 1.5 Test Cases

| Test                                      | Description                                           |
| ----------------------------------------- | ----------------------------------------------------- |
| `is_s1` canonical                | KLine with `sig == make_signature(nodes)` → True      |
| `is_s1` countersigned            | Two klines with mutual node references → True         |
| `is_s1` neither                  | KLine that is not canonical or countersigned → False  |
| `is_s1` all-literal              | All-literal kline → True (canonical, sig=1)           |
| `promote_participating` basic             | Query + candidate promoted                            |
| `promote_participating` with S4 identity  | S4 identity klines in STM also promoted               |
| `promote_participating` with S2/S3        | Partial klines in STM promoted                        |
| `promote_participating` no double-promote | Already-promoted klines not re-promoted               |
| Frame holds S4–S1                         | After ratification, frame contains mixed significance |
| Cogitator countersignature promotes all   | Countersignature discovery promotes participating     |
| Boundary S1 + structural check            | Boundary S1 on non-structural kline → no promotion    |
| Boundary S1 + structural S1               | Boundary S1 on structural kline → promotion           |
| Existing tests unchanged                  | All 329 tests still pass                              |

---

## 2. Phase A+: Extended Cogitation (Challenge 6b)

**Estimate:** 3–5 days  
**Risk:** Medium  
**Depends on:** Phase A

### 2.1 Misfit Classification

Add to `model.py`:

```python
def classify_misfit(self, kline: KLine) -> tuple[bool, bool]:
    """Classify a kline's misfit type.

    Returns (underfitting, overfitting):
    - underfitting: True if S & ~N != 0 (signature promises more than nodes deliver)
    - overfitting: True if N & ~S != 0 (nodes carry more than signature captures)
    """
    nodes_sig = make_signature(kline.nodes)
    underfit = (kline.signature & ~nodes_sig) != 0
    overfit = (nodes_sig & ~kline.signature) != 0
    return underfit, overfit
```

### 2.2 Expansion Operations

Add to `model.py`:

```python
def generate_expansions(
    self,
    kline: KLine,
    underfit_gap: int,
    overfit_mask: int,
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Generate expansion proposals for a misfit kline.

    Each yield is (proposal_kline, companion_klines) where:
    - proposal_kline is the expanded version of the input
    - companion_klines are klines formed from removed nodes (may be empty)

    Expansion proposals satisfy:
    - No invention: every signature used exists in the model
    - No orphan nodes: removed nodes form a companion kline

    The caller (Cogitator) is responsible for emitting proposals as
    frame events and for the teacher to ratify them.
    """
    nodes_sig = make_signature(kline.nodes)

    # Underfit expansion: add nodes to fill the gap
    if underfit_gap:
        yield from self._underfit_expansions(kline, underfit_gap)

    # Overfit expansion: remove excess nodes
    if overfit_mask:
        yield from self._overfit_expansions(kline, overfit_mask)

    # Dual misfit: both operations may apply
    if underfit_gap and overfit_mask:
        yield from self._dual_expansions(kline, underfit_gap, overfit_mask)


def _underfit_expansions(
    self, kline: KLine, gap: int
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Add nodes whose signatures contribute to the gap."""
    # Find klines in the model whose signatures contribute to the gap
    contributors = self.where(lambda k: (k.signature & gap) != 0)

    for contributor in contributors:
        # Build expanded nodes: original + contributor's nodes
        expanded_nodes = list(kline.nodes) + list(contributor.nodes)
        expanded_sig = kline.signature  # signature stays the same
        proposal = KLine(expanded_sig, expanded_nodes, kline.dbg_text)

        # Verify the expansion moves toward canonical
        new_nodes_sig = make_signature(expanded_nodes)
        if (new_nodes_sig & expanded_sig) != 0:  # closer to canonical
            yield (proposal, [])


def _overfit_expansions(
    self, kline: KLine, excess: int
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Remove nodes whose bits contribute to the excess."""
    # Find nodes contributing to excess
    excess_nodes = [n for n in kline.nodes
                     if not is_literal_node(n) and (n & excess) != 0]

    if not excess_nodes:
        return

    # Build trimmed kline: remaining nodes
    remaining = [n for n in kline.nodes if n not in excess_nodes]
    trimmed = KLine(kline.signature, remaining, kline.dbg_text)

    # Build companion kline from removed nodes
    companion_sig = make_signature(excess_nodes)
    companion = KLine(companion_sig, excess_nodes)

    yield (trimmed, [companion])


def _dual_expansions(
    self, kline: KLine, gap: int, excess: int
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Atomic replacement: swap excess nodes for gap-filling nodes."""
    excess_nodes = [n for n in kline.nodes
                     if not is_literal_node(n) and (n & excess) != 0]
    remaining = [n for n in kline.nodes if n not in excess_nodes]

    # Find contributors to fill the gap
    contributors = self.where(lambda k: (k.signature & gap) != 0)

    for contributor in contributors:
        replacement_nodes = remaining + list(contributor.nodes)
        replacement = KLine(kline.signature, replacement_nodes, kline.dbg_text)

        companion_sig = make_signature(excess_nodes)
        companion = KLine(companion_sig, excess_nodes)

        yield (replacement, [companion])
```

### 2.3 Cogitator Integration

Extend `Cogitator._process()` with S2 expansion:

```python
def _process(self, item: QueryCandidate) -> None:
    query, candidate, significance = item

    # S2 expansion only — ratification handled upstream in rationalise()
    candidate_sig = candidate.signature
    nodes_sig = make_signature(candidate.nodes)

    if candidate_sig == nodes_sig:
        return  # canonical — nothing to expand

    underfit, overfit = self._model.classify_misfit(candidate)

    if not underfit and not overfit:
        return  # neither — nothing to expand

    underfit_gap = candidate_sig & ~nodes_sig
    overfit_mask = nodes_sig & ~candidate_sig

    for proposal, companions in self._model.generate_expansions(
        candidate, underfit_gap, overfit_mask
    ):
        # Emit proposal as frame event for teacher ratification
        self._event_bus.publish(
            RationaliseEvent("frame", query, proposal, significance)
        )
        # Emit companion klines
        for companion in companions:
            self._event_bus.publish(
                RationaliseEvent("frame", query, companion, significance)
            )
```

### 2.4 Spec Updates

**specs/agent.md** — add countersigned check to Phase 3 (Assess), update
§S2 Expansion to reflect that `_process` handles only expansion (ratification
moved upstream to rationalise):

**specs/model.md** — add `classify_misfit` and `generate_expansions`:

> The model provides misfit classification and expansion proposal
> generation for the extended cogitation pipeline.

### 2.5 Test Cases

| Test                                 | Description                           |
| ------------------------------------ | ------------------------------------- |
| `classify_misfit` canonical          | `S == N` → (False, False)             |
| `classify_misfit` underfit           | `S & ~N != 0` → (True, False)         |
| `classify_misfit` overfit            | `N & ~S != 0` → (False, True)         |
| `classify_misfit` dual               | Both conditions → (True, True)        |
| `generate_expansions` underfit       | Returns proposal with added nodes     |
| `generate_expansions` overfit        | Returns trimmed + companion           |
| `generate_expansions` dual           | Returns replacement + companion       |
| `generate_expansions` no gap         | No expansion proposals emitted        |
| Cogitator expansion proposal         | frame event emitted for expansion     |
| Cogitator expansion companion        | frame event emitted for companion     |
| Cogitator no expansion for canonical | Canonical kline → no expansion        |
| KScript S2 template                  | `AB => A` produces underfitting kline |
| KScript sequencer                    | `A => A B` produces overfitting kline |
| KScript dual misfit                  | `WDMH => MHALL` produces dual misfit  |
| Existing tests unchanged             | All 329 tests + Phase A tests pass    |

---

## 3. Implementation Order

```
Phase A: Structural Grounding
  1. model.is_s1()                 — structural grounding check
  2. model.promote_participating()    — new method
  3. model.is_countersigned()         — renamed from _is_countersigned_in_model
  4. agent._publish()                 — remove auto-promote
  5. agent.rationalise() Phase 5      — use promote_participating
  6. agent.rationalise() Phase 3      — add ratification check
  7. Cogitator._run_work_item()       — structural check on boundary S1
  8. Tests for all new methods
  9. Verify all existing tests pass

Phase A+: Extended Cogitation
  1. model.classify_misfit()          — new method
  2. model.generate_expansions()      — new method + helpers
  3. Cogitator._process()             — S2 expansion
  4. Tests for misfit classification
  5. Tests for expansion proposals
  6. Tests for Cogitator integration
  7. Spec updates (agent.md, model.md)
  8. Verify all tests pass
```

---

## 4. Files Modified

| File                  | Change                                                                                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/kalvin/model.py` | Add `is_s1` (structural grounding), `promote_participating`, `is_countersigned`, `classify_misfit`, `generate_expansions`, `_underfit_expansions`, `_overfit_expansions`, `_dual_expansions` |
| `src/kalvin/agent.py` | Modify `_publish`, `rationalise()` Phase 3 (add ratification) and Phase 5, `Cogitator._process` (S2 expansion only), `Cogitator._run_work_item`                                  |
| `tests/test_model.py` | Add tests for new model methods                                                                                                                                                  |
| `tests/test_agent.py` | Add tests for structural grounding in cogitation                                                                                                                                 |
| `specs/agent.md`      | Update §Cogitation with S2 expansion phase                                                                                                                                       |
| `specs/model.md`      | Add misfit classification and expansion API                                                                                                                                      |
| `docs/roadmap.md`     | Update status of Challenge 6 and 6b                                                                                                                                              |

---

## 5. Open Questions for Phase A

1. **What counts as "participating" in ratification?** Current proposal:
   any STM kline whose signature appears in the union of query and candidate
   nodes. This may be too broad (promoting unrelated klines) or too narrow
   (missing klines involved in the expansion graph). May need empirical
   tuning.

2. **Should boundary S1 without structural S1 still call on_s1?** Yes —
   the re-rationalisation may discover new structure. But it won't promote
   participating klines (only structural S1 does that).

3. **Temperature-promoted S1 boundaries:** When τ > 1, the Cogitator may
   classify S2 QCs as S1. These should NOT trigger promotion unless the
   candidate is also structurally S1. The structural check guards against
   premature promotion.

4. **Performance of `promote_participating`:** Scanning all STM klines for
   matching signatures is O(STM size). For STM bound 256, this is fast.
   For larger bounds, may need index optimization.

## 6. Open Questions for Phase A+

5. **Expansion search depth:** How many contributors to consider for
   underfit expansion? The current proposal yields all model klines with
   overlapping signatures. This may be too many. May need a top-k limit.

6. **Companion kline emission:** Should companion klines be added to the
   model automatically, or only emitted as proposals? Current design:
   emitted as proposals only (teacher ratifies).

7. **Interaction with sampling:** Should expansion proposals count against
   top-k and top-p budgets? Current design: they're emitted after the
   expand() stream completes, so they don't interact with sampling.
   Future work may integrate them into the stream.
