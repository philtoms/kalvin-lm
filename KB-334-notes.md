# KB-334 — Audit Notes (working document, not a cascade artifact)

Audits the **other 6** curriculum files (`first-steps.md`,
`s3-auto-countersign.md`, `cascade-pressure.md`, `conflict-drill.md`,
`mhall-svo-single.md`, `mhall-svo-equivalence.md`) for the
**canonical-implies-not-a-misfit gap** KB-320 found in `first-steps-s2.md`
(rule 47). The audit method (compile + replay per-lesson through `KAgent`,
classify whether an S2/S3-intending lesson emits a candidate-bearing
`0 < significance < D_MAX` event) is the KB-320 `_RoutingAdapter` /
`_replay_first_steps_s2` harness, generalised to any curriculum path against a
**single persistent agent** (so earlier lessons' countersigns are available as
candidates for later lessons' misfits).

## Step 1 — Per-lesson verdict (empirical; controls validated)

Replay method: one `KAgent` per curriculum, hold the accumulated model across
lessons (mirroring `KAgentAdapter._handle_submit` and KB-320's
`_replay_first_steps_s2`). Per lesson: compile, pre-register every entry in STM,
`rationalise` each, `cogitate_drain(5.0)`. A lesson is **FLAGGED** iff (a) its
prose INTENDS S2/S3 routing AND (b) it emits NO event with `kind == "frame"`,
`candidate is not None`, `0 < significance < D_MAX`.

### Controls
- **Positive control** — `first-steps-s2.md` L5 (post-KB-320 `>` CONNOTED fix):
  **CLEAN-verified** (15 mid-band candidate events). ✓
- **Negative control** — pre-KB-320 `first-steps-s2.md` L5
  (`git show a1c40f1^:curricula/first-steps-s2.md`, the `MH => H A` CANONIZED
  version): **FLAGGED** (8 events, 0 candidates, 0 mid-band). ✓

### Verdict table

| curriculum | lesson | prose intent | compiled klines (sig/op/canonical?) | event trace | verdict |
|---|---|---|---|---|---|
| `first-steps.md` | 1,2,3 | S1/S4 (identity + countersign) | identities + `{M:[H]}`/`{H:[M]}` countersigns; no misfits | S4/S1 only | **CLEAN-by-design** |
| `s3-auto-countersign.md` | 1,2 | S1/S4 | identities + `M==H` countersign | S1/S4 | **CLEAN-by-design** |
| `s3-auto-countersign.md` | 3 | trigger S3, auto-countersign (zero ratify) | `H(alo) > A(lpha)` → genuine misfit `{Halo:[Alpha]}` (CONNOTED, `is_canon=False`, rule-47-compliant); `HPA => P X` canonical identity def; `A == A` co-entry | misfit reaches cogitation (**20 candidates**, `rationalise=False`), resolves via `on_s1` (structural S1, `candidate is None`) — the curriculum's **auto-countersign design** | **CLEAN w.r.t. rule 47** (correct operator; absence of mid-band event is the intended auto-countersign, NOT the canonical gap) |
| `cascade-pressure.md` | 1,2 | S1/S4 (10 identities + 5 pair countersigns) | identities/countersigns | S1/S4 | **CLEAN-by-design** |
| `cascade-pressure.md` | 3,4,5 | S2/S3 misfit — "Each entry creates a misfit", "maximum candidate overlap" | **every `=>` compound is canonical** (`op=S2` CANONIZED, `is_canon=True`, `misfit=False`); the `=> RHS` is **discarded entirely** | S1 only (AGT-14 short-circuit), **0 mid-band** | **FLAGGED** |
| `conflict-drill.md` | 1,2 | S1/S4 | identities + `A==B` countersign | S1/S4 | **CLEAN-by-design** |
| `conflict-drill.md` | 3,4 | S2/S3 multi-misfit — "zero overlap with any candidate", "multiple simultaneous misfits" | **every `=>` compound is canonical** (`op=S2` CANONIZED, `is_canon=True`, `misfit=False`); `=> RHS` discarded | S1 only, **0 mid-band** | **FLAGGED** |
| `mhall-svo-single.md` | 1 | S2/S3 (CONNOTED leaves `A>D`, `L>M`, `L>O`) | genuine misfits | **1675 mid-band** candidate events | **CLEAN-verified** |
| `mhall-svo-equivalence.md` | 1–4 | S1/S4 (identity/undersign/canon) | identities + `M=H` undersign + `ALL=>…` canon | S1/S4 | **CLEAN-by-design** |
| `mhall-svo-equivalence.md` | 5 | S2/S3 (CONNOTED leaves) | genuine misfits | **10430 mid-band** candidate events | **CLEAN-verified** |

### Deviation from the hypothesis (recorded, not force-fit)
The hypothesis named `s3-auto-countersign.md` L3 as CLEAN-verified ("the `H > A`
S3 trigger fires"). Empirically, `H > A` compiles to a genuine misfit
(`{Halo_packed:[Alpha]}`, CONNOTED, `is_canon=False`) and **does reach candidate
retrieval** (`rationalise` returns `False`, 20 candidates found) — so it is
rule-47-compliant. But `expand()` resolves it via `on_s1` (structural S1, no
`candidate` field), so no candidate-bearing mid-band event fires. This is the
curriculum's **auto-countersign design intent** (objective: "the proposal matches
co-entry `A == A`, zero ratify"), NOT the canonical-implies-not-a-misfit gap.
Classified **CLEAN w.r.t. rule 47** and excluded from the mid-band-routing
regression guard (which asserts mid-band expansion events; s3's design is
auto-countersign, covered by `tests/test_s3_auto_countersign.py` + the compile
smoke).

---

## Step 2 — Diagnosis (FLAGGED: `cascade-pressure.md` L3–5, `conflict-drill.md` L3–4)

For every FLAGGED lesson the mechanism is **identical to KB-320's root cause B**:
the lesson uses the `=>` CANONIZED operator to express what its prose claims is
an S2/S3 misfit, but a CANONIZED *compound* definition sets
`sig = make_signature(nodes)` by construction (§11.4), so the compiled kline is
canonical (`is_canon == True`) — it can never be a misfit, and
`KAgent.rationalise()` resolves it S1 via the AGT-14 self-grounded short-circuit
**before** any candidate is retrieved. Compounding this (KB-320's Step-1
deviation): for a *compound* (multi-char) signature, the CANONIZED right-hand
nodes are taken from the signature's own MTS resolution cache
(`ast_emitter._process_scope` → `_emit_mts`), so the explicit `=> RHS` nodes are
**discarded entirely**. `=>` cannot express a misfit for a compound sig because
it neither lets you pick a non-canonical sig nor pick the nodes.

### `cascade-pressure.md` — L3, L4, L5 (prose intent: S2/S3 misfit / high candidate density)

Prose quotes the intent explicitly:
- L3: "Each entry creates a misfit against the established pair klines."
- L4: "These entries should match many candidates simultaneously due to the
  accumulated model density."
- L5: "submit a highly compound entry whose nodes span many groups, creating
  maximum candidate overlap."

Code blocks + the kline actually produced (from the Step-1 trace):

| line | code block | discarded RHS | actual compiled kline | canonical? | event |
|---|---|---|---|---|---|
| L3 | `(Alpha Beta) AB => C(harlie)` | `=> Charlie` | `{AB: [Alpha, Beta]}` (sig `0x2c280a000001ff7`, nodes 2, op=S2 CANONIZED) | **True** | S1 (D_MAX) |
| L3 | `(Charlie Delta) CD => E(cho)` | `=> Echo` | `{CD: [Charlie, Delta]}` canonical | **True** | S1 |
| L3 | `(Echo Foxtrot) EF => G(olf)` | `=> Golf` | `{EF: [Echo, Foxtrot]}` canonical | **True** | S1 |
| L3 | `(Golf Hotel) GH => I(ndia)` | `=> India` | `{GH: [Golf, Hotel]}` canonical | **True** | S1 |
| L4 | `ABCD => E F` | `=> E F` | `{ABCD: [Alpha,Beta,Charlie,Delta]}` canonical | **True** | S1 |
| L4 | `EFGH => I J` | `=> I J` | `{EFGH: [Echo,Foxtrot,Golf,Hotel]}` canonical | **True** | S1 |
| L4 | `ACEGI => B D F H J` | `=> B D F H J` | `{ACEGI: [Alpha,Charlie,Echo,Golf,India]}` canonical | **True** | S1 |
| L5 | `ABCDEFGHIJ => A B … J` | `=> A…J` | `{ABCDEFGHIJ: [all 10]}` canonical | **True** | S1 |

Every node-bearing entry in L3/L4/L5 has `op=S2` (CANONIZED), `is_canon=True`,
`misfit=False`; every `rationalise` returns `True` on the S1 fast-path. **Zero
misfit klines compiled; zero mid-band events.**

### `conflict-drill.md` — L3, L4 (prose intent: S2/S3 multi-misfit)

Prose quotes:
- L3: "The entry `{AB: [D, C]}` has nodes `[D, C]` that have zero overlap with
  any candidate's nodes."
- L4: "ACE has bits for A, C, E. Its nodes `[B, D]` overlap with neither…"

| line | code block | discarded RHS | actual compiled kline | canonical? | event |
|---|---|---|---|---|---|
| L3 | `(Alpha Beta) AB => D(elta) C(harlie)` | `=> D C` | `{AB: [Alpha, Beta]}` (sig `0x2c280a000001ff7`, nodes 2, op=S2 CANONIZED) | **True** | S1 (D_MAX) |
| L4 | `(Alpha Charlie Echo) ACE => B(eta) D(elta)` | `=> B D` | `{ACE: [Alpha,Charlie,Echo]}` canonical | **True** | S1 |

Same pattern: canonical klines, AGT-14 S1 short-circuit, zero mid-band events.

### AGT-14 short-circuit path (re-confirmed on current `main` HEAD `a9a8479`)

`src/kalvin/agent.py` `KAgent.rationalise()` lines **203–212** (unchanged since
KB-320's citation):

```python
expected_sig = make_signature(kline.nodes)          # 203
if kline.signature == expected_sig:                 # 204  — True (canonical)
    all_resolved = all(                             # 205
        (node_kl := self._model.find(n)) is not None and self._model.grounded(node_kl)
        for n in kline.nodes
    )
    if all_resolved:                                # 209  — True
        self._model.add_to_ltm(kline)               # 210
        self._publish("frame", kline, kline, D_MAX) # 211  — S1
        return True                                 # 212
```

The canonical compound kline + grounded nodes (the identities were introduced in
L1/L2, pre-grounding the compound's MTS-resolved nodes) fires this branch and
returns S1 **before** candidate retrieval (`model.where()` at L228). Confirmed:
no FLAGGED lesson ever reached `rationalise=False`.

### Reconciliation with KB-320

This is **the same root cause B** KB-320 diagnosed for `first-steps-s2.md` L5:
§11.4 canonical ⇒ not a misfit ⇒ AGT-14 S1 short-circuit, compounded by (a) the
within-lesson pre-grounding amplifier (earlier-compiled identity entries land in
LTM before the compound is rationalised) and (b) the compound-sig MTS-resolution
override that discards the `=> RHS`. The only difference is **scale**: KB-320
found one lesson in one file; this audit finds the same gap across **5 lessons
in 2 files** (`cascade-pressure.md` L3/L4/L5, `conflict-drill.md` L3/L4).

### Auto-countersign backstop constraint (carried over from KB-320)

Even if a fixed lesson's S2/S3 proposal equals any loaded lesson entry,
`Reactor._auto_countersign` (`src/training/trainer/reactor.py`) silently
satisfies it (structural `KLine.__eq__` against `_current_entries`), emitting no
`ratify_request`. So a fix that needs to prove `ratify_request` fires must
produce a **reshaped** proposal distinct from every loaded entry. For the
regression test (Step 4), the in-process `_RoutingAdapter` candidate-bearing
mid-band event is sufficient; the `ratify_request` full-stack check follows the
`TestFirstStepsS2Routing::test_emits_ratify_request_full_stack` pattern only if a
curriculum's design requires it (not required here).

### KB-333 cross-check

KB-333 (the parallel *deletion/truncation* audit) has **NOT landed** as of this
task (`git log --oneline curricula/ | grep KB-333` empty). The two audits are
independent: KB-334 fixes the *operator* choice on misfit-intending lessons;
KB-333 restores *deleted content*. This task's fixes assume the current `main`
content. If a FLAGGED lesson is later found to also be deletion-truncated by
KB-333, KB-334's rule-47 operator fix takes precedence on the operator; KB-333
takes precedence on content restoration.

---

## Step 3 — Decision: validated CONNOTED constructions per FLAGGED curriculum

> Structural fact (settled contracts, confirmed empirically): `>`
> CONNOTED emits **one single-node kline per RHS node**
> (`ast_emitter._emit_operator_entries`, the `op == "CONNOTED"` branch loops
> `for node in nodes: emit(sig, [node], "CONNOTED")`). Only `=>` CANONIZED
> (and MTS CANONIZE) aggregates nodes into one multi-node kline — and
> CANONIZED sets `sig = make_signature(nodes)` (§11.4), so every multi-node
> kline is canonical by construction. **A genuine (non-canonical) misfit
> therefore always has exactly one node.** CONNOTED thus *decomposes* a
> multi-node `=>` lesson into N single-node misfits. This is the KB-320
> precedent (`MH => H A` → `MH > H A` → `{MH:[Halo]}` + `{MH:[Alpha]}`) and
> is the only way to express these misfits. The prose (Step 4) is reconciled
> to describe single-node misfits; candidate density is preserved because
> `model.where(sig)` is bit-overlap based (`kline.signature & sig != 0`), so
> each single-node misfit retrieves the established countersigns.

Block word lists bind by **first letter** from a word pool
(`binding_scope.resolve`); NATO phonetic words have unique first letters, so a
block listing every word whose first letter appears in the sig OR the RHS nodes
binds all of them. This follows KB-320's `(Mark Halo Alpha)` for `MH > H A`
(3 words for a 2-char sig) and rule 47 ("bind the intended node words with a
block word list rather than relying on the right-hand side").

All constructions were validated by replaying the full fixed curriculum through
the Step-1 harness (single persistent `KAgent`, default `max_candidates=8` —
**cannot** be lowered: `max_candidates=3` breaks the `first-steps-s2.md` control
to 0 mid-band events). Evidence recorded below.

### `cascade-pressure.md` — fix

| lesson | construction | misfit klines compiled | mid-band candidate events |
|---|---|---|---|
| L3 | `(Alpha Beta Charlie) AB > C` ×4 (one per cross-group pair) | 4 single-node overfit misfits `{AB:[Charlie]}`, `{CD:[Echo]}`, `{EF:[Golf]}`, `{GH:[India]}` (all `op=S3`, `is_canon=False`) | **113** |
| L4 | `(Alpha Beta Charlie Delta Echo Foxtrot) ABCD > E F` + `(Echo Foxtrot Golf Hotel India Juliet) EFGH > I J` + `(Alpha Charlie Echo Golf India Beta Delta Foxtrot Hotel Juliet) ACEGI > B D F H J` | 9 single-node misfits (`ABCD→{Echo},{Foxtrot}`; `EFGH→{India},{Juliet}`; `ACEGI→{Beta},{Delta},{Foxtrot},{Hotel},{Juliet}`) | **7060** |
| L5 | `(Alpha Beta Charlie Delta Echo Foxtrot Golf Hotel India Juliet) ABCDEFGHIJ > A C E G` | 4 single-node misfits `{ABCDEFGHIJ:[Alpha]}`, `…:[Charlie]`, `…:[Echo]`, `…:[Golf]}` (one per group; 10-atom compound sig → maximum candidate overlap) | **113486** |

L5 rationale: the original `ABCDEFGHIJ => A B … J` intended a single 10-node
canonical kline. The faithful CONNOTED decomposition would be 10 single-node
misfits → **839 396** expansion events (≈11 s replay) — pathological for both
the regression test and any real training run. The "maximum candidate overlap"
intent comes from the **10-atom compound sig** (it signifies every established
identity/countersign), not the node count. Four nodes spanning four of the five
groups (`A C E G`) preserve the intent (escalates above L4: 113 486 > 7060
events) while staying tractable (~3 s). The prose names this design choice.

### `conflict-drill.md` — fix

| lesson | construction | misfit klines compiled | mid-band candidate events |
|---|---|---|---|
| L3 | `(Alpha Beta Charlie Delta) AB > D C` | 2 single-node misfits `{AB:[Delta]}`, `{AB:[Charlie]}` (S3, disjoint from the `A==B` countersign) | **7** |
| L4 | `E(cho)` then `(Alpha Charlie Echo Beta Delta) ACE > B D` | 2 single-node misfits `{ACE:[Beta]}`, `{ACE:[Delta]}` (S3) | **13** |

`conflict-drill.md` is fast (<0.01 s); no node-count reduction needed.

### Auto-countersign backstop

The in-process `_RoutingAdapter` candidate-bearing mid-band event is sufficient
for the regression guard (Step 4). The `ratify_request` full-stack check
(`TestFirstStepsS2Routing::test_emits_ratify_request_full_stack` pattern) is not
added for cascade/conflict — their design is misfit-routing, not the
zero-ratify auto-countersign of `s3-auto-countersign.md`; a separate heavier
integration test is not required by the rule-47 audit.

### Rejected alternatives (consistent with KB-320-notes §Step 3)

- `=>` CANONIZED (baseline): canonical, AGT-14 S1 short-circuit, 0 S2/S3. (ruled out, rule 47)
- inline-annotated multi-node RHS (`AB > D(elta) C(harlie)`): garbles per
  KB-320 (inline annotation attaches to the first node only). Use block word
  lists instead.

### No agent-behaviour fallback

All constructions reach the S2/S3 band under the settled contracts. No
`agent.py`/`expand.py`/`ks/*` change. AGT-14, §11.4, and the MTS compound-sig
resolution are the baseline, not the target.
