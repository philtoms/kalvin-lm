# KB-320 — Investigation Notes (working document, not a cascade artifact)

## Step 1 — Captured event trace (current `main`, 2-lesson curriculum)

Direct KAgent replay (mirrors `KAgentAdapter._handle_submit`: compile per
lesson, pre-register every entry in STM, rationalise each, drain cogitator).
NLP tokenizer, dev=True.

### Lesson 1 — `M(ark)` → 1 entry
| # | entry (sig / nodes / dbg.op) | event | sig | candidate |
|---|------------------------------|-------|-----|-----------|
| 0 | `Mark` 0x4008000004d2e / [] / IDENTITY | frame (S4) | 0x0 | no |

### Lesson 5 — `(Mark Halo)\nMH => H A(lpha)` → 6 entries
| # | entry (sig / nodes / dbg.op / canonical?) | event | sig | candidate |
|---|--------------------------------------------|-------|-----|-----------|
| 0 | `Mark` 0x4008000004d2e / [] / IDENTITY | **ground** | D_MAX | no |
| 1 | `Halo` 0x80080000000048 / [] / IDENTITY | frame (S4) | 0x0 | no |
| 2 | `Halo` 0x2000040000014f / [] / IDENTITY | frame (S4) | 0x0 | no |
| 3 | `Halo` 0x8000020000006f / [] / IDENTITY | frame (S4) | 0x0 | no |
| 4 | `Halo`-packed 0xa008060000016f / [3 BPE toks] / CANONIZED / **canonical** | **frame (S1)** | D_MAX | no |
| 5 | `MH` 0xa4088600004d6f / [Mark, Halo_packed] / CANONIZED / **canonical** | **frame (S1)** | D_MAX | no |

**Summary (direct):** 7 events — 1 ground, 2 frame-S1, 4 frame-S4.
**0 candidate-bearing events; 0 genuine S2/S3 (0 < sig < D_MAX).**

### Full-stack (Trainer + KAgentAdapter + KAgent + MessageBus)
- progress statuses: ready → started → lesson_complete → lesson_complete → complete
- **`ratify_request` count = 0**; completes silently with only progress events
  (one structural auto-countersign of the per-token S4 identity against the
  loaded lesson entries).

### Deviation from the Step-1 hypothesis (IMPORTANT — re-scopes Step 3)
The hypothesis named the final kline `{MH: [H_node, A_node]}`. The observed
nodes are **[Mark, Halo_packed]** — i.e. the `=> H A(lpha)` right-hand side is
**discarded entirely**. Reason (`ast_emitter.py::_process_scope`):
`MH` is a multi-char sig, so `_emit_mts("MH")` runs first and caches its
resolved components (`(Mark Halo)` binding → `["Mark","Halo"]`); the
CANONIZED operator then takes the branch `if op == "CANONIZED" and
scope.sig.id in self._resolution_cache: resolved_nodes = cache["MH"]`, so
the explicit `=> H A` nodes never reach the entry. Two consequences:

1. The "Alpha" identity is **never introduced** by lesson 5 at all.
2. The kline is `MH => [Mark, Halo]`, canonical by §11.4
   (`sig = make_signature(node_values)`), so it can never be a misfit.

This is a stronger form of root cause B than hypothesised: for a *compound*
sig, `=>` does not even let you choose the nodes. Lesson 5's prose claim
("dual-misfit {MH:[H,A]}") is wrong on both counts — wrong nodes AND canonical.

## Step 1 — Hypothesis verdict
CONFIRMED: the current curriculum emits only ground/S1/S4 events for lesson 5,
no `candidate`, no `ratify_request`, matching KB-309 (findings.md §1 +
agent.log ~1848–1960). The mechanism (AGT-14 S1 short-circuit on the canonical
compound kline) is confirmed; the node identity is corrected above.

---

## Step 2 — Diagnosis (the two root causes)

### Root cause A — the curriculum is missing lessons 2–4 (accidental deletion)

`git show a1c40f1 -- curricula/first-steps-s2.md` shows commit `a1c40f1`
(2026-06-10, "feat(curricula): add NLP-first annotations to all 7 curriculum
files") collapsed lessons 1–4 into a single annotated `M(ark)` lesson,
**deleting** the `H`, `M == H`, and `A` lessons:

```
-### 2
-Introduce the identity H.
-```
-H
-```
-### 3
-Establish a bidirectional relationship between M and H.
-```
-M == H
-```
-### 4
-Introduce the identity A.
-```
-A
-```
```

The objective/approach/lesson-5 prose still describe a 5-lesson curriculum
("Introduce identities, countersign, add a new identity, then submit a
canonize…"), and lesson 5's prose claims the dual-misfit routes S2 "against
`{M: [H]}`" — a **countersign kline that was never taught**. The stale
`curricula/first-steps-s2.json` `lessons` array
(`["M","H","M == H","A","MH => H A"]`) records the intended structure.

**Consequence:** even if lesson 5's kline reached candidate retrieval,
`model.where(<query sig>)` would find no `{M: [H]}` / `{H: [M]}` countersign
to route against, because lesson 3 (`M == H`) — which compiles to those two
reciprocal countersign klines — was deleted. The routing target the lesson is
designed against does not exist in the model.

### Root cause B — `=>` CANONIZED yields a canonical kline, not a misfit; the AGT-14 S1 short-circuit fires

From the Step-1 trace, the final `{MH: [Mark, Halo_packed]}` entry (sig
`0xa4088600004d6f`, canonical) takes the
`if kline.signature == expected_sig:` branch in `KAgent.rationalise()`
(**`src/kalvin/agent.py` L203–211**):

```python
expected_sig = make_signature(kline.nodes)          # L203
if kline.signature == expected_sig:                 # L204  — True (canonical)
    all_resolved = all(                             # L205
        (node_kl := self._model.find(n)) is not None and self._model.grounded(node_kl)
        for n in kline.nodes
    )
    if all_resolved:                                # L209  — True
        self._model.add_to_ltm(kline)
        self._publish("frame", kline, kline, D_MAX)  # L211  — S1, return True
```

**Why the kline is canonical:** per `src/ks/token_encoder.py` §11.4, a
CANONIZED **compound definition** (`entry.op == "CANONIZED" and len(entry.sig) > 1`,
e.g. `MH`) sets `sig_uint64 = make_signature(node_values)` — so
`MH.signature == make_signature([Mark, Halo_packed])` **by construction**. The
kline is canonical (`signature == make_signature(nodes)`), never a misfit.

**Why `all_resolved` is True:** both nodes are grounded before the MH kline is
rationalised — see the within-lesson pre-grounding amplifier below. The branch
publishes `("frame", kline, kline, D_MAX)` and returns `True` **before**
candidate retrieval (L228), so no `candidate` ever appears.

This is codified as **AGT-14** in `specs/agent.md` (line 364: "Self-grounded
canonical: returns True when all nodes resolve, kline in LTM"; the "Canonical
— self-grounded" phase at line 172). It is the settled contract this task must
not change. `docs/kalvin-vision.md` line ~107 is the apex definition this
violates in spirit: **"S2 klines are misfits: their signature does not match
their nodes. … A dual misfit has both problems."** — but `=>` cannot produce a
misfit by construction, so the curriculum, not the agent, is wrong.

> **Note on the compound-sig node override (the Step-1 deviation):** the
> CANONIZED compound's nodes are `[Mark, Halo_packed]`, taken from the MTS
> resolution cache of the sig `MH` (`ast_emitter._process_scope`), **not** the
> `=> H A` right-hand side. So `MH => H A(lpha)` introduces neither `H` nor
> `A` as nodes — the explicit RHS is discarded for a compound sig. This is a
> *stronger* form of root cause B: `=>` cannot express a misfit for a compound
> sig because it neither lets you pick a non-canonical sig nor pick the nodes.

### The within-lesson pre-grounding amplifier

The pre-grounding that makes `all_resolved` True comes from **lesson 5's own
compilation**, via `KAgentAdapter._handle_submit`
(`src/training/harness/adapter.py` ~L187): it pre-registers **every** compiled
entry in STM (`agent.model.add_to_stm(entry)`) **before** rationalising any of
them, then rationalises them in compile order. The cascade **within a single
lesson's entry list**:

1. The per-BPE-token identities of "Halo" (`0x80080000000048`, `0x2000040000014f`,
   `0x8000020000006f`) are rationalised first → empty nodes → S4 → `add_to_ltm`
   → **grounded** (entries 1–3).
2. The Halo packed-sig CANONIZED kline (`0xa008060000016f`, nodes = those 3
   tokens) is rationalised next → canonical + all 3 nodes grounded → **AGT-14
   S1 short-circuit** → `add_to_ltm` → **grounded** (entry 4).
3. The MH CANONIZED kline (`0xa4088600004d6f`, nodes = [Mark, Halo_packed]) is
   rationalised last → canonical + Mark grounded (from lesson 1 **and** this
   lesson) + Halo_packed grounded (step 2) → **AGT-14 S1 short-circuit** (entry 5).

So even a single self-contained lesson 5 grounds its own compound nodes
before the final kline is rationalised. Restoring lessons 2–4 would
**additionally** pre-ground Mark and Halo from earlier lessons, strengthening
(not replacing) this amplifier — which is exactly why the fix cannot rely on
"un-grounded nodes" alone; it must break the *canonical* property itself.

### The auto-countersign backstop (constraint on the Step-3 fix)

`Reactor._auto_countersign()` (`src/training/trainer/reactor.py`) does a
structural `KLine.__eq__` match of the event's `proposal` against the loaded
lesson entries (`_current_entries`). Even if the agent emitted a genuine S2/S3
`"frame"` event whose `proposal` **equals** any loaded lesson entry, the
reactor would mark it satisfied **without** emitting a `ratify_request`. So for
the KB-309 observable (`ratify_request`) to surface, the dual-misfit must
**reach `expand()`** and yield a **reshaped** proposal **distinct from every
loaded lesson entry**. (In the Step-1 trace the S4 identity proposals equal the
loaded entries → auto-countersigned → no `ratify_request`; and the S1 events
never reach the reactor's S2/S3 path at all.) This is recorded as a hard
constraint on Step 3's fix construction.

### Reconciliation with KB-309's framing

KB-309 phrased it as "the trainer decomposes the dual-misfit into S1
token-frames" (`findings.md` §Follow-ups; `agent.log` ~L1852). This task's
diagnosis — "AGT-14 S1 short-circuit on the canonical compound kline" — is the
**same phenomenon, two layers**:

- The per-BPE-token identities emit S4 frames (auto-countersigned against loaded
  entries in the full stack) — KB-309's "token-frames".
- The canonical compound klines (Halo-packed, MH) hit the AGT-14 S1
  short-circuit — this task's "canonical-not-misfit".

Both layers must be addressed for an S2 event to surface: the kline must be a
genuine misfit (sig ≠ `make_signature(nodes)`, so AGT-14 is skipped) **and** it
must reach `expand()` with a reshaped proposal (so the auto-countersign
backstop does not swallow it).

### Captured event tuple (Step-1, for the record)
`[("ground",0x4008000004d2e,0x4008000004d2e,D_MAX,None),
 ("frame",0x80080000000048,0x80080000000048,0x0,None),
 ("frame",0x2000040000014f,0x2000040000014f,0x0,None),
 ("frame",0x8000020000006f,0x8000020000006f,0x0,None),
 ("frame",0xa008060000016f,0xa008060000016f,D_MAX,None),
 ("frame",0xa4088600004d6f,0xa4088600004d6f,D_MAX,None)]`
(lesson-1 Mark S4 frame omitted; 0 events carry a candidate.)


---

## Step 3 — Decision: the fix construction

### Restored lessons 2-4 (root cause A — non-negotiable)
From `git show a1c40f1^:curricula/first-steps-s2.md`, with NLP annotations:
- `### 2` — `H(alo)`
- `### 3` — `M(ark) == H(alo)` (the `{Mark:[Halo]}` / `{Halo:[Mark]}` countersign — the S2 routing target)
- `### 4` — `A(lpha)`

### Lesson 5 — `(Mark Halo Alpha)\nMH > H A` (CONNOTED; root cause B)
The `=>` CANONIZED operator is ruled out (§11.4 canonical-by-construction). The
chosen construction uses `>` CONNOTED with the `(Mark Halo Alpha)` block binding
so `H`→Halo, `A`→Alpha. Symbolic entries (verified via ASTEmitter):
`{MH:[Halo]}` CONNOTED and `{MH:[Alpha]}` CONNOTED, both with sig = the MH
compound (Mark|Halo) ≠ make_signature(nodes) → genuine misfit, AGT-14 skipped.

> Parsing note: inline node annotations attach to the FIRST node only, so
> `H A(lpha)` garbles to `Hlpha`+`A`. Using a block comment `(Mark Halo Alpha)`
> + bare `H A` resolves both nodes cleanly via the BindingScope.

### Empirical validation (Step 1 repro harness + FIFO StepBus full-stack)
1. **S2/S3 candidate-bearing events fire.** Direct-agent replay: 41 events with
   `0 < significance < D_MAX`, `candidate is not None`. `{MH:[Halo]}` routes S2
   against the restored `{Mark:[Halo]}` countersign (Halo node overlaps). ✓
2. **ratify_request fires (12) in the full-stack** — proving the auto-countersign
   backstop does NOT swallow every proposal. `{MH:[Alpha]}`'s escaping proposals
   (e.g. `{MH:[Halo_packed, <token>]}`) differ from every loaded lesson entry. ✓
3. **Reshaped proposal distinct from every loaded entry** — confirmed (many
   `in_l5=N(ESCAPES)` proposals). ✓
4. **Significance varies with UNRESOLVED_PENALTY** (the KB-309 unblock):
   penalty 5→0xff..f5 (d=10), 10→0xff..eb (d=20), 20→0xff..d7 (d=40) — distinct
   per value, so the auto-tune sweep will no longer be byte-identical. ✓

### Expected behaviour: lesson 5 does NOT auto-complete without a supervisor
Because `{MH:[Alpha]}` escapes the backstop, its entries stay unsatisfied and
emit `ratify_request` (12 of them) instead of auto-completing. This is CORRECT
— a genuine misfit that escapes auto-countersign REQUIRES supervisor
ratification (the whole point of S2). In production the supervisor participant
(delegated or auto-ratifying) handles it; in the no-supervisor test harness the
lesson legitimately stalls. The S2/S3 frame events (the KB-309 significance
observable) are emitted and relayed regardless.

### Rejected alternatives
- `MH => H A(lpha)` (CANONIZED): AGT-14 short-circuit, 0 S2/S3. (baseline, fails)
- `MH > H A(lpha)` (inline annotation): garbles `H`+`(lpha)` → `Hlpha`.
- `M(ark) > H(alo) A(lpha)` (single sig): {Mark:[Halo]} countersigns to S1
  (== lesson-3 kline); only {Mark:[Alpha]} misfits, routes S3 (no Halo overlap),
  0 ratify_request.
- `MH > H(alo)` (single node): completes cleanly but 0 ratify_request (the lone
  {MH:[Halo]} self-reconstructs and auto-countersigns).

### No agent-behaviour fallback needed
The agent/compiler contracts are NOT the obstacle — the S2 routing works as
designed once the curriculum expresses a genuine misfit. No `agent.py` change.
