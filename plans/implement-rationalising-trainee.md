# Implement the Rationalising Trainee — Plan

## Spec References

- `@specs/dialogue-driven-training.md` — WHAT this plan plugs into: the Actor
  interface (§Actor), the runner (§The Runner), and Validation (§Validation).
  This plan adds a stateful trainee actor that satisfies the existing contract.
  The spec is unchanged — its Out-of-Scope ("how a real trainee produces its
  responses") is where this plan lives.
- `@CONTEXT.md` — Identity, Canon (`signature == make_signature(nodes)`),
  Significance (S1 as the signal event), Structural State, KValue, Role.
- `@specs/kline.md`, `@specs/kvalue.md`, `@specs/agent.md` — KLine (signature +
  nodes), KValue, `RationaliseEvent`.
- `@plans/implement-synthesizing-trainer.md` — the sibling plan. This plan is
  the "next grill" that plan defers (its Out-of-Scope). The two are symmetric:
  a synthesizing **trainer** (stateless, script-derived) and a rationalising
  **trainee** (stateful, model-derived).

## Purpose

A loadable trainee that is a drop-in replacement for `TableTrainee`. Instead of
reading its next turn from the table, it **rationalises**: it maintains a
minimal model of what it has grounded and derives each turn from `(incoming,
model-state)`, never reading the table or the compiled script. The authored
dialogue table (`scripts/dialogue-mhall.json`) is the **golden master** — the
validation oracle the runner checks every emitted turn against; the rationaliser
reads it zero times.

The rationaliser is a **bootstrap double**: a genuine rationaliser in mechanism,
destined to be replaced only by another rationaliser (the full `KAgent`), never
by a synthesizer. The "bootstrap" is about cogitator complexity, not about the
rationalising nature — memory is load-bearing for its behaviour, and that does
not change when it is replaced.

## Terminology (load-bearing within this plan)

These distinctions are **not** glossary entries (they are temporary scaffolding
to keep trainer-side and trainee-side rationalisation separable), but they are
load-bearing for this plan and must be used consistently:

- **Grounded** — the *trainee's* own understanding; K's own significance
  reaching S1. Use this for K's state. _Avoid:_ bound (loose; replaced by
  grounded because grounded = understood).
- **Ratified** — the *trainer's* S1; the trainer's own understanding. Not a
  synonym for K being grounded. The two are kept distinct deliberately:
  conflating them collapses trainer-rationalisation and trainee-rationalisation
  into one, which is the confusion the prop terms exist to prevent.
- **Relationships** — the cross-canon operand pairings K proposes at Level 1.
  _Avoid:_ bindings (drift; the structural notion is a relationship between
  signatures and nodes).
- **Synthesizer / rationaliser** — descriptive props distinguishing *how* each
  side produces its turn (script-derived vs model-derived). Both sides
  ultimately rationalise; these are temporary scaffolding terms, not domain
  concepts, and do not belong in `CONTEXT.md`.
- **submit / emit / respond** — the verbs at the seams. The trainer **submits**
  on reaching S1; the trainee **emits** on reaching S1. Through the harness bus
  submit == emit; through the dialogue runner both reduce to **respond**.

A **convention** is a deterministic placeholder standing in for a future
significance (distance) computation. It is not a rule and must not be mistaken
for the model's semantics. Conventions are explicitly flagged where they appear.

## Design Decisions

**D1 — The table is the validation oracle, not a script.** The rationaliser is
"drop-in" in the precise sense that it satisfies the `Actor` protocol and can
be passed wherever `TableTrainee` goes. It never reads the table to decide what
to say; the runner validates every emitted turn against `decoded[cursor]` like
any real actor (spec DDT-11). MHALL's hand-authored K-rows are deliberately what
correct rationalisation yields, so the run completes. _Rejected:_ turning
validation off for a rationalising trainee (defeats the check that most needs to
run).

**D2 — `dbg`-free; structural predicates only.** Every decision derives from
`KLine.signature` + `KLine.nodes` + `signifier.make_signature` — the production
`KSignifier` interface — and never reads `dbg`. Mirrors the synthesizing
trainer's D2. Predicates: identity = `nodes == []` or self-referential
`{S:[S]}`; canon = `signature == make_signature(nodes)`; relationship shape =
node count.

**D3 — Minimal bespoke state that mirrors `KModel`, not the full `Model`.**
The rationaliser carries its own small state, not the full `Model` (STM/Frame/LTM
cascade). Rationale: (i) the rationaliser's simplified cogitation is explicitly
NOT `expand()` / `propose_expansions()` / the async `Cogitator`; pulling in
`Model` imports machinery deliberately not used; (ii) self-contained and
testable without standing up the memory cascade; (iii) the full `Model` is what
the *real* Kalvin uses — when the bootstrap is replaced, it swaps in `Model`,
but that is the replacement's concern. "Treat the current agent implementation
as high-level guidance only." The grounded memory **mimics `KModel`**: a dict
keyed by signature, each value the list of grounded klines under it. There is
no ordered distinction between identities and relationships — they are all
klines.

**D4 — Inputs: `(incoming, state, signifier)`.** `incoming` is the trainer's
last `RationaliseEvent` (or `None` for the opening, which the rationaliser has
no special case for — see D8). `state` is the rationaliser's mutable memory.
`signifier` is the production `KSignifier`. The rationaliser reads neither the
table nor the compiled script — this is the key difference from the trainer,
which reads `compiled`.

**D5 — Cogitation is simplified: synchronous, deterministic, inline.** No
background thread, no `expand()` distance computation, no `propose_expansions()`
misfit reshaping. The rationaliser's cogitation is two levels (Identity,
Relationships) plus a reactive grounding side-effect, all evaluated inline per
`respond` call. See §The Rationaliser Mechanism.

**D6 — A work-list, not a stack; selection is LIFO by convention.** The
rationaliser maintains a single work-list of outstanding S2/S3 entries. It
behaves like a stack on the MHALL happy path (LIFO / depth-first) but is a list
so K may select the most appropriate entry in future. LIFO is the **convention**
standing in for a future significance-based selection. A list, not a per-cascade
structure: a multi-cascade dialogue would simply put more entries in the one
list.

**D7 — Grounding is a reactive side-effect of receiving S1, not a cogitation
level.** Receiving an S1 query grounds it immediately in K's state and pops any
matching work-list entry; receiving an S4 query that matches an entry pops it
immediately. Cogitation itself only ever does two things — Level 0 (Identity)
and Level 1 (Relationships). There is no "Level 2 Grounding"; grounding fires
from the entry rule, independently of which entry is selected for cogitation.
Entry rule + cogitation together implement the principle: **process query, then
process next step** (each `respond` applies the entry rule as bookkeeping, then
emits exactly one event from cogitation, unless the work-list is empty).

**D8 — No opening special-case.** Turn 0 flows through the same entry rule as
every other turn: K receives the opening S2 → pushes it → enters Level 0 → emits
IDENTITY. This contrasts with the *trainer* synthesis (R1 is an explicit opening
rule); the asymmetry is fine — the trainer is script-driven and must know to
start, K is reactive and just responds to whatever arrives first.

**D9 — Rationalisation may construct synthetic signatures (cogitation).** When
the grouping convention (D10) requires it, K forms a signature by OR-reduction
of dangling nodes (e.g. `ALL = make_signature([a, little, lamb])`) and proposes
the relationship. Constructing a synthetic signature to jump between
deterministic states is cogitation, not a synthesizer move. The
rationaliser/synthesizer prop is violated only by *reading the table or script
to decide the turn* — never by constructing a signature from node state.

**D10 — Grouping is triggered by residual count imbalance (convention).** During
Level 1, after 1:1 relationship binding from the front, when one side's residual
reaches a single node the other side's entire residual is grouped into one
synthetic operand (`make_signature(remaining_nodes)`), and that relationship is
proposed. This is the **convention** standing in for a future significance-based
grouping (try-all-partitions, pick by distance). It reproduces MHALL's
`{ALL:[Object]}` with no extra grounding rows — ALL is constructed, not looked
up.

**D11 — Retry via group-size escalation (built, validated only against the table
trainer).** A pass groups at a fixed size; if the trainer responds S4 (refusal),
K increments the group size and restarts the pass from the first unmatched pair.
Escalation is pass-level. The rationaliser consumes the trainer's
ratification/refusal as the feedback signal — that is why relationship proposals
are emitted at S3 (1:1) or S2 (non-1:1): they are tentative, inviting
correction. _Coverage limit:_ the stateless `SynthesizingTrainer` cannot refuse
today, so retry/escalation is **built into the rationaliser but validated only
against `TableTrainer`** reading an authored golden master that includes
trainer-S4 refusal turns. A future grill extends the trainer's range of
expression to refuse; until then, the synthetic trainer exercises only the
happy path (group size 1).

**D12 — Termination is the runner's job, not K's.** When the work-list is empty
after the entry rule, K has no work and `respond` returns `None`. The runner
detects this and signals termination (spec DDT-13, DDT-15). The closing
COUNTERSIGNED S1 (MHALL turn 31) is not a special "I'm done" emission — it is
the last entry leaving the work-list as K grounds the open proposal, broadcast
because reaching S1 is always broadcast. No special-case termination logic.

## The Rationaliser Mechanism

### Entry rule (fires on every received query, before cogitation)

| Receive | Action |
| --- | --- |
| **S4** | Pop the matching identity work-item (the other side says "I don't know this either" — stalemate accepted, leaf bottomed out). S4 is a sentinel detected by value, not by band (see @agent spec §Phase 1b). |
| **S1** | Run **cleanup** (see §Cleanup): ground the kline and recursively ground every work-list kline it unblocks. Silent — an S1 emission (broadcast) happens only when K grounds the opening query, not on every grounding (Correction 1); so the S1 branch itself never emits. |
| **S2 or S3** | **Unpack**: push the query kline itself AND identity work-items for each unrecognised node (right-to-left), and — if the signature is unrecognised — `{signature: []}` last (LIFO top, so K asks about the signature before its nodes). The query kline is held pending in the work-list until cleanup grounds it; node identities are what K asks about. |

Then (D7): proceed to cogitation and emit exactly one event, unless the
work-list is empty (→ return `None`, runner terminates).

### Cleanup (recursive grounding engine)

Cleanup is the mechanism that grounds klines **structurally**, never by
promoting an S2 to S1. It is triggered by every received S1 and recurses.

Ground the triggering kline, then repeatedly ground any *groundable* work-list
kline, removing it and continuing (grounding one kline may unblock others).

**Groundability is keyed on the kline's kind, discriminated by MTS**
(@specs/signifier SIG-14; @specs/kscript §8 — a Multi-Token Signature is the
OR-reduction of two or more token values):

- **Identity** ``{sig: []}`` ≡ ``{sig: [sig]}`` — groundable iff its own
  signature is grounded.
- **Canon** ``{S:[nodes]}`` — groundable iff all its nodes are grounded.
- **Relationship** ``{S:[node]}`` (non-canon, non-identity) — groundable iff all
  its nodes are grounded **and ``S`` is not an MTS**.

`_is_mts(signature)` is true iff K has a grounded canon under the signature (K
knows a multi-token decomposition). The MTS distinction is the S2-query clarity
Level 1 surfaced: **a relationship to a single-token (non-MTS) signature is
terminal** — grounding its node fully resolves it (``a`` ↔ Det grounds `a` once
Det grounds). **A relationship to an MTS signature is non-terminal** — grounding
its node does not resolve it (understanding ``MHALL`` ↔ ``SVO`` requires the
operand binding, Level 1, not just ``SVO`` grounding). So an MTS relationship
grounds only by explicit ratification (an S1 for that kline).

- **No S2→S1 promotion.** A kline is grounded only structurally — an S1
  arrived for it, all its canon nodes grounded, or (for a non-MTS relationship)
  its node grounded.
- **Identities are self-referential.** ``{sig: []}`` ≡ ``{sig: [sig]}``; an
  identity retires exactly when its signature grounds.
- **Note on ``is_canon`` vs cleanup.** The codebase ``is_canon`` deliberately
  excludes identities (for retrieval/recency reasons). Cleanup's groundability
  includes identities (an identity is structurally a canon whose sole node is
  itself) and additionally admits non-MTS relationships. These are different
  questions on the same klines.

### Cogitation (Level 0 and Level 1, on the selected work-list entry)

Selection (D6): LIFO — the most-recently-added work-item first (**convention**,
placeholder for future significance-based selection). The work-list is
heterogeneous: it holds both identity asks `{sig: []}` and pending query klines
`{sig:[nodes]}`. Dispatch is by kline **shape** (structural):

**Level 0 — Identity.** The entry is an identity (`is_identity`). Emit
IDENTITY `{sig: []}` at S4. The entry stays in the list; it is retired by
cleanup when its signature grounds, or by the S4 branch on a matching
stalemate.

**Level 1 — Relationships.** The entry is a multi-node relationship
``{L:[R]}`` whose operands L and R are MTS signatures with grounded canons
(e.g. the opening ``{MHALL:[SVO]}``). K pairs the operands of L's canon and R's
canon left-to-right at group size 1, grouping one side's residual into a single
synthetic operand (``make_signature(residual)``) when the other reaches a
single node (D10). Each call emits the **first not-yet-ratified** binding as a
CONNOTED relationship; a binding is ratified once its kline is grounded (the
trainer replied S1). When every binding is ratified, K closes by emitting the
entry itself at S1 (the broadcast that K grounds the opening query, Correction
1) and removes it from the work-list.

Significance is the **emitted kline's signature-to-node count** (grill Q3):
1:1 → S3; multi-node → S2. MHALL's proposals are all 1:1 (grouping makes a
synthetic *lhs*, not multiple rhs nodes), so all are S3; the S2 branch is
coverage gap G1, unexercised by MHALL. Group-size escalation on a trainer S4
refusal (D11) is deferred — the bootstrap targets golden masters the convention
satisfies at size 1.

#### Recognition (the unpack push-decision)

A signature is **recognised** iff it is grounded OR already has an *identity*
work-item in flight. A pending multi-node query kline under the signature does
**not** count — K still wants to ask about that signature, so the query kline
must not recognise its own signature. Recognition drives unpack idempotency
(don't push a duplicate identity for a signature/node already known). The
Level-0 ask-decision needs no separate predicate: it is purely structural
(`is_identity`), not state-based.

#### Work-list discipline

The work-list holds two kinds of entry uniformly: **identity asks**
`{sig: []}` (things K emits about) and **pending query klines** `{sig:[nodes]}`
(held until cleanup grounds them). Both are KLines; cleanup and cogitation
distinguish them structurally. An S1 emission (broadcast) happens only when K
grounds the opening query (Correction 1); every other grounding is silent
bookkeeping.

### Verification against MHALL

The mechanism reproduces **all 15 K-rows** of `scripts/dialogue-mhall.json`
end-to-end with zero divergence: the 11 identity asks (Level 0), the three
operand-binding proposals ``{Mary:[Subject]}``, ``{had:[Verb]}``,
``{ALL:[Object]}`` (Level 1, ALL constructed synthetically via D10), and the
closing ``{MHALL:[SVO]}`` COUNTERSIGNED S1 (the opening grounded). The
rationaliser drives the full dialogue to completion against the table trainer.

## The Minimal State

```python
@dataclass
class _State:
    work_list: list[KLine]
    grounded: dict[int, list[KLine]]
```

`grounded` mirrors `KModel`: keyed by signature, each value the list of
grounded klines under that signature. Identities and relationships are stored
alike — there is no ordered distinction; everything is a kline.

Predicates:
- "K has a record of signature `S`" → `S in grounded`.
- "Kline `(S, nodes)` is grounded" → any kline in `grounded.get(S, [])` whose
  nodes equal `nodes`.
- "Entry matches an incoming query" (for pop rules) → entry's `(signature,
  nodes)` equals the query's.

S1 grounding writes the kline to `grounded[signature]` regardless of shape
(identity, canon, or relationship) — the shape is irrelevant to storage; it is
read structurally where a predicate needs it.

## Implementation Tasks

### Phase 1 — The rationaliser

**1.1 `src/training/dialogue/rationalise.py`.** The `Rationaliser` class — a
stateful `Actor` (`role="K"`, `kind="frame"`). Holds `_State` + `signifier`.
Implements `respond(incoming)` as: entry rule (bookkeeping) → cogitation
(one event) → return, or `None` if work-list empty. Implements Level 0, Level 1
(grouping per D10, escalation per D11), the entry rule, and S1/S4 pop/ground.
Constructs synthetic signatures via `signifier.make_signature`. Reads neither
table nor script nor `dbg`. → D1–D12.

### Phase 2 — Tests

**2.1 `tests/test_rationalise.py`.** Unit tests per mechanism branch, not only
through the golden master. Each test asserts the mechanism **as implemented**
(evolved beyond the plan's original sketch — e.g. S1 is recursive cleanup, not a
simple pop):
- Construction & Actor interface: role `"K"`; respond returns event or `None`.
- Entry rule: S2/S3 unpack (query + node identities + signature identity);
  recognised nodes not re-pushed; S4 pop; S1 ground + cleanup.
- Level 0: identity emission at S4; no opening special-case (D8).
- Cleanup & MTS discrimination: a non-MTS relationship grounds when its node
  grounds; an MTS relationship does not ground early (`_groundable` predicate).
- Level 1 grouping (D10): the 3-vs-1 residual constructs a synthetic ALL
  operand (`make_signature`), not looked up.
- Level 1 significance: every MHALL proposal is 1:1 (single node).
- Termination (D12): empty work-list → `None`.
- **G1 (S2 multi-node proposal) and G2 (escalation on trainer-S4 refusal)**
  are `pytest.skip` markers pointing at the coverage gap rather than tests of
  unimplemented behaviour.

**2.2 Runner integration test.** `Rationaliser` runs MHALL to exhaustion with
zero divergence against the golden master, driven through the runner's `run()`
like any Actor (the trainer stays a `TableTrainer` — the deterministic
oracle). **Prerequisite done:** the runner/Actor refactor (synthesizing-trainer
D7) — `Actor.respond` returns just `RationaliseEvent | None` (no cursor); the
runner validates against `decoded[cursor]`. The spec's §Actor/§Validation and
test matrix (DDT-9, DDT-16) updated to match. This is the canonical end-to-end
proof that a rationalising trainee is a drop-in `TableTrainee` replacement.

**2.3 Driver flag.** `scripts/dialogue_run.py` gains a `--rationalise` flag that
substitutes `Rationaliser` for `TableTrainee`, demonstrating the drop-in (trainer
stays `TableTrainer` for deterministic validation).

## File Structure

- `src/training/dialogue/rationalise.py` — new: the `Rationaliser` (Phase 1).
- `tests/test_rationalise.py` — new (Phase 2.1).
- `tests/test_dialogue_runner.py` — add the rationaliser integration test
  (Phase 2.2).
- `scripts/dialogue_run.py` — `--rationalise` flag (Phase 2.3).
- `scripts/dialogue-mhall.json` — unchanged. The golden master is reproduced
  as-is; no extra rows are added (ALL is constructed, not grounded — D10).

## Test Mapping

This plan introduces no new spec IDs (the spec is unchanged; the rationaliser
satisfies the existing DDT-\* Actor/Validation contract). Tests map to the
mechanism branches above and to the canonical end-to-end run.

| Mechanism / Spec | Test |
| --- | --- |
| Actor interface (DDT-9, DDT-16, DDT-17) | `Rationaliser.respond` returns `RationaliseEvent` with `role="K"` or `None` |
| Validation (DDT-11) | MHALL runs to exhaustion with zero divergence |
| Entry rule — S4 pop | unit |
| Entry rule — S1 ground + pop | unit |
| Entry rule — S2/S3 push | unit |
| Level 0 identity | unit |
| Level 1 1:1 → S3 | unit |
| Level 1 non-1:1 → S2 | unit (synthetic golden master — Coverage Gap G1) |
| Grouping D10 | unit |
| Escalation D11 | unit (authored trainer-S4 golden master — Coverage Gap G2) |
| No opening special-case (D8) | turn-0 path equals any other S2 path |
| Termination via empty work-list (D12) | work-list empty → `None` → runner terminates |

## Coverage Gaps (known, deferred)

- **G1 — The S2 (non-1:1) relationship branch is built but unexercised by
  MHALL.** Every MHALL relationship proposal is 1:1 → S3. A relationship
  asserting one operand fills multiple roles (e.g. `{Mary:[Subject, Object]}`)
  does not occur. Supply a synthetic golden master to cover this path.
- **G2 — Escalation requires a trainer-S4 refusal, which the stateless
  `SynthesizingTrainer` cannot produce.** Retry/escalation is validated only
  against `TableTrainer` reading an authored golden master. Extending the
  trainer's range of expression to refuse is a future grill.
- **G3 — Multi-cascade / multi-primary scripts.** Single active work-list;
  matches the spec's existing Out-of-Scope. A future grill.
- **G4 — The conceptual leap as a primary strategy.** This bootstrap reaches
  synthetic signatures only via the grouping convention (D10), triggered by
  count imbalance. A mature rationaliser leaps on significance when no scaffold
  exists; deferred to a later grill where true distance-based significance can
  inform the leap rather than a count heuristic.
- **G5 — Decoder label-collision (RESOLVED).** When a KScript label names
  both a canon and its atoms (e.g. `Det`, `Subject`, `Verb`, `Object` in
  MHALL), the decoder's IDENTITY resolution previously picked the atom (first
  compiled entry) instead of the canon. Fixed: the IDENTITY branch now prefers
  `canon_by_label` when the label names a canon, mirroring the
  constructed-relation branch. The rationaliser then reproduces all 11
  identity K-rows.

## Out of Scope

- The synthesizing trainer's range of expression (refusal/S4) — a future grill
  (needed to exercise escalation against the synthetic trainer, G2).
- Multi-primary / multi-cascade scripts (D6; matches the spec's existing
  Out-of-Scope).
- Replacing the bespoke state with the full `kalvin.Model` — that is the
  real-Kalvin replacement's concern, not the bootstrap's (D3).
- Significance-based selection and grouping (the conventions D6, D10 stand in
  for these).
- Bus integration, supervisor escalation — belong to the real harness trainer.
