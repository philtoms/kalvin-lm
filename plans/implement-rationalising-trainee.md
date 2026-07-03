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

**D9 — A synthesised signature is emitted as a canonical request, not a
binding assertion (the "leap too far" correction).** When the grouping
convention (D10) identifies a residual that would require synthesis, K forms
the signature by OR-reduction (`ALL = make_signature([a, little, lamb])`) but
emits `{ALL:[a,little,lamb]}` at S2 — a **canonical request** ("is this a
thing?"), the same shape as every other S2 query — NOT a relationship
assertion. K cannot legitimately assert a relationship to a signature it
invented; it must first confirm the signature exists. The trainer replies with
the scaffolding; K traverses it; async relationships (`{a:[Det]}`) ground by
elevation on re-receipt. Constructing a synthetic signature to pose as a
hypothesis is cogitation, not a synthesizer move; the rationaliser/synthesizer
prop is violated only by *reading the table or script to decide the turn*.
_Replaced:_ the earlier "synthesise-and-bind" (`{ALL:[Object]}`) which leapt
past the confirmation step.

**D10 — Grouping is triggered by residual count imbalance (convention).**
During Level 1, after 1:1 pairing from the front, when one side's residual
reaches a single node the other side's entire residual is grouped into one
synthetic operand (`make_signature(remaining_nodes)`). Per D9, that grouped
pair is emitted as a canonical request (S2), not a binding. This is the
**convention** standing in for a future significance-based grouping
(try-all-partitions, pick by distance).

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

**D12 — Termination is the runner's job, not K's.** When no entry is
*workable* after the entry rule (no identity, no Level-1-eligible opening —
though async-pending relationships may remain in the list), K has nothing to
emit and `respond` returns `None`. The runner detects this and signals
termination (spec DDT-13, DDT-15). The closing COUNTERSIGNED S1 (MHALL's last
K-row) is not a special "I'm done" emission — it is the opening relationship
leaving the work-list once all its pairs resolve (K grounds it via its own S1
broadcast, Correction 1). No special-case termination logic.

## The Rationaliser Mechanism

### Significance elevation (the async-grounding mechanism)

Not all grounding is synchronous. A relationship may arrive at S2 before its
node is grounded; it cannot ground then, but it may ground later, when its node
has grounded through some independent cascade. The stateless trainer, unable to
track K's state, resubmits the same S2 — and K, re-deriving its own significance
from current model state, finds the node now grounded and **elevates** the
relationship to S1.

**Elevation** (`_elevatable`): an incoming S2/S3 **relationship** (non-canon,
non-identity) whose nodes are all now grounded is elevated — K grounds it at S1.
This is the two-way significance dialog: K does not blindly accept the sender's
declared significance; it re-derives its own, which may be higher. Canons and
identities are never elevated (they have their own grounding paths).

This replaces the earlier MTS discriminator (which was contrived: it keyed on
signature shape, coincidental for single-token `a`). The honest rule is:
**relationships ground by elevation on re-receipt, never by node-resolution in
cleanup.** The over-grounding bug (the opening `{MHALL:[SVO]}` grounding early)
is prevented because the opening is *received once* and never re-received —
elevation never fires for it. The opening grounds by K's own closing S1
broadcast, not by elevation.

### Entry rule (fires on every received query, before cogitation)

| Receive | Action |
| --- | --- |
| **S4** | Pop the matching identity work-item (stalemate accepted). S4 is a sentinel detected by value, not by band. |
| **S1** | Run cleanup (ground + recurse). |
| **S2 or S3** | (1) **Elevation check**: if the query is an elevatable relationship, ground it via cleanup and return. (2) Otherwise unpack. (An identity asked at Level 0 is retired on *emission*, not on S2/S3 receipt — the `asked` set prevents re-asking; see §Cogitation.) |

Then proceed to cogitation (emit exactly one event, or return `None` when no
entry is workable — D12).

### Cleanup (recursive grounding engine)

Ground the triggering kline, then repeatedly ground any *groundable* work-list
kline, removing it and continuing (grounding one kline may unblock others).

**Groundability** (`_groundable`) — only **canons and identities**, never
relationships:

- **Identity** `{sig: []}` ≡ `{sig: [sig]}` — groundable iff its own signature
  is grounded.
- **Canon** `{S:[nodes]}` — groundable iff all its nodes are grounded.
- **Relationship** (non-canon, non-identity) — **never** groundable here. It
  grounds by elevation on re-receipt (above), not by node-resolution. This is
  what stops the opening relationship from grounding prematurely.

- **No S2→S1 promotion in cleanup.** A kline is grounded here only structurally
  (an S1 arrived for it, or all its canon nodes grounded).
- **Identities are self-referential.** `{sig: []}` ≡ `{sig: [sig]}`; an identity
  retires exactly when its signature grounds.
- **Note on `is_canon` vs cleanup.** The codebase `is_canon` deliberately
  excludes identities (for retrieval/recency reasons). Cleanup's groundability
  includes identities (an identity is structurally a canon whose sole node is
  itself). These are different questions on the same klines.

### Cogitation (Level 0 and Level 1)

**Selection: LIFO among *workable* entries** (a convention, placeholder for
future significance-based selection — the work-list is a list, not a queue,
because not every entry is workable at all times). Cogitation scans the
work-list top-down for the first workable entry:

- an **identity** (always askable), or
- a **Level-1-eligible opening** (a single-node relationship `{L:[R]}` whose
  operands L and R both have canons K has *seen* — in the work-list or grounded
  — so K can read their operands to pair).

**Async-pending relationships** (e.g. `{a:[Det]}` awaiting elevation) are **not
workable** — K cannot emit about them now; they are skipped, and removed later by
cleanup when elevation grounds them.

**Level 0 — Identity.** Emit IDENTITY `{sig: []}` at S4. The identity work-item
is **popped on emission** (fire-and-forget): the ask does not linger to block
cogitation under LIFO. This is what lets K move past an async-blocked signature
(e.g. `a`, whose `{a:[Det]}` awaits elevation) to ask the next workable entry,
rather than banging against a lingering `{a: []}`.

**Level 1 — Relationships.** The eligible opening `{L:[R]}`. K pairs the operands
of L's canon and R's canon left-to-right at group size 1, grouping one side's
residual into a single synthetic operand when the other reaches a single node
(D10). Each call emits the next unresolved pair:

- a **1:1 pair** `{lhs:[rhs]}` is emitted CONNOTED at S3 (a tentative connoted
  relationship, inviting ratification);
- a **grouped pair** (a synthesised residual) is emitted as a **canonical
  request** `{make_signature(residual): residual}` at S2 — *"is this a thing?"* —
  the same shape as every other S2 query. K cannot legitimately assert a
  relationship to a signature it invented (the "leap too far" was asserting the
  binding, not the synthesis). The trainer replies with the scaffolding; K
  traverses it; async relationships (`{a:[Det]}`) ground by elevation on
  re-receipt.

A pair is **resolved** when: a 1:1 pair is ratified (its kline grounded, trainer
replied S1), OR a grouped pair's synthesised canon is grounded (the
canonical-request traversal completed and any async relationships elevated).
When every pair is resolved, K closes by emitting the entry itself at S1
(COUNTERSIGNED) — the broadcast that K grounds the opening query (Correction 1)
— and removes it from the work-list. Group-size escalation on a trainer S4
refusal (D11) is deferred.

#### Recognition (the unpack push-decision)

`_recognised` has two modes:

- **Node** (re-traversal): recognised iff grounded, asked, or an identity is in
  flight. Re-traversing an already-asked node in a *new* canon is legitimate
  (the new context may resolve it — e.g. re-asking `a` while traversing ALL, so
  the re-receipt can elevate `{a:[Det]}`), so the `asked` set does **not**
  suppress node asks; it only suppresses duplicate identity work-items in flight.
- **Signature**: recognised iff grounded or asked. A signature K has already
  asked about is not re-asked when a later unpack encounters it as a signature
  (its identity work-item was popped on emission). The `asked` set is also
  populated when Level 1 emits a canonical request (the synthesised signature
  has been "asked about" via a different shape).

#### Work-list discipline

The work-list is heterogeneous: **identity asks** `{sig: []}` (transient —
popped on emission), **pending query/canon klines** `{sig:[nodes]}` (held until
cleanup grounds them), and **async-pending relationships** (held until elevation
grounds them). All are KLines; cleanup, cogitation, and elevation distinguish
them structurally. An S1 emission (broadcast) happens only when K grounds the
opening query (Correction 1) or via elevation; every other grounding is silent
bookkeeping.

### Verification against MHALL

The mechanism reproduces **all 16 K-rows** of the modified
`scripts/dialogue-mhall.json` end-to-end with zero divergence: the 11 identity
asks (Level 0), the two 1:1 CONNOTED proposals `{Mary:[Subject]}`,
`{had:[Verb]}` (ratified at S1), the **canonical request**
`{ALL:[a,little,lamb]}` at S2 with its scaffolding reply and async-elevation
cascade (the modified golden master turns 27–30), and the closing
`{MHALL:[SVO]}` COUNTERSIGNED S1 once all pairs resolve. The rationaliser drives
the full dialogue to completion against the table trainer.


## The Minimal State

```python
@dataclass
class _State:
    work_list: list[KLine]
    grounded: dict[int, list[KLine]]
    asked: set[int]
```

`grounded` mirrors `KModel`: keyed by signature, each value the list of
grounded klines under that signature. Identities and relationships are stored
alike — there is no ordered distinction; everything is a kline. `asked` tracks
signatures K has already asked about (as identities at Level 0, or as canonical
requests at Level 1) so unpack does not re-ask them as signature identities
(nodes may still be legitimately re-asked in a new traversal — see
§Recognition).

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
  recognised nodes not re-pushed; S4 pop; S1 ground + cleanup; S2/S3 elevation
  of an elevatable relationship.
- Level 0: identity emission at S4 (popped on emission); no opening special-case (D8).
- Async grounding & cleanup: a relationship grounds by **elevation on
  re-receipt** (`_elevatable`), not by node-resolution in cleanup; relationships
  are never `_groundable` (only canons/identities are).
- Level 1 grouping (D10): the 3-vs-1 residual emits a **canonical request**
  `{make_signature(residual): residual}` (S2), not a binding assertion; 1:1
  pairs carry no residual (CONNOTED S3).
- Termination (D12): no workable entry → `None`.
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

**2.3 Driver flag.** `scripts/dialogue_run.py` gains a `--rationalise` flag
that substitutes `Rationaliser` for `TableTrainee`, demonstrating the drop-in
(trainer stays `TableTrainer` for deterministic validation). Implemented and
verified: both modes run MHALL to completion (exit 0, 30 events);
`--rationalise --verbose` traces the full rationalised exchange through the
identity phase, the Level-1 relationship proposals and canonical request, and
the async-elevation cascade, to the closing S1.

## File Structure

- `src/training/dialogue/rationalise.py` — new: the `Rationaliser` (Phase 1).
- `tests/test_rationalise.py` — new (Phase 2.1).
- `tests/test_dialogue_runner.py` — add the rationaliser integration test
  (Phase 2.2).
- `scripts/dialogue_run.py` — `--rationalise` flag (Phase 2.3).
- `scripts/dialogue-mhall.json` — **modified**: the ALL grouped residual is
  now a canonical-request + scaffolding-reply + async-elevation sequence
  (turns 27–30) instead of a synthesised `{ALL:[Object]}` binding. 16 K-rows.

## Test Mapping

This plan introduces no new spec IDs (the spec is unchanged; the rationaliser
satisfies the existing DDT-\* Actor/Validation contract). Tests map to the
mechanism branches above and to the canonical end-to-end run.

| Mechanism / Spec | Test |
| --- | --- |
| Actor interface (DDT-9, DDT-16, DDT-17) | `Rationaliser.respond` returns `RationaliseEvent` with `role="K"` or `None` |
| Validation (DDT-11) | MHALL runs to exhaustion with zero divergence |
| Entry rule — S4 pop | unit |
| Entry rule — S1 cleanup | unit |
| Entry rule — S2/S3 unpack + elevation | unit |
| Level 0 identity (popped on emission) | unit |
| Async grounding — elevation on re-receipt (`_elevatable`) | unit |
| Relationships never cleanup-groundable (`_groundable`) | unit |
| Level 1 1:1 → S3 (no residual) | unit |
| Level 1 grouped → canonical request S2 (residual) | unit |
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

  **Exploration (post-implementation probe).** The signifier exposes a
  *semantic* signal — `signifies(a, b)` is true when two values share an NLP
  type-word bit (upper 32 bits). That is the natural significance signal a leap
  would act on. Probed against MHALL's grouped residual `[a, little, lamb]`:
  `a` does not signify `little` or `lamb`; `little` signifies `lamb`; the
  constructed `ALL` does not signify its target `Object`. For contrast, the
  correct 1:1 binding `Mary ↔ Subject` does **not** signify either. So on
  MHALL, `signifies` is a *weak, partial* signal — present for some true
  groupings, absent for others, and absent for correct bindings.

  Three consequences for any future leap strategy:

  1. **`signifies` alone is insufficient.** A leap keyed purely on `signifies`
     would miss `Mary ↔ Subject` (no shared type-word) and would not confirm
     `[a,little,lamb] → Object`. The count convention (D10) currently gets
     these right where significance is silent; a leap must *combine* significance
     with structure, not replace structure with it.
  2. **The leap's home is the relationship phase (Level 1), not the identity
     phase.** Synthetic signatures arise when K must relate two operands of
     different cardinality — i.e. inside Level 1's `_relationship_plan`. A
     primary leap strategy means Level 1 considers *candidate groupings*
     (partitions of the residual) scored by significance, rather than the
     single deterministic group-the-whole-tail rule. This is the
     "try-all-partitions, pick by distance" future form D10 names. (Under the
     async model, the chosen grouping is still emitted as a canonical request
     per D9 — the leap chooses *which* signature to hypothesise, not whether to
     assert it.)
  3. **Distance, not boolean significance, is the real signal.** The mature
     Kalvin uses `expand()`'s packed distance (the inverted significance) to
     rank candidates. The bootstrap rationaliser has no distance computation —
     it trades that for deterministic conventions. A G4 resolution therefore
     depends on giving the rationaliser *some* distance proxy (even a simple
     one: shared type-word count, node-overlap count), which is a larger change
     than the bootstrap's "no `expand()`" stance (D3, D5).

  **Recommendation for the next grill.** Frame G4 as: *replace D10's count-only
  grouping with a significance-scored partition search, accepting that the
  significance proxy will be approximate and must co-exist with the structural
  count rule as a tie-breaker.* The probe shows `signifies` is too weak to
  stand alone; the grill should decide which distance proxy (shared type-words?
  the real `expand()` distance? a bespoke node-overlap metric?) is appropriate
  for a bootstrap, and how significance and structure combine when they
  disagree. Do **not** attempt this before G1 (the S2 multi-node proposal
  branch) is covered — the partition search emits multi-node proposals, which
  the S2 branch exists to handle and is currently unexercised.
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
