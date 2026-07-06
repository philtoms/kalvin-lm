# Implement the Synthesizing Trainer — Plan

## Spec References

- `@specs/dialogue-driven-training.md` — WHAT this plan extends: the Actor
  interface (§Actor), the runner (§The Runner), and Validation (§Validation).
  This plan refactors the cursor contract those sections describe and adds a
  synthesizing trainer that satisfies it.
- `@CONTEXT.md` — Identity, Canon (`signature == make_signature(nodes)`),
  Significance, Structural State, KValue.
- `@specs/kline.md`, `@specs/kvalue.md`, `@specs/agent.md` — KLine (signature +
  nodes), KValue, `RationaliseEvent`.

## Purpose

A loadable trainer that is a drop-in replacement for `TableTrainer`. Instead of
reading its next turn from the table, it **synthesises** the next KValue from
two inputs:

1. the **compiled script** (`list[KValue]`), and
2. the **trainee's last KValue** (`incoming`).

The authored dialogue table (`scripts/dialogue-mhall.json`) is the **golden
master**: it remains the validation oracle the runner checks the synthesised
turns against, but the synthesizer reads it **zero** times. MHALL is the
canonical easy path — structurally interesting enough to bootstrap this trainer
(and, in a later grill, a new Kalvin cogitator).

## Design Decisions

**D1 — The table is the validation oracle, not a script.** The synthesizer is
"drop-in" in the precise sense that it satisfies the `Actor` protocol and can
be passed wherever `TableTrainer` goes. It never reads the table to decide what
to say; the runner validates every emitted turn against `decoded[cursor]` like
any real actor (spec DDT-11, design decision D2 of
`@plans/implement-dialogue-driven-training.md`). MHALL's hand-authored T-rows
are deliberately what correct synthesis yields, so the run completes.
_Rejected:_ turning validation off for a synthesizing trainer (defeats the
check that most needs to run); making the T-rows vestigial.

**D2 — The synthesizer is `dbg`-free.** This is a sub-goal: the function must
drop into a **production** trainer that does not rely on development
diagnostics. Every decision derives from `KLine.signature` + `KLine.nodes` +
`signifier.make_signature` — the production `KSignifier` interface — and never
reads `dbg`. _Rejected:_ reading `dbg.op` to distinguish canon from relation
(use the structural Canon test, D5); reading `dbg.label` to bridge identities
to canons (use signature equality, D6).

**D3 — Inputs: `(compiled, incoming, signifier)`.** `compiled` is the compiled
script as `list[KValue]` (production KValues; `dbg` may be absent). `incoming`
is the trainee's last KValue, or `None` for the opening. `signifier` is the
production `KSignifier`. The function is pure: it builds its indices from
`compiled` once and is otherwise a pure function of `incoming`.

**D4 — One level per call; the dialogue enacts recursion.** A signature is a
decode key **only for an identity** (`nodes == []`). Non-identity klines
decompose; the bridge from a kline to its nodes is
`signature == make_signature(nodes)` (the Canon definition). The synthesizer
emits **one level** of decomposition per `respond` and does not recurse
internally — when an emitted node is itself compound, the trainee re-asks about
it on a later turn and R2 fires again. MHALL exercises this (the trainer emits
compound nodes at S2; the trainee then asks about each). _Rejected:_ recursing
to identities within a single call (collapses the multi-turn dialogue into one
turn and loses the S2/S1 distinction that drives it).

**D5 — Canon vs. relation detected structurally.** A compiled kline is a canon
iff `signature == make_signature(nodes)` (`@CONTEXT.md` §Canon). This recovers
the CANONIZED/op distinction with zero `dbg` reads (verified against MHALL:
0 mismatches across 19 nodal entries).

**D6 — Bridging is signature equality, not label lookup.** Given the trainee's
identity `Q` (`{Q: []}`), the synthesizer finds compiled kline(s) whose
**signature == `Q`** and which have nodes, and emits that one-level
decomposition. There is no "identity signature ≠ canon signature" problem: a
kline that has nodes is not an identity, and its signature is the OR-reduction
of those nodes by construction. The earlier worry that `Det`/`Subject`/etc.
needed a `dbg`-label bridge was a terminology failure — those are not
identities. _Rejected:_ fixing the encoder so identity and canon signatures
agree (out of scope, possibly wrong by design); injecting an explicit
identity→canon bridge map (converges with reading `dbg`, which D2 forbids).

**D7 — Fix the historic cursor-leakage anomaly.** Today each actor holds a
cursor into a **role-filtered** subsequence and returns `(next_cursor, event)`,
leaking that subsequence-index into the `Actor.respond` contract and into
validation (`rows_by_role[role][next_cursor]`). A synthesizing actor has no
such index. The fix: the **runner owns the validation index into the full
`decoded` list** (it already walks `decoded` with its own `cursor`); the actor
returns just the event. Validation becomes `expected = decoded[cursor].value`
with a `event.role == decoded[cursor].role` check. This is a breaking change to
the `Actor` protocol, but its blast radius is bounded and internal (two default
actors + a handful of test doubles). _Rejected:_ having the synthesizer fake a
filtered-subsequence cursor (re-introduces the leakage it forces us to remove).

## The Synthesis Rules

Three rules, verified against all 16 trainer turns of the MHALL golden master.
Trigger conditions are detected structurally from `incoming.proposal`
(identity = `nodes == []` or self-referential `{S:[S]}`) plus its significance
band.

**R1 — Opening** (`incoming is None`): emit the **first compiled entry** at its
compiled op, **S2**. Scoped to single-primary, single-cascade scripts (matches
the spec's existing Out-of-Scope).

**R2 — Reply to IDENTITY,S4** (the trainee does not recognise signature `Q`):
find compiled kline(s) with `signature == Q` and non-empty nodes; emit the
first in op-precedence **CANONIZED > UNDERSIGNED > CONNOTED** (canon detected
per D5; relation direction is not needed as a trigger). Ties within a
canonicality class → first in compilation order. Response significance:
- **S2** if any emitted node is itself decomposable (a compound — the trainee
  will re-ask);
- **S1** if all emitted nodes are identities/leaves (bottomed out);
- **S4** if no decomposition exists for `Q` (stalemate — the trainer doesn't
  know it either; unreachable for a well-formed single-primary script).

**R3 — Reply to a non-identity proposal matching a compiled kline** (by
`(signature, nodes)`): emit **that compiled kline verbatim**. Response
significance depends on the match's kind — **S1** for a relation (the trainer
*ratifies* K's tentative proposal; the relation op is irrelevant to the
significance), **S2** for a canon (the trainer *confirms* K's hypothesised
canon). No match → CONNOTED,S4. (R3 fires on any non-identity proposal
regardless of its incoming band — a canon proposed at S2, e.g. K's
`{ALL:[a,little,lamb]}`, is confirmed at S2; a relation proposed at S3 is
ratified at S1.)

> Significance bands carry an intentional reading from the trainer (S1 =
> right/accept; S2 = essentially correct, cogitate; S3 = alternative reading;
> S4 = unknown). This is an anthropomorphic aid, **not** a domain definition —
> significance is technically a function of distance (`@CONTEXT.md` §Significance)
> — and the same band reads differently depending on emitter and context
> (trainer-S4 replying to a trainee-S4 means "I don't know it either," not
> "you're wrong"). It is deliberately not added to `CONTEXT.md`.

**R4 — Close (dialogue-level; lives in a shared `ScriptClose`, not `synthesize`).** A
script's dialogue is bookended: the trainer opens (R1 — primary at S2) and the
trainee closes (an **S1 on the primary**). The trainer recognises the trainee's
S1 on the primary signature as the close and **withholds** — it does not
R3-echo/ratify what the trainee has already grounded. This prevents the
trainer from emitting a spurious closing S1 on the primary (observed: the
synthesiser reproduced MHALL's middle perfectly but then re-emitted the primary
at S1 on the trainee's close, diverging in peer mode). R4 is the single-script
instance of general per-script **open-dialog-close** semantics: trainer opens
a script, trainee closes with S1 on that script's primary. A future multi-
script trainer generalises R4 to track a set of script primaries. The rule
lives in `ScriptClose` (a dialogue-contract object composed by every trainer
derivation, so the rule is written once) — `synthesize` stays
a pure function of `(compiled, incoming)`. `ScriptClose` is single-primary
today; the multi-script generalization (ordered primaries + current index +
`advance` on close) extends that object without touching the trainers.

## Implementation Tasks

### Phase 1 — The synthesis function

**1.1 `src/training/dialogue/synthesize.py`.** A pure function
`synthesize(compiled, incoming, signifier) -> KValue`. Builds (once, internally)
a `decompositions_by_signature: dict[int, list[KLine]]` (compiled klines with
nodes, keyed by signature, in compilation order) and captures the primary
(first compiled entry). Implements R1–R4 with structural predicates only
(identity, canon-via-`make_signature`, compound/leaf via membership in the
decomposition index). `dbg` is never read. → D2, D3, D4, D5, D6.

### Phase 2 — The actor/runner refactor

**2.1 `Actor` protocol.** `respond(incoming) -> RationaliseEvent | None` (drop
the cursor from the return). → D7.

**2.2 `_TableActor` / `TableTrainer` / `TableTrainee`.** Drop `_cursor`, the
`cursor` property, `exhausted`; `respond` returns just the event or `None`.
They become dumb iterators over their role-filtered rows (still never inspect
`incoming` to decide what to emit).

**2.3 `run()` / `_validate()`.** Drop `rows_by_role`. Validate via
`decoded[cursor]`: require `event.role == decoded[cursor].role` (else
`ActorDivergence`); require `event.proposal` (kline + significance) equals
`decoded[cursor].value`. The "actor returned None while the table expects a row"
divergence-by-omission check stays.

**2.4 Test doubles.** Update `_RecordingTrainer`, `_BadTrainee`,
`_SynthesisingTrainer`, `_WrongRoleTrainee`, `_RecordingTrainee` in
`tests/test_dialogue_runner.py` to the new `respond` signature.

### Phase 3 — The synthesizing actor

**3.1 `SynthesizingTrainer` in `runner.py`.** Holds `compiled` indices +
`signifier` (built at construction); implements `respond(incoming)` by calling
`synthesize`, wrapping the result in a `RationaliseEvent` (`proposal` = the
synthesised KValue, `query` = `incoming.proposal` or the synthesised value on
the opening, `role="T"`, `kind="frame"`). Constructor:
`SynthesizingTrainer(compiled, signifier)` — does **not** take `decoded`.
Non-exhausting: always returns an event (R1–R4 always produce a KValue, even
the S4 stalemate). Not produced by `default_actors`; callers wire it
explicitly. → D1.

### Phase 4 — Tests & driver

**4.1 Synthesis unit tests** (`tests/test_synthesize.py`): each rule in
isolation against the MHALL compiled script; the compound/leaf significance
cases; the stalemate (S4); the structural canon-detection. Cover the surface
MHALL leaves unstressed (R2 stalemate, R3 else-branch, relation precedence) so
the rules are not only validated through the golden master.

**4.2 Runner tests.** Update existing tests for the new contract; add a test
that `SynthesizingTrainer` runs MHALL to exhaustion with zero divergence
against the golden master (the trainee stays a `TableTrainee`).

**4.3 `scripts/dialogue_run.py`.** Optional: a `--synthesize` flag that
substitutes `SynthesizingTrainer` for `TableTrainer`, demonstrating the
drop-in.

## File Structure

- `src/training/dialogue/synthesize.py` — new: the pure synthesis function
  (Phase 1).
- `src/training/dialogue/runner.py` — `Actor` protocol simplified;
  `_TableActor`/`TableTrainer`/`TableTrainee` simplified; `run`/`_validate`
  reworked; `SynthesizingTrainer` added (Phases 2–3).
- `tests/test_synthesize.py` — new (Phase 4.1).
- `tests/test_dialogue_runner.py` — updated doubles + new synthesizer test
  (Phases 2.4, 4.2).
- `scripts/dialogue_run.py` — optional `--synthesize` flag (Phase 4.3).

## Out of Scope

- Multi-primary / multi-cascade scripts (R1 assumes a single primary; matches
  the spec's existing Out-of-Scope).
- The Kalvin cogitator (the trainee-side synthesizer) — the next grill.
- Bus integration, supervisor escalation — belong to the real harness trainer.
- Changing the encoder so identity and canon signatures agree (explicitly
  rejected, D6).
