# Dialogue-Driven Training — Specification

## Overview

This spec defines a training model in which a lesson is driven by an authored
**dialogue** between Trainer (T) and Kalvin (K), turn by turn. The dialogue is a
deterministic, table-driven artifact; the Trainer is a **stateless** responder;
Kalvin is driven by a deterministic contract double during bootstrap and by real
rationalisation thereafter. The Trainer's only verification of Kalvin is that K
emitted the expected kline (the K-side of two-sided validation); it never
assesses whether Kalvin has *learned* anything. The run terminates on dual
cursor exhaustion.

This is an **alternative** to `@specs/trainer-satisfaction.md` and its associated
planning. It replaces that spec's batch-partition + held-index framing with a
table-driven, pre-decoded turn sequence and an explicit stateless supply rule.
Where the two disagree, this spec governs. It fits into the existing harness
architecture (`@specs/harness-server.md`): same bus, same message actions, same
Trainer participant — only the trainer's internal loop and its configuration
source change.

## Dependencies

- `@CONTEXT.md` — Canon, Ratify, Significance, Structural State, Scaffolding.
- `@specs/harness-server.md` — Trainer participant, bus actions (`submit`,
  `countersign`, `rationalise`), supervisor messages, KAgent adapter.
- `@specs/agent.md` — RationaliseEvent shape; significance carried on the KValue.
- `@specs/kscript.md` — compiled-entry `op` field and declared bands; structural
  states (COUNTERSIGNED, CANONIZED, CONNOTED, UNDERSIGNED, IDENTITY).
- `@specs/kline.md` — `is_canon`, `is_identity`, KLine equality.
- `@specs/signifier.md` — `make_signature` (Canon predicate, canonical signatures).
- `@specs/stub-kagent.md` — the deterministic contract double the dialogue is
  validated against; the behaviour the real Kalvin must reproduce.
- `@specs/supervisor-decision.md` — what happens when the Trainer cannot resolve
  a proposal (escalation). Out of the bootstrap run.

## Definitions

### Dialogue Table

The source artifact for a lesson. A JSON object with two fields:

```
DialogueTable:
  script:  str          # KScript source — the authority for all kline structure
  turns:   list[Turn]   # ordered, the exact T/K exchange
```

```
Turn:
  actor:        "T" | "K"
  op:           str   # structural state: COUNTERSIGNED | CANON | CONNOTED |
                      #                UNDERSIGNED | IDENTITY
  signature:    str   # symbolic label, resolved by the decoder
  nodes:        list[str]   # symbolic labels, resolved by the decoder
  significance: "S1" | "S2" | "S3" | "S4"
  notes:        str   # human commentary; ignored by the decoder
```

`script` is the **single source of truth** for kline structure (canonical
signatures, atom values, subword composition). `turns` is the **exact exchange**:
the table is prescriptive, not predictive — the Trainer and the stub are
redesigned until their outcome agrees with it.

### Decoded Turn

A turn resolved to a submittable structure. Produced by the **decoder** (§Decoder)
at configuration time:

```
DecodedTurn:
  actor:  "T" | "K"
  op:     str       # passed through from the Turn
  value:  KValue    # KLine(signature, nodes) + significance, resolved to uint64
```

`actor` and `op` are carried alongside the KValue — never folded into it. `op`
and `significance` are independent axes: the op is the structural state; the
significance is the dialogic stance (§Significance Contract). They diverge in
most turns and neither reconstructs the other.

### Decoder

A single-stage, configuration-time function. The decoder is **not** part of the
training loop; it is a pre-loop configuration stage that turns a DialogueTable
into a flat ordered list of `DecodedTurn`.

```
decode(table) -> list[DecodedTurn]
```

Operation, per turn:

1. **Resolve the kline from `script`** (script is authority):
   - **Canon** — retrieve the compiled canon whose node-decoded labels match the
     turn's `nodes` list (node-list match). The turn's symbolic node names are
     the retrieval key.
   - **Identity / atom** — resolve by label.
   - **Constructed relation** (a relation K builds between two compounds, e.g.
     `{Mary:[subject]}`) — resolve each compound label to its canonical signature.
2. **Attach significance** by band lookup (`"S1"→SIG_S1`, …) from the turn.
3. **Pass through** `actor` and `op`; **ignore** `notes`.

Subword node names in the table must be the tokenizer's actual decoded subwords
(not pseudo-names); the table is maintained against real compiler output. Subword
atoms only ever appear as nodes of their own canon, so construction never needs
subword resolution — it only resolves compound and atom labels.

### The Trainer is Stateless

The Trainer carries **no temporal state** about the dialogue: no provenance
ledger, no open-proposal tracking, no record of which proposals it originated.
Its response to an incoming turn is a pure function of `(incoming turn, compiled
script)`.

Open-proposal state (e.g. that the primary `{MHALL:[SVO]}` opened at turn 1 and
closes only at the final countersign) is **Kalvin's** state, not the Trainer's.
The Trainer keeps answering requests until Kalvin closes the run.

> Principle: the Trainer is always functional and deterministic. Participants
> (supervisors and Kalvin) are rational, carry temporal state, and cogitate.

### Significance Contract

Significance is **symmetric** — either party may send any band — and turn-based.
A turn's significance is simultaneously the **turn contract** (whose move) and
the **semantic stance** (what the sender asserts):

| Band | Turn contract | Semantic stance |
|------|---------------|-----------------|
| S1   | terminal — no response expected; advance | asserted fact |
| S2   | proposal — the other party responds | aggregation / open whole |
| S3   | proposal — the other party responds | tentative association |
| S4   | request — "I do not recognise this"; supply | novel / ungrounded |

Symmetry is a design feature: there is no structural reason Kalvin cannot assume
a supervisor role. (Exercising K-originated S2 is deferred to a later grill; the
bootstrap run has K sending only S4 requests, S3 proposals, and the closing S1.)

### Partition and the Supply Rule

On lesson compile, entries are partitioned:

- **Withheld** — Identities (`op = IDENTITY`) and **Canons** (`is_canon`,
  including authored semantic canons and tokenizer subword canons — subword
  canons are not filtered; §Subword Canons). Held by the Trainer; supplied only
  on request.
- **Held index** — a lookup from **signature → held klines** for that signature,
  ordered by pull priority: **canon → relation → identity**.

The supply rule, on a K request for signature X (an S4 turn whose proposal is
`X:[]`):

> Look up X in the held index and supply the held kline(s) for X in priority
> order. **Canon-first, with full shadowing.**

**Full shadowing:** when a signature carries both a Canon and a relation, the
Canon is supplied and the relation is **never supplied** by the Trainer. A
relation sitting behind a canon is **K-discovered** (K proposes it as an S3
tentative association) and **T-ratified** (T responds with the reciprocal at
S1). This is how the Trainer makes Kalvin work harder: anything behind a canon
in the held index is invisible to supply and must be proposed by Kalvin.

A relation whose signature has **no** canon (e.g. `{a:[Det]}`) is not shadowed
and is supplied on request. The distinction between "supplied relation" and
"K-discovered relation" is therefore not a labelling rule — it is an emergent
consequence of canon-priority lookup.

### Significance the Trainer Attaches (node-terminality)

The significance the Trainer attaches to a supplied entry is derived from
**whether its nodes are terminal**:

- **Terminal nodes** (subword atoms / authored atoms that need no further
  decomposition) → **S1** (asserted fact).
- **Non-terminal nodes** (compounds/structures that themselves need resolving)
  → **S2** (open aggregation).

This rule is generative and self-validating: a table entry that violates it
(e.g. supplying `{a:[Det]}` at S1 when Det is non-terminal) is wrong by
construction, and correcting the table is how the rule is checked.

### Ratification

On a K **proposal** (an S3 turn), the Trainer ratifies: it responds with the
reciprocal at **S1** (UNDERSIGNED for a CONNOTED proposal, etc.). Ratification is
the Trainer's endorsement of a structure Kalvin constructed. Per `@CONTEXT.md`
§Ratify, ratification is the action of countersigning a selected proposal.

### Termination and K-verification

There is no satisfaction or "learned" computation. Kalvin learns continuously
(monotonic memory) and grounds under S1 ratification by updating LTM; it emits
**no grounding event**, so there is nothing for the Trainer to match a declared
band against. The Trainer never verifies whether Kalvin has learned an entry —
its only verification of Kalvin is structural: **did K emit the expected kline at
the cursor?** (the K-side of two-sided validation, §Training Loop).

The run ends on **dual cursor exhaustion** (§Training Loop): the trainer-side
cursor and the stub cursor both reach the end. The canonical table's final K is
the closing S1 countersign of the primary; dual-exhaustion requires that K-run
to have been emitted and matched, so the closing S1 is **verified by
construction, not trusted** (and not semantically detected — the Trainer stays
stateless).

### Training Loop (dispatch and cursors)

The loop is **dispatch-driven**, not replay. The pre-decoded table is the single
source of truth, read by **two self-cursored actors** — the trainer-side loop and
the stub — each consuming only its own rows. The trainer computes its turns via
the stateless supply function; the table **validates** the trainer's output and
**scripts** the stub's. The table never *drives* the trainer.

**Greedy cursor dispatch.** A cursor walks the table. A run of consecutive
same-actor rows is consumed whole by that actor: a T-run is submitted turn by
turn; a K-run is emitted by the stub. 1:1 alternation is the canonical table's
shape, **not a loop constraint** — a future table with longer T-runs or K-runs is
handled identically.

**Self-cursored actors.** The stub holds its own cursor over K-rows and emits its
next K-run in response to each `rationalise` call, exactly as real Kalvin will —
it owns its turn and never matches the submitted kline. The trainer-side loop
holds its own cursor over T-rows. They stay in sync via the bus protocol. This is
the bus-faithful reading: each actor owns its consumption of the shared table.

**Two-sided validation (Model A).** Each turn is validated on both sides:

1. The trainer-side receives the stub's K-run and **validates each emitted K ==
   table K** at the cursor (stub/table divergence if not).
2. It dispatches on the K significance and **computes the next T** via the supply
   function.
3. It **validates computed T == table T** at the cursor (supply-function bug if
   not).
4. It submits T; the stub emits its next K-run; repeat.

This fails fast and attributes precisely: a K mismatch is a stub/table problem;
a T mismatch is a supply-function problem.

**Dual-exhaustion termination.** The run ends only when **both** cursors are
exhausted. The greedy model guarantees they reach zero together under a
well-formed table: the table's final K-run rides the last T submission's
`rationalise`. Dual-exhaustion **verifies the final K by construction** — the
closing S1 must be in a K-run the stub actually emitted and Model A validated; it
is never trusted on trainer-exhaustion alone. A truncated table leaves the stub
cursor non-empty → the run fails, naming the un-emitted K-run. The closing S1's
*semantic identity* (it is the primary's countersign) is the table's
responsibility, not a trainer detection — so termination stays stateless.

**Synchronous execution** under the stub: submit → drain → validate, no event
loop. Multi-row K-run timing (whether a 3-row K-run emits on one `rationalise`)
is moot for the 1:1 canonical table and deferred until a non-1:1 table is
authored.

## Behavioural Rules

### Configuration (pre-loop)

1. The dialogue table is the lesson's source artifact (`script` + ordered
   `turns`).
2. The decoder pre-decodes every turn into a flat ordered list of `DecodedTurn`
   at configuration time. The training loop never touches `script` or the symbol
   map.
3. Decoding is single-stage: one function resolves the kline from `script`,
   attaches significance by lookup, and passes through `actor`/`op`; `notes` are
   ignored.
3a. Annotation-only turns (turns with `notes` but no structural fields) are
   **dropped at decode time** — they are not submittable klines. Decoding yields
   only structurally-complete turns.

### Opening

4. The Trainer emits exactly one non-reactive turn to start the run: the
   **primary** relationship (the root `==`), as a single half at **S2**. Opening
   the run is itself a proposal. The opening is **computed** by the supply
   function (an opening entry point: "if no prior context, emit the primary half
   at S2"), not read from the table — so every T turn, including turn 0, flows
   through the same function and is validated against the table.
5. After the opening, the Trainer is **purely reactive**: every subsequent T turn
   is a deterministic response to a K turn.

### Driving the dialogue (stateless response)

6. On a K **request** (S4, proposal `X:[]`): the Trainer looks up signature X in
   the held index and supplies the held kline(s) for X in priority order
   (canon → relation → identity), one per turn.
7. On a K **proposal** (S3): the Trainer ratifies — responds with the reciprocal
   at S1.
8. On a K **terminal** (S1): the Trainer takes no action; the run advances.
   Termination is governed by dual cursor exhaustion (§Training Loop), not by
   detecting the closing S1.
9. Canon-first with full shadowing: a relation whose signature carries a canon is
   never T-supplied; it is K-discovered (S3) then T-ratified (S1).
10. The significance the Trainer attaches to a supplied entry is derived from
    node-terminality (terminal → S1, non-terminal → S2).
11. The Trainer is stateless. Its response depends only on the incoming turn and
    the compiled script — never on dialogue history.

### K-verification and termination

12. The Trainer never verifies whether Kalvin has learned an entry. Its only
    verification of Kalvin is structural: the K it received equals the table's K
    at the cursor (the K-side of two-sided validation, §Training Loop).
13. Kalvin learns continuously (monotonic memory) and grounds under S1
    ratification by updating LTM. It emits **no grounding event** — there is no
    "declared band" event to match, and no `learned ⊇ submitted` computation.
14. The Trainer never verifies whether Kalvin has learned an entry; its only
    K-verification is structural (K == table K at the cursor).
15. The run ends on **dual cursor exhaustion** (trainer cursor AND stub cursor
    empty). The canonical table's final K is the closing S1 countersign of the
    primary; dual-exhaustion requires that K-run to have been emitted and
    validated, so the closing S1 is verified, not trusted.

### Training Loop (dispatch and validation)

22. The loop is **dispatch-driven**: the trainer computes each T turn via the
    stateless supply function in response to K's emission; the table validates T
    and scripts K. It is not a replayer that submits T turns read from the table.
23. Both actors are **self-cursored**: the trainer-side loop owns its cursor over
    T-rows, the stub owns its cursor over K-rows. Neither matches the other's
    klines; they stay in sync via the bus.
24. Dispatch is **greedy per actor**: a run of consecutive same-actor rows is
    consumed whole. 1:1 alternation is the canonical table's shape, not a
    constraint.
25. Each turn is **validated on both sides** (Model A): received K == table K
    (stub/table divergence), then computed T == table T (supply-function bug).
26. Termination is **dual-exhaustion**: both cursors empty. A table that ends
    without consuming its final K-run fails loudly (non-empty stub cursor), so a
    truncated table can never pass.
27. The closing S1 is verified, not detected: the trainer never semantically
    identifies "the primary's countersign" — it verifies the table's final K-run
    was consumed and matched, which is stateless-compatible.

### Subword Canons

16. Tokenizer subword canons are **not filtered** out of the withheld set. They
    are withheld and ratifiable, supplied at the node-terminality band like any
    Canon.
17. Subword structure is future-proofing: Kalvin needs it to reconstruct words
    for novel queries. All subword entries become grounded.

## Divergences and Stalls

### Stall (no held kline for a request)

18. A request for which the Trainer holds no kline is a proposal it cannot
    auto-ratify; it is escalated per `@specs/supervisor-decision.md`. There is no
    distinct event or escalation reason. (Supervisors are not invoked in the
    bootstrap run.)

### Divergence (K mismatch)

19. The only Kalvin divergence the Trainer recognises is structural: K emitted a
    kline that does not match the table's K at the cursor (a stub/table view
    mismatch, or — for real Kalvin — a kline the table did not prescribe). This
    is caught by the K-side of two-sided validation (§Training Loop) and fails
    fast with side attribution. There is no band-comparison divergence:
    "over-reach" and "under-reach" as reported-vs-declared-band concepts do not
    apply, because Kalvin emits no grounding event to compare.
20. The stub cannot diverge structurally either: it emits its own K-rows from its
    own cursor, so by construction it emits exactly its view of the table (§Out
    of Scope of the stub spec). §Divergence is exercised only by real Kalvin.

### "Kalvin never derives a withheld kline"

21. Deferred to a later grill. Under the stub, the table prescribes Kalvin's
    responses exactly, so an un-met request implies the table is wrong, not a
    Kalvin failure.

## Out of Scope

- How Kalvin produces its requests and grounds (cogitation, expand, misfit) —
  the Kalvin grill, validated against the same dialogue table.
- The global event-kind change (`ground`/`frame` → significance). This spec keys
  on `proposal.significance`, forward-compatible with that change.
- Reactive-scaffolding generation and the supervisor decision contract —
  `@specs/supervisor-decision.md`.
- Goal-based curriculum generation.
- Multi-primary scripts (a script with more than one top-level relationship) and
  whether the Trainer can open at S3 — deferred to a later grill.
- The Trainer signalling intentionality by submitting at any band S1–S4 —
  deferred to a later grill (the bootstrap run uses the node-terminality rule).

## Canonical Example

The reference dialogue is the "Mary had a little lamb" exchange
(`scripts/dialogue-mhall.json`): a single depth-first cascade opened by the
primary `{MHALL:[SVO]}` at S2, driven by K's S4 requests, with the role-binding
relations (Mary↔subject, had↔verb, ALL↔object) K-discovered as S3 proposals and
T-ratified as S1, closing on K's S1 countersign of the primary. K emits only
requests, proposals, and the closing countersign — it emits no grounding events.
The run terminates on dual cursor exhaustion; the deterministic driver is
`@specs/stub-kagent.md`. (Kalvin's learning — entries entering LTM under S1
ratification — is monotonic memory, not a Trainer-verified gate.)

## Test Matrix

| ID  | Criterion | Origin ref |
|-----|-----------|------------|
| DDT-1 | A dialogue table has `script` + ordered `turns`; each turn has actor, op, signature, nodes, significance, notes | §Dialogue Table |
| DDT-2 | The decoder pre-decodes every turn into `(actor, op, KValue)` at configuration time | §Decoder |
| DDT-3 | Decoding is single-stage; it resolves the kline from `script`, attaches significance by lookup, passes through actor/op, ignores notes | §Decoder |
| DDT-4 | Canons are retrieved by node-list match against compiled canons | §Decoder |
| DDT-5 | The Trainer is stateless: its response depends only on the incoming turn and the compiled script | §The Trainer is Stateless |
| DDT-6 | Significance is symmetric and turn-based (S2/S3 proposal, S4 request, S1 terminal) | §Significance Contract |
| DDT-7 | The Trainer opens with exactly one turn: the primary `==` half at S2; then is purely reactive | §Opening |
| DDT-8 | On a K S4 request for signature X, the Trainer supplies the held kline(s) for X in priority order (canon → relation → identity) | §Partition and the Supply Rule |
| DDT-9 | Canon-first full shadowing: a relation whose signature carries a canon is never T-supplied | §Partition and the Supply Rule |
| DDT-10 | A shadowed relation is K-discovered (S3) then T-ratified (S1) | §Partition and the Supply Rule |
| DDT-11 | A relation whose signature has no canon is supplied on request (not shadowed) | §Partition and the Supply Rule |
| DDT-12 | The Trainer attaches S1 to supplied entries with terminal nodes, S2 to those with non-terminal nodes | §Significance the Trainer Attaches |
| DDT-13 | On a K S3 proposal, the Trainer ratifies with the reciprocal at S1 | §Ratification |
| DDT-14 | On a K S1 terminal, the Trainer takes no action; the run advances | §Driving the dialogue |
| DDT-15 | (retired) The Trainer never verifies Kalvin's learning; its only K-verification is structural (K == table K at the cursor) | §Termination and K-verification |
| DDT-16 | (retired) Kalvin learns continuously and grounds under S1 by updating LTM; it emits no grounding event | §Termination and K-verification |
| DDT-17 | (retired) No `learned ⊇ submitted` computation; the run terminates on dual cursor exhaustion | §Termination and K-verification |
| DDT-18 | The run ends on dual cursor exhaustion (trainer AND stub); the canonical table's final K is the closing S1, verified not trusted | §Training Loop, §Termination and K-verification |
| DDT-19 | Subword canons are not filtered; they are withheld, ratifiable, supplied at the node-terminality band | §Subword Canons |
| DDT-20 | A request with no held kline is escalated per supervisor-decision (no distinct event) | §Stall |
| DDT-21 | (retired) No band-comparison divergence (over/under-reach); the only K divergence is structural (K != table K), caught by two-sided validation | §Divergence |
| DDT-22 | The loop is dispatch-driven: T computed by the supply function, table validates; not a replayer | §Training Loop |
| DDT-23 | Both actors are self-cursored (trainer over T-rows, stub over K-rows); no kline-matching between them | §Training Loop |
| DDT-24 | Dispatch is greedy per actor: a run of consecutive same-actor rows is consumed whole; 1:1 is not a constraint | §Training Loop |
| DDT-25 | Each turn is validated two-sided (Model A): K==table then T==table, attributing failures to the right side | §Training Loop |
| DDT-26 | Termination requires both cursors empty; a truncated table (final K-run never emitted) fails with a non-empty stub cursor | §Training Loop |
| DDT-27 | The closing S1 is verified (table's final K-run consumed + matched), not semantically detected (stateless) | §Training Loop |
| DDT-28 | Annotation-only turns are dropped at decode time (not submittable) | §Configuration |
