# Dialogue Cogitation — Specification

## Overview

Cogitation is an agents's act of working a work-list entry toward S1. Each
`rationalise` call applies the entry rule to the incoming query as bookkeeping,
then emits a **batch** of values from cogitation. Cogitation dispatches a
workable entry on **structure-as-significance** into one of two paths:

- the **S3 countersignature path** — a single-node relationship `{L:[R]}` whose
  operands both have seen canons: K pairs the two canons' operands into proposals
  for ratification, then establishes the S1 countersignature (both directions of
  the reciprocal pair); and
- the **S2 misfit-origination path** — a multi-node misfit: K originates a single
  proposal by recombining grounded klines and offers it for ratification.

Both paths strive toward S1 by **ratification** — another participant
countersigns what K proposes. The mechanism (algorithm, accumulation, candidate
resolution) is HOW and lives in `@plans/implement-rationalising-trainee.md`
(§The Rationaliser Mechanism); this spec owns the contracts and boundaries on
the acts.

## Dependencies

- `@CONTEXT.md` — **Proposal**, **Misfit (proposal)**, **Canon**, **Ratify**,
  **Structural State** (COUNTERSIGNED is bidirectional)
- `@specs/dialogue-driven-training.md` — the actor contract this cogitation
  satisfies (the trainee side); the dialogue table and runner
- `@specs/cogitator.md` — the real async `expand()` / `propose_expansions()` slow
  path that dialogue cogitation deliberately simplifies (synchronous, inline)
- `@specs/supervisor-decision.md` — the ratifier for Trainer-originated misfits
  (the sole asymmetry between K-side and T-side cogitation; §Symmetry)
- `@plans/implement-rationalising-trainee.md` — the mechanism: significance
  routing, pairing, accumulated shaping, `must_match` resolution

## Definitions

### Countersignable entry

A single-node relationship `{L:[R]}` whose operands L and R each have a **seen
canon** — a canon under L (resp. R) in grounded memory or the work-list, so its
operands are readable. A single-node relationship whose operand canons are not
yet seen is S3-_structure_ but not _countersignable_; it awaits the entry rule's
elevation/cleanup and is skipped by cogitation.

### Misfit-origination entry

A **multi-node** misfit — a non-identity, non-canon entry with two or more nodes
whose signature does not OR-reduce to its nodes. K cannot pair its operands
(there is no second canon to pair against); it must _originate substitutions_
onto the entry's own nodes.

### S3 countersignature path

The cogitation path for a countersignable entry: relate the two canons at S1 by
pairing their operands, then emit the reciprocal pair.

### S2 misfit-origination path

The cogitation path for a misfit-origination entry: shape a single proposal by
recombining grounded klines and emit it at S2 for ratification.

## Behavioural Rules

### Routing

**COG-1.** The entry rule handles **S1** (ground/cleanup) and **S4** (pop the
matching identity ask) before cogitation; only **S2/S3** entries reach
cogitation.

**COG-2.** Cogitation routes a workable non-identity entry on structure:
a countersignable entry takes the S3 path; a misfit-origination entry takes the
S2 path. A single-node relationship whose operand canons are not yet seen is
skipped (not countersignable, not S2); it does **not** route to the S2 path.

**COG-3.** The S2 and S3 bands overlap (the boundary is `S2|S3`, not a clean
split): a 1:1 structure that is typically S3 may stall — the trainer did not
ratify a pair — and behave misfit-like. S3-stall → S2 migration is out of scope
(§Out of Scope).

### S3 countersignature path

**COG-4.** K pairs the operands of the two canons left-to-right at group size 1,
grouping one side's residual into a single synthesised operand when the other
reaches a single node. K emits **every unresolved pairing in one batch**: a 1:1
pair `{lhs:[rhs]}` is CONNOTED at S3; a grouped residual is emitted as a
canonical request `{make_signature(residual): residual}` at S2 (K cannot assert
a relationship to a signature it invented — it must first confirm it).

**COG-5.** A pairing is **resolved** when a 1:1 pair is ratified (its kline
grounded) or a grouped pair's synthesised canon is grounded.

**COG-6.** When every pairing is resolved, K establishes the **S1
countersignature**: it removes the entry from the work-list, grounds and emits
**both directions of the reciprocal pair** — the entry `{L:[R]}` and its
reciprocal `{R:[L]}` at S1. A COUNTERSIGNED state is bidirectional
(`@CONTEXT.md`, Structural State); emitting only one direction is a
half-countersignature. Grounding the reciprocal is what lets `is_countersigned`
re-recognise the pair at S1 on retrieval.

### S2 misfit-origination boundaries

**COG-7 (Trigger).** K may originate a misfit proposal only when the entry it is
working is a misfit-origination entry (COG-2). Honest entries (identities,
canons), single-node relationships (S3-structure), and countersignable entries
never originate misfits; they take their own paths. The licence is permissive —
K does honest work (identity asks, canon grounding) on an S2 entry before, or
instead of, originating a misfit reply.

**COG-8 (No invention).** Every **substituted** node in an originated misfit
proposal must be a node of a kline K has grounded. The entry's own nodes are
received, not substituted; only what the shaping rules introduce counts, and
they draw exclusively from grounded klines. K recombines grounded klines; it
never fabricates a node value it has not grounded into a substitution. (Mirrors
the cogitator's Universal Constraint, `@specs/cogitator.md` §S2 Expansion.)
COG-8 is satisfied by construction — no guard is needed beyond the rules
sourcing from grounded klines.

**COG-9 (Candidate admission).** A grounded kline `C` is a candidate for entry
`E` iff `C` shares at least one **node value** with `E.nodes`
(`node_overlap(C.nodes, E.nodes) ≠ ∅`). Admission is keyed on the entry's
_nodes_, not its head signature — this avoids the over-admission that single-bit
NLP type words would cause under `signifies`.

**COG-10 (Drop already-grounded proposals).** If the proposal K shapes is
already grounded (an isomorphic kline exists in K's memory), K drops it rather
than emitting it.

### Termination

**COG-11.** K cogitates the work-list **LIFO**. A misfit entry persists in the
work-list (cleanup grounds canons and identities; it never removes a non-canon
relationship).

**COG-12.** K has **no notion of closing** an S2 entry. The entry is not
"closed" by a successful proposal; K keeps cogitating. A run terminates because
an emission matches the master's terminal row, not because K decided it was done.

**COG-13.** After K emits a misfit proposal and it is ratified (grounded), the
next cogitation re-runs shaping against the same entry. Already-grounded
proposals are dropped (COG-10), so K advances naturally — it does not re-emit the
ratified shape.

## Test Matrix

Tests live in `tests/test_rationalise.py`; the MHALL golden master
(`scripts/dialogue-mhall.json`) is exercised end-to-end by the runner
integration test in `tests/test_dialogue_runner.py`.

| ID     | Criterion                                                                                                                                                                    | Origin ref     |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| COG-1  | S1 grounds/cleans and S4 pops the identity ask in the entry rule; neither reaches cogitation                                                                                 | §Routing       |
| COG-2  | A countersignable entry routes to the S3 path; a multi-node misfit routes to the S2 path; a single-node relationship with unseen operand canons is skipped, not routed to S2 | §Routing       |
| COG-4  | S3 path emits every unresolved pairing in one batch: 1:1 CONNOTED at S3, grouped residual as a canonical request at S2                                                       | §S3 path       |
| COG-6  | S3 close grounds and emits both directions of the reciprocal pair at S1                                                                                                      | §S3 path       |
| COG-7  | Only a misfit-origination entry originates a misfit; S3-structure and countersignable entries do not                                                                         | §S2 boundaries |
| COG-8  | Every substituted node in an originated proposal is a node of a grounded kline                                                                                               | §S2 boundaries |
| COG-9  | Candidate admission requires a shared node value (not signature); identities never admit                                                                                     | §S2 boundaries |
| COG-10 | A shaped proposal isomorphic to a grounded kline is dropped, not emitted                                                                                                     | §S2 boundaries |
| COG-13 | After ratification the next cogitation advances (the ratified shape is not re-emitted)                                                                                       | §Termination   |

## Out of Scope

- **Trainer-side cogitation.** The Trainer may originate misfits under the
  identical boundaries (COG-7–COG-10) and the identical mechanism; the sole
  asymmetry is the **ratifier** — K-originated misfits are ratified by T,
  T-originated misfits by the supervisor (`@specs/supervisor-decision.md`).
  Specifying and validating T's real-actor cogitation is deferred.
- **Observed misfits.** A received kline the recipient classifies as a misfit
  but did not author (`classify_misfit`, `@specs/cogitator.md` §S2 Expansion) —
  distinct from the _originated_ misfits bounded here.
- **Supervisor escalation.** When no candidate admits, the reactive decision
  path (`@specs/supervisor-decision.md`) applies; this spec bounds the act of
  originating, not the resolution when origination fails.
- **S3-stall → S2 migration.** When a 1:1 S3 pair is not ratified the entry
  stalls and may migrate to the S2 path. The migration condition and state
  transition are deferred until real stall behaviour can be observed against the
  golden master (COG-3).
