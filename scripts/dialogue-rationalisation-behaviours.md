# S2-Misfit Rationalisation Behaviours

This document specifies how a participant (Kalvin, K; symmetrically the Trainer, T)
**originates misfit proposals** during rationalisation. It is referenced by
`@specs/dialogue-driven-training.md` (§Decoder, §Out of Scope) as the home of the
S2-misfit pedagogy and is the output of the rationalisation grill (2026-07-07).

It is pre-cascade: it establishes the model a future spec and plan will be written
from. Terms are defined in `@CONTEXT.md` (**Proposal**, **Misfit (proposal)**,
**Canon**, **Ratify**); this document owns the _mechanism_ and the _boundaries_ on
the act, not the glossary.

The reference dialogue is `scripts/dialogue-wdmh.json` ("What Did Mary Have") — a
two-script dialogue whose second script encodes a question as a deliberate misfit
(`WDMH` with nodes that do not OR-reduce to `WDMH`).

## 1. The problem

A **Canon** can self-close at S1 during rationalisation: its signature is the
OR-reduction of its nodes, so when its nodes ground, the kline is structurally
self-grounded and K re-derives it as S1. A **Misfit** (`@CONTEXT.md`) cannot — its
signature carries information its nodes do not, so no amount of node-grounding
yields a structural warrant for S1. The only path to S1 for a misfit is
**ratification**: another participant countersigns it.

K nonetheless always strives to S1. When the level-1 entry K is working is a
misfit, K cannot self-close; instead K must **originate a misfit proposal** — a
candidate whose signature it knows does not reduce to its nodes — and offer it for
ratification. This document bounds that act.

## 2. Scope

- **In scope:** originated misfits — klines a participant _constructs and emits_
  whose own signature does not reduce to its own nodes. The grill establishes
  Kalvin's behaviour in full.
- **Out of scope (this document):** the Trainer originating misfits. T's generation
  follows the **same boundaries** (the principle is symmetric — see §9); only the
  ratifier differs. T's real-actor cogitation is deferred to a later grill.
- **Out of scope (other documents):** observed misfits (a received kline classified
  via `classify_misfit` — `@specs/cogitator.md` §S2 Expansion); supervisor escalation
  when no candidate admits (`@specs/supervisor-decision.md`).

## 3. Boundaries

These are the rules that govern when and how K may originate a misfit proposal.

### 3a. Routing — cogitation dispatches on significance

Cogitation routes work-list entries on **significance**, not on a numeric
"level." Rationalisation performs the first split: **S1** grounds/cleans and
**S4** pops the identity ask in the entry rule (never reaching cogitation);
**S2 / S3** reach cogitation. Cogitation then performs the second split by
structure-as-significance:

- **S3 path (operand pairing).** A single-node relationship `{L:[R]}`.
  When its operands L and R have seen canons, K relates the two canons by
  pairing their operands left-to-right (`Mary↔Subject`, `had↔Verb`), emitting
  each unresolved pair as a proposal. This is the existing `_emit_s3_pairing` /
  `_pair_resolved` machinery — the S3 path. A pair is resolved when ratified
  (the trainer countersigned it). The path self-closes at S1 when all pairs
  resolve. A single-node relationship whose operand canons are not yet seen is
  S3-_structure_ but not _workable_ — cogitation skips it (it awaits
  elevation/cleanup) and does **not** route it to S2.
- **S2 path (misfit origination).** A **multi-node** misfit entry whose
  signature does not reduce to its nodes (`{WDMH:[Mary,had,what]}`). K cannot
  pair operands (there is no second canon to pair against); it must _originate
  substitutions_ — the accumulation mechanism in §4. Single-node relationships
  are never S2 (they are S3-structure); only multi-node misfits route here.

The S2 and S3 bands **overlap** (the boundary is `S2|S3`, not a clean split).
A 1:1 structure that is typically S3 may stall — the trainer did not ratify the
pair — and behave misfit-like. The S3 path's `_pair_resolved` already encodes
this roughly ("trainer did not immediately ratify"). A stalled S3 entry may
**migrate** to the S2 path; the migration is in scope for this implementation
(§10) and is specified once both paths exist and real stall behaviour can be
observed against the WDMH golden master.

**B1 — Trigger (S2 routing).** K may originate a misfit proposal only when the
entry it is working is an **S2 (multi-node misfit) structure** — routed to the
S2 origination path, not the S3 pairing path. The routing is structural: a
non-identity, non-canon entry with two or more nodes is an S2 structure.
Honest entries (identities, canons), single-node relationships (S3-structure),
and S3-pairable entries never originate misfits; they take their own paths.
The licence is **permissive** — K does honest work (identity asks, canon
grounding) on an S2 entry before, or instead of, originating a misfit reply.
(B1 was originally framed as "the canon self-close path is blocked";
investigation showed the trigger is _eligibility/routing_, not the close — a
misfit entry never reaches the S3 close path because it never enters the S3
path.)

**B2 — No invention.** Every **substituted** node in an originated misfit proposal
must be a node of a kline K has grounded. The entry's own nodes (the substrate
`target` starts from) are received, not substituted; only what rule 1
(expansion) and rule 2 (graft) introduce counts, and both draw exclusively from
grounded klines. K recombines grounded klines; it never fabricates a node value
it has not grounded into a substitution. This mirrors the cogitator's Universal
Constraint (`@specs/cogitator.md` §S2 Expansion): every signature/node
generated must already exist in the model. B2 is satisfied by construction — no
guard is needed beyond the rules sourcing from grounded klines.

**B3 — Candidate admission (shared nodes).** A grounded kline `C` is a candidate
for entry `E` iff `C` shares at least one **node value** with `E.nodes`:
`node_overlap(C.nodes, E.nodes) ≠ ∅`. Admission is keyed on the entry's _nodes_,
not its head signature — this avoids the over-admission that single-bit NLP type
words would cause under `signifies`. (Both klines having a `Mary` node is the
intended commonality.)

**B4 — Drop already-grounded proposals.** If the proposal K shapes is already
grounded (an isomorphic kline exists in K's memory), K drops it rather than
emitting it.

## 4. The generation mechanism — one proposal, accumulated shaping

K shapes a **single proposal** by processing admitted candidates in preference
order, mutating one target as each candidate fires. There is no meta-proposal set
and no selection step; the proposal is _built_, not _chosen_.

**Initialise.** `target = copy(entry.nodes)`. A node with no admitted candidate is
**open** — a slot a later match may fill.

**Process candidates in preference order:**

1. **Node-expansion** (preferred). For each grounded kline `C` whose signature
   is in `target.nodes` (`C.signature ∈ target.nodes`): replace that one node in
   `target` with `C.nodes`. The matched node is consumed; `target`'s other nodes
   persist. (Rule 1 sources its candidates by signature-in-target-nodes, a
   separate scan from B3 — a kline need not share a node value to fire rule 1.)
2. **Node-graft** (with `must_match`). For each B3-admitted candidate `C` (shared
   nodes) that did not fire rule 1:
   - Compute `must_match` from the **current accumulated** `target`: the nodes that
     prior matches have established (resolved/expanded), **excluding still-open
     nodes**. Open nodes are not required to match — they are slots to be filled.
   - Resolve `must_match` against `C` by **recursive canon-resolution to fixed
     point** (§5). If `must_match` is fully matched, graft: substitute `C`'s
     difference (`C.nodes − shared`) into the open slots. If `must_match` cannot be
     fully matched, `C` does not fire.

**Accumulation.** Each match mutates `target`, so the next candidate sees the
changed target. `must_match` reflects everything prior matches have established
(rule 1's expansions are _preserved_ — rule 2 must respect them, possibly by
resolving them back through grounded klines). The proposal is the accumulated
`target` when no more candidates fire.

## 5. Recursive canon-resolution (`must_match`)

`must_match` is the set of accumulated nodes a node-graft candidate must account
for. Direct match first; failures resolve through grounded klines:

1. **Partition** the failed nodes into maximally-coverable subsets, each matching a
   grounded kline's `nodes` exactly. Uncoverable nodes are retained
   in `must_match`.
2. **Replace** each coverable subset with its grounded kline's signature;
   `must_match`
   becomes shallower (the resolved signature replaces its constituent nodes — e.g.
   `[did, have] → had`).
3. **Re-check** the new `must_match` against `C`. Resolved signatures may now form
   _new_ coverable subsets, so **recurse** — re-partition, re-resolve, re-check —
   until either `must_match` is fully matched (graft proceeds) or a resolution pass
   produces no change (candidate rejected).

`must_match` updates cumulatively: the shallow resolved form replaces the deeper
form for any subsequent candidate. Resolution is full-depth (the combinatorial cost
of the partition search at each level is accepted for completeness); termination is
the fixed point.

The purpose of `must_match` is to **balance graft**: without it, a candidate
sharing a single node (`Mary`) could licence wholesale substitution of its entire
surplus, over-powering the prior node-expansion work. `must_match` forces the graft
to _earn_ its substitution by accounting for the accumulated structure, directly or
via canon-resolution.

## 6. Sequence and termination

- K cogitates the work-list **LIFO**. A misfit entry persists in the work-list
  (cleanup grounds canons and identities; it never removes a non-canon
  relationship, so a CANONIZED-but-misfit entry survives).
- K has **no notion of closing**. The level-1 entry is not "closed" by a successful
  proposal; K keeps cogitating. A run terminates because an emission matches the
  master's terminal row, not because K decided it was done.
- After K emits a misfit proposal and it is ratified (grounded), the next
  cogitation re-runs generation against the same entry. Already-grounded proposals
  are dropped (B4), so K advances naturally — it does not re-emit the ratified shape.

## 7. Worked example — WDMH

Entry (the misfit; the canon form T also sends resolves and clears):
`E = {WDMH: [Mary, had, what]}`. Open: `what` (no candidate yet).

Rule-1 candidates (signatures in `E.nodes`): the canon `{had: [did, have]}`
(`had ∈ E.nodes`). Sourced separately from B3 — rule 1 finds klines whose
_signature_ the entry references as a node.

B3 candidates (shared _nodes_, feeding rule 2): `{MHALL: [Mary, had, a,
little, lamb]}` (shares `Mary, had` as nodes). Note `{had: [did, have]}` is
_not_ a B3 candidate — its nodes `[did, have]` share nothing with `E.nodes`.

**Node-expansion:** `had ∈ target` → `target = [Mary, did, have, what]`.
Accumulated: `[Mary, did, have]`. Open: `what`.

**Node-graft (`MHALL`):**

- `must_match = [Mary, did, have]` (accumulated; `what` open, excluded).
- Direct: `Mary`✓, `did`✗, `have`✗. Failed `{did, have}`.
- Resolve: `{did, have}` matches grounded kline `{had: [did, have]}` → `had`.
  `must_match = [Mary, had]`.
- Re-check: `Mary`✓, `had`✓. Fully matched.
- Graft: shared `{Mary, had}`, difference `{a, little, lamb}` → substitute into
  open `what`. `target = [Mary, had, a, little, lamb]`.

No more candidates fire. **Proposal = `{WDMH: [Mary, had, a, little, lamb]}` at
S2.** T ratifies (countersigns) → S1.

Note that the node-expansion's intermediate shape `[Mary, did, have, what]` is never
emitted — it is a step in the accumulation, absorbed and resolved back into the
final shape. There is one proposal, shaped by two successive matches.

## 8. Consequence for `scripts/dialogue-wdmh.json`

The table currently encodes a **two-step** misfit sequence (rows `#47` and `#48` in
the trace), predating this mechanism. Under the locked mechanism the proposal is
shaped in a single cogitation: `#47`'s shape is an intermediate accumulation, not
an emission. **The table should be collapsed to a single misfit proposal** — the
`#48`-equivalent `WDMH(Mary, had, a, little, lamb)` at S2, ratified by T at S1 —
with the redundant `#47` row removed.

_Table edit deferred to a follow-up (awaiting go-ahead)._

## 9. Symmetry with the Trainer

The Trainer may originate misfits under the **identical** boundaries (B1–B4) and
the identical generation mechanism (§4–§5). The sole asymmetry is the **ratifier**:
K-originated misfits are ratified by T (the Trainer's countersign);
T-originated misfits are ratified by the supervisor (the third role), consistent
with the existing escalation path (`@specs/supervisor-decision.md`). Specifying and
validating T's real-actor cogitation is deferred.

## 10. Open questions

These were identified by the grill and are **not** decided here:

- **Work-list state at the terminal proposal.** Cleanup preserves the opening
  misfit but grounds every S1 kline T sends. The precise work-list state at the
  moment K must emit the proposal matching the master's terminal row needs
  verification against the new rationaliser code once it is testable.
- **`must_match` edge cases.** Multiple node-graft candidates; resolution chains
  that revisit signatures (termination is the fixed point, but pathological
  grounded sets warrant test coverage).
- **Accumulation bookkeeping.** The exact definition of "accumulated vs open" when
  several node-expansion and node-graft candidates interleave needs test
  fixtures beyond the WDMH example.
- **S3-stall → S2 migration (in scope).** When a 1:1 S3 pair is not ratified, the
  entry stalls and may migrate to the S2 path. The migration condition and the
  state transition are specified once the clean S2 and S3 paths exist and real
  stall behaviour can be observed against the WDMH golden master (§3a).
- **T-side generation.** Symmetric in principle (§9); implementation deferred.
