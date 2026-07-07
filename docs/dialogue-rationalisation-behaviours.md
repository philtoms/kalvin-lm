# Dialogue Rationalisation Behaviours

A catalogue of the rationalisation behaviours the dialogue actors
(`training/dialogue/`) exhibit. Each entry describes a behaviour as it is
implemented, or — where marked — as it is specified but not yet
implemented.

**Scope.** These are **dialogue actor** behaviours: the simple
table-reading and rationalising doubles under `training/dialogue/`. They
are not the production rationalisation pipeline under `kalvin/` (the
model, the cogitator, `misfit.py`), which is out of scope for this branch.
**Simplicity is the byword** — dialogue actors stay simple; the
production pipeline will be revisited separately.

Status:
- ✅ **Implemented** — the behaviour is in the code today.
- 📋 **Specified** — documented; not yet implemented.

---

## B2 — S1 emission: consistency or prior ratification

An S1 response is admissible only if **either**:

- (a) the signature **is** the canon the nodes retrieve (the
  self-consistent case), **or**
- (b) the (signature, nodes) misfit was **previously ratified** at S1 in
  the same exchange — a once-novel misfit promoted to grounded.

This is a rationalisation rule — a constraint on what a participant may
legitimately *emit* at S1. It is not a decode rule: the decoder builds any
kline the author writes. The table runner does not enforce it either; it
validates an emitted event against the decoded row (role + kline +
significance equality), trusting the author's authored significance. B2
binds the *rationaliser* when it produces a turn, not the runner that
checks one.

Case (b) is what lets an S2 misfit be ratified to S1 in a later turn
without rewriting the kline.

---

## B3 — `close` marks where the run concluded

A turn marked `close: true` has one mechanical meaning: **the run
concluded at this row.** The runner's validation at that row reports
whether the cursor landed on it and its event was emitted as authored.
That is the whole of what `close` asserts.

It carries no claim about whether the open proposal was ratified, whether
the closing turn is a countersign, whether its kline is grounded or a
misfit, or whether anything is "left for later." All of that is the
author's intent, expressed by which rows precede the close — not by the
marker. The marker is a full-stop; the rows are the author's sentences.

Both an S1 countersign close (MHALL) and an unanswered S2 misfit close
(the WDMH table's closing turn) are well-formed: the marker fires, the
runner validates the boundary row, done. Exploring how T should respond
to a K-substituted answer is an authoring act — remove the close, add a T
row with its own close, run again. No code changes.

---

## B4 — Answer substitution

📋 Specified — not yet implemented. The WDMH table
(`scripts/dialogue-wdmh.json`) closing turn is its specification.

When the rationaliser cannot progress a query by identity-asking or
operand-pairing, it may answer the query by **substituting a different
grounded kline** as the response. K proposes a whole known kline A as the
answer to a query Q, where A is selected by sharing at least one node
with Q — not by reshaping Q's nodes under Q's signature.

Answer substitution is distinct from signature-preserving expansion.
Adding, removing, or replacing nodes under a query's own signature (the
production cogitator's underfit/overfit/dual modes) tries to make the
nodes faithfully cover the query's bits. Answer substitution instead
*abandons* the query's signature and offers a different kline entirely.
A signature-preserving expander can never yield MHALL from a WDMH query,
because MHALL's atoms do not cover WDMH's bits — they answer WDMH as a
different kline.

The emitted kline is the grounded candidate **verbatim** at **S2** — no
reshaping, no signature invention. Ratification is not the behaviour's
concern: if T later ratifies it, that arrives as a normal S1 incoming and
grounds via B2(b); if T never responds, the run closes (B3) and the
substituted answer remains an open proposal.

---

## B5 — Substitution candidates are grounded klines sharing a node with the query

📋 Specified — not yet implemented.

The rationaliser does not employ distance heuristics to significance; the
only discriminator is **grounded vs not-grounded**. Candidacy for
substitution is therefore bounded to **grounded klines** that share at
least one node with the query.

A kline is a substitution candidate for query Q when it:

- shares ≥1 node with **Q's signature** (not Q's nodes — K is answering
  "what is Q?" so the query's signature is the thing being answered), and
- is **grounded** in K's memory.

K-grounding is the gate on substitution: the answers K can reach depend
entirely on what K has *grounded*, not what K has merely *seen*. This is
why the curriculum grounds the answer before posing the question — MHALL
is script 1, WDMH is script 2. Were MHALL not grounded, the WDMH query
would yield no substitution candidate and K would emit no answer.

---

## B6 — K-generated and T-generated misfits are the same kline

✅ Implemented (a consequence of the closed structural-state set; nothing
to add).

A misfit kline is structurally identical whether the Trainer authored it
as scaffolding or Kalvin synthesised it by answer substitution. The kline
`{WDMH: [Mary,had,a,little,lamb]}` is the same objective structure in
both cases. Origin is a **role-level** fact, not a kline fact:

- The five structural states (`@CONTEXT.md`) stay closed — no sixth
  state, no origin flag on the kline.
- Origin is recorded by the **role** the turn carries (`role: K` vs
  `role: T`). The table already encodes it; nothing structural is added.
- The two origins differ only in **ratification path**: a T-generated
  misfit is ratified by the next K response; a K-generated misfit awaits
  a later T ratification (or remains open — B3).

Answer substitution (B4) therefore adds a new behaviour of the dialogue
rationaliser, not a new structural state. It emits ordinary klines that
flow through the same countersignature machinery as any other proposal.

---

## B7 — Answer-substitution placement: a Level-2 fallback in `Rationaliser._cogitate`

📋 Specified — not yet implemented.

Answer substitution is a new **Level-2 fallback branch** in
`Rationaliser._cogitate` (`src/training/dialogue/rationalise.py`), firing
only after Levels 0–1 cannot progress the top work-list entry:

- **Level 0** — identity asks (S4).
- **Level 1** — operand-pair relationship emission (S3/S2) and S1 close.
- **Level 2 (answer substitution)** — when no entry is workable by
  Levels 0–1, scan `self._state.grounded` for a kline sharing a node with
  the stalled query's signature (B5) and emit it verbatim at S2 (B4).

`grounded` is the only memory scanned, so the grounded-only constraint
(B5) holds for free — there is no other memory to consult. One KValue is
emitted per call (ordered regime); the table's `close` selects which
candidate is the expected terminal (B3).

This placement preserves Levels 0–1 and the MHALL golden-master sequence
untouched — MHALL never substitutes; only a query that exhausts Levels
0–1 falls through to Level 2. The trigger is mechanical and local:
`_cogitate` already scans the work-list and `_find_canon_nodes` already
scans `grounded`; answer substitution reuses both and adds no new state.
