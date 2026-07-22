# Implement the Synthesizing Trainer — Plan

> **Status: implemented.** Kept as a build record, not an active spec. It was
> last revised to track the compound-word-identity refactor (`is_canon` /
> `is_identity` / `COMPOUND_TOKEN`) and to engage the `RationalisingTrainee`
> end-to-end. See `@specs/dialogue-driven-training.md` for the active contract.

## What was built

`src/training/dialogue/synthesize.py` — the `synthesize(compiled, incoming,
signifier, grounded) -> KValue` function — plus the `SynthesizingTrainer` actor
(`actors.py`) that wraps it. A drop-in for `ScriptTrainer`: it derives each turn
from the compiled script and the trainee's last KValue. The decoded table
remains the validation oracle the runner checks against.

Three rules:

- **R1 — Opening** (`incoming is None`): emit the current primary at S2.
- **R2 — Reply to an identity** (`is_identity(proposal)`): reply by precedence:
  1. **Canon** — a real canon of the signature (`is_canon`): teach its parts.
     Significance is pedagogical — S1 when K has grounded every node, else S2.
  2. **CONNOTES** — a single-node connotation: a teachable gloss (e.g.
     `a:[Det]`), emitted at S2.
  3. **otherwise** — the signature has only a compound-word identity, only
     DENOTES relations, or nothing: ratify the identity at S1 (`{sig: []}`).

  A **DENOTES** is deliberately not supplied: it is a role-binding (e.g.
  `Mary:[Subject]`) that belongs to the S3 phase, where K proposes the binding
  and T ratifies it. Supplying it on the S4 ask would pre-empt K's proposal.
  Compound-word identities (`{Mary: [CT, M, ary]}`) are `is_identity` and are
  never supplied — their subwords are tokeniser artefacts, not pedagogical
  structure.

- **R3 — Reply to a non-identity proposal**: echo an exact compiled match
  verbatim (S1 for a relation — ratify; S2 for a canon — confirm); otherwise
  emit the proposal back at S4 (T cannot endorse it here).

The actor wraps these rules with two behaviours the pure function does not own:

- **Per-event burst reply.** The actor answers *each* event in the incoming
  burst (not only the last). A real trainee emits bursts of several asks;
  answering only the last drops the rest and stalls the exchange.
- **A `_grounded` view** of what K has ratified (signatures emitted at S1 by
  either side), fed to R2 for canon significance.
- **Scripted fallback** (see Design decisions).

> **Note on the residual pairing.** The reference dialogue is not a frozen
> oracle; the script, the code, and the rules co-evolve toward agreement with
> the author (see `@specs/dialogue-driven-training.md` §Purpose). An earlier
> reading treated a grouped S3 residual as a canonical request emitted at S2;
> the corrected rule synthesises the residual into a left-operand signature and
> emits the grouped pair as a CONNOTES at S3, identical to a 1:1 pair. The
> MHALL script's `ALL` pairing was updated to match.

## Spec References

- `@specs/dialogue-driven-training.md` — the actor contract this satisfies.

## Design decisions (rationale, not contract)

- **Structure plus op.** Derivation mixes structural predicates (`is_canon` /
  `is_identity`, via `signifier.make_signature`) with relation-op reads
  (`dbg.op`). Structure classifies a kline (canon / identity / misfit); the op
  distinguishes a teachable CONNOTES from a role-binding DENOTES within the
  same structural class. An earlier design was purely structural (`dbg`-free),
  but a single-node CONNOTES and a single-node DENOTES are structurally
  identical — only the op separates "supply this gloss" from "leave this
  role-binding for the S3 phase". The op read is the minimal, honest way to
  make that distinction.
- **Stateful view of K.** `synthesize` takes a `grounded` set (the trainer's
  view of what K has ratified), so a canon's pedagogical significance reflects
  K's actual knowledge (S1 once K has grounded all the parts, S2 while some
  remain unknown). The actor owns this view; the pure function stays a pure
  function of its arguments.
- **Scripted fallback.** A synthesizing trainer is reactive — it derives a
  reply from K's proposal. When K PASSes (no substantive proposal), the trainer
  has nothing to synthesise against, yet it may still owe a driving move (a
  close, the next script's opening) that has no structural derivation from K's
  state. In that case the decoded `table` supplies the next T proposal: the
  earliest T coverage row not yet emitted to its full multiplicity (a close
  that recurs as coverage needs as many emissions as it has authored copies).
  This is a scoped exception to script-blindness — synthesis drives every real
  exchange; the script steps in only for the trainer's driving moves. Without a
  `table` the trainer PASSes back.
- One level per call; the dialogue enacts recursion (an emitted compound node
  is re-asked on a later turn).
