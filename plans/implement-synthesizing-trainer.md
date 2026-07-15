# Implement the Synthesizing Trainer — Plan

> **Status: implemented.** Kept as a build record, not an active spec.

## What was built

`src/training/dialogue/synthesize.py` — the pure `synthesize(compiled,
incoming, signifier) -> KValue` function, plus the `SynthesizingTrainer` actor
(`actors.py`) that wraps it. A drop-in for `TableTrainer`: it derives each turn
from the compiled script and the trainee's last KValue, never reading the
dialogue table. The table remains the validation oracle the runner checks
against.

Three rules, verified against the MHALL golden master:

- **R1 — Opening** (`incoming is None`): emit the current primary at S2.
- **R2 — Reply to an identity** (`nodes == []`): emit the first decomposition
  by op-precedence, S2/S1/S4 by whether nodes are themselves decomposable.
- **R3 — Echo a matching compiled kline**: emit it verbatim (S1 for a relation,
  S2 for a canon; S4 on no match).

The synthesizer is `dbg`-free — every decision derives from `signature` +
`nodes` + `signifier.make_signature`. Multi-script opening is the trainer's
responsibility (it holds ordered `primaries` from `primaries_from_source` and
advances on each open).

## Spec References

- `@specs/dialogue-driven-training.md` — the actor contract this satisfies.

## Design decisions (rationale, not contract)

- The synthesizer is `dbg`-free so it can drop into a production trainer.
- Canon vs. relation is detected structurally (`signature ==
  make_signature(nodes)`); no `dbg` reads.
- One level per call; the dialogue enacts recursion (an emitted compound node
  is re-asked on a later turn).
