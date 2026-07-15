# Implement Dialogue-Driven Training — Plan

> **Status: implemented.** Kept as a build record, not an active spec. The
> authoritative current contract is `@specs/dialogue-driven-training.md`.

## What was built

The dialogue runner (`src/training/dialogue/`): a dialogue-table decoder
(configuration-time) plus a coverage-tracking runner that drives two actors
over the harness `MessageBus`.

- **Decoder** (`decoder.py`) — `decode(table) -> list[DecodedTurn]`: resolves
  each turn's kline from `script`, attaches significance by band lookup, drops
  annotation-only turns. `load_table` parses the JSON table (with `priors`
  composition and single-close collapse).
- **Actors** (`actors.py`) — `TableTrainer` / `TableTrainee` (table-reading
  defaults) plus the real `SynthesizingTrainer` and `RationalisingTrainee`.
- **Runner** (`runner.py`) — a wildcard subscriber over `MessageBus` that
  tracks a per-key coverage budget, intercepts PASS before matching, and
  terminates on close / coverage exhaustion / mutual PASS / divergence.
- **Driver** — `scripts/dialogue_run.py`.

## Spec References

- `@specs/dialogue-driven-training.md` — the current contract.

## Design decisions (rationale, not contract)

- Both actors are validated against the table; the table-reading defaults
  cannot diverge — the match exists for the real actors.
- The runner holds coverage bookkeeping only; whose-turn, cursors, and pacing
  live in the actors. The relay lives in the bus.
- The actor's role rides on the event; the runner routes on the self-declared
  role (the shape an async real actor will use).
- A PASS satisfies `burst >= 1` when an actor has nothing substantive; mutual
  PASS is terminal.

This plan superseded `@plans/implement-rationalise-trainer-significance.md`
for dialogue-table lessons.
