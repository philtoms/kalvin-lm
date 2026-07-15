# Rationalise Trainer Significance (S4-Drop MVP) — Plan

> **Status: superseded.** This MVP was superseded by
> `@plans/implement-dialogue-driven-training.md` for dialogue-table lessons.
> Kept as a historical record only.

## What this was

The first piece of the two-way significance dialog: Kalvin consuming the
trainer's declared significance during rationalisation. It added a significance
gate to `KAgent.rationalise` (drop on declared-S4; pass otherwise) and a
`"rationalise"` adapter action, with recurrence on declared-S4 routed to the
reactor's escalation safety net.

## Spec References

- `@specs/agent.md` §Rationalisation (the significance gate).
- Superseded by `@plans/implement-dialogue-driven-training.md`.
