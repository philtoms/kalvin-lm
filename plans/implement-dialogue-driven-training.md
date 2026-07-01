# Implement Dialogue-Driven Training — Plan

## Spec References

- `@specs/dialogue-driven-training.md` — WHAT this plan implements (all DDT-\* criteria).
- `@specs/kscript.md`, `@specs/kline.md`, `@specs/kvalue.md`, `@specs/agent.md` — compile, KLine equality, KValue, `RationaliseEvent`.

This plan supersedes `@plans/implement-rationalise-trainer-significance.md` for
dialogue-table lessons. It does not delete that plan; both coexist until the
dialogue-driven model is chosen as the path forward.

## Implementation Tasks

### Phase 1 — Decoder (configuration time)

**1.1 Dialogue-table loader.** Parse `script` + `turns[]` into a typed
`DialogueTable`/`Turn`. Annotation-only turns (notes, no `op`) are retained on
`Turn` and dropped at decode. → DDT-1.

**1.2 Script resolution.** Compile `script` once (`ks.compiler.compile_source`).
Build the canon-retrieval index: node-decoded-label tuple → compiled canon
KValue; plus a label → KLine index for atom/relation resolution. → DDT-4, DDT-5.

**1.3 Single-stage decode.** `decode(table) -> list[DecodedTurn]`. Per turn:
CANONIZED → canon by node-list match; IDENTITY → atom by label; constructed
relation → rebuild from node canonical signatures; attach significance by band
lookup; pass through role/op; drop annotation-only turns; ignore notes.
→ DDT-3, DDT-5.

### Phase 2 — The runner

**2.1 Actor interface & event role.** Add an optional `role: str | None` field
to `RationaliseEvent` (default `None`; existing non-dialogue emitters are
untouched). A `respond(incoming: RationaliseEvent | None) -> (int,
RationaliseEvent) | None` protocol. Each actor holds its own cursor (starting
before its first row); `respond` advances it by one and returns
`(next_cursor, event)` with the actor's role on the event, or `None` when
exhausted. → DDT-9, DDT-16, DDT-17.

**2.2 Table-reading actors.** `TableTrainer` and `TableTrainee`, each holding
its own cursor into its own rows (the decoded table filtered to its role).
`respond` advances by one and returns `(next_cursor, event)` (`proposal` = the
row's KValue, `query` = the incoming event's proposal, `role` = the actor's
key); `None` when exhausted. Neither inspects the incoming event. The actor is
dumb: it does not know about alternation or run boundaries. → DDT-7, DDT-8,
DDT-9, DDT-10.

**2.3 The run loop.** `run(decoded, trainer, trainee) -> RunResult`. The runner
owns the table cursor: each step reads `decoded[cursor].role`, asks that actor
for its next row, and advances. Greediness is automatic — consecutive
same-actor rows go to the same actor. Ends when the cursor passes the table
end. → DDT-6, DDT-7, DDT-13, DDT-14.

**2.4 Validation (router).** After each response, validate by the **role the
actor declared on its event**: look up the decoded rows for that role and check
the emitted `proposal` equals the row at the cursor the actor returned. A role
with no table rows or a kline/significance mismatch raises
`ActorDivergence(role, cursor, expected, emitted)`. Keying on the event's
self-declared role (not the table key) is what lets a real, possibly
asynchronous actor announce itself. → DDT-11, DDT-17.

### Phase 3 — Driver

**3.1 `scripts/dialogue_run.py`.** Load a table, decode it, construct the
default actor pair (`default_actors`), call `run`, print a PASS/FAIL summary
(`--verbose` traces the exchange). Exit 0 on completion, 1 on divergence.
No bus, no adapter.

## File Structure

- `src/kalvin/events.py` — `RationaliseEvent` gains an optional `role: str | None` field (Phase 2).
- `src/training/dialogue/decoder.py` — table structures + `decode` + `load_table` (Phase 1).
- `src/training/dialogue/runner.py` — `Actor` protocol, `_TableActor`, `TableTrainer`, `TableTrainee`, `default_actors`, `run`, `RunResult`, `ActorDivergence` (Phase 2).
- `scripts/dialogue_run.py` — driver (Phase 3).
- Deleted: `src/training/dialogue/supply.py`, `loop.py`, `stub_kagent.py` (the supply rule / held index / two-sided Model A / dual-exhaustion / `_KAgentLike` stub they encoded are gone).
- `scripts/dialogue-mhall.json` — the canonical example dialogue.

## Test Mapping

| Spec ID | Test                                                                                                                   |
| ------- | ---------------------------------------------------------------------------------------------------------------------- |
| DDT-1   | loader parses table/turn fields                                                                                        |
| DDT-2   | loader rejects an unknown `op`                                                                                         |
| DDT-3   | `decode` returns a flat ordered `list[DecodedTurn]`                                                                    |
| DDT-4   | decode resolves kline from script + significance lookup + role/op pass-through; annotation-only dropped; notes ignored |
| DDT-5   | CANONIZED retrieved by node-list match                                                                                 |
| DDT-6   | runner owns the table cursor; reads whose row is next and asks that actor; first row is the trainer's                  |
| DDT-7   | greedy: consecutive same-actor rows go to the same actor (e.g. `T,T,K` → trainer twice, then trainee)                  |
| DDT-8   | trainer and trainee are symmetric cursor readers of the same table                                                     |
| DDT-9   | actor yields one event per `respond`, returning `(next_cursor, event)`; `proposal`/`query`/`role` wired correctly      |
| DDT-10  | default actors do not inspect the incoming event                                                                       |
| DDT-11  | actor divergence raises `ActorDivergence` keyed on the event's declared role, naming role/cursor/expected/emitted      |
| DDT-12  | (removed) both actors are now validated (DDT-11)                                                                       |
| DDT-13  | run ends when the cursor passes the end of the table                                                                   |
| DDT-14  | runner has no harness message-bus dependency                                                                           |
| DDT-15  | no learned/grounding notion or signal in the runner                                                                    |
| DDT-16  | actor cursor starts before its first row; `respond` advances by one and returns `(next_cursor, event)` or `None`       |
| DDT-17  | runner routes/validates on the event's self-declared `role` (router shape for async real actors)                       |

Canonical end-to-end test: the full "Mary had a little lamb" dialogue
(`scripts/dialogue-mhall.json`) runs to exhaustion through the default actor
pair, every run validated.

## Design Decisions

**D1 — Both actors emit `RationaliseEvent`s.** The exchange shape is fixed: each
turn is a `RationaliseEvent(kind, query, proposal)`. Significance is carried on
the `proposal` KValue. This is the shape a real trainer and a real trainee will
produce, so the default actors produce it too — replacement is a like-for-like
swap. _Rejected:_ a bare-KValue exchange (loses the query/proposal pairing real
actors rely on).

**D2 — Both actors are validated against the table.** The runner checks every
emitted response — trainer and trainee — against the authored table. A real
trainer is expected to synthesise its training responses, and a real trainee
its responses; both are validated the same way. The table-reading doubles cannot
diverge (they read the table the runner validates against); the checks exist for
the real actors. _Rejected:_ trainee-side-only validation (left the trainer
unchecked, and a synthesising real trainer is exactly the case that most needs
checking).

**D3 — Bus-agnostic runner.** The runner is a plain Python loop. Bus integration
arrives with the real actors. _Rejected:_ routing the default actors through the
real `KAgentAdapter` + `MessageBus` (couples a simple loop to harness
complexity, and misrepresents the default actors as `_KAgentLike` peers).

**D4 — The runner owns the table cursor; greediness is the runner's.** The
runner reads `decoded[cursor].role` and asks that actor for its next row; while
consecutive rows share an actor, it asks the same actor again. The actor itself
is dumb: it holds its own cursor into its own rows and yields one row per
`respond`, knowing nothing about alternation or run boundaries. For a strictly
alternating table each actor is asked one row at a time; a `T,T,K` table asks
the trainer twice, then the trainee. _Rejected:_ the actor emitting a whole run
per `respond` (couples the actor to run-boundary detection and forces the two
actors to share a position — a synchronisation problem); per-actor sub-tables
with independent cursors over the shared sequence (same problem).

**D5 — No supply rule.** The trainer's "correct next response" is the table's
trainer rows at the cursor. No held index, no canon-first shadowing, no
node-terminality significance, no ratification dispatch, no opening entry-point.
All of that was the trainer's internal cleverness, belonging to a real trainer,
not to the loop. _Rejected:_ a stateless supply function computing trainer turns
(ENTANGLES the loop with script structure and invents a shadowing model the
table already prescribes).

**D6 — The actor's role rides on the event; the runner is a router.**
`RationaliseEvent` gains an optional `role` field; dialogue actors self-declare
it (`"T"`/`"K"`), and the runner validates by that role rather than by the
table key. This makes the event self-describing — the shape a real, possibly
asynchronous actor uses to announce itself — without forcing the 30-odd
non-dialogue emitters (cogitator, agent, supervisor, …) to supply a role.
_Rejected:_ a required `role` field (breaks every emitter outside the dialogue);
a separate dialogue-event type (duplicates `RationaliseEvent` and splits the
exchange shape the real actors must converge on).

**Deferred.** Real trainer / real trainee; bus integration; supervisor
escalation; multi-cascade and multi-primary lessons.
