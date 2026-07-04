# 0001 — The peer runner is a sink, not a driver

After the trainer delivers the opening entry to the trainee, `run_peer`
receives emissions via a push `receive(event)` contract and holds coverage
bookkeeping only. It does not call into actors, decide whose turn it is, or
pace the exchange. This is a deliberate departure from the synchronous
`run`, which drives the exchange via `Actor.respond()`.

## Considered options

- **Cursor-driven with relaxed ordering** — keep `respond()` and validate each
  turn against a cursor that may advance out of order. Rejected: requires the
  runner to decide whose turn it is next, which is actor-coupling state, and
  cannot express back-to-back same-actor turns (`T T K T K K`) without that
  state.
- **Pull-via-generator** (`respond -> Iterator[event]`) — rejected: still
  paces the actors from the runner, reintroducing dispatch logic.
- **Threaded/asyncio bus** — rejected: that is the harness server's job;
  importing concurrency into the bus-agnostic runner would duplicate it.

## Consequences

The synchronous `run` and the peer `run_peer` are now structurally different
regimes sharing a table and a content-equality notion of a match, not two
variants of one loop. `PeerDivergence` and `PeerRunResult` are separate from
`ActorDivergence` and `RunResult` because the peer regime's data
(covered subset, arrival-ordered events) has no cursor to sit on. Delivering
the opening to the trainee is the caller's responsibility, since a pure sink
cannot perform the one asymmetric priming act.
