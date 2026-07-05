# 0002 — The peer runner is a MessageBus subscriber, not a standalone sink

**Status:** accepted. **Supersedes** ADR-0001.

After the trainer delivers the opening entry, the peer run is driven by the
harness `MessageBus` (`src/training/harness/bus.py`). The bus *is* the sink and
the relay; the peer runner is a **coverage-tracking wildcard subscriber** plus a
thin driver that seeds the opening and runs the bus until the closing is seen.
Actors reply by `bus.send(Message(role=<other>, ...))` from any thread, which
delivers true non-blocking behaviour for free — no `asyncio`, no second
concurrency model. The peer runner depends on `training.harness`.

## Why this supersedes ADR-0001

ADR-0001 made the peer runner a standalone, bus-agnostic sink that "imports no
concurrency primitive" and rejected the threaded bus as "the harness server's
job." Two later decisions overturned that:

1. **The Actor contract became fire-and-forget `accept(event, sink) -> None`
   with zero-or-many replies, and `accept` must be non-blocking** (true async).
   A standalone sink has no concurrency primitive to make `accept` non-blocking;
   it would have to introduce one (asyncio, or a local thread/queue duplicating
   the bus).
2. **The peer runner belongs next to the harness** — it is a training
   application. Reusing the existing thread-safe, role-routed `MessageBus` as the
   sink+relay is both the non-blocking mechanism *and* the correct locality. The
   bus already does exactly what the peer relay needs (role-based dispatch,
   thread-safe `send`, single-dispatch `run()` loop), and the existing
   `KAgent` adapter already replies from a cogitation thread via the bus.

## Considered options (for the non-blocking mechanism)

- **`asyncio` throughout `peer_runner.py`** — `accept`/`receive` become
  coroutines, an event loop drives the relay. Rejected: introduces a *second*
  concurrency model into a codebase that already uses threads (the bus),
  infects the whole module and its callers with `async`/`await`, and
  duplicates what the bus already does.
- **Threads + a local queue in the runner** — rejected: reimplements the bus.
- **Reuse `MessageBus`** — accepted: the bus is already the sink+relay; the
  runner shrinks to a coverage subscriber. Non-blocking comes from the bus's
  existing thread-safety (actors reply from cogitation threads), with no new
  concurrency primitive.

## Consequences

- The peer runner depends on `training.harness.bus` and `Message`. It is no
  longer bus-agnostic; it is a harness component, consistent with its nature as
  a training application.
- The synchronous ordered `run` (`runner.py`) stays bus-agnostic and
  synchronous. The two regimes now differ not only in control (loop vs. bus) but
  in dependency: `runner.py` is standalone; `peer_runner.py` is harness-bound.
  This is the natural seam.
- The idle timeout (Q7) is the bus's `queue.get(timeout=...)` — no custom timer.
- The relay rule ("route to the non-emitting actor", Q4) is expressed as bus
  addressing: each actor addresses its reply to the *other* role; the bus
  delivers; the runner-as-wildcard-subscriber does coverage bookkeeping and
  never reroutes.
- `PeerDivergence` / `PeerRunResult` remain peer-specific types (covered subset,
  arrival-ordered events) distinct from the synchronous `ActorDivergence` /
  `RunResult`.
