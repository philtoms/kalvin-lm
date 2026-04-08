# Plan: Async Rationalisation via Pub/Sub

## Context

`rationalise()` currently returns `list[KLine]` synchronously. Fast stream matches are processed inline; slow stream matches are accumulated in `__backlog` for `cogitate()` to process later. `cogitate()` is an important design feature — it ranges over all slow model queries regardless of frame or query order. The goal is to make both rationalise (fast) and cogitate (slow) emit generated klines as they occur via a single-channel publish-subscribe pattern, with cogitate running on its own thread.

## Summary of Changes

1. New `events.py` - single-channel EventBus + RationaliseEvent
2. Rewrite `rationalise()` - returns `None`, emits fast events, accumulates slow in backlog
3. Rewrite `cogitate()` - runs on its own thread, emits slow events as it processes the backlog
4. Update `KModel` abstract - fix return type
5. Update TUI and scripts - subscribe to single event channel

## Step 1: Create `src/kalvin/events.py`

New file with:

- **`RationaliseEvent`** - simple class:
  - `kind: str` - `"fast"`, `"slow"`, `"complete"`
  - `kline: KLine` - the generated/matched kline
  - `query: KLine` - original query that triggered this

- **`EventBus`** - single-channel pub/sub:
  - `subscribe(callback)` - register a callback that receives ALL events
  - `publish(event)` - invoke all callbacks synchronously
  - Single channel: fast, slow, and complete events all delivered to the same subscription
  - Thread-safe for cross-thread publishing (cogitate runs on its own thread)

## Step 2: Add EventBus to Kalvin

In `kalvin.py`:

- Add `self._event_bus = EventBus()` to `__init__`
- Add `events` property returning `self._event_bus`
- Add `_emit(kind, kline, query)` helper for constructing and publishing events
- Start cogitate thread automatically in `__init__`

## Step 3: Rewrite `rationalise()` (kalvin.py:188-227)

Change signature from `-> list[KLine]` to `-> None`.

**New flow:**

1. Top-level call (`frame is None`): allocate frame via `get_frame()`
2. If frame already has kline: emit `"complete"` and return
3. Recursive node rationalisation (same as current)
4. `frame.query(kline)` returns fast/slow generators
5. Accumulate slow generator in `__backlog` (thread-safe, for cogitate to consume)
6. For each fast match: emit `"fast"` event with matched kline; on S1, upgrade and return
7. If no S1: add kline to frame
8. Emit `"complete"` from top-level return

Reentrant-safe: each top-level call gets its own frame. `Model.add` is idempotent.

## Step 4: Rewrite `cogitate()` - background thread

`cogitate()` is the slow-stream processor. It ranges over ALL accumulated slow queries regardless of frame or query order. It runs on its own thread, started automatically when Kalvin is initialized.

**New flow:**

1. Runs in a background daemon thread
2. Loops: waits for entries in `__backlog`, then processes them
3. For each slow match: emit `"slow"` event to the same EventBus channel
4. On S1 significance: upgrade the model kline and emit `"slow"` event
5. Continues until Kalvin is shut down

Thread safety:

- `__backlog` access must be synchronized (e.g. `threading.Lock` + `threading.Condition` or `queue.Queue`)
- EventBus.publish must be safe to call from the cogitate thread
- Model mutations (`upgrade`) must be thread-safe or synchronized

## Step 5: Update KModel abstract (abstract.py:372)

Change: `def rationalise(...) -> KLine | None` → `def rationalise(...) -> None`

## Step 6: Update consumers

**`encode()` (kalvin.py:91-127)** - No change needed. Already ignores return value, relies on side effects only.

**TUI `app.py` (lines 364, 390)** - Subscribe to the single event channel:

```python
def on_event(event):
    if event.kind == "complete":
        display(buffer, event.query)
        buffer.clear()
    else:
        buffer.append(event.kline)

kalvin.events.subscribe(on_event)
```

Then `rationalise(kline)` becomes fire-and-forget. Slow results arrive asynchronously via the cogitate thread.

**`scripts/kalvin_test.py`** - Subscribe to events instead of using return value.

## Step 7: Update exports

Add `EventBus` and `RationaliseEvent` to `src/kalvin/__init__.py`.

## Files to modify

| File                     | Change                                                            |
| ------------------------ | ----------------------------------------------------------------- |
| `src/kalvin/events.py`   | **New** - single-channel EventBus + RationaliseEvent              |
| `src/kalvin/kalvin.py`   | Add EventBus, rewrite rationalise() and cogitate() with threading |
| `src/kalvin/abstract.py` | Fix rationalise return type                                       |
| `ui/kscript/app.py`      | Subscribe to events instead of collecting return value            |
| `scripts/kalvin_test.py` | Subscribe to events                                               |
| `src/kalvin/__init__.py` | Export new types                                                  |

## Verification

1. `uv run pytest` - existing tests pass (encode-based tests unaffected)
2. `python scripts/kalvin_test.py` - events emitted and printed
3. TUI: run script, verify responses display via callbacks (both fast and slow events)
