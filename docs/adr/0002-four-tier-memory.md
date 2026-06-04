# Four-tier memory: STM → Frame → LTM → Base

Kalvin's model has four tiers forming a cascade: STM → Frame → LTM → Base. Each tier serves a distinct role: STM is the event register (all klines pass through), Frame is compressed working context (only recognised klines), LTM is persistent knowledge (ratified and novel klines), and Base is read-only shared knowledge.

The key design decision is **selective write**: `Model.add()` is removed in favour of three explicit methods — `add_stm()`, `add_frame()`, `add_ltm()` — that cascade downwards. `add_stm()` always refreshes FIFO position (removes-if-present then adds). `add_frame()` writes to Frame and cascades to STM. `add_ltm()` writes to LTM and cascades through Frame to STM. The caller (KAgent) selects the appropriate entry point based on the significance outcome of rationalisation.

**Write destinations by rationalisation outcome:**

| Outcome | STM | Frame | LTM |
|---------|-----|-------|-----|
| Ground (already exists) | ✓ | — | — |
| S4 (identity, novel) | ✓ | ✓ | ✓ |
| S1 (canonical, countersigned, all-literal, resolved) | ✓ | ✓ | ✓ |
| S2/S3 slow path query | ✓ | — | — |
| S1 discovered during cogitation | ✓ | ✓ | ✓ |
| S2/S3 expansion proposal | ✓ | ✓ | — |

This makes the Frame a compression of STM activity — only klines that Kalvin "recognises" through matching, expansion, or ratification reach the Frame. Of those, only ratified (S1) and novel (S4) klines reach LTM for cross-session persistence.

**Literal dedup** is the sole write guard: if `kline.is_literal()` and an equal kline exists in any tier, the entire cascade is skipped. Non-literal writes are unconditional within their target tiers. Frame and LTM are monotonic (append-only, never remove).

**Considered alternatives:**

- LTM replaces Base (single persistent tier): rejected because Base serves a different purpose — shared, read-only, potentially curated knowledge that sessions layer on top of. Merging would lose that boundary.
- LTM merges into Base at session end: rejected because it mutates the shared base and prevents clean session isolation.
- Automatic STM read-through (re-insert all queried klines): rejected in favour of always-refresh `add_stm()` — removes-if-present then adds unconditionally, absorbing the old `refresh_stm()` method.
- Single `add()` writing to all tiers: rejected because it prevents the agent from selectively routing klines to the correct tier based on significance.
- Conditional promotion from Frame to LTM: rejected in favour of direct `add_ltm()` cascade. `promote()` and `promote_participating()` are removed — the agent calls `add_ltm()` directly, which cascades through Frame and STM.

**Consequences:**

- `Model.add()` and `Model.refresh_stm()` are removed. `Model.promote()` is removed. The write API is `add_stm()`, `add_frame()`, `add_ltm()`.
- `expand.promote_participating()` becomes a loop of `add_ltm()` calls.
- Frame no longer receives all klines — only those from significant outcomes. The slow-path query kline goes to STM only.
- `add_stm()` absorbs `refresh_stm()`: always removes-if-present then adds, refreshing FIFO position unconditionally.
- All three mutable tiers (STM, Frame, LTM) are persisted in one file. Normal session start loads Frame and LTM; STM starts empty.
- Frame and LTM share identical internal storage structure (KLineStore).
