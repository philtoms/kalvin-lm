# Four-tier memory: STM → Frame → LTM → Base

Kalvin's model gains a fourth tier (LTM) between Frame and Base to enable persistence of accumulated grounded knowledge across sessions. The previous three-tier model (STM → Frame → Base) had no mechanism for carrying session-confirmed knowledge forward — Frame was per-session, Base was read-only, so every grounded kline discovered during a session was lost on session end. LTM is structurally identical to Frame but semantically distinct: it holds klines promoted from Frame after confirmation (S1/S4), persisted across sessions, and loaded at session start. Frame remains working context; LTM is the cross-session accumulator. Base is unchanged (read-only, set at construction).

**Considered alternatives:**

- LTM replaces Base (single persistent tier): rejected because Base serves a different purpose — shared, read-only, potentially curated knowledge that sessions layer on top of. Merging would lose that boundary.
- LTM merges into Base at session end: rejected because it mutates the shared base and prevents clean session isolation.
- Automatic STM read-through (re-insert all queried klines): rejected in favour of caller-driven `refresh_stm()` — only klines actively used in cogitation should receive recency precedence.

**Consequences:**

- `promote()` shifts from STM→Frame to Frame→LTM. `add()` now writes to both STM and Frame directly.
- `promote_all()` is removed — promotion is selective, significance-driven.
- All three mutable tiers (STM, Frame, LTM) are persisted in one file. Normal session start loads Frame and LTM; STM starts empty.
- Frame and LTM share identical internal storage structure — refactoring opportunity for a shared Tier/KLineStore class.
