## Context

The Kalvin project has two core subsystems:
1. **KScript** - A DSL that compiles `.ks` files into KLine entries (knowledge graph nodes)
2. **Kalvin** - A knowledge graph engine that processes KLines via `rationalise()` and responds

Currently, the `ui/chat/` TUI exists but doesn't properly integrate these subsystems - it shows JSONL output without interactive exploration. We need a new TUI that enables continuous development: write script → compile → execute → see responses → iterate.

## Goals / Non-Goals

**Goals:**
- MVP TUI with horizontal layout (editor | responses)
- Compile KScript and feed KLines to Kalvin one-at-a-time
- Display Kalvin responses as scrollable JSON list
- Click response → halt + append JSON to editor
- Auto/step execution modes with halt/resume
- Persistent Kalvin state via save/load dialogs
- Modular architecture for future expansion (graph visualization, cogitate integration)

**Non-Goals:**
- Graph visualization (future)
- Cogitate background process (future)
- KScript syntax highlighting (MVP: plain text)
- Modifying existing `ui/chat/` TUI
- Error handling (catastrophic errors halt with standard trace)

## Decisions

### 1. Application Module Structure

**Decision:** Create new `ui/kscript/` module mirroring `ui/chat/` structure but independent.

```
ui/kscript/
├── __init__.py
├── __main__.py        # Entry point: python -m ui.kscript
├── app.py             # Main KScriptApp
├── dialogs.py         # File dialogs (new implementation)
├── regions/
│   ├── __init__.py
│   ├── editor.py      # EditorRegion: script editing
│   ├── responses.py   # ResponsesRegion: KLine output list
│   └── toolbar.py     # ToolbarRegion: action buttons
```

**Rationale:** Clean separation from existing TUI, modular regions for future expansion.

### 2. Execution State Machine

**Decision:** State machine with 3 states: IDLE → RUNNING → HALTED

```
IDLE ──[Run]──▶ RUNNING ──[Halt]──▶ HALTED
  ▲                │                    │
  └────────────────┴──[Resume]─────────┘
  │                                     │
  └─────────────────[Step]──────────────┘  (stays in HALTED)
```

**Rationale:** Clear state transitions, prevents conflicting actions, enables async execution with interruption.

### 3. KLine Feeding Strategy

**Decision:** Use Textual's async worker pattern with cancellation token.

```python
async def feed_klines(self, entries: list[CompiledEntry]) -> None:
    for entry in entries:
        if self._cancelled:
            break
        response = self._kalvin.rationalise(entry_to_kline(entry))
        self._responses.append(response)
        self._refresh_responses()
        await asyncio.sleep(0)  # Yield to event loop
```

**Rationale:** Allows UI to remain responsive, supports cancellation, simple for MVP.

### 4. Response Click Behavior

**Decision:** Click response item triggers:
1. Halt execution (if running)
2. Append clicked item's JSON to editor content

**Rationale:** Supports exploration loop - user sees interesting response, clicks to incorporate into script.

### 5. Kalvin State Management

**Decision:** Kalvin instance lives at app level, accumulates across all Run operations. Save/Load via binary (default) or JSON.

**Rationale:** Knowledge graph should grow organically. Persistence enables session continuity.

### 6. Error Handling

**Decision:** No error handling in the application layer. Catastrophic errors (file I/O, memory, etc.) halt the app with standard Python traceback.

**Rationale:** This is an error-free development loop. The KScript language and Kalvin engine handle all semantic cases gracefully.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              KScriptApp                                    │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          ToolbarRegion                              │  │
│  │  [Load.ks] [Save.k] [Load.k] [Run] [Halt] [Step] [Clear]  Status: ● │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────┬────────────────────────────────────────┐  │
│  │                            │                                        │  │
│  │       EditorRegion         │           ResponsesRegion              │  │
│  │                            │                                        │  │
│  │  ┌────────────────────┐    │    ┌────────────────────────────────┐  │  │
│  │  │                    │    │    │  ListView (clickable)          │  │  │
│  │  │   TextArea         │    │    │    - ListItem: {"A": "B"}      │  │  │
│  │  │   (plain text)     │    │    │    - ListItem: {"B": "A"}      │  │  │
│  │  │                    │    │    │    - ...                       │  │  │
│  │  └────────────────────┘    │    └────────────────────────────────┘  │  │
│  │                            │                                        │  │
│  └────────────────────────────┴────────────────────────────────────────┘  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  App State                                                          │  │
│  │  - _kalvin: Kalvin (persistent, accumulating)                       │  │
│  │  - _execution_state: IDLE | RUNNING | HALTED                        │  │
│  │  - _pending_entries: list[CompiledEntry] (for step mode)            │  │
│  │  - _last_script_dir: Path (sticky default)                          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Large scripts could slow UI during execution | Yield to event loop after each KLine; show progress indicator |
| Click during run could race with append | State machine ensures halt completes before click handler runs |
| Kalvin model grows unbounded | Future: add model stats display and prune controls |
