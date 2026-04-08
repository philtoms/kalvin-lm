# Design: Dev Mode Hot Reload for KScript UI

## Context

The KScript TUI app (`ui/kscript/app.py`) is a Textual application for interactive KScript development. It maintains:
- A `Kalvin` model instance with learned state
- Editor content (KScript text)
- Execution state (running/halted/idle)
- UI state (scroll positions, last directories)

Currently, any code change requires manually restarting the app, losing all in-memory state. This design adds hot reload capability for development mode.

## Goals / Non-Goals

**Goals:**
- Automatically restart app when source files change
- Preserve Kalvin model state across restarts
- Preserve UI state (editor content, cursor position, scroll position)
- Enable via `--dev` flag (opt-in, no production impact)

**Non-Goals:**
- Live code injection without restart (too complex for Textual)
- State diffing/merging (full state replace is sufficient)
- Multi-window state sync
- Production hot reload

## Decisions

### 1. File Watching: `watchdog` with Debounce

**Choice:** Use `watchdog` library with a 200ms debounce window.

**Rationale:**
- `watchdog` is cross-platform and well-maintained
- Debounce prevents rapid restarts when multiple files change (e.g., git operations)
- Alternative considered: `inotify` (Linux-only), polling (inefficient)

**Implementation:**
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

class HotReloadHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[], None], debounce_ms: int = 200):
        self._callback = callback
        self._debounce_ms = debounce_ms
        self._last_trigger = 0

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.src_path.endswith('.py'):
            now = time.time() * 1000
            if now - self._last_trigger > self._debounce_ms:
                self._last_trigger = now
                self._callback()
```

### 2. State Serialization: Hybrid Approach

**Choice:** Use Kalvin's native `save()`/`load()` for model state + JSON for UI state.

**Rationale:**
- Kalvin already has binary serialization via `save(path)` and `load(path)`
- UI state (editor text, cursor position) is simple and fits JSON well
- Single temp file for model (`.tmp.bin`) + one for UI (`.tmp.json`)

**State file format:**
```
.tmp.bin   → Kalvin model (binary, via Kalvin.save())
.tmp.json  → UI state (editor content, cursor, scroll, directories)
```

**UI state schema:**
```json
{
  "editor_content": "...",
  "cursor_line": 5,
  "cursor_column": 12,
  "scroll_y": 0,
  "last_script_dir": "data/scripts",
  "last_state_dir": "data",
  "execution_state": "idle"
}
```

### 3. Restart Mechanism: `os.execv()`

**Choice:** Use `os.execv()` to replace current process with new one.

**Rationale:**
- Clean process replacement - no orphan processes
- Preserves PID and terminal attachment
- Alternative considered: subprocess + sys.exit (leaves orphan, terminal issues)

**Implementation:**
```python
def restart_app() -> None:
    """Restart the current process with same arguments."""
    import sys
    import os
    os.execv(sys.executable, [sys.executable] + sys.argv)
```

### 4. Dev Mode Flag: `--dev` CLI Argument

**Choice:** Add `--dev` flag via argparse in `__main__.py`.

**Rationale:**
- Explicit opt-in for development mode
- Can be passed through to app via environment variable or constructor
- No overhead when not in dev mode

**Implementation:**
```python
# __main__.py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='Enable hot reload')
    args = parser.parse_args()

    app = KScriptApp(dev_mode=args.dev)
    app.run()
```

### 5. Integration Point: `KScriptApp` Hooks

**Choice:** Add lifecycle hooks in `KScriptApp` for state save/restore.

**Rationale:**
- Minimal changes to existing app structure
- Hooks are called at appropriate lifecycle points
- Hot reload logic isolated in `hotreload.py` module

**Hooks:**
- `on_mount()`: Check for temp files, restore state if present
- `on_unmount()`: Save state (called before restart)
- `_save_hot_reload_state()`: Internal method to serialize state
- `_restore_hot_reload_state()`: Internal method to deserialize state

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| State corruption on crash | Use atomic writes (write to temp, then rename) |
| Rapid restarts during batch edits | 200ms debounce on file changes |
| State files left behind | Clean up on successful restore; ignore stale files |
| Restart during active execution | Save execution state; user can resume with Step |
| Large model state slows restart | Document as expected behavior; user can clear model |

## Migration Plan

1. Add `watchdog` to dependencies (pyproject.toml)
2. Create `ui/kscript/hotreload.py` module
3. Update `__main__.py` to add `--dev` argument
4. Update `app.py` to add state save/restore hooks
5. Test: make code change → verify restart → verify state restored

**Rollback:** Remove `--dev` flag and `hotreload.py` module. No data migration needed (temp files are disposable).
