# Proposal: Dev Mode Hot Reload for KScript UI

## Why

Hot reload is essential for round-trip project development. Without it, developers must manually restart the TUI application after every code change, losing context and state each time. This friction significantly slows iteration cycles when building and refining UI components.

A hot reload capability allows the app to automatically detect code changes, preserve application state, restart with updated code, and restore state—enabling rapid development feedback loops.

## What Changes

- **File watcher**: Monitor Python source files in `ui/kscript/` and `src/` for changes when running in dev mode
- **State serialization**: Use Kalvin's native `save()`/`load()` to persist model state to a temp file (`.tmp.bin`) before restart; capture UI state (editor content, cursor position, open files, scroll position)
- **Graceful restart**: Trigger app restart when changes detected, preserving the TUI context
- **State restoration**: Reload serialized state via Kalvin `load()` after restart to resume exactly where the user left off
- **Dev mode flag**: Add `--dev` CLI flag to enable hot reload behavior (off by default)

## Capabilities

### New Capabilities

- `hot-reload`: File watching, state serialization, graceful restart, and state restoration for development mode

### Modified Capabilities

- None (this is a new standalone capability)

## Impact

**Affected code:**

- `ui/kscript/app.py` - dev mode flag, state save/restore hooks
- `ui/kscript/__main__.py` - CLI argument parsing for `--dev`
- New module: `ui/kscript/hotreload.py` (watcher, state manager, restart logic)

**Watched paths:**

- `ui/kscript/**/*.py` - UI source files
- `src/**/*.py` - Core library source files

**State persistence:**

- Uses Kalvin's existing `save()`/`load()` serialization
- Temp file: `.tmp.bin` (project root)

**Dependencies:**

- `watchdog` library for file system monitoring (or similar)
- Leverages existing Kalvin serialization

**Systems:**

- Affects `ui/kscript/` TUI application
- Watches `src/` for changes (triggers reload on core lib changes)
- Development-only feature, no production impact
