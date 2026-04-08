# Tasks: Dev Mode Hot Reload

## 1. Setup

- [x] 1.1 Add `watchdog` dependency to pyproject.toml

## 2. Hot Reload Module

- [x] 2.1 Create `ui/kscript/hotreload.py` module
- [x] 2.2 Implement `HotReloadHandler` class with debounced file watching (200ms)
- [x] 2.3 Implement `restart_app()` function using `os.execv()`
- [x] 2.4 Implement `HotReloadManager` class to coordinate watcher and restart

## 3. State Serialization

- [x] 3.1 Implement `save_state()` function to persist Kalvin model to `.tmp.bin`
- [x] 3.2 Implement `save_ui_state()` function to persist UI state to `.tmp.json`
- [x] 3.3 Implement `load_state()` function to restore Kalvin model from `.tmp.bin`
- [x] 3.4 Implement `load_ui_state()` function to restore UI state from `.tmp.json`
- [x] 3.5 Implement `cleanup_state_files()` function to remove temp files after restore
- [x] 3.6 Add error handling for corrupted state files (log warning, start fresh)

## 4. App Integration

- [x] 4.1 Update `ui/kscript/__main__.py` to add `--dev` CLI argument via argparse
- [x] 4.2 Update `KScriptApp.__init__()` to accept `dev_mode` parameter
- [x] 4.3 Update `KScriptApp.on_mount()` to start file watcher and restore state in dev mode
- [x] 4.4 Update `KScriptApp` to save state before restart (hook into hot reload trigger)
- [x] 4.5 Add `_save_hot_reload_state()` method to `KScriptApp`
- [x] 4.6 Add `_restore_hot_reload_state()` method to `KScriptApp`

## 5. Testing

- [x] 5.1 Test: start app with `--dev` flag, verify file watcher is active
- [x] 5.2 Test: modify `.py` file in `ui/kscript/`, verify restart with state restored
- [x] 5.3 Test: modify `.py` file in `src/`, verify restart with state restored
- [x] 5.4 Test: rapid file changes only trigger single restart
- [x] 5.5 Test: corrupted `.tmp.json` is handled gracefully with warning

> **Note:** Tasks 5.1-5.5 are manual verification tests. Implementation is complete.
