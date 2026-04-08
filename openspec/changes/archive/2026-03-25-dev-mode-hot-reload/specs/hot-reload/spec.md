# Spec: Hot Reload

## ADDED Requirements

### Requirement: Dev mode flag enables hot reload

The system SHALL provide a `--dev` CLI flag that enables hot reload behavior when starting the KScript TUI application. When `--dev` is not specified, the application SHALL run normally without any hot reload functionality.

#### Scenario: Dev mode enabled
- **WHEN** user starts app with `--dev` flag
- **THEN** hot reload is active and file watcher is started

#### Scenario: Dev mode disabled (default)
- **WHEN** user starts app without `--dev` flag
- **THEN** app runs normally without file watching or state persistence

### Requirement: File changes trigger app restart

The system SHALL monitor Python source files in `ui/kscript/` and `src/` directories for modifications when running in dev mode. When a `.py` file modification is detected, the system SHALL trigger a restart after a 200ms debounce period.

#### Scenario: Python file modified in ui/kscript
- **WHEN** a `.py` file in `ui/kscript/` is modified
- **THEN** the app restarts after 200ms debounce

#### Scenario: Python file modified in src
- **WHEN** a `.py` file in `src/` is modified
- **THEN** the app restarts after 200ms debounce

#### Scenario: Non-Python file modified
- **WHEN** a non-`.py` file is modified in watched directories
- **THEN** no restart is triggered

#### Scenario: Multiple rapid file changes
- **WHEN** multiple files are modified within 200ms
- **THEN** only one restart is triggered after changes stop

### Requirement: Kalvin model state is persisted across restarts

The system SHALL save the Kalvin model state to a temporary file (`.tmp.bin`) before restarting. After restart, the system SHALL restore the Kalvin model from this file if it exists.

#### Scenario: Save model state before restart
- **WHEN** a restart is triggered
- **THEN** Kalvin model state is saved to `.tmp.bin`

#### Scenario: Restore model state after restart
- **WHEN** app starts in dev mode and `.tmp.bin` exists
- **THEN** Kalvin model is restored from `.tmp.bin`

#### Scenario: No state file on first run
- **WHEN** app starts in dev mode and `.tmp.bin` does not exist
- **THEN** app initializes with fresh Kalvin instance

### Requirement: UI state is persisted across restarts

The system SHALL save UI state (editor content, cursor position, scroll position, last directories, execution state) to a temporary file (`.tmp.json`) before restarting. After restart, the system SHALL restore UI state from this file if it exists.

#### Scenario: Save UI state before restart
- **WHEN** a restart is triggered
- **THEN** UI state is saved to `.tmp.json` including editor content, cursor position, scroll position, and directory preferences

#### Scenario: Restore UI state after restart
- **WHEN** app starts in dev mode and `.tmp.json` exists
- **THEN** editor content, cursor position, scroll position, and directory preferences are restored

#### Scenario: Restore editor content exactly
- **WHEN** UI state is restored
- **THEN** editor displays the same content as before restart

#### Scenario: Restore cursor position
- **WHEN** UI state is restored
- **THEN** cursor is positioned at the same line and column as before restart

### Requirement: Clean process restart

The system SHALL use `os.execv()` to restart the application, replacing the current process with a new instance using the same command-line arguments.

#### Scenario: Process is replaced on restart
- **WHEN** restart is triggered
- **THEN** current process is replaced by new process with same arguments
- **AND** PID remains the same
- **AND** terminal attachment is preserved

### Requirement: State files are cleaned up

The system SHALL remove temporary state files (`.tmp.bin`, `.tmp.json`) after successfully restoring state. If state files exist but are corrupted, the system SHALL log a warning and start fresh.

#### Scenario: State files removed after restore
- **WHEN** state is successfully restored from temp files
- **THEN** temp files are deleted

#### Scenario: Corrupted state file handled gracefully
- **WHEN** `.tmp.json` exists but is malformed JSON
- **THEN** app logs warning and starts with fresh UI state
- **AND** corrupted file is removed
