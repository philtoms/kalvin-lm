## Why

The project needs an interactive development environment for the KScript → Kalvin pipeline. Currently, there's no way to explore how Kalvin responds to KScript compilation in real-time, observe the knowledge graph evolution, or iterate on scripts through a continuous feedback loop.

## What Changes

- **New TUI application**: A dedicated Textual TUI for KScript development and Kalvin exploration
- **KScript compilation pipeline**: Integration with KScript API to compile `.ks` files into Kalvin-compatible KLines
- **Interactive execution model**: Auto-feed KLines to Kalvin with ability to halt, step, and resume
- **Response-driven exploration**: Click any response KLine to append it to the editor, enabling continuous development loops
- **Persistent Kalvin state**: Save/Load Kalvin model state to preserve knowledge graph across sessions

## Capabilities

### New Capabilities

- `kscript-editor`: KScript source editing with plain text display, load/save `.ks` files, sticky default directory
- `kline-execution`: Compile KScript to KLines, feed to Kalvin one-at-a-time, automatic or manual step-through with halt/resume controls
- `response-browser`: Scrollable list of Kalvin response KLines in JSON format, click-to-select interaction that halts execution and appends to editor
- `kalvin-persistence`: Save and load Kalvin model state (`.bin`/`.json`), accumulating state across runs

### Modified Capabilities

None - this is a new application.

## Impact

- **New module**: `ui/kscript/` - complete new TUI application
- **Dependencies**: `kscript` module (existing), `kalvin` module (existing), `textual` framework
- **File storage**: New default directory `data/scripts/` for `.ks` files
- **No changes to existing code**: The existing `ui/chat/` TUI remains unchanged
