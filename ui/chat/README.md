# Kalvin Chat

A Textual TUI application for interacting with the Kalvin language model.

## Installation

```bash
uv sync
```

## Usage

Run the application:

```bash
python -m ui.chat
```

## Configuration

The app has two configurable fields in the **Config** region:

| Field   | Default                                       | Description                   |
| ------- | --------------------------------------------- | ----------------------------- |
| Model   | `~/dev/ai/kalvin/data/kalvin.bin`             | Path to the Kalvin model file |
| Grammar | `data/tokenizer/simplestories-1_grammar.json` | Path to the grammar file      |

### File Selection

- Click **Browse** button or double-click the input field to open a file dialog
- Use mount point buttons (**Root**, **Volumes**, **Home**) to quickly navigate
- The dialog opens at the current file's location with the file pre-selected

## Key Bindings

| Key      | Action                                                |
| -------- | ----------------------------------------------------- |
| `Ctrl+Q` | Quit the application                                  |
| `Ctrl+R` | Restart the app (full module reload)                  |
| `Ctrl+L` | Clear the response area                               |
| `Ctrl+S` | Save chat (saves directly if file open, else Save As) |
| `Ctrl+A` | Save As (always prompts for filename)                 |
| `Ctrl+O` | Open a saved chat file                                |
| `Enter`  | Send message                                          |

## Chat Management

Chats are saved as JSON files in `data/chats/` with the following format:

```json
{
  "chats": [
    {"chat": "user input", "response": "model response"},
    ...
  ]
}
```

- **Save**: If a chat is already open, saves directly. Otherwise opens Save As dialog.
- **Save As**: Always prompts for a new filename.
- **Open**: Browse and select a `.json` chat file to load.

## Structure

```
ui/chat/
├── __init__.py         # Package exports
├── __main__.py         # Entry point
├── app.py              # Main KalvinApp class
├── dialogs.py          # FileDialog, OpenDialog, SaveDialog
├── README.md           # This file
└── regions/
    ├── __init__.py     # Region exports
    ├── config.py       # ConfigRegion component
    ├── chat.py         # ChatRegion component
    └── history.py      # ChatHistoryRegion component
```

## Layout

```
┌─────────────────────────────────────────────────────────┐
│ Configuration                                           │
│ Model: [path________________] [Browse]                  │
│ Grammar: [path_______________] [Browse]                  │
├─────────────────────────────┬───────────────────────────┤
│ Chat                        │ History                   │
│                             │                           │
│ [Response Area - Editable]  │ [Previous chats list]     │
│                             │                           │
│ [Input____________] [Send]  │                           │
│                    [Submit] │                           │
└─────────────────────────────┴───────────────────────────┘
```

- **Left panel**: Chat input and editable response area
- **Right panel**: Chat history - click to load a previous response

## Components

| Component           | Module               | Description                                 |
| ------------------- | -------------------- | ------------------------------------------- |
| `KalvinApp`         | `app.py`             | Main application with save/load actions     |
| `FileDialog`        | `dialogs.py`         | Base class for file dialogs                 |
| `OpenDialog`        | `dialogs.py`         | File browser for opening files              |
| `SaveDialog`        | `dialogs.py`         | File browser with filename input for saving |
| `ConfigRegion`      | `regions/config.py`  | Model/grammar path inputs with browse       |
| `ChatRegion`        | `regions/chat.py`    | Chat input and editable response display    |
| `ChatHistoryRegion` | `regions/history.py` | Selectable list of chat history             |

## Development

The app is built with [Textual](https://textual.textualize.io/).

To integrate model inference, modify the `_handle_send()` method in `app.py`.
