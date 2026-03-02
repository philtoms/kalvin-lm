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

| Field   | Default                                                                    | Description                   |
| ------- | -------------------------------------------------------------------------- | ----------------------------- |
| Model   | `~/dev/ai/kalvin/data/kalvin.bin`                                          | Path to the Kalvin model file |
| Grammar | `/Volumes/USB-Backup/ai/data/tidy-ts/simplestories-1_grammar.json`        | Path to the grammar file      |

### File Selection

- Click **Browse** button or double-click the input field to open a file dialog
- Use mount point buttons (**Root**, **Volumes**, **Home**) to quickly navigate
- The dialog opens at the current file's location with the file pre-selected

## Key Bindings

| Key      | Action                  |
| -------- | ----------------------- |
| `Ctrl+Q` | Quit the application    |
| `Ctrl+L` | Clear the response area |
| `Enter`  | Send message            |

## Structure

```
ui/chat/
├── __init__.py         # Package exports
├── __main__.py         # Entry point
├── app.py              # Main KalvinApp class
├── dialogs.py          # FileLoadDialog component
├── README.md           # This file
└── regions/
    ├── __init__.py     # Region exports
    ├── config.py       # ConfigRegion component
    └── chat.py         # ChatRegion component
```

## Components

| Component       | Module                 | Description                                |
| --------------- | ---------------------- | ------------------------------------------ |
| `KalvinApp`     | `app.py`               | Main application                           |
| `FileLoadDialog`| `dialogs.py`           | Modal file browser with mount navigation   |
| `ConfigRegion`  | `regions/config.py`    | Model/grammar path inputs with browse      |
| `ChatRegion`    | `regions/chat.py`      | Chat input and response display            |

## Development

The app is built with [Textual](https://textual.textualize.io/).

To integrate model inference, modify the `_handle_send()` method in `app.py`.
