## Why

When running KScript in the TUI, the rationalise response is currently displayed as raw JSON (e.g., `{"12345": [67890]}`). This is hard to read and doesn't show the semantic relationships. Users need to see the decompiled KScript source to understand what Kalvin matched and returned. Additionally, the current single-line selection model doesn't support selecting multi-construct scripts.

## What Changes

- **Decompile rationalise responses**: Use the existing `Decompiler` class to convert returned KLines back to readable KScript source
- **Display decompiled source**: Show responses as KScript text instead of JSON in the responses region
- **Script-level selection**: Enable selecting entire decompiled scripts (multi-construct output) rather than individual lines

## Capabilities

### New Capabilities
- `response-decompilation`: Capability to decompile Kalvin response KLines into human-readable KScript source format in the TUI

### Modified Capabilities
- `response-browser`: Change display format from JSON items to decompiled KScript blocks, and update selection model from line-based to script-based

## Impact

- **Files affected**: `ui/kscript/app.py`, `ui/kscript/regions/responses.py`
- **Dependencies**: Uses existing `kscript.decompiler.Decompiler` and `kalvin.mod_tokenizer.ModTokenizer`
- **UI behavior**: Response items change from single JSON lines to multi-line KScript blocks
