## Context

The KScript TUI currently displays rationalise responses as raw JSON (`{"12345": [67890]}`), which is hard to read. There's already a `Decompiler` class in `src/kscript/decompiler.py` that converts KLines back to KScript source. The `Kalvin.rationalise()` method returns `list[KLine]` from the frame.

Current flow:
1. User runs script → entries compiled → fed to `Kalvin.rationalise()`
2. Response (list of KLines) displayed as individual JSON items in ResponsesRegion
3. Click on item → appends JSON to editor

Target flow:
1. User runs script → entries compiled → fed to `Kalvin.rationalise()`
2. Response decompiled to KScript source → displayed as multi-line block
3. Click on block → appends KScript source to editor

## Goals / Non-Goals

**Goals:**
- Decompile rationalise response KLines into readable KScript format
- Display responses as KScript blocks (multi-line) instead of JSON items
- Enable selecting entire decompiled scripts on click
- Reuse existing `Decompiler` class

**Non-Goals:**
- Changing the rationalise algorithm or semantics
- Adding syntax highlighting for decompiled KScript
- Persisting decompiled output to files

## Decisions

### Decision 1: Decompile per-response
Each call to `Kalvin.rationalise()` returns `list[KLine]`. We decompile this list immediately and store the result.

**Rationale:** Decompiler expects ordered KLines where first is primary. Each rationalise call produces a coherent script fragment.

**Alternative considered:** Batch decompile all responses at display time. Rejected because it complicates the display model and loses per-response granularity.

### Decision 2: ResponseItem stores both KLine and decompiled source
Extend `ResponseItem` to hold the decompiled KScript string alongside the original KLine data.

**Rationale:** Need original KLine for potential future operations while displaying decompiled source. Avoids re-decompiling on every render.

### Decision 3: Shared Decompiler instance in KScriptApp
Initialize `Decompiler` once in `KScriptApp.__init__` with a `Mod32Tokenizer`.

**Rationale:** `Decompiler` is stateless between calls (resets in `decompile()`). Shared instance avoids repeated tokenizer initialization.

**Alternative considered:** Create Decompiler on-demand. Rejected for efficiency - tokenizer init has overhead.

### Decision 4: Selection appends KScript source (not JSON)
When user clicks a response, append the decompiled KScript source to editor.

**Rationale:** Consistency - user sees KScript, clicks KScript, gets KScript in editor.

## Risks / Trade-offs

- **Large responses could flood UI** → Decompiler output is typically compact (few lines per response). If needed, add truncation later.
- **Decompiler errors surface as `!!! BROKEN` markers** → Acceptable - these are informative for debugging.
- **ResponseItem changes from single-line to multi-line** → ListView handles variable-height items, but may need CSS tuning.
