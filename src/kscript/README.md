# KScript

A domain-specific scripting language for creating and manipulating Kalvin knowledge graph models.

## Overview

KScript provides a concise syntax for defining KLines (knowledge lines) with significance relationships, enabling you to build knowledge graphs that can be interpreted into Kalvin models.

## Installation

KScript is part of the Kalvin package. Install with:

```bash
uv sync
```

## Quick Start

```python
from kscript import parse, interpret_script

# Parse and interpret a script
source = """
    greeting = hello world
    question > greeting ?
"""
result = interpret_script(source)

print(f"Interpreted {len(result.model)} KLines")
print(f"Symbols: {list(result.symbol_table.keys())}")
```

## Syntax

### Basic KLine

A KLine is defined by an identifier (the sig):

```kscript
hello
```

### Significance Relationships

KLines can have relationships to other nodes using significance operators:

| Operator | Name        | Meaning              |
| -------- | ----------- | -------------------- |
| `=`      | S1          | Prefix match         |
| `=>`     | S2          | Partial positional   |
| `>`      | S3 Forward  | Unordered (forward)  |
| `<`      | S3 Backward | Unordered (backward) |
| `!=`     | S4          | No match             |

```kscript
# S1: Prefix match
greeting = hello world

# S2: Partial positional
sentence => the cat sat

# S3: Unordered
related > concept1 concept2
inverse < source target

# S4: No match (exclusion)
different != opposite
```

### Multiple Relationships

A KLine can have chained relationships:

```kscript
MHALL = SVO => OVH
```

### Multi-line KLines

Use Python-style indentation for nested structures:

```kscript
MHALL = SVO =>
    S < M
    V < H
    O < ALL
```

This creates:

- `MHALL` with S1 relationship to `SVO`
- `MHALL` with S2 relationship to the indented KLines
- Nested KLines: `S < M`, `V < H`, `O < ALL`

### Comments

Comments are enclosed in parentheses and can appear anywhere, even mid-identifier:

```kscript
sit > V(erb)
N(oun) = cat dog
hel(comment)lo    # becomes: hello
```

### Load and Save

```kscript
load /path/to/model.bin

# Define KLines...
greeting = hello

save /path/to/output.bin
```

## CLI Usage

Run a KScript file with `ks_run.py`:

```bash
# Create a new model from script
uv run python scripts/ks_run.py my-script.ks

# Update an existing model
uv run python scripts/ks_run.py my-script.ks --model existing.bin

# Save as JSON format
uv run python scripts/ks_run.py my-script.ks --format json

# Verbose output
uv run python scripts/ks_run.py my-script.ks -v
```

## API Reference

### `parse(source: str) -> KScript`

Parse KScript source code into an AST.

```python
from kscript import parse

ast = parse("greeting = hello")
```

### `interpret_script(source: str, agent=None) -> InterpretResult`

Interpret KScript source using a Kalvin agent.

**Parameters:**

- `source`: KScript source code
- `agent`: Optional Kalvin agent instance. If not provided, creates a new Kalvin agent with default settings.

```python
from kscript import interpret_script
from kalvin import Kalvin

# Use existing Kalvin agent
kalvin = Kalvin.load("model.bin")
result = interpret_script(source, agent=kalvin)
print(f"KLines: {len(result.model)}")

# Or create a new agent automatically
result = interpret_script(source)
```

### `InterpretResult`

| Field          | Type             | Description                  |
| -------------- | ---------------- | ---------------------------- |
| `model`        | `Model`          | Kalvin model with KLines     |
| `symbol_table` | `dict[str, int]` | Maps sig names to signatures |
| `load_paths`   | `list[str]`      | Paths from load statements   |
| `save_path`    | `str \| None`    | Path from save statement     |

## Examples

### Simple Knowledge Graph

```kscript
# Animals
animal = mammal bird fish
mammal = dog cat whale
bird = eagle sparrow penguin

# Relationships
dog > loyal
cat > independent
whale > ocean
```

### NLP-style Parsing

```kscript
# Parts of speech
V(erb) = sit stand run
N(oun) = cat dog mouse
Adj(ective) = big small fast

# Sentence patterns
SVO => N V N
SV => N V
```

### Complex Structure

```kscript
# Multi-level categorization
entity = physical abstract

physical =>
    object > tangible
    living > animate

abstract =>
    concept > idea
    relation > connection
```

## Grammar

```
script      ::= statement*
statement   ::= load_stmt | save_stmt | kline_expr
load_stmt   ::= "load" IDENTIFIER
save_stmt   ::= "save" [IDENTIFIER]
kline_expr  ::= KSig kline_tail*
kline_tail  ::= significance nodes
nodes       ::= inline_nodes | indented_klines
inline_nodes ::= IDENTIFIER+
indented_klines ::= INDENT kline_expr+ DEDENT
significance ::= "=" | "=>" | ">" | "<" | "!="
```

## File Extension

KScript files use the `.ks` extension.
