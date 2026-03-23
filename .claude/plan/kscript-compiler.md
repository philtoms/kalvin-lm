# KScript Compiler Implementation Plan

## Overview

Build a **new** KScript compiler in `src/ksc/` that compiles `.ks` script files into KLine graphs, outputting JSONL format.

## Key Decisions

- **New module**: Fresh implementation in `src/ksc/` folder (separate from existing `src/kscript/`)
- **Output**: JSONL format
- **Signatures**: Literal A-Z identifiers (e.g., `M`, `ALL`), not tokenized
- **Operators**: Use spec operators with defined semantics

## Grammar (from spec)

```
script      ::= signature construct*

construct   ::= countersign | canonize | connotate | undersign

countersign ::= "==" node
canonize    ::= "=>" node+ | "<=" node+
connotate   ::= ">" node | "<" node
undersign   ::= "=" node

node        ::= signature | string | number | comment | subscript
subscript   ::= INDENT script+ DEDENT
signature   ::= [A-Z]+ comment?
string      ::= "\"" printable ascii chars "\""
number      ::= [0-9]+
comment     ::= "(" printable ascii chars ")"
```

## Clarifications

- **Comments ignored**: Standalone comments (on their own line) and inline comments are discarded, never producing output
- **Operators require nodes**: An operator is only a construct if followed by nodes (inline or in subscript). If no nodes follow, the operator is ignored and the signature becomes an identity
  - `X==` with no nodes → `==` ignored → `{"X": None}`
- **Constructs**: Canonization is the only construct that accepts multiple nodes
- **Numbers and Strings**: Only used as nodes, not as signatures
- **Compilation order**: Multi-pass; source order is NOT a semantic requirement
- **Identity scripts**: A signature-only script/subscript creates `{sig: None}`
  - `A` → `{"A": None}`
- **Duplicate signatures**: Always output as separate lines (never merged)
- **Nested subscripts**: Arbitrary depth
- **Node ordering**: Nodes in output preserve source order
- **Node binding**: Nodes bind the the immediate LHS signature:
  - `A => B => C` → `{A: ["B"]}` AND `{"B": ["C"]}`

## Construct Semantics

| Syntax               | Output                                                   | Comments                                  |
| -------------------- | -------------------------------------------------------- | ----------------------------------------- |
| `A`                  | `{A: None}`                                              | identity kline                            |
| `A == B`             | `{A: B}` AND `{B: A}`                                    | countersigned (signature)                 |
| `A ==`               | `{A: B}` AND `{B: A}` AND `{B: None}`                    | multi-line with indented subscript        |
| `  B`                |                                                          | subscripted identity                      |
| `AB => C D`          | `{AB: [C, D]}`                                           | multi-node canonization (nodes)           |
| `C D <= AB`          | `{AB: [C, D]}`                                           | multi-node canonization (nodes)           |
| `AB =>`              | `{AB: [C, D]}` AND `{C: None}` AND `{D: 1}` AND `{1: D}` | multi-line canonization                   |
| ` C`                 |                                                          | subscripted identity                      |
| ` D == 1`            |                                                          | subscripted countersignature              |
| `A > B`              | `{A: [B]}`                                               | connotated (nodes)                        |
| `A < B`              | `{B: [A]}`                                               | connotated (nodes)                        |
| `A = B`              | `{A: B}`                                                 | undersigned (signature)                   |
| `A = "\"hello\""`    | `{A: "\"hello\""}`                                       | literal string (with escape char)         |
| `(comment (nested))` |                                                          | nested comment: greedy match              |
| `(comment `          |                                                          | unterminated comment: greedy match to EOL |
| `()`                 |                                                          | empty comment                             |

### Node Output Types

| Construct   | Operator  | Output Type         | Example           |
| ----------- | --------- | ------------------- | ----------------- |
| Identity    | (none)    | `None`              | `{A: None}`       |
| Countersign | `==`      | `str` (signature)   | `{A: B}`          |
| Undersign   | `=`       | `str` (signature)   | `{A: B}`          |
| Connotate   | `>` `<`   | `list[str]` (nodes) | `{A: ["B"]}`      |
| Canonize    | `=>` `<=` | `list[str]` (nodes) | `{A: ["B", "C"]}` |

## Error handling strategy

KScripts are always valid. This does not mean that parsers should ignore bad syntax, but rather should recover valid output from bad syntax

## Recovery Semantics

| Syntax    | Output                    | Comments                                                    |
| --------- | ------------------------- | ----------------------------------------------------------- |
|           |                           | empty file: empty output                                    |
| `A ==`    | `{A: None}`               | incomplete construct: recover identity                      |
| `A =>`    | `{A: None}`               | incomplete construct: recover identity                      |
| `1 > A`   | `{1: None}`               | unsupported literal in signature position: recover identity |
| `A < 1`   | `{A: None}`               | unsupported literal in signature position: recover identity |
| `A > 1`   | `{A: [1]}` AND `{B: [2]}` | unexpected indent across multi-lines: revert to previous    |
| `  B > 2` |                           | indent if any                                               |

## Module Structure: `src/ksc/`

```
src/ksc/
├── __init__.py        # Module exports
├── __main__.py        # CLI and API entry points
├── token.py           # Token types and Token dataclass
├── lexer.py           # Lexer with indentation support
├── ast.py             # AST node definitions
├── parser.py          # Recursive descent parser
├── compiler.py        # Compiles AST to entries
└── output.py          # JSONL output writer
```

## Implementation Phases

### Phase 1: Token Types (`token.py`)

```python
from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    # Construct operators
    COUNTERSIGN = auto()   # ==
    CANONIZE_FWD = auto()  # =>
    CANONIZE_BWD = auto()  # <=
    CONNOTATE_FWD = auto() # >
    CONNOTATE_BWD = auto() # <
    UNDERSIGN = auto()     # =

    # Literals
    SIGNATURE = auto()     # [A-Z]+
    STRING = auto()        # "..."
    NUMBER = auto()        # [0-9]+
    COMMENT = auto()       # (...)

    # Structure
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()

@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int
    column: int
```

### Phase 2: Lexer (`lexer.py`)

- Multi-char operators: `==`, `=>`, `<=` (check before single-char)
- Single-char operators: `=`, `>`, `<`
- Signatures: `[A-Z]+` with optional inline comment
- String literals: `"printable ascii chars"`
- Comments: `(...)` can be standalone or inline with signature
- Python-style indentation (INDENT/DEDENT tokens)
  - spaces and or tabs, flexible indent count

### Phase 3: AST (`ast.py`)

```python
@dataclass
class Signature:
    name: str              # A-Z+ identifier
    comment: str | None    # optional (...)
    line: int
    column: int

@dataclass
class StringLiteral:
    value: str
    line: int
    column: int

@dataclass
class NumberLiteral:
    value: str
    line: int
    column: int

Node = Signature | StringLiteral | NumberLiteral

class ConstructType(Enum):
    COUNTERSIGN = "=="
    CANONIZE_FWD = "=>"
    CANONIZE_BWD = "<="
    CONNOTATE_FWD = ">"
    CONNOTATE_BWD = "<"
    UNDERSIGN = "="

@dataclass
class Construct:
    type: ConstructType
    nodes: list[Node]
    line: int

@dataclass
class Script:
    signature: Signature
    constructs: list[Construct]
    subscripts: list["Script"]
    line: int

@dataclass
class KScriptFile:
    scripts: list[Script]
```

### Phase 4: Parser (`parser.py`)

Recursive descent parser handling:

1. Multiple scripts per file (column 1 starts new script)
2. Signature optionally followed by one or more constructs
3. Construct: operator followed by node (optionally followed by more nodes if canonize construct)
4. Multiple Constructs: adhere to immediate binding semantics (see clarification above)
5. Nodes can be signatures, literals or the signatures of indented subscripts
6. Subscripts should be parsed recursively AFTER processing all of the nodes of a construct

### Phase 5: Compiler (`compiler.py`)

```python
@dataclass
class CompiledEntry:
    """A single compiled KLine entry.

    nodes semantics:
    - None: identity entry (sig exists with no children)
    - str: single signature link (countersign, undersign)
    - list[str]: nodes list (connotate, canonize)
    """
    signature: str
    nodes: str | None | list[str]

class Compiler:
    def __init__(self):
        self.entries: list[CompiledEntry] = []

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile KScript file."""

    def compile_script(self, script: Script, is_top_level: bool) -> None:
        """Compile a script, adding entries."""
        # If script has no constructs, emit identity: {sig: None}
```

### Phase 6: Output (`output.py`)

```python
def write_jsonl(entries: list[CompiledEntry], path: Path) -> None:
    """Write entries to JSONL file.

    Each entry becomes a separate line:
    {"L": ["M"]}
    {"L": ["O"]}
    """
    with open(path, 'w') as f:
        for entry in entries:
            f.write(json.dumps({entry.signature: entry.nodes}) + '\n')
```

### Phase 7: CLI and API (`__main__.py`, `__init__.py`)

```python __main__.py
def main():
    import argparse
    parser = argparse.ArgumentParser(description="KScript compiler")
    parser.add_argument("input", help="Input .ks file")
    parser.add_argument("-out", dest="output", help="Output file (.jsonl)")
    args = parser.parse_args()

    source = Path(args.input).read_text()

    try:
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()
        entries = Compiler().compile(kscript_file)

        output_path = Path(args.output) if args.output else Path(args.input).with_suffix('.jsonl')
        write_jsonl(entries, output_path)

    except (LexerError, ParseError, CompileError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

```python __init__.py
class KScript:
    """Compiled KScript model supporting incremental construction."""

    def __init__(self, source: str | Path, base: KScript | None = None):
        """Compile KScript from string or file path.

        Args:
            source: KScript source string or path to file.
                    Supported: .ks, .json, .jsonl
            base: Optional existing KScript to extend
        """
        ...

    @property
    def entries(self) -> list[CompiledEntry]:
        """Return all compiled entries."""
        ...

    def output(self, path: str | Path) -> None:
        """Write entries to JSON or JSONL file based on suffix."""
        ...

    def to_model(self) -> list[dict[str, str | None | list[str]]]:
        """Return entries as list of dicts (preserves duplicate signatures)."""
        ...

    def to_dict(self) -> dict[str, str | None | list[str]]:
        """Merge entries into dict format (later entries override earlier)."""
        ...
```

## Files to Create

| File                  | Purpose                      |
| --------------------- | ---------------------------- |
| `src/ksc/__init__.py` | Module exports + KScript API |
| `src/ksc/__main__.py` | CLI entry point              |
| `src/ksc/token.py`    | Token types and dataclass    |
| `src/ksc/lexer.py`    | Lexer with indentation       |
| `src/ksc/ast.py`      | AST node definitions         |
| `src/ksc/parser.py`   | Recursive descent parser     |
| `src/ksc/compiler.py` | AST to compiled entries      |
| `src/ksc/output.py`   | JSON/JSONL reader/writer     |
| `tests/test_ksc.py`   | Test suite                   |

## Example Compilation

**Input** (`example.ks`):

```
(Mary had a little lamb)
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A = D(et)
     L = M(od)
     L > O
X==
```

**Output** (`mhall.jsonl`):

```json
{"MHALL": "SVO"}
{"SVO": "MHALL"}
{"SVO": ["S", "V", "O"]}
{"S": "M"}
{"V": "H"}
{"O": "ALL"}
{"ALL": ["A", "L", "L"]}
{"A": "D"}
{"L": "M"}
{"L": ["O"]}
{"X": null}
```

## Usage

```bash
# Compile and output to terminal
python -m ksc script.ks

# Compile to JSON (array format, preserves duplicates)
python -m ksc script.ks -out output.json

# Compile to JSONL (line-delimited)
python -m ksc script.ks -out output.jsonl

# Extend existing model from JSON/JSONL
python -m ksc script.ks -base base.json -out extended.json

# Load and extend JSON/JSONL models
python -m ksc model.json -base base.jsonl -out merged.json
```

### API Usage Examples

In addition to CLI usage, the compiler can be used programmatically:

```python
from ksc import KScript

# Inline model generation from string
model1 = KScript("A == B")
# -> KScript object containing: [{A: "B"}, {B: "A"}]

# Load from JSON/JSONL file
model2 = KScript("path/to/model.json")

# Extend model with script
model3 = KScript("path/to/script.ks", model1)

# Save to JSON or JSONL (based on suffix)
model3.output("/path/to/output.json")
```

```python
# Simple inline script
model = KScript("A == B C")
model.output("output.json")  # -> JSON array
model.output("output.jsonl") # -> JSONL

# Chain multiple sources
base = KScript("A == B")
extended = KScript("C => A D", base)
extended.output("extended.json")

# Load from JSON/JSONL
loaded = KScript("existing_model.json")
extended = KScript("more.ks", base=loaded)

# Get model as list (preserves duplicates)
model.to_model()  # -> [{"A": "B"}, {"B": "A"}]

# Get model as dict (last wins)
model.to_dict()   # -> {"A": "B", "B": "A"}

# Inspect entries
for entry in model.entries:
    print(f"{entry.signature}: {entry.nodes}")
```
