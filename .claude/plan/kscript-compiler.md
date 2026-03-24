# KScript Compiler Implementation Plan

## Overview

Build a **new** KScript compiler in `src/kscipt/` that compiles `.ks` script files into KLine graphs, outputting JSONL or binary format.

## Key Decisions

- **New module**: Fresh implementation in `src/kscript/` folder)
- **Output**: JSONL format (default) or binary format (`.bin` suffix)
- **Signatures**: Literal A-Z identifiers (e.g., `M`, `ALL`), tokenized for binary output
- **Operators**: Use spec operators with defined semantics
- **Tokenizer**: `Mod32Tokenizer` from `src/kalvin/mod_tokenizer.py` for encoding signatures to token IDs

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
- **Duplicate signatures**: Always output as separate klines (never merged)
- **Nested subscripts**: Arbitrary depth
- **Node ordering**: Nodes in output preserve source order
- **Node binding**: Nodes bind to the immediate LHS signature:
  - `A => B C => D` → `{A: ["B", "C]}` AND `{"C": ["D"]}`

## Construct Semantics

| Syntax               | Output                     | Comments                                  |
| -------------------- | -------------------------- | ----------------------------------------- |
| `A`                  | `{A: None}`                | identity kline                            |
| `A == B`             | `{A: B}` AND `{B: A}`      | countersigned (signature)                 |
| `A ==`               | `{A: B}` AND `{B: A}`      | multi-line with indented subscript        |
| `  B`                | AND `{B: None}`            | + subscripted identity                    |
| `AB => C D`          | `{AB: [C, D]}`             | multi-node canonization (nodes)           |
| `C D <= AB`          | `{AB: [C, D]}`             | multi-node canonization (nodes)           |
| `AB =>`              | `{AB: [C, D]}`             | multi-line canonization                   |
| ` C`                 | AND `{C: None}`            | + subscripted identity                    |
| ` D == 1`            | AND `{D: 1}`               | + subscripted under-signature (recovery)  |
| `A > B`              | `{A: [B]}` AND `{B: None}` | connotated (nodes) + identity             |
| `A < B`              | `{B: [A]}` AND `{A: None}` | connotated (nodes) + identity             |
| `A = B`              | `{A: B}`AND `{B: None}`    | undersigned (signature) + identity        |
| `A = "\"hello\""`    | `{A: "\"hello\""}`         | literal string (with escape char)         |
| `(comment (nested))` |                            | nested comment: greedy match              |
| `(comment `          |                            | unterminated comment: greedy match to EOL |
| `()`                 |                            | empty comment                             |

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

| Syntax    | Output         | Comments                                                    |
| --------- | -------------- | ----------------------------------------------------------- |
|           |                | empty file: empty output                                    |
| `A ==`    | `{A: None}`    | incomplete construct: recover identity                      |
| `A =>`    | `{A: None}`    | incomplete construct: recover identity                      |
| `1 > A`   | `{1: None}`    | unsupported literal in signature position: recover identity |
| `A < 1`   | `{A: None}`    | unsupported literal in signature position: recover identity |
| `A > 1`   | `{A: [1]}`     | unexpected indent across multi-lines: revert to previous    |
| `  B > 2` | AND `{B: [2]}` | indent if any                                               |
| `A == 1`  | `{A: 1}`       | literal countersign: recover undersign                      |

## Module Structure: `src/kscript/`

```
src/kscript/
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
from kalvin.mod_tokenizer import ModTokenizer, Mod32Tokenizer

@dataclass
class CompiledEntry:
    """A single compiled KLine entry.

    nodes semantics:
    - None: identity entry (sig exists with no children)
    - int: single token ID link (countersign, undersign)
    - list[int]: nodes list (connotate, canonize)
    """
    signature: int
    nodes: int | None | list[int]

    @classmethod
    def encode(cls, sig: str, nodes: str | None | list[str], tokenizer: ModTokenizer) -> "CompiledEntry":
        """Encode string signature/nodes to token IDs."""
        sig_id = tokenizer.encode(sig)[0]
        if nodes is None:
            return cls(signature=sig_id, nodes=None)
        elif isinstance(nodes, str):
            return cls(signature=sig_id, nodes=tokenizer.encode(nodes)[0])
        else:
            return cls(signature=sig_id, nodes=tokenizer.encode("".join(nodes)))

    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings."""
        sig = tokenizer.decode([self.signature])
        if self.nodes is None:
            return sig, None
        elif isinstance(self.nodes, int):
            return sig, tokenizer.decode([self.nodes])
        else:
            return sig, [tokenizer.decode([n]) for n in self.nodes]

class Compiler:
    def __init__(self, tokenizer: ModTokenizer | None = None):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile KScript file."""

    def compile_script(self, script: Script, is_top_level: bool) -> None:
        """Compile a script, adding entries."""
        # If script has no constructs, emit identity: {sig: None}
```

### Phase 6: Output (`output.py`)

```python
import struct
from pathlib import Path

def write_jsonl(entries: list[CompiledEntry], path: Path, tokenizer: ModTokenizer) -> None:
    """Write entries to JSONL file.

    Each entry becomes a separate line:
    {"L": ["M"]}
    {"L": ["O"]}
    """
    with open(path, 'w') as f:
        for entry in entries:
            sig, nodes = entry.decode(tokenizer)
            f.write(json.dumps({sig: nodes}) + '\n')

def write_bin(entries: list[CompiledEntry], path: Path) -> None:
    """Write entries to binary file.

    Binary format:
    - Header: 4 bytes magic "KSC1"
    - Entry count: 4 bytes (uint32, little-endian)
    - Per entry:
      - signature: 8 bytes (uint64, little-endian)
      - node_type: 1 byte (0=None, 1=int, 2=list)
      - if int: 8 bytes (uint64)
      - if list: 4 bytes count + N*8 bytes (uint64 each)
    """
    with open(path, 'wb') as f:
        f.write(b'KSC1')
        f.write(struct.pack('<I', len(entries)))
        for entry in entries:
            f.write(struct.pack('<Q', entry.signature))
            if entry.nodes is None:
                f.write(struct.pack('<B', 0))
            elif isinstance(entry.nodes, int):
                f.write(struct.pack('<B', 1))
                f.write(struct.pack('<Q', entry.nodes))
            else:
                f.write(struct.pack('<B', 2))
                f.write(struct.pack('<I', len(entry.nodes)))
                for node_id in entry.nodes:
                    f.write(struct.pack('<Q', node_id))

def read_bin(path: Path) -> list[CompiledEntry]:
    """Read entries from binary file."""
    entries = []
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'KSC1':
            raise ValueError(f"Invalid magic: {magic}")
        count = struct.unpack('<I', f.read(4))[0]
        for _ in range(count):
            sig = struct.unpack('<Q', f.read(8))[0]
            node_type = struct.unpack('<B', f.read(1))[0]
            if node_type == 0:
                nodes = None
            elif node_type == 1:
                nodes = struct.unpack('<Q', f.read(8))[0]
            else:
                list_len = struct.unpack('<I', f.read(4))[0]
                nodes = [struct.unpack('<Q', f.read(8))[0] for _ in range(list_len)]
            entries.append(CompiledEntry(signature=sig, nodes=nodes))
    return entries
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
from kalvin.mod_tokenizer import ModTokenizer, Mod32Tokenizer

class KScript:
    """Compiled KScript model supporting incremental construction."""

    def __init__(self, source: str | Path, base: KScript | None = None, tokenizer: ModTokenizer | None = None):
        """Compile KScript from string or file path.

        Args:
            source: KScript source string or path to file.
                    Supported: .ks, .json, .jsonl, .bin
            base: Optional existing KScript to extend
            tokenizer: Tokenizer for encoding (default: ModTokenizer)
        """
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self._entries: list[CompiledEntry] = []

        path = Path(source)
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".bin":
                self._entries = read_bin(path)
            elif suffix in (".json", ".jsonl"):
                self._entries = self._read_json(path)
            elif suffix == ".ks":
                self._compile(path.read_text())
        else:
            # Treat as inline source
            self._compile(str(source))

        if base:
            self._entries = base._entries + self._entries

    def _read_json(self, path: Path) -> list[CompiledEntry]:
        """Read JSON/JSONL and encode to CompiledEntry."""
        # ... decode JSON strings to token IDs
        pass

    @property
    def entries(self) -> list[CompiledEntry]:
        """Return all compiled entries."""
        ...

    def output(self, path: str | Path) -> None:
        """Write entries to file based on suffix.

        Suffix determines format:
        - .json: JSON array
        - .jsonl: JSONL (line-delimited)
        - .bin: Binary format
        """
        path = Path(path)
        if path.suffix == ".bin":
            write_bin(self._entries, path)
        elif path.suffix == ".jsonl":
            write_jsonl(self._entries, path, self.tokenizer)
        else:
            # JSON array format
            ...

    def to_model(self) -> list[dict[str, str | None | list[str]]]:
        """Return entries as list of dicts (preserves duplicate signatures)."""
        ...

```

## Files to Create

| File                      | Purpose                             |
| ------------------------- | ----------------------------------- |
| `src/kscript/__init__.py` | Module exports + KScript API        |
| `src/kscript/__main__.py` | CLI entry point                     |
| `src/kscript/token.py`    | Token types and dataclass           |
| `src/kscript/lexer.py`    | Lexer with indentation              |
| `src/kscript/ast.py`      | AST node definitions                |
| `src/kscript/parser.py`   | Recursive descent parser            |
| `src/kscript/compiler.py` | AST to compiled entries (tokenized) |
| `src/kscript/output.py`   | JSON/JSONL/binary reader/writer     |
| `tests/test_kscript.py`   | Test suite                          |

## Dependencies

- `kalvin.mod_tokenizer.ModTokenizer` - Tokenizer for encoding signatures to token IDs

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
{"M": null}
{"V": "H"}
{"H": null}
{"O": "ALL"}
{"ALL": null}
{"ALL": ["A", "L", "L"]}
{"A": "D"}
{"D": null}
{"L": "M"}
{"M": null}
{"L": ["O"]}
{"O": null}
{"X": null}
```

## Usage

```bash
# Compile and output to terminal
python -m kscript script.ks

# Compile to JSON (array format, preserves duplicates)
python -m kscript script.ks -out output.json

# Compile to JSONL (line-delimited)
python -m kscript script.ks -out output.jsonl

# Compile to binary format (tokenized)
python -m kscript script.ks -out output.bin

# Load binary model
python -m kscript model.bin

# Extend existing model from JSON/JSONL/binary
python -m kscript script.ks -base base.json -out extended.json

# Load and extend JSON/JSONL models
python -m kscript model.json -base base.jsonl -out merged.json
```

### API Usage Examples

In addition to CLI usage, the compiler can be used programmatically:

```python
from kscript import KScript

# Inline model generation from string
model1 = KScript("A == B")
# -> KScript object containing: [{sig: token_id_A, nodes: token_id_B}, ...]

# Load from JSON/JSONL file
model2 = KScript("path/to/model.json")

# Load from binary file (tokenized)
model3 = KScript("path/to/model.bin")

# Extend model with script
model4 = KScript("path/to/script.ks", model1)

# Save to JSON or JSONL (based on suffix)
model4.output("/path/to/output.json")

# Save to binary (tokenized)
model4.output("/path/to/output.bin")
```

```python
# Simple inline script
model = KScript("A == B C")
model.output("output.json")  # -> JSON array
model.output("output.jsonl") # -> JSONL
model.output("output.bin")   # -> Binary (tokenized)

# Chain multiple sources
base = KScript("A == B")
extended = KScript("C => A D", base)
extended.output("extended.bin")

# Load from binary
loaded = KScript("existing_model.bin")
extended = KScript("more.ks", base=loaded)

# Get model as list (preserves duplicates)
model.to_model()  # -> [{"A": "B"}, {"B": "A"}]

# Inspect entries (token IDs)
for entry in model.entries:
    sig, nodes = entry.decode(model.tokenizer)
    print(f"{sig}: {nodes}")
```
