# Spec: KScript Compiler

## Purpose

A compiler that transforms KScript source files (`.ks`) into KLine graph representations. KScript is a domain-specific language for defining knowledge graph relationships using a concise, pythonesque indentation-based syntax.

## Requirements

### Requirement: Module Structure
The compiler SHALL be organized as a Python package with the following structure:

```
src/kscript/
├── __init__.py        # Module exports + KScript API class
├── __main__.py        # CLI entry point
├── token.py           # TokenType enum and Token dataclass
├── lexer.py           # Lexer with indentation support
├── ast.py             # AST node definitions
├── parser.py          # Recursive descent parser
├── compiler.py        # Compiles AST to CompiledEntry list
└── output.py          # JSON/JSONL/binary I/O
```

#### Scenario: Package imports
- **WHEN** a user imports `from kscript import KScript`
- **THEN** the KScript API class is available
- **AND** all public types (CompiledEntry, Lexer, Parser, Compiler) are accessible

---

### Requirement: Token Types
The system SHALL define the following token types:

| Type | Pattern | Description |
|------|---------|-------------|
| COUNTERSIGN | `==` | Bidirectional link |
| CANONIZE_FWD | `=>` | Forward canonization |
| CANONIZE_BWD | `<=` | Backward canonization |
| CONNOTATE_FWD | `>` | Forward connotation |
| CONNOTATE_BWD | `<` | Backward connotation |
| UNDERSIGN | `=` | Undersign link |
| SIGNATURE | `[A-Z]+` | Uppercase identifier |
| STRING | `"..."` | Double-quoted string |
| NUMBER | `[0-9]+` | Numeric literal |
| COMMENT | `(...)` | Parenthesized comment |
| NEWLINE | `\n` | Line ending |
| INDENT | - | Increased indentation |
| DEDENT | - | Decreased indentation |
| EOF | - | End of file |

#### Scenario: Token dataclass
- **WHEN** a token is created
- **THEN** it has `type` (TokenType), `value` (str), `line` (int), `column` (int)

---

### Requirement: Lexer - Multi-character Operators
The lexer SHALL tokenize multi-character operators before single-character ones.

#### Scenario: Tokenize equality operators
- **WHEN** lexer encounters `==`
- **THEN** it produces a single COUNTERSIGN token (not two UNDERSIGN tokens)

#### Scenario: Tokenize arrow operators
- **WHEN** lexer encounters `=>` or `<=`
- **THEN** it produces CANONIZE_FWD or CANONIZE_BWD respectively

---

### Requirement: Lexer - Signatures
The lexer SHALL tokenize uppercase letter sequences as SIGNATURE tokens.

#### Scenario: Single-letter signature
- **WHEN** lexer encounters `A`
- **THEN** it produces SIGNATURE token with value `"A"`

#### Scenario: Multi-letter signature
- **WHEN** lexer encounters `MHALL`
- **THEN** it produces SIGNATURE token with value `"MHALL"`

#### Scenario: Signature with inline comment
- **WHEN** lexer encounters `S(ubject)`
- **THEN** it produces SIGNATURE token with value `"S"`
- **AND** the comment `(ubject)` is consumed but not attached

---

### Requirement: Lexer - String Literals
The lexer SHALL tokenize double-quoted strings with escape sequence support.

#### Scenario: Simple string
- **WHEN** lexer encounters `"hello"`
- **THEN** it produces STRING token with value `'"hello"'` (including quotes)

#### Scenario: String with escape
- **WHEN** lexer encounters `"\"quoted\""`
- **THEN** it produces STRING token with value `'"\"quoted\""'`

#### Scenario: Unterminated string
- **WHEN** lexer encounters `"no closing quote` followed by newline
- **THEN** it produces STRING token with value `'"no closing quote'`

---

### Requirement: Lexer - Comments
The lexer SHALL tokenize parenthesized comments with greedy matching.

#### Scenario: Simple comment
- **WHEN** lexer encounters `(this is a comment)`
- **THEN** it produces COMMENT token with value `(this is a comment)`

#### Scenario: Nested comment
- **WHEN** lexer encounters `(comment (nested))`
- **THEN** it produces COMMENT token with value `(comment (nested))`

#### Scenario: Unterminated comment
- **WHEN** lexer encounters `(no closing` followed by newline
- **THEN** it produces COMMENT token with value `(no closing`

#### Scenario: Empty comment
- **WHEN** lexer encounters `()`
- **THEN** it produces COMMENT token with value `()`

---

### Requirement: Lexer - Indentation
The lexer SHALL emit INDENT and DEDENT tokens based on indentation changes using a stack-based approach.

#### Scenario: Increased indentation
- **WHEN** a line has more leading whitespace than the previous
- **THEN** an INDENT token is emitted before that line's content

#### Scenario: Decreased indentation
- **WHEN** a line has less leading whitespace than the previous
- **THEN** one or more DEDENT tokens are emitted to match the indentation stack

#### Scenario: Same indentation
- **WHEN** a line has the same indentation as the previous
- **THEN** no INDENT or DEDENT tokens are emitted

#### Scenario: End of file dedents
- **WHEN** lexer reaches end of file with remaining indent stack depth > 1
- **THEN** DEDENT tokens are emitted to close all levels

---

### Requirement: AST - Node Types
The system SHALL define the following AST node types:

```python
@dataclass
class Signature:
    id: str              # A-Z+ identifier
    comment: str | None  # optional (...)
    line: int
    column: int

@dataclass
class StringLiteral:
    id: str              # includes quotes
    line: int
    column: int

@dataclass
class NumberLiteral:
    id: str              # string representation
    line: int
    column: int

Node = Signature | StringLiteral | NumberLiteral
```

---

### Requirement: AST - Construct Types
The system SHALL define a ConstructType enum:

| Type | Operator | Description |
|------|----------|-------------|
| COUNTERSIGN | `==` | Bidirectional signature link |
| CANONIZE_FWD | `=>` | Forward multi-node composition |
| CANONIZE_BWD | `<=` | Backward multi-node composition |
| CONNOTATE_FWD | `>` | Forward single-node annotation |
| CONNOTATE_BWD | `<` | Backward single-node annotation |
| UNDERSIGN | `=` | Unidirectional signature link |

---

### Requirement: AST - Construct Node
The system SHALL define a Construct dataclass:

```python
@dataclass
class Construct:
    type: ConstructType
    nodes: list[Node]
    line: int
    has_leading_nodes: bool = False  # True for backward canonize with nodes before operator
```

---

### Requirement: AST - Script Node
The system SHALL define a Script dataclass representing a complete script:

```python
@dataclass
class Script:
    signature: Signature
    constructs: list[Construct]
    subscripts: list["Script"]  # Nested scripts
    line: int
```

---

### Requirement: AST - File Node
The system SHALL define a KScriptFile dataclass:

```python
@dataclass
class KScriptFile:
    scripts: list[Script]
```

---

### Requirement: Parser - Multiple Scripts
The parser SHALL support multiple top-level scripts in a single file, each starting at column 1.

#### Scenario: Parse multiple scripts
- **WHEN** parser processes `A\nB\nC`
- **THEN** the KScriptFile contains 3 Script nodes

---

### Requirement: Parser - Top-level Scripts
The parser SHALL only start a new script when a SIGNATURE token appears at the outermost indentation level (column 1).

#### Scenario: Script at column 1
- **WHEN** parser encounters a SIGNATURE at indent level 0
- **THEN** a new top-level Script is created

---

### Requirement: Parser - Immediate Binding
The parser SHALL implement immediate binding semantics where each construct binds to the most recent signature.

#### Scenario: Chained constructs
- **WHEN** parser processes `A => B => C`
- **THEN** first construct binds A to B
- **AND** second construct binds B to C

---

### Requirement: Parser - Backward Canonize with Leading Nodes
The parser SHALL support backward canonize patterns where nodes appear before the `<=` operator.

#### Scenario: Multiple nodes before backward canonize
- **WHEN** parser processes `B C D <= A`
- **THEN** a CANONIZE_BWD construct is created with nodes [B, C, D, A]
- **AND** `has_leading_nodes` is True

---

### Requirement: Parser - Subscripts
The parser SHALL recursively parse indented blocks as subscripts.

#### Scenario: Simple subscript
- **WHEN** parser processes:
  ```
  A =>
    B
    C
  ```
- **THEN** Script A has 2 subscripts (B and C)
- **AND** B and C each have no constructs (identity scripts)

#### Scenario: Nested subscripts
- **WHEN** parser processes:
  ```
  A =>
    B =>
      C
  ```
- **THEN** Script A has 1 subscript (B)
- **AND** Script B has 1 subscript (C)

---

### Requirement: Compiler - Identity Semantics
The compiler SHALL emit an identity entry `{sig: None}` for scripts with no valid constructs.

#### Scenario: Standalone signature
- **WHEN** compiler processes `A`
- **THEN** output contains `{"A": null}`

#### Scenario: Incomplete construct
- **WHEN** compiler processes `A ==`
- **THEN** output contains `{"A": null}` (operator ignored)

---

### Requirement: Compiler - Countersign Semantics
The compiler SHALL emit bidirectional entries for countersign constructs with signature nodes.

#### Scenario: Countersign with signature
- **WHEN** compiler processes `A == B`
- **THEN** output contains `{"A": "B"}` AND `{"B": "A"}`

#### Scenario: Countersign with literal (recovery)
- **WHEN** compiler processes `A == 1`
- **THEN** output contains `{"A": "1"}` only (no reverse entry for literals)

---

### Requirement: Compiler - Canonize Semantics
The compiler SHALL emit multi-node entries for canonize constructs.

#### Scenario: Forward canonize
- **WHEN** compiler processes `AB => C D`
- **THEN** output contains `{"AB": ["C", "D"]}`

#### Scenario: Backward canonize with trailing node
- **WHEN** compiler processes `X <= A B`
- **THEN** output contains `{"B": ["A"]}` (last node is parent, rest are children)

#### Scenario: Backward canonize with leading nodes
- **WHEN** compiler processes `B C D <= A`
- **THEN** output contains `{"A": ["B", "C", "D"]}`

---

### Requirement: Compiler - Connotate Semantics
The compiler SHALL emit entries with identity for the node.

#### Scenario: Forward connotate
- **WHEN** compiler processes `A > B`
- **THEN** output contains `{"A": ["B"]}` AND `{"B": null}`

#### Scenario: Backward connotate
- **WHEN** compiler processes `A < B`
- **THEN** output contains `{"B": ["A"]}` AND `{"A": null}`

---

### Requirement: Compiler - Undersign Semantics
The compiler SHALL emit unidirectional entries with identity for the node.

#### Scenario: Undersign with signature
- **WHEN** compiler processes `A = B`
- **THEN** output contains `{"A": "B"}` AND `{"B": null}`

#### Scenario: Undersign with literal
- **WHEN** compiler processes `A = "hello"`
- **THEN** output contains `{"A": "\"hello\""}` AND no reverse entry

---

### Requirement: Compiler - Subscripts as Nodes
The compiler SHALL use subscript signatures as nodes when a construct has no inline nodes.

#### Scenario: Canonize with subscript nodes
- **WHEN** compiler processes:
  ```
  A =>
    B
    C
  ```
- **THEN** output contains `{"A": ["B", "C"]}`
- **AND** `{"B": null}` AND `{"C": null}` (identity for subscripts)

---

### Requirement: Compiler - Literal Encoding
The compiler SHALL encode literals (strings, numbers) without the PACKED_BIT to distinguish them from signatures.

#### Scenario: String literal encoding
- **WHEN** compiler encodes `"hello"` as a node
- **THEN** each character is encoded as `(ord(char) << 1)` with bit 0 clear

#### Scenario: Number literal encoding
- **WHEN** compiler encodes `42` as a node
- **THEN** each digit character is encoded as `(ord(digit) << 1)`

---

### Requirement: Compiler - Duplicate Signatures
The compiler SHALL preserve duplicate signatures as separate entries (never merge).

#### Scenario: Multiple entries same signature
- **WHEN** compiler processes:
  ```
  A > B
  A > C
  ```
- **THEN** output contains both `{"A": ["B"]}` AND `{"A": ["C"]}`

---

### Requirement: CompiledEntry - Decode
The CompiledEntry class SHALL support decoding token IDs back to strings.

#### Scenario: Decode signature entry
- **WHEN** entry has signature with PACKED_BIT set
- **THEN** decode returns the signature string

#### Scenario: Decode literal entry
- **WHEN** entry has nodes with PACKED_BIT clear
- **THEN** decode returns the literal string (characters shifted right)

---

### Requirement: Output - JSON Format
The output module SHALL support writing entries as a JSON array.

#### Scenario: Write JSON
- **WHEN** `write_json(entries, path, tokenizer)` is called
- **THEN** file contains a JSON array of entry objects
- **AND** each entry is `{"sig": nodes}` where nodes is null, string, or array

---

### Requirement: Output - JSONL Format
The output module SHALL support writing entries as JSONL (line-delimited JSON).

#### Scenario: Write JSONL
- **WHEN** `write_jsonl(entries, path, tokenizer)` is called
- **THEN** file contains one JSON object per line
- **AND** each line is `{"sig": nodes}`

---

### Requirement: Output - Binary Format
The output module SHALL support writing entries in a compact binary format.

#### Scenario: Binary format header
- **WHEN** `write_bin(entries, path)` is called
- **THEN** file starts with 4-byte magic: `KSC1`
- **AND** followed by 4-byte entry count (uint32, little-endian)

#### Scenario: Binary entry encoding
- **WHEN** writing an entry
- **THEN** signature is 8 bytes (uint64)
- **AND** node_type is 1 byte (0=None, 1=int, 2=list)
- **AND** if int: 8 bytes (uint64)
- **AND** if list: 4-byte count + N*8 bytes

---

### Requirement: Output - Binary Reading
The output module SHALL support reading binary format files.

#### Scenario: Read binary
- **WHEN** `read_bin(path)` is called with valid KSC1 file
- **THEN** list of CompiledEntry objects is returned

---

### Requirement: Output - JSON/JSONL Reading
The output module SHALL support reading JSON and JSONL files.

#### Scenario: Read JSON
- **WHEN** `read_json(path, tokenizer)` is called
- **THEN** entries are decoded from JSON objects
- **AND** strings are encoded to token IDs

---

### Requirement: KScript API - Construction
The KScript class SHALL accept multiple input types.

#### Scenario: Construct from string
- **WHEN** `KScript("A == B")` is called
- **THEN** the source is compiled to entries

#### Scenario: Construct from .ks file
- **WHEN** `KScript("path/to/script.ks")` is called with existing file
- **THEN** file contents are compiled to entries

#### Scenario: Construct from .json/.jsonl file
- **WHEN** `KScript("model.json")` is called
- **THEN** entries are loaded from JSON

#### Scenario: Construct from .bin file
- **WHEN** `KScript("model.bin")` is called
- **THEN** entries are loaded from binary format

---

### Requirement: KScript API - Extension
The KScript class SHALL support extending an existing model.

#### Scenario: Extend model
- **WHEN** `KScript("C == D", base=existing_model)` is called
- **THEN** new entries are appended to base entries

---

### Requirement: KScript API - Output
The KScript class SHALL support output to different formats based on file suffix.

#### Scenario: Output JSON
- **WHEN** `model.output("out.json")` is called
- **THEN** file is written as JSON array

#### Scenario: Output JSONL
- **WHEN** `model.output("out.jsonl")` is called
- **THEN** file is written as JSONL

#### Scenario: Output binary
- **WHEN** `model.output("out.bin")` is called
- **THEN** file is written in binary format

---

### Requirement: CLI - Basic Compilation
The CLI SHALL compile .ks files to output formats.

#### Scenario: Default output
- **WHEN** `python -m kscript script.ks` is called
- **THEN** output is written to `script.jsonl`

#### Scenario: Specified output
- **WHEN** `python -m kscript script.ks -out custom.json` is called
- **THEN** output is written to `custom.json`

---

### Requirement: Error Recovery
The compiler SHALL recover from syntax errors and produce valid output.

#### Scenario: Unsupported literal in signature position
- **WHEN** compiler processes `1 > A`
- **THEN** output contains `{"1": null}` (recover to identity)

#### Scenario: Unexpected indent
- **WHEN** compiler encounters unexpected indentation
- **THEN** processing continues with previous valid state

---

### Requirement: Default Tokenizer
The KScript compiler SHALL use Mod32Tokenizer as the default tokenizer for signature encoding.

#### Scenario: KScript API default tokenizer
- **WHEN** `KScript("A == B")` is called without tokenizer argument
- **THEN** Mod32Tokenizer is used for encoding
- **AND** signatures are encoded in bits 1-32

#### Scenario: Compiler default tokenizer
- **WHEN** `Compiler()` is instantiated without tokenizer argument
- **THEN** Mod32Tokenizer is used for encoding

#### Scenario: CLI default tokenizer
- **WHEN** `python -m kscript script.ks` is called
- **THEN** Mod32Tokenizer is used for encoding
- **AND** output binary files use 32-bit character encoding

---

### Requirement: Tokenizer Override
The system SHALL allow explicit tokenizer override when needed.

#### Scenario: Custom tokenizer in API
- **WHEN** `KScript("A == B", tokenizer=custom_tokenizer)` is called
- **THEN** the provided tokenizer is used instead of the default

#### Scenario: Custom tokenizer in Compiler
- **WHEN** `Compiler(tokenizer=custom_tokenizer)` is instantiated
- **THEN** the provided tokenizer is used for compilation

---

## Examples

### Example: Complete Compilation

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

**Output** (decoded):
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
