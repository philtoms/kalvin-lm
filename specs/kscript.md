# KScript Language Specification

**Status:** Authoritative reference for KScript implementation  
**Builds on:**
- `docs/kscript-intro.md` — what KScript is, the six operators, and their role in training
- `docs/tokenizer-significance.md` — encoding: packed vs. literal nodes, `is_literal()`, bit-level algebra
- `docs/learning-and-training.md` — significance levels S1–S4, ratification, the training loop

---

## 1. Compilation Pipeline

```
Source (.ks) → Lexer → Token Stream → Parser → AST → Compiler → [CompiledEntry]
                                                                        ↓
                                                            KLine knowledge graph entries
```

Strictly one-directional. A decompiler exists for best-effort source reconstruction from compiled klines.

**Compilation target:** Each compiled entry is a `KLine` (`specs/kline.md`) — an identified, ordered sequence of `uint64` nodes, where the signature is the OR-reduction of nodes (`specs/signature.md`) and encoding is provided by the Mod tokenizer (`specs/tokenizer.md`).

---

## 2. Lexical Structure

### 2.1 Tokens

| Token | Pattern | Category |
|-------|---------|----------|
| `SIGNATURE` | `[A-Z]+` | Node (uppercase alpha only) |
| `LITERAL` | `[0-9]+` or `"..."` | Node (non-uppercase) |
| `COUNTERSIGN` | `==` | Inline operator |
| `CANONIZE_FWD` | `=>` | Chain operator |
| `CANONIZE_BWD` | `<=` | Chain operator |
| `CONNOTATE_FWD` | `>` | Inline operator (always; see §2.2) |
| `CONNOTATE_BWD` | `<` | Chain operator |
| `UNDERSIGN` | `=` | Inline operator |
| `COMMENT` | `(...)` with nested parens | Insignificant |
| `NEWLINE` | `\n` | Insignificant |
| `INDENT` | Increased indentation | Structure |
| `DEDENT` | Decreased indentation | Structure |
| `EOF` | End of file | Sentinel |

### 2.2 Lexing Rules

1. **Multi-character operators** (`==`, `=>`, `<=`) matched before single-character (`=`, `>`, `<`).
2. **Identifiers** `[A-Z][A-Z0-9]*` classified as `SIGNATURE` only if all chars are uppercase alpha (`isupper() and isalpha()`). Non-uppercase identifiers are invalid — use quoted strings.
3. **Numbers** `[0-9]+` → `LITERAL`.
4. **Quoted strings** `"..."` support backslash escapes → `LITERAL`. Unterminated strings stop at newline.
5. **Comments** `(...)` support nested parens, may span lines. Inline comments after an identifier are consumed without emitting a token.
6. **Indentation** uses Python-style INDENT/DEDENT tokens based on leading whitespace. Each space/tab counts as one unit.
7. **Unknown characters** raise `LexerError`.

### 2.3 Operator Placement

**Inline operators** appear within a primary construct, between signature and node:
`==` (COUNTERSIGN), `>` (CONNOTATE_FWD), `=` (UNDERSIGN)

**Chain operators** appear between construct groups, introducing a right-hand side:
`=>` (CANONIZE_FWD), `<=` (CANONIZE_BWD), `<` (CONNOTATE_BWD)

> `>` is classified as both inline and chain in the grammar, but the parser **always greedily consumes it as inline** after any primary construct. The chain-`>` code path exists in the compiler but is unreachable from normal parsing.

---

## 3. Grammar

```
script      ::= construct+
construct   ::= block | literal | primary_construct+ ( chain_op construct )?
block       ::= INDENT construct+ DEDENT
primary_construct ::= sig ( inline_op node )?
node        ::= sig | literal
sig         ::= SIGNATURE
literal     ::= LITERAL
chain_op    ::= CANONIZE_FWD | CANONIZE_BWD | CONNOTATE_FWD | CONNOTATE_BWD
inline_op   ::= COUNTERSIGN | CONNOTATE_FWD | UNDERSIGN
```

### Constraints

- Only `SIGNATURE` can own constructs. `LITERAL` in construct position is a bare unsigned identity — no inline or chain operators.
- `NEWLINE` and `COMMENT` are skipped between constructs.
- Multiple `primary_construct` at the same indentation form an implicit group (used for BWD chain emission).
- Empty source produces empty output (no error).

### Parse Errors

| Situation | Error |
|-----------|-------|
| `LITERAL` in position requiring `SIGNATURE` | `ParseError` |
| `1 => A` — literal owning a chain | `ParseError` |
| `A => 1 => B` — chaining through a literal | `ParseError` |

---

## 4. AST

```
KScriptFile
  └── scripts: [Script]
        └── constructs: [Construct]
              ├── inner: Block | Literal | [PrimaryConstruct]
              ├── chain_op: TokenType?
              └── chain_right: Construct?

Block
  └── constructs: [Construct]

PrimaryConstruct
  ├── sig: Signature (id, line, column)
  ├── op: TokenType?
  └── node: Node? (Signature | Literal)

Signature  — id: str (uppercase)
Literal    — id: str (any value)
```

---

## 5. Compilation

### 5.1 CompiledEntry

Extends `KLine` with encode/decode:

```python
class CompiledEntry(KLine):
    @classmethod
    def encode(cls, sig, nodes, tokenizer, *, sig_level, significance, dbg_text) -> CompiledEntry
    def decode(self, tokenizer) -> tuple[str, str | None | list[str]]
```

**Singleton rule:** `nodes` list with exactly one element is unwrapped to a single `uint64`. Empty nodes become `None`.

### 5.2 Encoding

Via the Mod tokenizer (`specs/tokenizer.md`):

- **Signatures** (uppercase alpha): *packed* — characters OR'd into one `uint64` with bit 0 clear. Order and multiplicity are lost.
- **Literals** (everything else): *literal* — one `uint64` per character with lower 32 bits = `0xFFFFFFFF`. Order and identity preserved.

Literal test: `(node & 0xFFFFFFFF) == 0xFFFFFFFF` — the standalone `is_literal()` function.

### 5.3 MCS (Multi-Character Signature) Expansion

When a signature has **more than one character**, the compiler automatically emits:

1. **Component identities:** One unsigned entry per character: `{A: None}, {B: None}, {C: None}`
2. **MCS canonization:** `{ABC: [A, B, C]}` (CANONIZE_FWD)
3. **Unsigned compound:** `{ABC: None}`

Single-character signatures do **not** trigger MCS expansion.

### 5.4 Deduplication

Entries are deduplicated by `(signature, nodes)` pair. Second emission of the same pair is silently dropped.

### 5.5 Per-Operator Compilation

#### Unsigned (bare)
```
A       → {A: None}        (plus MCS if multi-char)
hello   → {hello: None}
```

#### COUNTERSIGN (`==`)
```
A == B  → {A: B}, {B: A}   (bidirectional)
```
MCS applied to node if signature. Reverse direction NOT emitted for literal nodes.

#### UNDERSIGN (`=`)
```
A = B   → {A: B}           (unidirectional)
A = A   → {A: None}        (self-identity collapses to unsigned)
```

#### CONNOTATE_FWD (`>` inline)
```
A > B   → {A: B}           (unidirectional)
```

#### CANONIZE_FWD (`=>` chain)
Owner is the last primary construct's node (if present), or its signature. One entry **per right-hand item**:
```
A => B C       → {A: B}, {A: C}
A > X => B C   → {A: X}, {X: B}, {X: C}    (owner is X, the node)
```
Right-hand construct compiled recursively.

#### CANONIZE_BWD (`<=` chain)
One entry per left-side primary construct:
```
A B <= CD  → {CD: A}, {CD: B}
```
Right-hand construct compiled recursively.

#### CONNOTATE_BWD (`<` chain)
First right-hand item becomes the owner; last left-side owner is the node:
```
A B < C    → {C: last_owner_of(A B)}
```

### 5.6 Subscript Blocks

Indented block after a chain operator is flattened to extract all items. Per-item emission applies. Block items compiled recursively.

### 5.7 Significance Level Assignment

| Operator | Level | Internal Voice |
|----------|-------|---------------|
| COUNTERSIGN (`==`) | S1 | "I know that I know this." |
| UNDERSIGN (`=`) | S1 | "I know that I know this." |
| CANONIZE_FWD (`=>`) | S2 | "I understand some of it." |
| CANONIZE_BWD (`<=`) | S2 | "I understand some of it." |
| CONNOTATE_FWD (`>`) | S3 | "I recognise aspects of it." |
| CONNOTATE_BWD (`<`) | S3 | "I recognise aspects of it." |
| UNSIGNED (bare) | S4 | "I do not understand this at all." |

> Significance bits are NOT encoded into token IDs during compilation. The level is used for debug text and decompiler inference. Significance is a runtime property of Kalvin's rationalisation pipeline, not a compiled attribute.

---

## 6. Decompilation

Best-effort source reconstruction from compiled klines. **Lossy** — significance levels are heuristically inferred, not stored.

### 6.1 MCS Name Recovery

Packed encoding loses character order. The decompiler detects MCS entries — klines where `signature == OR(node_values)` — to recover names. Two patterns:

1. **Legacy:** Single entry with multiple packed single-char nodes
2. **Per-char:** Consecutive entries with same signature, each with one packed single-char node

### 6.2 Level Inference

| Condition | Inferred Level |
|-----------|---------------|
| No nodes | S4 |
| `(sig & node) != 0` | S2 |
| `(sig & node) == 0` | S3 |
| MCS entry (sig == OR of nodes) | S2 |

> S1 cannot be reliably distinguished from S3 without explicit significance bits.

---

## 7. File Formats

### 7.1 Source (`.ks`)
Plain text.

### 7.2 JSON (`.json`)
```json
[{"MHALL": "SVO"}, {"SVO": "MHALL"}, {"ABC": ["A", "B", "C"]}]
```
Each entry: `{"sig": nodes}` where `nodes` is `null`, a string, or a list of strings.

### 7.3 JSONL (`.jsonl`)
One JSON object per line (same format as JSON array items).

### 7.4 Binary (`.bin`) — KSC1 format, little-endian
```
Header:
  4 bytes: magic "KSC1"
  4 bytes: entry count (uint32)

Per entry:
  8 bytes: signature (uint64)
  1 byte:  node_type (0=None, 1=int, 2=list)
  if type==1: 8 bytes (uint64 node)
  if type==2: 4 bytes count + N × 8 bytes (uint64 each)
```

---

## 8. Public API

### Python

```python
from kscript import KScript

model = KScript("A == B")                    # from source string
model = KScript("script.ks")                 # from file (.ks, .json, .jsonl, .bin)
extended = KScript("C = D", base=model)      # extend existing model
model.output("output.json")                  # output (format by suffix)
lines = model.to_jsonl()                     # JSONL lines

for entry in model.entries:
    sig, nodes = entry.decode(tokenizer)
```

### CLI

```bash
python -m kscript script.ks                        # → .jsonl (default)
python -m kscript script.ks -out output.json        # → .json
python -m kscript script.ks -out output.bin         # → .bin
python -m kscript script.ks -dev                    # include debug text
```

### Pipeline

```python
from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler, compile_source
from kscript.decompiler import Decompiler

tokens = Lexer(source).tokenize()
kfile = Parser(tokens).parse()
entries = Compiler(tokenizer, dev=True).compile(kfile)
decompiled = Decompiler(tokenizer).decompile(entries)
```

---

## 9. Module Structure

```
kscript/
├── __init__.py       # KScript class (public API), re-exports
├── __main__.py       # CLI entry point
├── token.py          # TokenType enum, Token dataclass
├── lexer.py          # Lexer (source → tokens)
├── ast.py            # AST node dataclasses
├── parser.py         # Parser (tokens → AST)
├── compiler.py       # Compiler (AST → [CompiledEntry])
├── decompiler.py     # Decompiler ([KLine] → [DecompiledEntry])
└── output.py         # JSON/JSONL/binary I/O
```

### Dependencies

| Dependency | Used By | Origin |
|------------|---------|--------|
| `kalvin.kline.KLine` | `CompiledEntry` base | `specs/kline.md` |
| `kalvin.mod_tokenizer.ModTokenizer` | Encoding/decoding | `specs/tokenizer.md` |
| `kalvin.mod_tokenizer.Mod32Tokenizer` | Default tokenizer | `specs/tokenizer.md` |
| `kalvin.signature.make_signature` | Decompiler MCS detection | `specs/signature.md` |
