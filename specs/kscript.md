# KScript Language Specification

**Version:** 2.0  
**Date:** 2026-04-29  
**Status:** Reverse-engineered from reference implementation

---

## 1. Overview

KScript is a domain-specific language for constructing **knowledge graphs** (ordered sequences of identified nodes called **KLines**). It compiles declarative scripts into a flat list of encoded graph entries that can be loaded into a **Kalvin Agent** for rationalisation.

### 1.1 Compilation Pipeline

```
Source (.ks) → Lexer → Token Stream → Parser → AST → Compiler → [CompiledEntry]
                                                                        ↓
                                                            KLine knowledge graph entries
```

The pipeline is strictly one-directional. There is also a **decompiler** that best-effort reconstructs KScript source from compiled KLines.

### 1.2 Core Concepts

| Concept                 | Description                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| **Node**                | An opaque `uint64` value — the universal atom                                    |
| **KLine**               | An identified, ordered sequence of zero or more nodes: `{signature, nodes[]}`    |
| **Signature**           | A `uint64` identity key computed via OR-reduction of nodes                       |
| **Significance**        | A four-level classification (S1–S4) of how strongly one KLine relates to another |
| **Signature (lexical)** | An uppercase identifier like `ABC` — becomes a packed node                       |
| **Literal (lexical)**   | Anything not uppercase alpha — numbers, quoted strings                           |

---

## 2. Lexical Structure

### 2.1 Token Types

| Token         | Pattern                    | Category                    |
| ------------- | -------------------------- | --------------------------- |
| `SIGNATURE`   | `[A-Z]+`                   | Node (uppercase identifier) |
| `LITERAL`     | `[0-9]+` or `"..."`        | Node (non-uppercase)        |
| `COUNTERSIGN` | `==`                       | Construct operator          |
| `CANONIZE`    | `=>`                       | Chain operator              |
| `CONNOTATE`   | `>`                        | Inline operator             |
| `UNDERSIGN`   | `=`                        | Construct operator          |
| `COMMENT`     | `(...)` with nested parens | Insignificant               |
| `NEWLINE`     | `\n`                       | Insignificant               |
| `INDENT`      | Increased indentation      | Structure                   |
| `DEDENT`      | Decreased indentation      | Structure                   |
| `EOF`         | End of file                | Sentinel                    |

### 2.2 Operator Classification

Operators are divided into two groups by where they may appear:

**Inline operators** (appear within a primary construct, between sig and node):

- `==` (COUNTERSIGN) — bidirectional link
- `>` (CONNOTATE) — connotation
- `=` (UNDERSIGN) — unconditional link

**Chain operators** (appear between construct groups, introducing a right-hand side):

- `=>` (CANONIZE) — canonization

### 2.3 Lexing Rules

1. **Multi-character operators** (`==`, `=>`) are matched before single-character operators (`=`, `>`).
2. **Identifiers** `[A-Z][A-Z0-9]*` are classified as:
   - `SIGNATURE` if all characters are uppercase alpha (`isupper() and isalpha()`)
   - Non-uppercase identifiers are **not valid** as literals — use quoted strings instead
3. **Numbers** `[0-9]+` are always `LITERAL`.
4. **Quoted strings** `"..."` support backslash escapes and are `LITERAL`. Unterminated strings stop at newline.
5. **Comments** `(...)` support nested parentheses and may span multiple lines.
6. **Inline comments** after an identifier are consumed without emitting a token.
7. **Indentation** uses Python-style INDENT/DEDENT tokens based on leading whitespace.
8. **Unknown characters** (not matching any rule, including `<`) raise a `LexerError`.

### 2.4 Indentation Rules

- Indentation is counted as the number of leading spaces/tabs.
- At each line start, the indent level is compared to the stack:
  - Greater → emit `INDENT`, push new level
  - Less → emit one or more `DEDENT` tokens, pop to matching level
  - Equal → no structural token
- At EOF, remaining levels are closed with `DEDENT` tokens.

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
chain_op    ::= CANONIZE
inline_op   ::= COUNTERSIGN | CONNOTATE | UNDERSIGN
```

### 3.1 Key Constraints

1. **Only SIGNATUREs can own constructs.** A `LITERAL` in construct position is a bare unsigned identity — it cannot have inline operators or chain operators.
2. **NEWLINE and COMMENT tokens are insignificant** — they are skipped between constructs.
3. **Multiple primary_constructs at the same indentation level** form an implicit group.
4. **Literal on the right of a chain operator** is valid: `A => 1` compiles, but `1 => A` is a parse error.

### 3.2 Parse Errors

| Situation                                                           | Error                            |
| ------------------------------------------------------------------- | -------------------------------- |
| `LITERAL` in position requiring `SIGNATURE` (e.g., construct owner) | `ParseError`                     |
| `1 => A` — literal owning a chain                                   | `ParseError`                     |
| `A => 1 => B` — chaining through a literal                          | `ParseError`                     |
| Empty source                                                        | Produces empty script (no error) |

---

## 4. AST Structure

```
KScriptFile
  └── scripts: [Script]
        └── constructs: [Construct]
              ├── inner: Block | Literal | [PrimaryConstruct]
              ├── chain_op: TokenType? (chain operator)
              └── chain_right: Construct? (right side of chain)

Block
  └── constructs: [Construct]

PrimaryConstruct
  ├── sig: Signature (id, line, column)
  ├── op: TokenType? (inline operator)
  └── node: Node? (Signature | Literal)

Signature  — id: str (uppercase)
Literal    — id: str (any value)
```

---

## 5. Compilation Semantics

### 5.1 Entry Model

Each compiled entry is a **KLine**:

```python
CompiledEntry:
    signature: uint64    # encoded identity key
    nodes: uint64 | list[uint64] | None  # encoded node values
    dbg_text: str        # debug label (dev mode)
```

**Singleton rule:** If `nodes` is a list with exactly one element, it is unwrapped to a single `uint64`. Empty nodes become `None`.

> **Implementation note:** `KLine.__init__` normalizes `nodes=None` to `nodes=[]`. Code that inspects compiled entries should treat an empty node list as unsigned (equivalent to `None`). The `decode()` method returns `""` for unsigned entries due to this normalization.

### 5.2 Encoding

Strings are encoded to `uint64` values via a **Mod Tokenizer**:

- **Signatures** (uppercase alpha): _packed_ encoding — characters are OR'd into a single `uint64` with bit 0 clear. Lossy: order and multiplicity are lost.
  - `encode("ABC") → [bit_A | bit_B | bit_C]`
- **Literals** (everything else): _literal_ encoding — one `uint64` per character with lower 32 bits = `0xFFFFFFFF`. Preserves order and identity.
  - `encode("hello") → [(104<<32)|0xFFFFFFFF, (101<<32)|0xFFFFFFFF, ...]`

**Literal test:** `(node & 0xFFFFFFFF) == 0xFFFFFFFF`

> **Note:** This is the standalone `is_literal` function defined in the
> @kline spec. It is no longer a tokenizer method.

### 5.3 MCS (Multi-Character Signature) Expansion

When a signature has **more than one character**, the compiler automatically emits:

1. **Component identities:** One unsigned entry per character.
   ```
   ABC → emits: {A: None}, {B: None}, {C: None}
   ```
2. **MCS canonization:** One entry mapping the compound to its components.
   ```
   ABC → emits: {ABC: [A, B, C]}  (CANONIZE)
   ```
3. **Unsigned compound:** The compound itself as unsigned.
   ```
   ABC → emits: {ABC: None}
   ```

Single-character signatures do NOT trigger MCS expansion.

### 5.4 Deduplication

The compiler deduplicates entries by `(signature, nodes)` pair. If the same encoded entry would be emitted twice, the second is silently dropped.

### 5.5 Construct Compilation Rules

#### Unsigned (bare signature or literal)

```
A       → {A: None}        (plus MCS if multi-char)
hello   → {hello: None}
```

#### COUNTERSIGN (`==`)

```
A == B  → {A: B}, {B: A}   (bidirectional)
```

If the node is a signature, MCS expansion is applied to it too. Reverse direction is NOT emitted for literal nodes.

#### UNDERSIGN (`=`)

```
A = B   → {B: A}           (unidirectional, value becomes signature)
A = A   → {A: None}        (self-identity collapsed to unsigned)
```

Self-identity (`A = A`) is emitted with the internal op name `IDENTITY` (level S1), which collapses to an unsigned entry.

#### CONNOTATE (`>`)

```
A > B   → {A: B}           (unidirectional)
```

#### CANONIZE chain (`=>`)

The **owner** is the last primary construct's node (if present), or its signature. A single entry is emitted with **all right-hand items** as the node list:

```
A => B C       → {A: [B, C]}              (single entry, all items)
A > X => B C   → {A: X}, {X: [B, C]}      (owner is X, the node)
A = X => B C   → {X: A}, {X: [B, C]}      (owner is X, the node)
```

Singleton unwrapping: if there is only one right-hand item, the list is unwrapped: `A => B → {A: B}`.

The right-hand construct is then compiled recursively.

### 5.6 Subscript Blocks

A subscript block (indented construct after a chain operator) is flattened to extract all items:

```
A =>
  B
  C = D
```

Flattens to items `[B, C = D]`. Then CANONIZE emits a single entry with all items:

- `{A: [B, C]}` plus recursive compilation of C's inline op: `{D: C}`

Blocks may mix literals and primary constructs:

```
A =>
  1
  B
  "hello"
```

Flattens to `[Literal(1), PrimaryConstruct(B), Literal("hello")]` → single entry: `{A: [1, B, "hello"]}`

### 5.7 Significance Level Assignment

Each emitted entry is tagged with a significance level based on the operator:

| Operator           | Level | Meaning                |
| ------------------ | ----- | ---------------------- |
| COUNTERSIGN (`==`) | S1    | Mutual / bidirectional |
| UNDERSIGN (`=`)    | S1    | Unconditional          |
| CANONIZE (`=>`)    | S2    | Canonical              |
| CONNOTATE (`>`)    | S3    | Connotative            |
| UNSIGNED (bare)    | S4    | Identity only          |

> **Note:** In the current implementation, significance bits are NOT encoded into the token IDs during compilation. The level is used only for debug text and decompiler inference. The decompiler uses heuristic bit-overlap analysis to re-infer levels from compiled data.

---

## 6. Decompilation

The decompiler converts compiled KLines back to KScript entries:

### 6.1 MCS Name Recovery

Multi-character signatures encoded with Mod tokenizers lose character order (packed = OR of bits). The decompiler recovers names by detecting **MCS entries**:

- An MCS entry is a KLine where `signature == OR(node_values)`.
- MCS entries provide the name: the packed token maps to the original multi-char string.
- Two patterns are detected:
  1. **Legacy:** Single entry with multiple packed single-char nodes
  2. **Per-char:** Consecutive entries with same signature, each with one packed single-char node

### 6.2 Level Inference

Without explicit significance bits, levels are heuristically inferred:

| Condition                                     | Level                    |
| --------------------------------------------- | ------------------------ |
| No nodes                                      | S4 (unsigned)            |
| Single node with `(sig & node) != 0`          | S2 (canonize)            |
| Single node with `(sig & node) == 0`          | S3 (connotate/undersign) |
| Multi-node list with `(sig & nodes_sig) != 0` | S2 (canonize)            |
| Multi-node list with `(sig & nodes_sig) == 0` | S3 (connotate)           |
| MCS entry (sig == OR of nodes)                | S2                       |

> **Caveat:** S1 (countersign/undersign) cannot be reliably distinguished from S3 (connotate) without explicit significance bits, since singleton unwrapping collapses the structural distinction.

---

## 7. File Formats

### 7.1 Source Files (`.ks`)

Plain text files containing KScript source.

### 7.2 JSON Output (`.json`)

```json
[
  { "MHALL": "SVO" },
  { "SVO": "MHALL" },
  { "S": "M" },
  { "ABC": ["A", "B", "C"] }
]
```

Each entry is `{"sig": nodes}` where `nodes` is `null`, a string, or a list of strings.

### 7.3 JSONL Output (`.jsonl`)

One JSON object per line (same format as JSON array items).

### 7.4 Binary Output (`.bin`)

**KSC1 format** (little-endian):

```
Header:
  4 bytes: magic "KSC1"
  4 bytes: entry count (uint32)

Per entry:
  8 bytes: signature (uint64)
  1 byte:  node_type (0=None, 1=int, 2=list)
  if type==1: 8 bytes (uint64 node)
  if type==2: 4 bytes count + N * 8 bytes (uint64 each)
```

---

## 8. Public API

### 8.1 Python API (`KScript` class)

```python
from kscript import KScript

# Compile from source string
model = KScript("A == B")

# Compile from file (.ks, .json, .jsonl, .bin)
model = KScript("script.ks")

# Extend existing model
extended = KScript("C = D", base=model)

# Output to file (format by suffix: .json, .jsonl, .bin)
model.output("output.json")
model.output("output.bin")

# Get JSONL lines
lines = model.to_jsonl()

# Access compiled entries
for entry in model.entries:
    sig, nodes = entry.decode(tokenizer)
```

### 8.2 CLI

```bash
# Compile .ks to .jsonl (default)
python -m kscript script.ks

# Specify output format
python -m kscript script.ks -out output.json
python -m kscript script.ks -out output.bin

# Dev mode (include debug text)
python -m kscript script.ks -dev
```

### 8.3 Low-level Pipeline

```python
from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler, compile_source
from kscript.decompiler import Decompiler

# Manual pipeline
tokens = Lexer(source).tokenize()
kfile = Parser(tokens).parse()
entries = Compiler(tokenizer, dev=True).compile(kfile)

# Convenience function (default tokenizer is Mod32Tokenizer)
entries = compile_source(source, dev=True)

# Or specify a tokenizer explicitly
from kalvin.mod_tokenizer import Mod64Tokenizer
entries = compile_source(source, tokenizer=Mod64Tokenizer(), dev=True)

# Decompile
decompiled = Decompiler(tokenizer).decompile(entries)
```

---

## 9. Complete Examples

### 9.1 Minimal

```
A
```

Compiled: `{A: None}` (unsigned identity)

### 9.2 Bidirectional Link

```
A == B
```

Compiled: `{A: B}, {B: A}`

### 9.3 Canonization with Subscript

```
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A = D
     L = M
     L > O
```

Compiled (in order):

```
{MCS for MHALL}: {M: None}, {H: None}, {A: None}, {L: None}, {MHALL: [M,H,A,L,L]}, {MHALL: None}
{MCS for SVO}: {S: None}, {V: None}, {O: None}, {SVO: [S,V,O]}, {SVO: None}
{MHALL: SVO}           (countersign, S1)
{SVO: MHALL}           (countersign reverse, S1)
{SVO: [S, V, O]}       (canonize, single entry, S2 — deduplicated against MCS)
{M: S}                 (undersign, S1)
{H: V}                 (undersign, S1)
{MCS for ALL}: {A: None}, {L: None}, {ALL: [A,L,L]}, {ALL: None}
{ALL: O}               (undersign, S1)
{ALL: [A, L, L]}       (canonize, single entry, S2 — deduplicated against MCS)
{D: A}                 (undersign, S1)
{M: L}                 (undersign, S1)
{L: O}                 (connotate fwd, S3)
```

> **Note on deduplication:** Duplicate `{A: None}` and `{L: None}` from ALL's MCS are silently dropped by the compiler (§5.4), as they were already emitted by MHALL's MCS. The CANONIZE entries `{SVO: [S, V, O]}` and `{ALL: [A, L, L]}` are also deduplicated against the MCS canonization entries, which encode identically.

### 9.4 Mixed Literal Block

```
A =>
  1
  B
  "hello"
```

Compiled:

```
{A: ["1", B, "hello"]} (canonize, single entry with all items, S2)
{1: None}              (unsigned 1, from recursive block compilation)
{B: None}              (unsigned B, from recursive block compilation)
{"hello": None}        (unsigned "hello", from recursive block compilation)
```

> **Note:** CANONIZE emits a single entry with all right-hand items as the node list. The subscript block is then compiled recursively; each bare item emits its own unsigned identity entry.

### 9.5 Chained Constructs

```
A => B => C
```

Compiled:

```
{A: B}                 (canonize fwd, singleton)
{B: C}                 (canonize fwd, singleton)
{C: None}              (unsigned C, from recursive compilation of right side)
```

> **Note:** The right side of a chain is always compiled recursively. Bare signatures emit their unsigned identity.

### 9.6 Complex Nested

```
MHALL == SVO =>
  S(ubject) = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O
    MOD => A B
```

Key structures:

- `MHALL` countersigns `SVO` (bidirectional)
- `S`, `V`, `O` undersign their values → `{M: S}`, `{H: V}`, `{ALL: O}`
- `ALL` canonizes `A`, `L` (subscript items) → `{ALL: [A, L, L]}` (deduplicated against MCS)
- `L > O` connotates
- `MOD => A B` canonizes both items: `{MOD: [A, B]}`

---

## 10. Implementation Components

### 10.1 Module Structure

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

### 10.2 Dependencies

| Dependency                            | Used By                    | Required |
| ------------------------------------- | -------------------------- | -------- |
| `kalvin.kline.KLine`                  | `CompiledEntry` base class | Yes      |
| `kalvin.mod_tokenizer.ModTokenizer`   | Encoding/decoding          | Yes      |
| `kalvin.mod_tokenizer.Mod32Tokenizer` | Default tokenizer          | Yes      |
| `kalvin.signature.make_signature`     | Decompiler MCS detection   | Yes      |

### 10.3 Compiler Class

```python
class Compiler:
    def __init__(self, tokenizer: ModTokenizer | None = None, dev: bool = False)
    def compile(self, file: KScriptFile) -> list[CompiledEntry]
```

Internal state:

- `self.entries` — accumulated output list
- `self._seen` — deduplication set of `(sig_id, nodes_tuple)` pairs
- `self._sig_levels` — mapping from operator name to significance level string

### 10.4 CompiledEntry

Extends `KLine` with encode/decode support:

```python
class CompiledEntry(KLine):
    @classmethod
    def encode(cls, sig, nodes, tokenizer, *, sig_level, significance, dbg_text) -> CompiledEntry

    def decode(self, tokenizer) -> tuple[str, str | None | list[str]]
```

---

## 11. Design Rationale & Gotchas

### 11.1 Why Two Node Types?

Signatures (uppercase) use **packed encoding** — multiple characters OR'd into one `uint64`. This makes them usable as **bitmask signatures** for fast overlap-based candidate retrieval. Literals preserve exact content via literal encoding.

### 11.2 Why Singleton Unwrapping?

Single-element node lists are indistinguishable from single nodes in the KLine model. Unwrapping keeps the representation canonical and avoids ambiguity.

### 11.3 Why MCS Expansion?

Packed encoding loses character order. `ABC` and `CBA` may produce the same token. MCS entries store `[A, B, C]` as ordered nodes, enabling the decompiler to recover the original name.

### 11.4 Deduplication is Semantic

The compiler deduplicates by **encoded** `(signature, nodes)` pair, not by source text. This means `AB => A B` and the MCS expansion of `AB` may share entries.

### 11.5 Decompiler is Lossy

The decompiler uses **heuristic inference** for significance levels. Without explicit bits:

- S1 and S3 are easily confused (both produce single nodes with zero bit overlap)
- The best-effort reconstruction is useful for debugging but not for round-tripping

### 11.6 Indentation is Structural

Like Python, indentation defines block structure. Mixed tabs and spaces will work but are discouraged. The lexer counts each space/tab as one unit.

---

## 12. Build-From-Scratch Implementation Plan

### Phase 1: Token Types & Lexer

**Estimate:** 0.5 day

1. Define `TokenType` enum (11 values)
2. Define `Token` frozen dataclass
3. Implement `Lexer` with:
   - Multi-char operator priority (`==`, `=>`, `<=` before `=`, `>`, `<`)
   - Identifier → SIGNATURE/LITERAL classification
   - Number, quoted string, comment parsing
   - Python-style INDENT/DEDENT tracking
4. **Tests:** Token classification, operator lexing, indent/dedent, comments, edge cases

### Phase 2: AST & Parser

**Estimate:** 0.5 day

1. Define AST nodes: `Signature`, `Literal`, `PrimaryConstruct`, `Block`, `Construct`, `Script`, `KScriptFile`
2. Implement recursive descent `Parser` matching the grammar
3. Handle: inline ops vs chain ops, block constructs, literal constructs, insignificant tokens
4. **Tests:** AST structure for each construct type, parse errors for invalid input

### Phase 3: Encoder (Mod Tokenizer dependency)

**Estimate:** 0.5 day (assumes Mod Tokenizer already exists)

1. Define `CompiledEntry` extending `KLine`
2. Implement `encode()` and `decode()` with packed/literal discrimination
3. **Tests:** Encode/decode round-trips for signatures, literals, mixed

### Phase 4: Compiler

**Estimate:** 1.5 days

1. Implement `Compiler.compile()` traversing AST
2. Implement per-operator emission rules (§5.5)
3. Implement MCS expansion (§5.3)
4. Implement deduplication (§5.4)
5. Implement chain processing with flattening
6. **Tests:** Each operator type, MCS expansion, chains, nested subscripts, literals, dedup, complex examples

### Phase 5: Output Module

**Estimate:** 0.5 day

1. Implement `write_json`, `write_jsonl`, `write_bin`, `read_json`, `read_bin`
2. Binary format: KSC1 with magic header
3. **Tests:** Write/read round-trips for each format, invalid magic detection

### Phase 6: Decompiler

**Estimate:** 1 day

1. Implement MCS name recovery (both patterns)
2. Implement level inference heuristics
3. Implement `DecompiledEntry` with `to_kscript()` and `to_dict()`
4. **Tests:** Round-trip for each operator, MCS name recovery, complex nested scripts

### Phase 7: Public API & CLI

**Estimate:** 0.5 day

1. Implement `KScript` class with source/file loading and output
2. Implement `__main__.py` CLI
3. Implement `compile_source()` convenience function
4. **Tests:** API usage, file loading, output format selection, base extension

**Total Estimate:** 5 days

### Build Order Dependency

```
[Mod Tokenizer] ──→ Phase 3 (Encoder) ──→ Phase 4 (Compiler)
                                                       ↓
Phase 1 (Lexer) ──→ Phase 2 (Parser) ─────────────→ Phase 4
                                                       ↓
                                              Phase 5 (Output)
                                              Phase 6 (Decompiler)
                                              Phase 7 (API/CLI)
```

Phases 1 and 2 can proceed in parallel with Phase 3. Phase 4 requires all three. Phases 5–7 are independent of each other but depend on Phase 4.

---

## 13. Test Matrix Summary

| Category           | Count   | Key Tests                                                   |
| ------------------ | ------- | ----------------------------------------------------------- |
| Lexer              | 14      | Token types, operators, comments, indent/dedent, edge cases |
| Parser AST         | 5       | Chains, blocks, literals, parse errors                      |
| Compiler Basic     | 8       | Each operator, literals, quoted strings                     |
| MCS Expansion      | 4       | Multi-char sigs, no single-char MCS, countersign MCS        |
| Chains             | 4       | CANONIZE chains, per-item emission, subscript blocks        |
| Nested Subscripts  | 2       | Nested blocks, mixed inline ops                             |
| Complex Examples   | 3       | AB=>A B, AB==CD, AB>C                                       |
| Literal Edge Cases | 7       | Bare literals, block mixing, parse error for literal owners |
| Decompiler         | 14      | Round-trips, MCS recovery, level inference, Mod32 compat    |
| Output I/O         | 4       | Binary/JSON/JSONL round-trips                               |
| Encode/Decode      | 4       | Sig-only, sig-to-sig, sig-to-literal, list                  |
| KScript API        | 6       | Inline source, output formats, base extension, file loading |
| **Total**          | **~72** |                                                             |
