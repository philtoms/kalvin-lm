# KScript Build-From-Scratch Implementation Plan

**Source:** @kscript.md v2.0  
**Date:** 2026-04-29  
**Assumes:** Kalvin core specs are implemented (`kalvin.kline`, `kalvin.mod_tokenizer`, `kalvin.signature`)  
**Does NOT assume:** Any existing `kscript/` code

---

## Dependency Graph

```
[Kalvin Core: KLine, ModTokenizer, make_signature]
          │
          ▼
   ┌── Task 1: token.py (0.5h)
   │       │
   │       ▼
   │   Task 2: lexer.py (2h)
   │       │
   │       ▼
   │   Task 3: ast.py (0.5h)
   │       │
   │       ▼
   │   Task 4: parser.py (2h)
   │       │
   ├───────┼──────────────────────┐
   │       ▼                      ▼
   │  Task 5: compiler.py    Task 6: output.py
   │       (4h)                   (1.5h)
   │       │                      │
   │       ├──────────────────────┘
   │       ▼
   │  Task 7: decompiler.py (2h)
   │       │
   │       ▼
   │  Task 8: public API + CLI (1.5h)
   │       │
   │       ▼
   └── Task 9: integration tests (1h)

Total: ~15h (2 days)
```

Tasks 5 and 6 are independent of each other once Task 4 is done.

---

## Task 1: Token Types

**File:** `kscript/token.py`  
**Spec ref:** §2.1  
**Time:** 30 min

### Deliverable

```python
from enum import Enum, auto
from dataclasses import dataclass

class TokenType(Enum):
    COUNTERSIGN = auto()    # ==
    CANONIZE_FWD = auto()   # =>
    CANONIZE_BWD = auto()   # <=
    CONNOTATE_FWD = auto()  # >
    CONNOTATE_BWD = auto()  # <
    UNDERSIGN = auto()      # =
    SIGNATURE = auto()      # [A-Z]+
    LITERAL = auto()        # numbers and quoted strings
    COMMENT = auto()        # (...)
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

### Acceptance

- [ ] 13 enum values defined
- [ ] `Token` is frozen (immutable, hashable)
- [ ] All fields present: type, value, line, column

---

## Task 2: Lexer

**File:** `kscript/lexer.py`  
**Spec ref:** §2.1–2.4  
**Time:** 2h

### Interface

```python
class Lexer:
    def __init__(self, source: str): ...
    def tokenize(self) -> list[Token]: ...
```

### Rules (in priority order)

1. At line start: count leading spaces/tabs → emit INDENT/DEDENT (§2.4)
2. Multi-char operators **before** single-char: `==`, `=>`, `<=` before `=`, `>`, `<` (§2.3.1)
3. Identifiers `[A-Z][A-Z0-9]*`: `SIGNATURE` if all uppercase alpha (`isupper() and isalpha()`), else **LexerError** (§2.3.2). Non-uppercase identifiers are not valid — use quoted strings instead.
4. After an identifier, consume inline `(...)` comment without emitting (§2.3.6)
5. Numbers `[0-9]+` → `LITERAL` (§2.3.3)
6. Quoted strings `"..."` with backslash escapes, stop at newline if unterminated → `LITERAL` (§2.3.4)
7. Comments `(...)` with nested parens, multi-line → `COMMENT` (§2.3.5)
8. Any other character → `LexerError`
9. At EOF: close all remaining indent levels with DEDENT, emit EOF (§2.4)

### Indentation algorithm

```
indent_stack = [0]

at each line start:
    indent = count_leading_whitespace()
    if indent > indent_stack[-1]:
        emit INDENT
        push indent onto stack
    elif indent < indent_stack[-1]:
        while stack[-1] > indent:
            pop stack
            emit DEDENT
    # else: same level, no token

at EOF:
    while len(stack) > 1:
        pop, emit DEDENT
    emit EOF
```

### Error class

```python
class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int): ...
```

### Test cases

| # | Input | Expected tokens (types, ignoring NEWLINE/COMMENT) |
|---|-------|---------------------------------------------------|
| 1 | `"A"` | `[SIGNATURE, EOF]` |
| 2 | `"ABC"` | `[SIGNATURE(value="ABC"), EOF]` |
| 3 | `"42"` | `[LITERAL(value="42"), EOF]` |
| 4 | `"hello"` | `[LexerError]` — lowercase identifiers not valid, use quoted `"hello"` instead |
| 5 | `'"hello world"'` | `[LITERAL(value='"hello world"'), EOF]` |
| 6 | `"A == B"` | `[SIGNATURE, COUNTERSIGN, SIGNATURE, EOF]` |
| 7 | `"A => B"` | `[SIGNATURE, CANONIZE_FWD, SIGNATURE, EOF]` |
| 8 | `"A <= B"` | `[SIGNATURE, CANONIZE_BWD, SIGNATURE, EOF]` |
| 9 | `"A > B"` | `[SIGNATURE, CONNOTATE_FWD, SIGNATURE, EOF]` |
| 10 | `"A < B"` | `[SIGNATURE, CONNOTATE_BWD, SIGNATURE, EOF]` |
| 11 | `"A = B"` | `[SIGNATURE, UNDERSIGN, SIGNATURE, EOF]` |
| 12 | `"A =>\n  B\n  C"` | `[SIGNATURE, CANONIZE_FWD, NEWLINE, INDENT, NEWLINE, SIGNATURE, NEWLINE, SIGNATURE, DEDENT, EOF]` |
| 13 | `"A (inline) => B"` | `[SIGNATURE, COMMENT, CANONIZE_FWD, SIGNATURE, EOF]` |
| 14 | `"A => B\n(multi\nline)\nC => D"` | COMMENT spans both newlines, value contains "multi" and "line" |
| 15 | `"(outer (inner) outer)"` | Single COMMENT token with nested parens |
| 16 | `""` | `[EOF]` only |
| 17 | `"A = ="` | `[SIGNATURE, UNDERSIGN, UNDERSIGN, EOF]` — `=` vs `==` disambiguation |
| 18 | `'A => "unterminated'` | LITERAL value stops at newline, no closing quote |

### Gotchas

- **`==` vs `=`**: Two-char match first. Source `A == B` must not produce SIGNATURE, UNDERSIGN, UNDERSIGN, SIGNATURE.
- **Inline comments**: After reading an identifier, check for `(` and consume the whole comment without emitting. This means `A(comment)` produces just SIGNATURE("A").
- **NEWLINE resets at_line_start**: After emitting NEWLINE, set flag so next token dispatch handles indentation.
- **Pending INDENT/DEDENT**: Decreased indent may produce multiple DEDENT tokens. Buffer them and emit one per `_next_token()` call.

---

## Task 3: AST Nodes

**File:** `kscript/ast.py`  
**Spec ref:** §4  
**Time:** 30 min

### Deliverable

```python
@dataclass
class Signature:
    id: str       # uppercase
    line: int
    column: int

@dataclass
class Literal:
    id: str       # any value
    line: int
    column: int

Node = Signature | Literal
ConstructItem = PrimaryConstruct | Literal

@dataclass
class PrimaryConstruct:
    sig: Signature
    op: TokenType | None = None       # inline operator
    node: Node | None = None          # right-hand node

@dataclass
class Block:
    constructs: list["Construct"]

@dataclass
class Construct:
    inner: Block | Literal | list[PrimaryConstruct]
    chain_op: TokenType | None = None
    chain_right: "Construct | None" = None

@dataclass
class Script:
    constructs: list[Construct]

@dataclass
class KScriptFile:
    scripts: list[Script]
```

### Acceptance

- [ ] All 8 dataclasses defined
- [ ] Type aliases `Node` and `ConstructItem` defined
- [ ] No logic — pure data containers

---

## Task 4: Parser

**File:** `kscript/parser.py`  
**Spec ref:** §3  
**Time:** 2h

### Interface

```python
class ParseError(Exception):
    def __init__(self, message: str, token: Token): ...

class Parser:
    def __init__(self, tokens: list[Token]): ...
    def parse(self) -> KScriptFile: ...
```

### Grammar (from spec §3)

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

### Parse algorithm (recursive descent)

```
parse():
    skip_insignificant()
    if at_end(): return KScriptFile([Script([])])
    return KScriptFile([parse_script()])

parse_script():
    return Script(parse_constructs_until(EOF))

parse_construct():
    skip_insignificant()
    if peek() == INDENT:       return parse_block()
    if peek() == LITERAL:      return parse_literal_construct()
    # otherwise: primary_construct+
    indent = peek().column
    primaries = [parse_primary_construct()]
    while is_primary_construct_start() and peek().column >= indent:
        primaries.append(parse_primary_construct())
    # optional chain
    chain_op = try_chain_op()
    if chain_op:
        right = parse_construct()
        return Construct(primaries, chain_op, right)
    return Construct(primaries)

parse_block():
    expect(INDENT)
    constructs = parse_constructs_until(DEDENT)
    expect(DEDENT)
    return Construct(Block(constructs))

parse_literal_construct():
    return Construct(parse_literal())

parse_primary_construct():
    sig = parse_sig()
    op = try_inline_op()
    if op:
        node = parse_node()
        return PrimaryConstruct(sig, op, node)
    return PrimaryConstruct(sig)
```

### Key decisions

1. **`>` is always inline.** `try_inline_op()` checks for COUNTERSIGN, CONNOTATE_FWD, UNDERSIGN — in that order. It is called before `try_chain_op()`, so `>` is consumed inline first. (§2.2)

2. **Multiple primaries at same indent.** After parsing the first primary_construct, keep parsing more while the next token is SIGNATURE and at the same/greater column. These form the implicit group for BWD chains.

3. **Insignificant tokens.** `skip_insignificant()` consumes NEWLINE and COMMENT tokens between grammar elements. Never skip inside a primary_construct (between sig and its operator).

4. **Literal constructs are bare.** A LITERAL in construct position wraps as `Construct(Literal(...))` with no chain_op. It cannot own inline or chain operators.

### Error cases (§3.2)

| Input | Must raise ParseError |
|-------|----------------------|
| `"1 => A"` | LITERAL where SIGNATURE expected |
| `"A => 1 => B"` | Chaining through a literal |

### Test cases

| # | Input | AST assertion |
|---|-------|---------------|
| 1 | `"A"` | `constructs[0].inner` is `[PrimaryConstruct(sig="A")]`, no chain |
| 2 | `"A => B"` | `chain_op == CANONIZE_FWD`, `chain_right.inner` is `[PrimaryConstruct(sig="B")]` |
| 3 | `"A < B"` | `chain_op == CONNOTATE_BWD` |
| 4 | `"A > 1 < B"` | `inner[0].op == CONNOTATE_FWD`, `chain_op == CONNOTATE_BWD` |
| 5 | `"A => B C <= D"` | Top-level `chain_op == CANONIZE_FWD`, `chain_right.chain_op == CANONIZE_BWD` |
| 6 | `"A =>\n  B\n  C"` | `chain_right.inner` is `Block` with ≥1 construct |
| 7 | `"A B <= CD"` | `inner` has 2 primaries |
| 8 | `"1 => A"` | `ParseError` raised |
| 9 | `""` | Empty constructs list, no error |

---

## Task 5: Compiler

**File:** `kscript/compiler.py`  
**Spec ref:** §5  
**Time:** 4h

This is the largest task. Build in layers: entry model → MCS → inline ops → chains → dedup.

### 5A: CompiledEntry (30 min)

**Spec ref:** §5.1, §5.2

```python
class CompiledEntry(KLine):
    def __init__(self, signature: int, nodes, dbg_text: str = ""):
        super().__init__(signature=signature, nodes=nodes, dbg_text=dbg_text)

    @classmethod
    def encode(cls, sig_str, nodes_str, tokenizer, *, sig_level="S4", significance=None, dbg_text="") -> CompiledEntry:
        """Encode string names to token IDs.
        - Uppercase alpha strings → packed (single uint64)
        - Everything else → literal encoding (one uint64 per char, lower 32 bits = 0xFFFFFFFF)
        """

    def decode(self, tokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to string names."""
```

**Encoding rules (§5.2):**
- Signature string → `tokenizer.encode(sig)[0]` → single packed uint64, bit 0 clear (all-uppercase-alpha auto-detected)
- Literal string → `tokenizer.encode(lit)` → list of `(char_cp << 32) | 0xFFFFFFFF` (non-uppercase auto-detected)
- **Literal test:** `(node & 0xFFFFFFFF) == 0xFFFFFFFF`
- **No `pack` parameter needed** — the tokenizer determines encoding mode internally

**Singleton rule (§5.1):**
- If nodes is a list with 1 element → unwrap to single int
- If nodes is empty/None → store as None (but KLine normalizes to `[]`)

**Test cases for encode/decode:**

| # | sig | nodes | encode produces | decode recovers |
|---|-----|-------|-----------------|-----------------|
| 1 | `"A"` | `None` | signature=packed_A, nodes=[] | `("A", "")` |
| 2 | `"A"` | `"B"` | signature=packed_A, nodes=packed_B | `("A", ["B"])` |
| 3 | `"A"` | `"hello"` | signature=packed_A, nodes=[literal_h, literal_e, ...] | `("A", "hello")` |
| 4 | `"AB"` | `["A", "B"]` | signature=packed_AB, nodes=[packed_A, packed_B] | `("AB", ["A", "B"])` |

> **Note:** Due to KLine normalization, `nodes=None` becomes `nodes=[]`. `decode()` returns `""` for empty nodes, not `None`. The decompiler handles this correctly.

### 5B: Compiler core — unsigned + inline ops (1h)

**Spec ref:** §5.3, §5.4, §5.5

```python
class Compiler:
    def __init__(self, tokenizer=None, dev=False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self._seen: set[tuple[int, ...]] = set()
        self._sig_levels = {
            "COUNTERSIGN": "S1", "UNDERSIGN": "S1", "IDENTITY": "S1",
            "CANONIZE_FWD": "S2", "CANONIZE_BWD": "S2",
            "CONNOTATE_FWD": "S3", "CONNOTATE_BWD": "S3",
            "UNSIGNED": "S4",
        }

    def compile(self, file: KScriptFile) -> list[CompiledEntry]: ...
```

**`_emit(sig_str, nodes_str, op_name)` — the central emission method:**

```
1. Apply singleton rule: if nodes is list with 1 element → unwrap to str
2. Encode sig and nodes to token IDs
3. Dedup check: if (sig_id, nodes_key) in _seen → return
4. Add to _seen
5. Format debug text if dev mode
6. Append CompiledEntry to self.entries
```

**`_emit_mcs(sig_str)` — MCS expansion (§5.3):**

```
if len(sig) <= 1: return False
for each char in sig:
    _emit(char, None, "UNSIGNED")
_emit(sig, list(sig_chars), "CANONIZE_FWD")    # {ABC: [A,B,C]}
# Note: the unsigned compound {ABC: None} is emitted by the caller
# via the normal unsigned path after _emit_mcs returns
return True
```

> **Gotcha (§5.3):** MCS emits 3 things: components, canonization, unsigned compound. The unsigned compound is NOT emitted inside `_emit_mcs` — it happens naturally when the caller processes the bare signature. If `_emit_mcs` emitted the compound too, it would duplicate the caller's unsigned emission. Duplicates are caught by dedup, but it's cleaner to let the caller handle it.

**Inline operator emission (`_emit_primary`):**

```
emit_mcs(pc.sig.id)

if op is None:
    emit(sig, None, "UNSIGNED")

if op == COUNTERSIGN:
    emit_mcs(node) if node is signature
    emit(sig, node, "COUNTERSIGN")
    emit(node, sig, "COUNTERSIGN")  # reverse — only if node is signature

if op == UNDERSIGN:
    if sig == node: emit(sig, None, "IDENTITY")
    else: emit(sig, node, "UNDERSIGN")

if op == CONNOTATE_FWD:
    emit(sig, node, "CONNOTATE_FWD")
```

**Test cases for basic ops:**

| # | Source | Entries must include |
|---|--------|---------------------|
| 1 | `"A"` | `{A: None}` |
| 2 | `"A == B"` | `{A: B}`, `{B: A}` |
| 3 | `"A = B"` | `{A: B}` |
| 4 | `"A = A"` | `{A: None}` (identity, deduped to one entry) |
| 5 | `"A > B"` | `{A: B}` |
| 6 | `'A = "42"'` | `{A: '"42"'}` (literal node — quoted string) |
| 7 | `'A > "x"'` | `{A: '"x"'}` (quoted string literal) |
| 8 | `'A => "hello"'` | `{A: '"hello"'}` |

**MCS test cases:**

| # | Source | Entries must include |
|---|--------|---------------------|
| 9 | `"ABC"` | `{A: None}`, `{B: None}`, `{C: None}`, `{ABC: [A,B,C]}`, `{ABC: None}` |
| 10 | `"A => X"` | No MCS for single-char A |
| 11 | `"ABC == X"` | MCS for ABC, `{ABC: X}`, `{X: ABC}` |

### 5C: Chain operators (1.5h)

**Spec ref:** §5.5, §5.6

**`_process_chain(left_primaries, chain_op, right_construct)`:**

First: emit inline ops for all left primaries that have them.

Then branch by chain_op:

**CANONIZE_FWD (`=>`):**
```
owner = last_primary.node if present, else last_primary.sig
emit_mcs(owner) if owner is signature
right_items = flatten_to_items(right_construct)
for each item in right_items:
    emit(owner, item_id(item), "CANONIZE_FWD")
compile_construct(right)   # recursive
```

**CANONIZE_BWD (`<=`):**
```
owner = first right item's sig
emit_mcs(owner)
for each left_primary:
    emit(owner, left_primary.sig, "CANONIZE_BWD")
compile_construct(right)   # recursive
```

**CONNOTATE_BWD (`<`):**
```
owner = first right item's sig
emit_mcs(owner)
last_left_owner = get_owner(left_primaries[-1])
emit(owner, last_left_owner, "CONNOTATE_BWD")
compile_construct(right)   # recursive
```

**CONNOTATE_FWD chain (`>`):**
```
# UNREACHABLE in practice (§2.2 note), but implement for grammar completeness:
owner = get_owner(left_primaries[-1])
nodes = [item_id(item) for item in right_items]
emit(owner, nodes, "CONNOTATE_FWD")
compile_construct(right)   # recursive
```

**`_flatten_to_items(construct)` — block flattening (§5.6):**
```
if Block: recursively flatten all child constructs
if Literal: return [literal]
if [PrimaryConstruct]: return the list
```

**`_get_owner(primary_construct)`:**
```
return primary.node if present, else primary.sig
```

**`_item_id(item)`:**
```
if PrimaryConstruct: return item.sig.id
if Literal: return item.id
```

**Chain test cases:**

| # | Source | Entries must include |
|---|--------|---------------------|
| 1 | `"A => B => C"` | `{A: B}`, `{B: C}`, `{C: None}` |
| 2 | `"A => B <= C"` | `{A: B}`, `{C: B}` |
| 3 | `"A > B > C"` | `{A: B}`, `{B: C}` |
| 4 | `"A > B < C"` | `{A: B}`, `{C: B}` |
| 5 | `"A => B C"` | `{A: B}`, `{A: C}` (per-item) |
| 6 | `"A =>\n  B\n  C"` | `{A: B}`, `{A: C}` (subscript block) |
| 7 | `"A B <= CD"` | `{CD: A}`, `{CD: B}` |
| 8 | `"AB > C"` | `{AB: C}` (with MCS for AB) |
| 9 | `"C < AB"` | `{AB: C}` (with MCS for AB) |

### 5D: Subscript blocks and nested chains (1h)

**Spec ref:** §5.6

The right-hand construct is always compiled recursively after chain emission. This means bare items in subscript blocks emit their own unsigned entries.

**Nested subscript test cases:**

| # | Source | Entries must include |
|---|--------|---------------------|
| 1 | `"A =>\n  B =>\n    C\n    D"` | `{A: B}`, `{B: C}`, `{B: D}` |
| 2 | `"A =>\n  B\n  C = D"` | `{A: B}`, `{A: C}`, `{C: D}` |

### 5E: Full example verification (30 min)

**Spec ref:** §9.3, §9.6

Compile these sources and verify entry-by-entry against the spec's expected output:

**§9.3:**
```
MHALL == SVO =>
   S = M
   V = H
   O = ALL =>
     A = D
     L = M
     L > O
```

**§9.6:**
```
MHALL == SVO =>
  S = M
  V = H
  O = ALL =>
    A = D
    L = M < MOD => A B
    L > O < BS =>
      B = "baby"
      S = "sheep"
```

> **Tip:** Use Mod64Tokenizer for tests to avoid bit collisions in Mod32's smaller range. Write a helper `compile_and_dump(source)` that prints all decoded entries for comparison.

### Gotchas

- **Dedup is by encoded pair, not source text.** `{A: B}` encoded twice = one entry. But `{A: B}` and `{A: C}` are different entries even if B and C have the same packed value.
- **MCS dedup: `{L: None}` from MHALL's MCS is emitted once** even though L appears twice in "MHALL". Second is dropped by dedup.
- **`_encode_node` for multi-char literals** must encode each character individually via `tokenizer.encode(char)`, not just the first character. The tokenizer auto-detects non-uppercase and uses literal encoding.
- **Reverse countersign** is NOT emitted for literal nodes. `A == 42` → `{A: "42"}` only, no `{"42": A}`.

---

## Task 6: Output Module

**File:** `kscript/output.py`  
**Spec ref:** §7  
**Time:** 1.5h

### Interface

```python
def write_json(entries, path, tokenizer): ...
def write_jsonl(entries, path, tokenizer): ...
def write_bin(entries, path): ...
def read_json(path, tokenizer) -> list[CompiledEntry]: ...
def read_bin(path) -> list[CompiledEntry]: ...
```

### JSON format (§7.2)

```json
[
  {"MHALL": "SVO"},
  {"SVO": "MHALL"},
  {"S": "M"},
  {"ABC": ["A", "B", "C"]}
]
```

Each entry: `{"sig_name": nodes}` where nodes is `null`, a string, or a list of strings.

- Decode each entry to get string names
- `json.dump(data, f, indent=2)`

### JSONL format (§7.3)

One JSON object per line. Same structure as JSON array items.

### Binary format — KSC1 (§7.4)

```
Header:
  4 bytes: magic b"KSC1"
  4 bytes: entry count (uint32 LE)

Per entry:
  8 bytes: signature (uint64 LE)
  1 byte:  node_type
           0 = None (no nodes)
           1 = int  (single node)
           2 = list (multiple nodes)
  if type==1: 8 bytes (uint64 LE, the single node)
  if type==2: 4 bytes count (uint32 LE) + N × 8 bytes (uint64 LE each)
```

### Read functions

- `read_bin`: verify magic == b"KSC1", raise `ValueError` otherwise
- `read_json`: try `json.loads` as array, fall back to JSONL (line-by-line)

### Test cases

| # | Test | Assertion |
|---|------|-----------|
| 1 | Write bin → read bin | Same signature and nodes for every entry |
| 2 | Write json → read json | Same number of entries |
| 3 | Write jsonl | Each line is valid JSON, total lines == entries |
| 4 | Read bin with bad magic | Raises `ValueError("Invalid magic")` |

---

## Task 7: Decompiler

**File:** `kscript/decompiler.py`  
**Spec ref:** §6  
**Time:** 2h

### Interface

```python
@dataclass
class DecompiledEntry:
    level: str           # "S1"–"S4"
    sig: str
    nodes: str | list[str] | None

    def to_dict(self) -> dict: ...
    def to_kscript(self) -> str: ...

class Decompiler:
    def __init__(self, tokenizer=None): ...
    def decompile(self, klines: list[KLine]) -> list[DecompiledEntry]: ...
```

### Algorithm

**Pass 1: Build MCS name map (§6.1)**

```
mcs_names: dict[int, str] = {}   # packed_token → original_name

# Pattern 1: Legacy — single entry with multiple packed single-char nodes
for each kline:
    nodes = kline.as_node_list()
    if len(nodes) >= 2 AND all nodes are packed single-char:
        nodes_or = OR of all node values
        if kline.signature == nodes_or:
            name = concat of decoded chars
            mcs_names[signature] = name

# Pattern 2: Per-char — consecutive entries with same sig, each with one packed single-char
for each consecutive group of klines with same signature:
    if all have exactly 1 packed single-char node:
        nodes_or = OR of all those node values
        if nodes_or == group_signature:
            name = concat of decoded chars
            mcs_names[signature] = name
```

**`_try_decode_packed_single_char(node)`:**
```
if is_literal(node): return None
decoded = tokenizer.decode([node])
if decoded and len(decoded) == 1 and decoded.isupper():
    return decoded
return None
```

**Pass 2: Decompile each kline**

```
for each kline:
    sig_str = decode_sig(kline.signature)    # use mcs_names if available
    nodes = kline.as_node_list()

    if is_mcs_entry(kline):
        emit S2 entry with decoded nodes

    elif nodes is empty:
        emit S4 unsigned entry

    else:
        level = infer_level(kline)
        node_strs = decode_nodes(nodes)
        emit entry with level, sig_str, node_strs
```

**Level inference (§6.2):**

```
if nodes empty:          → S4
if single int node:
    if (sig & node) != 0: → S2
    else:                  → S3
if node list:
    nodes_sig = make_signature(nodes)
    if (sig & nodes_sig) != 0: → S2
    else:                        → S3
```

> **Caveat:** S1 and S3 are indistinguishable by bit overlap alone. Countersign `A == B` produces `{A: B}` where A and B have zero bit overlap → inferred as S3, not S1. This is a known limitation.

**`decode_nodes` — grouping consecutive literal chars:**
```
for each node in nodes:
    if is_literal(node):
        buffer literal node IDs
    else:
        flush literal buffer as one decoded string
        decode packed node (use mcs_names if available)
flush remaining literal buffer
```

### Test cases

| # | Source | Decompile assertion |
|---|--------|-------------------|
| 1 | `"A"` | Entry with sig="A", nodes=None, level="S4" |
| 2 | `"A = B"` | Entry with sig="A", nodes="B" |
| 3 | `"A == B"` | Entries for A→B and B→A present |
| 4 | `"A > B"` | Entry with sig="A", nodes="B" |
| 5 | `'A = "hello"'` | Entry with nodes='"hello"' |
| 6 | `""` | Empty list |
| 7 | `"A =>\n  B\n  C"` | Entries for A, B, C present |
| 8 | `"ABC"` | MCS name "ABC" recovered (not bit-ordered garble) |
| 9 | `"AB == CD"` | Both "AB" and "CD" names recovered |
| 10 | §9.3 full example | "MHALL" and "SVO" names recovered |

---

## Task 8: Public API + CLI

**File:** `kscript/__init__.py`, `kscript/__main__.py`  
**Spec ref:** §8  
**Time:** 1.5h

### KScript class (`__init__.py`)

```python
class KScript:
    def __init__(self, source, base=None, tokenizer=None, dev=False):
        """
        source: str (inline KScript) or Path (.ks, .json, .jsonl, .bin)
        base: optional KScript to extend
        tokenizer: defaults to Mod32Tokenizer
        """
    @property
    def entries(self) -> list[CompiledEntry]: ...

    def output(self, path) -> None:
        """Write to file. Format by suffix: .json, .jsonl, .bin"""

    def to_jsonl(self) -> list[str]:
        """Return list of JSON lines."""
```

**Loading logic:**
- If `Path(source).exists()`: load by suffix
  - `.ks` → read text, compile
  - `.json` / `.jsonl` → `read_json`
  - `.bin` → `read_bin`
- Else: treat as inline source string

**Extending:** Copy entries from `base`, then compile new source and append.

**Re-exports:** `KScript`, `CompiledEntry`, `Lexer`, `Parser`, `Compiler`, `KScriptFile`

### CLI (`__main__.py`)

```python
# Usage:
#   python -m kscript script.ks              # → script.jsonl
#   python -m kscript script.ks -out out.json
#   python -m kscript script.ks -out out.bin
#   python -m kscript script.ks -dev
```

Arguments:
- `input`: required positional, input file path
- `-out`: optional output path (default: input with `.jsonl` suffix)
- `-dev`: flag, enable debug text

### Convenience function (in `compiler.py`)

```python
def compile_source(source: str, tokenizer=None, dev=False) -> list[CompiledEntry]:
    tokens = Lexer(source).tokenize()
    kfile = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kfile)
```

### Test cases

| # | Test | Assertion |
|---|------|-----------|
| 1 | `KScript("A = B")` | ≥1 entries |
| 2 | `KScript("A = B").output(path.json)` then read | Valid JSON array |
| 3 | `KScript("A = B").output(path.bin)` then read | Starts with b"KSC1" |
| 4 | `KScript("A = B").to_jsonl()` | Each line is valid JSON dict |
| 5 | `KScript("C = D", base=KScript("A = B"))` | More entries than base |
| 6 | `KScript(path.bin)` from saved | Same entry count |
| 7 | `KScript(path.json)` from saved | Same entry count |

---

## Task 9: Integration Tests

**File:** `tests/test_kscript.py`  
**Time:** 1h

### Test structure

Organize into classes matching the spec sections:

```
TestTokenType        — §2.1 (2 tests)
TestLexer            — §2   (16 tests)
TestParserAST        — §3   (9 tests)
TestCompilerBasic    — §5.5 (10 tests)
TestMCSExpansion     — §5.3 (4 tests)
TestChains           — §5.5 (6 tests)
TestNestedSubscripts — §5.6 (2 tests)
TestComplexExamples  — §9   (5 tests)
TestLiteralEdgeCases — §3.1 (8 tests)
TestDecompiler       — §6   (12 tests)
TestDecompilerMCS    — §6.1 (3 tests)
TestOutputIO         — §7   (4 tests)
TestCompiledEntry    — §5.2 (4 tests)
TestKScriptAPI       — §8   (7 tests)
```

### Test helpers

```python
_tok64 = Mod64Tokenizer()

def compile64(source):
    return compile_source(source, tokenizer=_tok64, dev=True)

def _md(entries):
    """sig → [decoded_node_values...]"""
    result = {}
    for e in entries:
        sig, nodes = e.decode(_tok64)
        result.setdefault(sig, []).append(nodes)
    return result

def _has(md, sig, node_value):
    return node_value in md.get(sig, [])
```

### Spec example validation (§9.3)

Write a single test that compiles the §9.3 source and asserts **every entry** in the expected output list. This is the master regression test:

```python
def test_spec_93_complete(self):
    source = """MHALL == SVO =>
       S = M
       V = H
       O = ALL =>
         A = D
         L = M
         L > O"""
    entries = compile64(source)
    md = _md(entries)

    # MCS for MHALL
    assert _has(md, "M", "")
    assert _has(md, "H", "")
    assert _has(md, "A", "")
    assert _has(md, "L", "")       # only once (dedup drops second L)
    assert _has(md, "AHLM", ["M", "H", "A", "L", "L"])
    assert _has(md, "AHLM", "")    # unsigned compound

    # MCS for SVO
    assert _has(md, "S", "")
    assert _has(md, "V", "")
    assert _has(md, "O", "")
    assert _has(md, "OSV", ["S", "V", "O"])
    assert _has(md, "OSV", "")

    # Countersign
    assert _has(md, "AHLM", ["OSV"])
    assert _has(md, "OSV", ["AHLM"])

    # Canonize fwd per-item from subscript
    assert _has(md, "OSV", ["S"])
    assert _has(md, "OSV", ["V"])
    assert _has(md, "OSV", ["O"])

    # Undersign
    assert _has(md, "S", ["M"])
    assert _has(md, "V", ["H"])

    # MCS for ALL + undersign
    assert _has(md, "AL", ["A", "L", "L"])
    assert _has(md, "AL", "")
    assert _has(md, "O", ["AL"])

    # Canonize per-item from ALL subscript
    assert _has(md, "AL", ["A"])
    assert _has(md, "AL", ["L"])

    # Leaf undersigns and connotate
    assert _has(md, "A", ["D"])
    assert _has(md, "L", ["M"])
    assert _has(md, "L", ["O"])
```

> **Note:** Decoded names use bit-position order (e.g., `"AHLM"` for `"MHALL"`, `"OSV"` for `"SVO"`, `"AL"` for `"ALL"`). This is correct — packed encoding is lossy. The decompiler recovers the original names via MCS mapping, but `CompiledEntry.decode()` returns bit-order names.

---

## Module Structure (final)

```
kscript/
├── __init__.py       # KScript class, re-exports
├── __main__.py       # CLI entry point
├── token.py          # TokenType enum, Token dataclass
├── lexer.py          # Lexer
├── ast.py            # AST node dataclasses
├── parser.py         # Parser
├── compiler.py       # Compiler, CompiledEntry, compile_source()
├── decompiler.py     # Decompiler, DecompiledEntry
└── output.py         # JSON/JSONL/binary I/O
```

**Total files:** 9  
**Total LOC estimate:** ~1800  
**Total test count:** ~92

---

## External Dependencies

KScript depends on these from Kalvin core. They must be built first.

| Import | What it provides | Used in |
|--------|-----------------|---------|
| `kalvin.kline.KLine` | Base class for CompiledEntry (signature, nodes, equality, hashing) | `compiler.py` |
| `kalvin.kline.KNodes` | Type alias for node representations | `compiler.py` |
| `kalvin.kline.KSig` | Type alias for uint64 signatures | `compiler.py` |
| `kalvin.mod_tokenizer.ModTokenizer` | `encode(text)`, `decode(ids)` — mode auto-detected from content | `compiler.py`, `decompiler.py`, `output.py` |
| `kalvin.mod_tokenizer.Mod32Tokenizer` | Default tokenizer variant (31 bit positions) | `compiler.py` |
| `kalvin.kline.is_literal` | `is_literal(node) → bool` (standalone function) | `compiler.py`, `decompiler.py` |
| `kalvin.signature.make_signature` | `make_signature(nodes) → uint64` | `decompiler.py` |

**Not depended on:** `kalvin.significance.D_BOUNDARY`, `D_MAX` — these are not used by KScript despite being imported in some implementations.
