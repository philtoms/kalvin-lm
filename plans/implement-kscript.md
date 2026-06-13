# KScript Build-From-Scratch Implementation Plan

**Spec:** @specs/kscript.md v3.0  
**Date:** 2026-06-12  
**Status:** Plan  
**Assumes:** Kalvin core (`kalvin.kline.KLine`, `kalvin.mod_tokenizer`, `kalvin.signature.make_signature`, `kalvin.nlp_tokenizer.NLPTokenizer`)

---

> **Archival Note (added 2026-06-13):** This plan was written before the compiled-entry `op` field
> terminology was updated. The code described here used token-name op strings; the current code uses
> structural-state names. The mapping:
>
> - `"UNSIGNED"` → `"IDENTITY"` — the op value for bare-node identity klines was renamed (per the
>   `plans/impl/rename-unsigned-to-identity.md` plan, completed before ADR-0006).
> - `"COUNTERSIGN"` → `"COUNTERSIGNED"`, `"UNDERSIGN"` → `"UNDERSIGNED"`,
>   `"CONNOTATE"` → `"CONNOTED"`, `"CANONIZE"` → `"CANONIZED"` — compiled-entry op values
>   now use past-participle structural-state names (per ADR-0006, implemented in KB-209).
>
> **Old terms present in this file:** `"UNSIGNED"`, `"COUNTERSIGN"`, `"UNDERSIGN"`, `"CONNOTATE"`,
> `"CANONIZE"` (as compiled-entry op values in pseudocode and significance-level dicts).
>
> Note: `TokenType` enum members (`TokenType.COUNTERSIGN`, `TokenType.UNDERSIGN`, etc.) are
> **unchanged** — they name lexer tokens, not structural states. References to `TokenType.X` in this
> plan are still current.
>
> For the authoritative current terminology, see CONTEXT.md glossary entries **Structural State**
> and **Identity**, and `docs/adr/0006-op-is-structural-state-not-token.md`.

## Overview

This plan builds the KScript compiler from scratch against the consolidated spec (v3.0). The implementation lives in `src/ks/` (new module folder). The existing `src/kscript/` code is not modified and will be removed once the new implementation is verified.

- **Scope model** as the central organising principle (§3)
- **Four-stage pipeline**: Lexer → Parser → ASTEmitter → TokenEncoder (§1.1)
- **`nodes: list[uint64]`** everywhere — no None, no singleton unwrapping (§6.1)
- **BPE annotations** replacing comments (§9)
- **No decompiler, no file format I/O, no CLI** — compilation output is the product

## Dependency Graph

```
[Kalvin Core: KLine, NLPTokenizer, make_signature]
          │
          ▼
   Task 1: token.py (0.5h)
          │
          ▼
   Task 2: lexer.py (2h)
          │
          ▼
   Task 3: ast.py (0.5h)
          │
          ▼
   Task 4: parser.py (2h)
          │
          ├──────────────────────┐
          ▼                      ▼
   Task 5: binding_scope.py   Task 6: ast_emitter.py
          (1.5h)                    (4h)
          │                      │
          └──────────┬───────────┘
                     ▼
              Task 7: token_encoder.py (2h)
                     │
                     ▼
              Task 8: compiler.py (1h)
                     │
                     ▼
              Task 9: integration tests (2h)

Total: ~15.5h (2 days)
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
    CANONIZE = auto()       # =>
    CONNOTATE = auto()      # >
    UNDERSIGN = auto()      # =
    SIGNATURE = auto()      # [A-Z]+
    ANNOTATION = auto()     # (...)
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

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-1 | All 10 token types defined, Token is frozen |

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

### Rules (priority order)

1. At line start: count leading spaces/tabs → emit INDENT/DEDENT (§2.4)
2. Multi-char operators before single-char: `==`, `=>` before `=`, `>` (§2.3)
3. Identifiers `[A-Z][A-Z0-9]*`: `SIGNATURE` if all uppercase alpha, else `LexerError` (§2.3)
4. After an identifier, preserve inline `(...)` as `ANNOTATION` token attached to preceding signature
5. Standalone `(...)` with nested parens, multi-line → `ANNOTATION` (§2.3)
6. `<` is invalid → `LexerError`
7. Any other unknown character → `LexerError`
8. At EOF: close all remaining indent levels with DEDENT, emit EOF

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-1 | All token types recognized |
| KS-2 | `==`, `=>` matched before `=`, `>`: `A == B` produces SIGNATURE, COUNTERSIGN, SIGNATURE |
| KS-3 | `(...)` with nested parens preserved as ANNOTATION |
| KS-4 | Python-style INDENT/DEDENT tokens |
| KS-5 | Empty input → [EOF]; unknown char → LexerError |

---

## Task 3: AST Nodes

**File:** `kscript/ast.py`  
**Spec ref:** §4–5  
**Time:** 30 min

### Deliverable

```python
@dataclass
class Signature:
    id: str       # uppercase
    line: int
    column: int

@dataclass
class Annotation:
    """BPE annotation node — renamed from Comment."""
    text: str
    line: int
    column: int

@dataclass
class OperatorScope:
    """A scope created by an operator (§3)."""
    sig: Signature
    op: TokenType | None = None       # None = bare signature (unsigned)
    items: list[ConstructItem] = field(default_factory=list)
    child_block: Block | None = None
    inline_annotation: Annotation | None = None      # sig-side
    node_inline_annotation: Annotation | None = None  # node-side

@dataclass
class Block:
    constructs: list[ConstructItem]

ConstructItem: TypeAlias = "Annotation | OperatorScope | Block"
```

### Key structural points

- **`OperatorScope`** models operator-delimited scopes. No `chain_right` field.
- **`Annotation`** provides BPE encoding word text.
- **`items`** list holds nodes and child constructs within the scope.

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-6 | AST structure reflects scope model: OperatorScope nodes with sig, op, items, child_block |
| KS-8 | Annotations preserved as AST Annotation nodes |
| KS-9 | Inline annotation attachment: sig-side and node-side |

---

## Task 4: Parser

**File:** `kscript/parser.py`  
**Spec ref:** §3–4  
**Time:** 2h

### Interface

```python
class Parser:
    def __init__(self, tokens: list[Token]): ...
    def parse(self) -> KScriptFile: ...
```

### Grammar (§4)

```
script      ::= construct+
construct   ::= block | annotation | operator_scope
block       ::= INDENT construct+ DEDENT
annotation  ::= ANNOTATION
operator_scope ::= sig ( operator items )?
items       ::= item*
item        ::= sig | annotation | operator_scope
sig         ::= SIGNATURE
operator    ::= COUNTERSIGN | CANONIZE | CONNOTATE | UNDERSIGN
```

### Scope model in the parser (§3)

The parser enforces:
- **S2**: The identifier immediately preceding an operator is the scope's signature.
- **S3**: Identifiers succeeding the operator are items (nodes) in that scope.
- **S4**: INDENT creates child scope items appended to the current scope.
- **S5**: DEDENT closes the child scope.

### Parse algorithm

```
parse_script():
    return Script(parse_constructs_until(EOF))

parse_construct():
    skip NEWLINE
    if peek() == ANNOTATION:  return Annotation node
    if peek() == INDENT:      return parse_block()
    # otherwise: operator_scope
    sig = parse_sig()
    inline = try_inline_annotation()
    op = try_operator()
    if op:
        items = parse_items_until(eol or next_operator or DEDENT)
        child_block = try_child_block()
        return OperatorScope(sig, op, items, child_block, inline)
    return OperatorScope(sig, inline_annotation=inline)

parse_items():
    items = []
    while is_item_start() and not at_eol_or_dedent():
        if peek() == ANNOTATION: items.append(Annotation node)
        elif peek() == SIGNATURE:
            sig = parse_sig()
            node_inline = try_inline_annotation()
            op = try_operator()
            if op:
                # This sig is the preceding identifier for the new scope
                items.append(parse_operator_scope_from(sig, op, node_inline))
            else:
                items.append(sig)  # bare node
        else: break
    return items
```

### Inline annotation handling

- **Sig-side**: `S(ubject) = M` — ANNOTATION token immediately after SIGNATURE, before operator. Attached as `inline_annotation`.
- **Node-side**: `A = D(et)` — ANNOTATION token immediately after node SIGNATURE. Attached as `node_inline_annotation`.

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-6 | `A == B > C = D` produces chained OperatorScope nodes |
| KS-7 | INDENT/DEDENT creates Block nodes |
| KS-8 | ANNOTATION tokens preserved in AST |
| KS-9 | `S(ubject) = M` has inline_annotation on sig |
| KS-10 | Empty source → empty script |

---

## Task 5: BindingScope

**File:** `kscript/binding_scope.py`  
**Spec ref:** §10  
**Time:** 1.5h

### Interface

```python
class BindingScope:
    def push_scope(self) -> None: ...
    def pop_scope(self) -> None: ...
    def add_words(self, words: list[str]) -> None: ...
    def resolve(self, char: str) -> str | None: ...
```

### Internal structure

```python
@dataclass
class _Scope:
    word_lists: list[list[str]]     # accumulated word lists
    counters: dict[str, int]        # per-character occurrence counter (keyed lowercase)
```

### Resolution algorithm (§10.1)

```
resolve(char):
    for scope in scopes (innermost first):
        for word_list in scope.word_lists (most-recent-first):
            matches = [w for w in word_list if w[0].lower() == char.lower()]
            if matches:
                counter = scope.counters.get(char.lower(), 0)
                if counter >= len(matches): continue  # exceeded, try next list
                word = matches[counter]
                if len(matches) > 1: counter += 1  # ambiguous: increment
                scope.counters[char.lower()] = counter
                return word
    return None  # unbound
```

### Binding rules (§10.1)

- **B1**: Once bound in a scope, cannot be re-bound within that scope.
- **B2**: Characters seek bindings: inline first, then scope stack.
- **B3**: First-letter matching, case-insensitive, occurrence counter for duplicates.
- **B4**: Inline override — patches parent scope's MCS CANONIZE entry.

Note: B1 and B4 enforcement happens in the ASTEmitter, not in BindingScope. The scope provides resolution only.

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-23 | Block annotation first-letter matching with `(Mary Had A Little Lamb)` |
| KS-24 | Occurrence counter: duplicate L resolves to different words |
| KS-25 | Inline binding bypasses counter |
| KS-27 | Scope inheritance: inner to outer |
| KS-28 | Scope shadowing: inner shadows outer |
| KS-29 | Counter reset: each new scope starts at zero |
| KS-30 | Unbound character returns None |
| KS-31 | Inert annotation: no matching characters |

---

## Task 6: ASTEmitter

**File:** `kscript/ast_emitter.py`  
**Spec ref:** §3, §6–8, §10  
**Time:** 4h

This is the largest task. Build in layers: unsigned → inline ops → multi-item ops → MCS → binding → scope.

### Interface

```python
class SymbolicEntry(NamedTuple):
    sig: str
    nodes: list[str]       # always a list
    op: str                # COUNTERSIGN, CANONIZE, CONNOTATE, UNDERSIGN, UNSIGNED
    component_labels: list[str] | None = None  # resolved words per sig char

class ASTEmitter:
    def __init__(self, scope: BindingScope | None = None, dev: bool = False): ...
    def emit(self, file: KScriptFile) -> list[SymbolicEntry]: ...
```

### 6A: Entry emission and significance levels (30 min)

```python
_sig_levels = {
    "COUNTERSIGN": "S1",
    "UNDERSIGN": "S1",
    "CANONIZE": "S2",
    "CONNOTATE": "S3",
    "UNSIGNED": "S4",
}
```

`_emit_entry(sig, nodes, op)`:
- `nodes` is always a `list[str]`. Empty list for unsigned.
- No singleton unwrapping.
- No dedup (MCS dedup is separate, see 6D).

### 6B: Scope walk and unsigned/inline operators (1h)

Walk the AST, processing OperatorScope nodes:

```python
def _process_scope(self, scope: OperatorScope) -> None:
    sig = scope.sig.id
    resolved_sig = self._resolve_inline_or_scope(sig, scope.inline_annotation)
    self._emit_mcs(resolved_sig)

    if scope.op is None:
        self._emit_entry(resolved_sig, [], "UNSIGNED")
        return

    items = scope.items
    child_block = scope.child_block

    # Collect all node identifiers for this scope
    node_ids = self._collect_node_ids(items, child_block)

    if scope.op == TokenType.COUNTERSIGN:
        for node_id in node_ids:
            self._emit_entry(resolved_sig, [node_id], "COUNTERSIGN")
            self._emit_entry(node_id, [resolved_sig], "COUNTERSIGN")

    elif scope.op == TokenType.UNDERSIGN:
        for node_id in node_ids:
            if node_id == resolved_sig:
                self._emit_entry(resolved_sig, [], "UNSIGNED")
            else:
                self._emit_entry(node_id, [resolved_sig], "UNDERSIGN")

    elif scope.op == TokenType.CONNOTATE:
        for node_id in node_ids:
            self._emit_entry(resolved_sig, [node_id], "CONNOTATE")

    elif scope.op == TokenType.CANONIZE:
        self._emit_entry(resolved_sig, node_ids, "CANONIZE")

    # Recursively compile child items
    self._compile_children(items, child_block)
```

### 6C: Operator chains (1h)

When items contain OperatorScope nodes, the last node of the current scope becomes the signature for the next scope (§3 Rule S3):

```
A == B > C = D
```

The parser produces three OperatorScope nodes in sequence. The emitter processes each, and the last node of each scope feeds the next scope's signature.

For CANONIZE, items in child_block are flattened into the CANONIZE node list:

```python
def _collect_node_ids(self, items, child_block) -> list[str]:
    ids = []
    for item in items:
        if isinstance(item, Signature):
            ids.append(item.id)
        elif isinstance(item, OperatorScope):
            ids.append(item.sig.id)
    if child_block:
        # Flatten child block items into CANONIZE node list
        ids.extend(self._collect_block_ids(child_block))
    return ids
```

### 6D: MCS expansion (1h)

**Spec ref:** §8

```python
def _emit_mcs(self, sig: str) -> int | None:
    """Emit MCS entries for multi-character identifiers."""
    if len(sig) <= 1:
        return None

    chars = [self._resolve_char(c) for c in sig]
    for resolved_char in chars:
        self._emit_entry(resolved_char, [], "UNSIGNED")
    self._emit_entry(sig, chars, "CANONIZE")
    return len(self.entries) - 1  # index of CANONIZE entry for Rule B4
```

**MCS deduplication (§8.3):**

Track emitted `(sig, tuple(nodes))` pairs for CANONIZE entries only. If a subsequent CANONIZE entry would produce the same pair, skip it. This prevents duplicate MCS canonization from subscript blocks.

```python
def _emit_entry(self, sig, nodes, op):
    key = (sig, tuple(nodes))
    if op == "CANONIZE":
        if key in self._mcs_canonize_seen:
            return
        self._mcs_canonize_seen.add(key)
    # emit...
```

### 6E: Binding integration (30 min)

**Spec ref:** §10

Character resolution for MCS and signature encoding:

```python
def _resolve_char(self, char: str) -> str:
    if self._scope:
        word = self._scope.resolve(char)
        if word is not None:
            return word
    return char
```

Rule B4 override: when inline annotation fires inside a subscript, patch the parent's MCS CANONIZE entry:

```python
def _apply_inline_override(self, char: str, word: str, parent_canonize_idx: int | None):
    if parent_canonize_idx is not None:
        entry = self.entries[parent_canonize_idx]
        if isinstance(entry.nodes, list):
            for i, node in enumerate(entry.nodes):
                if node == char:
                    entry.nodes[i] = word
                    break
```

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-11 | COUNTERSIGN per-item: `A == B C` → `{A:[B]}, {B:[A]}, {A:[C]}, {C:[A]}` |
| KS-12 | UNDERSIGN per-item reversed: `A = B C` → `{B:[A]}, {C:[A]}` |
| KS-13 | CONNOTATE per-item: `A > B C` → `{A:[B]}, {A:[C]}` |
| KS-14 | CANONIZE aggregates: `A => B C D` → `{A:[B,C,D]}` |
| KS-15 | Operator chain: `A == B > C = D` → correct signatures per scope |
| KS-16 | Indent extends scope: child block items belong to parent operator |
| KS-17 | DEDENT returns to parent scope |
| KS-18 | Non-CANONIZE with indent: per-item extends into child block |
| KS-19 | MCS expansion: multi-char produces components + canonization + unsigned |
| KS-20 | No MCS for single-char identifiers |
| KS-21 | MCS on node side: `A == MHALL` triggers MCS for MHALL |
| KS-22 | Node count invariant: MCS node count equals character count |
| KS-26 | Rule B4 override: inline patches parent MCS CANONIZE |
| KS-33 | Self-identity: `A = A` → `{A:[]}` with op=UNSIGNED |
| KS-34 | Nodes always a list: `A => B` → `{A:[B]}`, `A` → `{A:[]}` |

---

## Task 7: TokenEncoder

**File:** `kscript/token_encoder.py`  
**Spec ref:** §11  
**Time:** 2h

### Interface

```python
class CompiledEntry(KLine):
    """Encoded compilation output. nodes is always list[uint64]."""
    signature: int
    nodes: list[int]          # always a list, may be empty

class TokenEncoder:
    def __init__(self, tokenizer: KTokenizer, dev: bool = False): ...
    def encode_entries(self, symbolic: list[SymbolicEntry]) -> list[CompiledEntry]: ...
```

### Encoding rules (§11)

- **Signature string** → `tokenizer.encode(sig)` → single `uint64` (or packed from multi-token).
- **Node strings** → each encoded via tokenizer → `list[uint64]`. Always a list.
- **Empty nodes** → `[]`. Never None.
- **Multi-token words** (§11.4): When a resolved word BPE-encodes to multiple tokens, run full MCS at the BPE-token level: emit unsigned entries per token, CANONIZE mapping packed signature → tokens, return packed signature as the single node.

### Multi-token word handling

```python
def _encode_node(self, word: str) -> tuple[int, list[SymbolicEntry]]:
    """Encode a word to a single uint64 node, emitting MCS entries for multi-token words."""
    tokens = self.tokenizer.encode(word)
    if len(tokens) == 1:
        return tokens[0], []
    # Multi-token: run MCS at BPE subword level
    extras = []
    for tok in tokens:
        extras.append(SymbolicEntry(tok, [], "UNSIGNED"))
    packed = 0
    for tok in tokens:
        packed |= tok
    extras.append(SymbolicEntry(packed, tokens, "CANONIZE"))
    return packed, extras
```

### Mod32 fallback (§11.3)

When a character is unbound (raw character string, not a word), encode via Mod32 bit-packed encoding. The tokenizer's `encode()` handles this based on input — single uppercase characters are Mod32-packed.

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-32 | Unbound characters use Mod32 fallback encoding |
| KS-34 | Nodes always a list (enforced by CompiledEntry type) |

---

## Task 8: Compiler (Orchestrator)

**File:** `kscript/compiler.py`, `kscript/__init__.py`  
**Spec ref:** §1.1, §12  
**Time:** 1h

### Interface

```python
class Compiler:
    def __init__(self, tokenizer: KTokenizer | None = None, dev: bool = False): ...
    def compile(self, file: KScriptFile) -> list[CompiledEntry]: ...

def compile_source(source: str, tokenizer=None, dev=False) -> list[CompiledEntry]: ...
```

### Pipeline (§1.1)

```python
def compile(self, file):
    # Create BindingScope — always active
    scope = BindingScope()
    scope.push_scope()  # Root scope

    emitter = ASTEmitter(scope=scope, dev=self.dev)
    symbolic = emitter.emit(file)

    encoder = TokenEncoder(tokenizer=self.tokenizer, dev=self.dev)
    self.entries = encoder.encode_entries(symbolic)
    return self.entries
```

The Compiler creates the BindingScope and passes it to the ASTEmitter. It then passes the symbolic output to the TokenEncoder. No encoding logic lives in the Compiler.

### Public API (§13)

```python
class KScript:
    def __init__(self, source: str, tokenizer=None, dev=False): ...
    
    @property
    def entries(self) -> list[CompiledEntry]: ...
```

### Test mapping

| Spec ID | Test |
|---------|------|
| KS-35 | Complex nested example (spec §14.11) produces correct complete entry list |
| KS-36 | NLP-bound example (spec §14.12) produces correct resolved entries |
| KS-37 | Mixed NLP/Mod32 klines: bound NLP, unbound Mod32 |

---

## Task 9: Integration Tests

**Files:** `tests/test_kscript.py`  
**Time:** 2h

### Test structure

```
TestTokenType              — §2.1 (1 test)
TestLexer                  — §2   (5 tests)
TestParserAST              — §3-4 (5 tests)
TestBindingScope           — §10  (9 tests)
TestEmitterUnsigned        — §7.1 (1 test)
TestEmitterCountersign     — §7.2 (2 tests)
TestEmitterUndersign       — §7.3 (2 tests)
TestEmitterConnotate       — §7.4 (1 test)
TestEmitterCanonize        — §7.5 (2 tests)
TestEmitterScopeChains     — §3.2 (4 tests)
TestEmitterMCS             — §8   (4 tests)
TestEmitterBinding         — §10  (4 tests)
TestComplexExamples        — §14  (5 tests)
```

### Spec example validation (§14.11)

Write a single test that compiles the §14.11 source and asserts every entry against the expected output table. This is the master regression test.

### NLP-bound example validation (§14.12)

Compile the §14.12 source with NLPTokenizer and verify binding resolution, MCS with resolved words, and Rule B4 override.

### Test helpers

```python
def compile_source(source, tokenizer=None, dev=True):
    """Compile and return entries."""
    tokens = Lexer(source).tokenize()
    kfile = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kfile)

def entries_to_dict(entries, tokenizer):
    """sig → [decoded_node_lists...] for assertion."""
    result = {}
    for e in entries:
        sig = tokenizer.decode([e.signature])
        nodes = [tokenizer.decode([n]) for n in e.nodes]
        result.setdefault(sig, []).append(nodes)
    return result

def has_entry(md, sig, nodes):
    """Check if sig has a matching node list."""
    return nodes in md.get(sig, [])
```

### Test mapping (complete)

| Spec ID | Category | Test |
|---------|----------|------|
| KS-1 | Lexer | All token types defined |
| KS-2 | Lexer | Multi-char operator priority |
| KS-3 | Lexer | BPE annotation parsing |
| KS-4 | Lexer | INDENT/DEDENT tracking |
| KS-5 | Lexer | Edge cases and errors |
| KS-6 | Parser | Scope model AST structure |
| KS-7 | Parser | Block parsing |
| KS-8 | Parser | Annotations preserved |
| KS-9 | Parser | Inline annotation attachment |
| KS-10 | Parser | Empty source |
| KS-11 | Scope | COUNTERSIGN per-item |
| KS-12 | Scope | UNDERSIGN per-item reversed |
| KS-13 | Scope | CONNOTATE per-item |
| KS-14 | Scope | CANONIZE aggregates |
| KS-15 | Scope | Operator chain |
| KS-16 | Scope | Indent extends scope |
| KS-17 | Scope | DEDENT returns to parent |
| KS-18 | Scope | Non-CANONIZE with indent |
| KS-19 | MCS | Multi-char expansion |
| KS-20 | MCS | No single-char expansion |
| KS-21 | MCS | MCS on node side |
| KS-22 | MCS | Node count invariant |
| KS-23 | Binding | Block first-letter matching |
| KS-24 | Binding | Occurrence counter |
| KS-25 | Binding | Inline binding |
| KS-26 | Binding | Rule B4 override |
| KS-27 | Binding | Scope inheritance |
| KS-28 | Binding | Scope shadowing |
| KS-29 | Binding | Counter reset |
| KS-30 | Binding | Unbound characters |
| KS-31 | Binding | Inert annotation |
| KS-32 | Encoding | Mod32 fallback |
| KS-33 | Operators | Self-identity |
| KS-34 | Structure | Nodes always a list |
| KS-35 | Integration | §14.11 complex nested |
| KS-36 | Integration | §14.12 NLP-bound |
| KS-37 | Integration | Mixed NLP/Mod32 |

---

## Module Structure (final)

```
kscript/
├── __init__.py         # KScript class, re-exports
├── token.py            # TokenType enum, Token dataclass
├── lexer.py            # Lexer
├── ast.py              # AST node dataclasses (Annotation, OperatorScope, Block)
├── parser.py           # Parser (scope-model aware)
├── binding_scope.py    # BindingScope
├── ast_emitter.py      # ASTEmitter (AST → symbolic entries)
├── token_encoder.py    # TokenEncoder + CompiledEntry
└── compiler.py         # Compiler orchestrator + compile_source()
```

**Total files:** 9  
**Deleted from old codebase:** `decompiler.py`, `output.py`, `__main__.py`, `nlp_types.py`
**Estimated LOC:** ~1800  
**Total test count:** ~47

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| No decompiler, no file I/O, no CLI | Spec scope reduction. Compilation output is the product. Scripts handle format conversion. |
| No `chain_right` | Scope model (§3). Each operator creates a scope boundary. |
| `nodes: list[uint64]` always | Eliminates None checks, singleton unwrapping, type branching. Uniform structure. |
| BindingScope always active | Mod32 is the fallback for unbound characters. |
| MCS dedup on CANONIZE only | Prevents duplicate entries from MCS + subscript CANONIZE overlap. Not a general dedup mechanism. |
| `Annotation` replaces `Comment` | Terminology reflects purpose: these are BPE encoding inputs, not inert documentation. |
| Multi-token MCS in TokenEncoder | BPE subword decomposition is an encoding concern. The ASTEmitter works with symbolic strings. |
