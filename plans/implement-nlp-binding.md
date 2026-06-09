# KScript NLP Binding Implementation Plan

**Specs:** @kscript-nlp-binding, @kscript-nlp, @kscript, @signature  
**Date:** 2026-06-09  
**Status:** Planning  
**Assumes:** KScript compiler pipeline (lexer → parser → ASTEmitter → TokenEncoder), NLPTokenizer, existing NLP-BPE encoding support

---

## Dependency Graph

```
[Existing: Parser, AST, ASTEmitter, TokenEncoder, Decompiler, NLPTokenizer]
          │
          ▼
   Task 1: AST changes — Comment node type, inline comment field (1h)
          │
          ▼
   Task 2: Parser changes — preserve comments in AST (1.5h)
          │
          ▼
   Task 3: NLPSymbolTable data structure (0.5h)
          │
          ▼
   Task 4: BindingResolver — scope walk, binding algorithm (3h)
          │
          ├───┐
          ▼   ▼
   Task 5: Compiler integration (1h)    Task 7: Decompiler NLP updates (1h)
          │
          ▼
   Task 6: ASTEmitter integration — consult symbol table (1.5h)
          │
          ▼
   Task 8: Integration tests — full MHALL example (1.5h)
          │
          ▼
   Task 9: Compatibility tests — Mod32 unchanged, same source both modes (1h)

Total: ~12h (1.5 days)
```

Tasks 5 and 7 are independent once Task 4 is done.

---

## Task 1: AST Changes

**Files:** `src/kscript/ast.py`  
**Spec ref:** @kscript-nlp-binding §5.1  
**Test matrix:** NB-20, NB-21

### Deliverable

Add `Comment` dataclass to AST:

```python
@dataclass
class Comment:
    """A comment preserved in the AST for NLP binding.

    Attributes:
        text: Raw comment text including parentheses, e.g. "(ubject)"
        line: 1-based line number
        column: 1-based column number
    """
    text: str
    line: int
    column: int
```

Update `ConstructItem` type alias to include `Comment`:

```python
ConstructItem: TypeAlias = "PrimaryConstruct | Literal | Comment"
```

Update `PrimaryConstruct` to carry optional inline comment:

```python
@dataclass
class PrimaryConstruct:
    sig: Signature
    op: TokenType | None = None
    node: Node | None = None
    inline_comment: Comment | None = None  # NEW
```

Update `Construct.inner` type to include `Comment` in the union — comments can appear as standalone constructs in the block sequence.

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-20 | `test_comment_ast_node` — parse comment produces Comment node | ○ |
| NB-21 | `test_inline_comment_attached` — `S(ubject)` produces PrimaryConstruct with inline_comment | ○ |

---

## Task 2: Parser Changes

**Files:** `src/kscript/parser.py`  
**Spec ref:** @kscript-nlp-binding §5.2, §5.3  
**Test matrix:** NB-20, NB-21, NB-22

### Deliverable

Modify `_skip_insignificant` to stop skipping COMMENT tokens:

```python
def _skip_insignificant(self) -> None:
    """Skip NEWLINE tokens only. COMMENT tokens are now preserved."""
    while not self._at_end() and self._peek().type == TokenType.NEWLINE:
        self._advance()
```

Add comment handling in `_parse_construct`:

```python
def _parse_construct(self) -> Construct:
    self._skip_insignificant()

    if self._check(TokenType.INDENT):
        return self._parse_block()

    if self._check(TokenType.COMMENT):
        return self._parse_comment_construct()

    # ... rest unchanged
```

Add `_parse_comment_construct`:

```python
def _parse_comment_construct(self) -> Construct:
    """Parse a COMMENT token as a Comment AST node."""
    token = self._expect(TokenType.COMMENT)
    comment = Comment(text=token.value, line=token.line, column=token.column)
    return Construct(comment)
```

Modify `_parse_primary_construct` to detect inline comments (COMMENT token immediately after SIGNATURE, before any operator):

```python
def _parse_primary_construct(self) -> PrimaryConstruct:
    sig = self._parse_sig()

    # Check for inline comment: SIGNATURE followed by COMMENT (no operator between)
    inline_comment = None
    if self._check(TokenType.COMMENT):
        token = self._advance()
        inline_comment = Comment(text=token.value, line=token.line, column=token.column)

    op = self._try_inline_op()
    if op is not None:
        node = self._parse_node()
        return PrimaryConstruct(sig, op, node, inline_comment=inline_comment)

    return PrimaryConstruct(sig, inline_comment=inline_comment)
```

Update `Construct.inner` type union to include `Comment` alongside `Block`, `Literal`, and `list[PrimaryConstruct]`.

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-20 | `test_comment_in_construct_sequence` — comment appears between signatures in AST | ○ |
| NB-21 | `test_inline_comment_on_primary` — `S(ubject) = M` has inline_comment on S | ○ |
| NB-22 | `test_grammar_unchanged` — existing parser tests pass without modification | ○ |

---

## Task 3: NLP Symbol Table

**Files:** new `src/kscript/symbol_table.py`  
**Spec ref:** @kscript-nlp-binding §6.1  
**Test matrix:** NB-4, NB-10

### Deliverable

```python
@dataclass
class Binding:
    """A single character→NLP word binding."""
    char: str          # Single character, e.g. "M"
    word: str          # NLP word, e.g. "Mary"
    consumed: bool = False

class Scope:
    """A lexical scope containing bindings."""

    def __init__(self, parent: Scope | None = None):
        self.parent = parent
        self.bindings: dict[str, list[Binding]] = {}
        self.pending_comment: list[str] | None = None  # Pending word list

    def bind(self, char: str, word: str) -> None:
        """Add a binding for char in this scope."""
        ...

    def resolve(self, char: str) -> str | None:
        """Resolve a character to its NLP word.

        Searches this scope first, then parent. Returns None if unbound.
        Does NOT consume bindings — consumption only applies to word list claiming.
        """
        ...

    def claim_next(self, char: str) -> Binding | None:
        """Claim the next unconsumed binding for char in this scope.

        Used for word list positional consumption. Returns None if all
        bindings for this char are consumed.
        """
        ...


class NLPSymbolTable:
    """Maps single-character signatures to NLP words."""

    def __init__(self):
        self._scopes: list[Scope] = []  # Stack
        self._resolved: dict[tuple[str, int], str] = {}  # (char, ast_node_id) → word

    def push_scope(self) -> Scope:
        """Push a new scope (inheriting current top)."""
        ...

    def pop_scope(self) -> None:
        """Pop the current scope."""
        ...

    def current_scope(self) -> Scope:
        """Return the current (top) scope."""
        ...

    def bind(self, char: str, word: str) -> None:
        """Bind char to word in current scope."""
        ...

    def resolve(self, char: str) -> str | None:
        """Resolve char through scope chain. Returns None if unbound."""
        ...

    def is_active(self) -> bool:
        """Whether this symbol table has been populated (NLP mode)."""
        return len(self._resolved) > 0
```

### Design decisions

- `Scope` supports duplicate characters via `list[Binding]` — each position is a separate binding.
- `claim_next` consumes in order for positional word list resolution.
- `resolve` searches current scope first, then parent (standard lexical scoping).
- The symbol table is populated by BindingResolver and read by ASTEmitter.

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-4 | `test_positional_binding` — `(Mary had a little lamb)` + MHALL binds 5 chars | ○ |
| NB-10 | `test_consumption_order` — `(Alice Alpha)` + AA binds A#0→Alice, A#1→Alpha | ○ |

---

## Task 4: Binding Resolver

**Files:** new `src/kscript/binding_resolver.py`  
**Spec ref:** @kscript-nlp-binding §6  
**Test matrix:** NB-1 through NB-13

### Deliverable

```python
class BindingResolver:
    """Walks KScript AST and builds NLP symbol table from comment word lists."""

    def resolve(self, file: KScriptFile) -> NLPSymbolTable:
        """Build symbol table from AST."""
        table = NLPSymbolTable()
        table.push_scope()  # Root scope
        for script in file.scripts:
            self._resolve_constructs(script.constructs, table)
        table.pop_scope()
        return table

    def _resolve_constructs(self, constructs: list[Construct], table: NLPSymbolTable) -> None:
        """Process a sequence of constructs."""
        for construct in constructs:
            self._resolve_construct(construct, table)

    def _resolve_construct(self, construct: Construct, table: NLPSymbolTable) -> None:
        """Process a single construct."""
        inner = construct.inner

        if isinstance(inner, Comment):
            self._handle_comment(inner, table)
            return

        if isinstance(inner, Block):
            table.push_scope()
            self._resolve_constructs(inner.constructs, table)
            table.pop_scope()
            return

        if isinstance(inner, Literal):
            return  # Literals don't participate in binding

        # list[PrimaryConstruct]
        for pc in inner:
            self._resolve_primary(pc, table)

        # Chain right — subscript block gets its own scope
        if construct.chain_right is not None:
            table.push_scope()
            self._resolve_construct(construct.chain_right, table)
            table.pop_scope()

    def _resolve_primary(self, pc: PrimaryConstruct, table: NLPSymbolTable) -> None:
        """Process a primary construct."""
        sig = pc.sig.id

        # Inline binding: S(ubject) → bind S to "Subject"
        if pc.inline_comment is not None:
            word = self._extract_inline_word(sig, pc.inline_comment)
            table.bind(sig, word)

        # Try to claim pending word list for this signature
        # (multi-char MCS claiming)
        elif len(sig) > 1:
            self._try_claim_word_list(sig, table)

        # For single-char signatures without inline binding:
        # resolution happens at encoding time via table.resolve()

    def _handle_comment(self, comment: Comment, table: NLPSymbolTable) -> None:
        """Store comment as pending word list."""
        words = self._extract_words(comment.text)
        if words:
            table.current_scope().pending_comment = words

    def _try_claim_word_list(self, sig: str, table: NLPSymbolTable) -> None:
        """Try to claim the pending word list for this multi-char signature."""
        scope = table.current_scope()
        if scope.pending_comment is None:
            return
        words = scope.pending_comment
        if len(words) != len(sig):
            return  # Mismatch — comment is inert
        # Positional zip
        for char, word in zip(sig, words):
            table.bind(char, word)
        scope.pending_comment = None  # Consumed

    def _extract_inline_word(self, sig_char: str, comment: Comment) -> str:
        """Extract word from inline comment: S + ubject → Subject."""
        # Strip parentheses from comment text: "(ubject)" → "ubject"
        text = comment.text
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return sig_char + text

    def _extract_words(self, comment_text: str) -> list[str]:
        """Extract word list from comment: "(Mary had a little lamb)" → ["Mary", "had", ...]"""
        text = comment_text.strip()
        if text.startswith("("):
            text = text[1:]
        if text.endswith(")"):
            text = text[:-1]
        return text.split()
```

### Algorithm walkthrough (MHALL example)

```
Input AST:
  Comment("(Mary had a little lamb)")
  Construct([MHALL == SVO] => Block([
    PrimaryConstruct(S, =, M, inline=Comment("(ubject)"))
    PrimaryConstruct(V, =, H, inline=Comment("(erb)"))
    PrimaryConstruct(O, =, ALL) => Block([
      PrimaryConstruct(A, >, D, inline=Comment("(et)"))
      PrimaryConstruct(L, >, M, inline=Comment("(od)"))
      PrimaryConstruct(L, >, O)
    ])
  ]))

Walk:
  1. Comment → pending = ["Mary", "had", "a", "little", "lamb"]
  2. MHALL == SVO (MCS) → claim pending: M→Mary, H→had, A→a, L→little, L→lamb
  3. Push scope (subscript)
     4. S(ubject) → inline bind: S→Subject
     5. M resolves via table.resolve("M") → "Mary" (from parent scope)
     6. V(erb) → inline bind: V→Verb
     7. O(bject) → inline bind: O→Object
     8. Push scope (ALL subscript)
        9. A resolves → "a"
        10. D(et) → inline bind: D→Det (shadows nothing, new binding)
        11. L resolves → L#0 "little" (first unconsumed L in parent)
        12. M(od) → inline bind: M→Mod (shadows M→"Mary" in this scope)
        13. L resolves → L#1 "lamb" (next unconsumed L in parent)
        14. O resolves → "Object" (from parent scope)
     15. Pop scope (M→"Mary" restored)
  16. Pop scope
```

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-1 | `test_inline_binding_subject` | ○ |
| NB-2 | `test_inline_binding_verb` | ○ |
| NB-3 | `test_inline_binding_det` | ○ |
| NB-4 | `test_block_word_list_claiming` | ○ |
| NB-5 | `test_word_list_mismatch_inert` | ○ |
| NB-6 | `test_orphan_comment_inert` | ○ |
| NB-7 | `test_multiple_pending_comments` | ○ |
| NB-8 | `test_upward_traversal` | ○ |
| NB-9 | `test_downward_traversal` | ○ |
| NB-10 | `test_binding_consumption` | ○ |
| NB-11 | `test_duplicate_char_disambiguation` | ○ |
| NB-12 | `test_lexical_shadowing` | ○ |
| NB-13 | `test_scope_restoration` | ○ |

---

## Task 5: Compiler Integration

**Files:** `src/kscript/compiler.py`  
**Spec ref:** @kscript-nlp-binding §1.1  
**Test matrix:** NB-18, NB-19

### Deliverable

Wire BindingResolver into the Compiler pipeline:

```python
class Compiler:
    def __init__(self, tokenizer: KTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        # Build NLP symbol table (only if NLP tokenizer)
        symbol_table = NLPSymbolTable()
        if not self.tokenizer.supports_mcs:
            # NLP mode — run binding resolver
            from .binding_resolver import BindingResolver
            resolver = BindingResolver()
            symbol_table = resolver.resolve(file)

        emitter = ASTEmitter(dev=self.dev, skip_mcs=not self.tokenizer.supports_mcs,
                             symbol_table=symbol_table)
        symbolic = emitter.emit(file)

        encoder = TokenEncoder(tokenizer=self.tokenizer, dev=self.dev,
                               symbol_table=symbol_table)
        self.entries = encoder.encode_entries(symbolic)
        return self.entries
```

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-18 | `test_mod32_compilation_unchanged` — existing tests pass without regression | ○ |
| NB-19 | `test_same_source_both_modes` — MHALL script compiles under Mod32 and NLP | ○ |

---

## Task 6: ASTEmitter Integration

**Files:** `src/kscript/ast_emitter.py`  
**Spec ref:** @kscript-nlp-binding §7.4  
**Test matrix:** NB-15, NB-16, NB-17

### Deliverable

Extend ASTEmitter to accept and consult the NLPSymbolTable:

```python
class ASTEmitter:
    def __init__(self, dev: bool = False, skip_mcs: bool = False,
                 symbol_table: NLPSymbolTable | None = None):
        ...
        self._symbol_table = symbol_table

    def _resolve_sig_word(self, sig_char: str) -> str:
        """Resolve a single-char signature to its NLP word, or return the char itself."""
        if self._symbol_table:
            word = self._symbol_table.resolve(sig_char)
            if word is not None:
                return word
        return sig_char
```

Modify `_emit_entry` to use resolved words when building symbolic entries. When a binding exists, the symbolic entry's `sig` field carries the NLP word instead of the raw character. The TokenEncoder then encodes that word through the NLP tokenizer.

For MCS decomposition (`_emit_mcs`), each component character is resolved individually before constructing the canonization entry.

### Key insight

The ASTEmitter stays in the symbolic (string) domain. It does not encode tokens. It merely changes *which string* it passes downstream — the raw character or the resolved NLP word. The TokenEncoder handles all numeric encoding.

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-15 | `test_nlp_bound_signature_has_nlp_type_bits` | ○ |
| NB-16 | `test_nlp_bound_node_has_bpe_id` | ○ |
| NB-17 | `test_mixed_mcs_signature` | ○ |

---

## Task 7: Decompiler NLP Updates

**Files:** `src/kscript/decompiler.py`  
**Spec ref:** @kscript-nlp-binding §8  
**Test matrix:** NB-24, NB-25

### Deliverable

Extend `_decode_sig` to handle NLP-type-only signatures:

```python
def _decode_sig(self, sig: int) -> str:
    """Decode signature to string."""
    if sig in self._mcs_names:
        return self._mcs_names[sig]

    if is_literal_node(sig):
        return self._decode_node(sig)

    # Try direct decode (works for Mod32 and single-token NLP)
    result = self.tokenizer.decode([sig])
    if result:
        return result

    # NLP type signature (OR'd types, not a single token)
    if is_nlp_node(sig):
        return self._describe_nlp_type(sig)

    return f"<{sig}>"

def _describe_nlp_type(self, sig: int) -> str:
    """Describe the set NLP type bits in a signature."""
    # Extract high 32 bits and describe which POS/DEP/MORPH flags are set
    # Returns something like "<PROPN|VERB|DET|ADJ|NOUN>"
    ...
```

Node decoding already works for NLP-BPE nodes via `_decode_node` (which calls `tokenizer.decode([node])`). No changes needed there.

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-24 | `test_decompile_nlp_node_readable` — node side decodes to word | ○ |
| NB-25 | `test_decompile_nlp_signature_type_description` — sig side shows type bits | ○ |

---

## Task 8: Integration Tests

**Files:** `tests/test_nlp_binding.py`  
**Spec ref:** @kscript-nlp-binding §10  
**Test matrix:** NB-23

### Deliverable

Full end-to-end test of the MHALL script:

```python
def test_mhall_nlp_binding_full():
    """The complete MHALL example produces correct NLP-bound entries."""
    source = """(Mary had a little lamb)
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A > D(et)
     L > M(od)
     L > O"""
    entries = compile_nlp(source)

    # Verify binding resolver produced correct symbol table
    # Verify compiled entries have correct NLP signatures and nodes
    # Verify all 11 bindings are resolved
    ...
```

Additional integration tests:

- Script with unbound characters (mixed NLP/Mod32)
- Script with shadowing and scope restoration
- Script with mismatched comment (inert)
- Same source compiled under Mod32 produces existing output

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-23 | `test_mhall_nlp_binding_full` | ○ |

---

## Task 9: Compatibility Tests

**Files:** `tests/test_nlp_binding.py`  
**Spec ref:** @kscript-nlp-binding §10  
**Test matrix:** NB-18, NB-19, NB-26

### Deliverable

- All existing KScript tests pass without modification under Mod32
- All existing KScript tests pass under NLP without binding (no comments in source)
- Same source with comments compiles identically under Mod32 (comments are inert)
- Significance routing works on NLP-bound klines

### Test mapping

| Spec ID | Test | Status |
|---------|------|--------|
| NB-18 | `test_existing_mod32_tests_pass` | ○ |
| NB-19 | `test_same_source_both_modes` | ○ |
| NB-26 | `test_significance_routing_nlp_bound` | ○ |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Symbol table operates on strings, not token IDs | Keeps binding resolver tokenizer-agnostic; encoding happens in TokenEncoder |
| Binding resolver is a separate pass, not folded into ASTEmitter | Clean separation: resolver builds bindings, emitter consumes them |
| Scope stack in symbol table, not resolver | Symbol table owns scoping state; resolver drives the walk |
| Mod32 fallback for unbound characters | Graceful degradation; no compilation errors for unbound chars |
| Comments as AST nodes, not attached metadata | Uniform treatment; binding resolver walks construct sequence naturally |

---

## Status

| Task | Status | Notes |
|------|--------|-------|
| Task 1: AST changes | ○ Not started | |
| Task 2: Parser changes | ○ Not started | |
| Task 3: NLPSymbolTable | ○ Not started | |
| Task 4: BindingResolver | ○ Not started | Core algorithm |
| Task 5: Compiler integration | ○ Not started | |
| Task 6: ASTEmitter integration | ○ Not started | |
| Task 7: Decompiler NLP updates | ○ Not started | |
| Task 8: Integration tests | ○ Not started | |
| Task 9: Compatibility tests | ○ Not started | |
