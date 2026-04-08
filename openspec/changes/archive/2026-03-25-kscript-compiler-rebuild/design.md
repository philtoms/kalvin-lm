## Context

The KScript compiler is a core component that transforms `.ks` source files into KLine graph representations. The current implementation exists but lacks formal verification against a specification. A comprehensive spec now exists at `openspec/specs/kscript-compiler/spec.md` covering 28 requirements across lexer, parser, AST, compiler, and output modules.

This rebuild implements the compiler from the spec as the source of truth, enabling:
- Verifiable correctness through spec-conformance testing
- Maintainable codebase with clear module boundaries
- Onboarding documentation via the spec itself

## Goals / Non-Goals

**Goals:**
- Implement all 28 requirements from `openspec/specs/kscript-compiler/spec.md`
- Maintain API compatibility with existing `KScript` class
- Preserve CLI interface (`python -m kscript`)
- Support all output formats: JSON, JSONL, binary (KSC1)
- Implement error recovery semantics per spec

**Non-Goals:**
- New language features beyond the spec
- Performance optimization (correctness first)
- IDE/language server integration
- Source map generation

## Decisions

### D1: Module structure follows spec domains
**Decision:** Organize modules by spec domain (lexer, parser, compiler, output) rather than by feature.

**Rationale:** The spec is organized by domain, making verification straightforward. Each module maps to a spec section.

**Alternatives considered:**
- Pipeline-oriented (stages.py, passes.py) - harder to map to spec requirements

### D2: CompiledEntry extends KLine
**Decision:** Keep `CompiledEntry` as a subclass of `kalvin.abstract.KLine`.

**Rationale:** Existing integration with Kalvin's `rationalise()` expects KLine objects. The inheritance provides:
- `signature: int` - token ID
- `nodes: int | None | list[int]` - child references
- `dbg_text: str` - debug info

**Alternatives considered:**
- Composition over inheritance - would require adapter layer

### D3: PACKED_BIT for signature vs literal distinction
**Decision:** Use bit 0 (PACKED_BIT) to distinguish signatures from literals at decode time.

**Rationale:**
- Signatures: encoded with `PACKED_BIT` set (bit 0 = 1)
- Literals: encoded as `(ord(char) << 1)` (bit 0 = 0)
- Enables lossless round-trip decode without external metadata

**Alternatives considered:**
- Separate type field in entry - increases binary size
- Prefix byte in encoding - breaks token ID alignment

### D4: Recursive descent parser
**Decision:** Use hand-written recursive descent parser.

**Rationale:**
- Clear mapping to grammar rules in spec
- Easy to implement immediate binding semantics
- Good error recovery opportunities
- No external parser generator dependency

**Alternatives considered:**
- PEG parser library - overkill for this grammar
- Regex-based parsing - can't handle nested subscripts cleanly

### D5: Multi-pass compilation
**Decision:** Compile in single pass, emitting entries immediately.

**Rationale:** The grammar is designed for single-pass compilation with immediate binding. No forward references needed since source order isn't a semantic requirement.

**Alternatives considered:**
- Two-pass (collect then emit) - unnecessary complexity

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KScript Compiler                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Source String                                                     │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────┐     ┌─────────┐     ┌───────────┐     ┌────────┐     │
│   │  Lexer  │────▶│  Parser │────▶│ Compiler  │────▶│ Output │     │
│   └─────────┘     └─────────┘     └───────────┘     └────────┘     │
│        │               │                 │               │          │
│      tokens          AST             entries         .json/.bin    │
│   (list[Token])   (KScriptFile)   (list[Entry])                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Data Flow:
  source → Lexer.tokenize() → Parser.parse() → Compiler.compile() → entries
                                                                     ↓
                                                              Output.write_*()
```

## Module Interfaces

### token.py
```python
class TokenType(Enum):
    COUNTERSIGN, CANONIZE_FWD, CANONIZE_BWD,
    CONNOTATE_FWD, CONNOTATE_BWD, UNDERSIGN,
    SIGNATURE, STRING, NUMBER, COMMENT,
    NEWLINE, INDENT, DEDENT, EOF

@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int
    column: int
```

### lexer.py
```python
class Lexer:
    def __init__(self, source: str): ...
    def tokenize(self) -> list[Token]: ...
```

### ast.py
```python
@dataclass
class Signature:
    id: str
    comment: str | None
    line: int
    column: int

@dataclass
class Construct:
    type: ConstructType
    nodes: list[Node]
    line: int
    has_leading_nodes: bool = False

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

### parser.py
```python
class Parser:
    def __init__(self, tokens: list[Token]): ...
    def parse(self) -> KScriptFile: ...
```

### compiler.py
```python
class CompiledEntry(KLine):
    @classmethod
    def encode(cls, sig: str, nodes: ..., tokenizer: ModTokenizer,
               *, nodes_are_literals: bool = False) -> "CompiledEntry": ...
    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]: ...

class Compiler:
    def __init__(self, tokenizer: ModTokenizer | None = None): ...
    def compile(self, file: KScriptFile) -> list[CompiledEntry]: ...
```

### output.py
```python
def write_json(entries: list[CompiledEntry], path: Path, tokenizer: ModTokenizer) -> None: ...
def write_jsonl(entries: list[CompiledEntry], path: Path, tokenizer: ModTokenizer) -> None: ...
def write_bin(entries: list[CompiledEntry], path: Path) -> None: ...
def read_json(path: Path, tokenizer: ModTokenizer) -> list[CompiledEntry]: ...
def read_bin(path: Path) -> list[CompiledEntry]: ...
```

### __init__.py (API)
```python
class KScript:
    def __init__(self, source: str | Path, base: KScript | None = None,
                 tokenizer: ModTokenizer | None = None): ...
    @property
    def entries(self) -> list[CompiledEntry]: ...
    def output(self, path: str | Path) -> None: ...
    def to_jsonl(self) -> list[str]: ...
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Breaking existing TUI integration | Keep `KScript` API identical; add integration tests |
| Literal encoding edge cases | Test all escape sequences; verify round-trip |
| Indentation handling differences | Follow Python's model exactly; test mixed tabs/spaces |
| Binary format versioning | Use KSC1 magic; reserve header space for future versions |
| Duplicate signature handling | Never merge; always emit separate entries |
