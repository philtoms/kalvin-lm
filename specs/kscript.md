# KScript Language Specification

**Version:** 3.0  
**Date:** 2026-06-12  
**Status:** Authoritative

---

## 1. Overview

KScript is a domain-specific language for constructing **knowledge graphs** — ordered sequences of identified nodes called **KLines**. It compiles declarative scripts into a list of compiled entries that can be loaded into a Kalvin Agent for rationalisation.

### 1.1 Compilation Pipeline

```
Source (.ks) → Lexer → Parser → AST → ASTEmitter → TokenEncoder → [CompiledEntry]
```

| Stage | Responsibility |
|-------|---------------|
| **Lexer** | Source text → token stream |
| **Parser** | Token stream → AST (structural) |
| **ASTEmitter** | AST → symbolic entries (strings). Resolves BPE annotations inline via BindingScope. |
| **TokenEncoder** | Symbolic entries → encoded entries (uint64 values). Encoding-agnostic. |

The pipeline is strictly one-directional. The ASTEmitter operates on symbolic strings — no encoding happens there. The TokenEncoder converts strings to opaque `uint64` values via a pluggable tokenizer.

### 1.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Node** | An opaque `uint64` value — the universal atom |
| **KLine** | An identified, ordered sequence of zero or more nodes: `{signature, nodes[]}` |
| **Signature** | A `uint64` identity key computed via OR-reduction of node values |
| **Significance** | A four-level classification (S1–S4) of how strongly one KLine relates to another |
| **Identifier** | An uppercase alpha string like `ABC` — the lexical form of a signature or node |
| **BPE Annotation** | A parenthesised annotation `(...)` providing word text for BPE token encoding |
| **Scope** | An operator-delimited region that determines kline structure and binding resolution |

---

## 2. Lexical Structure

### 2.1 Token Types

| Token | Pattern | Category |
|-------|---------|----------|
| `SIGNATURE` | `[A-Z]+` | Identifier |
| `COUNTERSIGN` | `==` | Operator |
| `CANONIZE` | `=>` | Operator |
| `CONNOTATE` | `>` | Operator |
| `UNDERSIGN` | `=` | Operator |
| `ANNOTATION` | `(...)` with nested parens | BPE Annotation |
| `NEWLINE` | `\n` | Insignificant |
| `INDENT` | Increased indentation | Structural |
| `DEDENT` | Decreased indentation | Structural |
| `EOF` | End of file | Sentinel |

### 2.2 Operator Classification

| Operator | Symbol | Behaviour |
|----------|--------|-----------|
| COUNTERSIGN | `==` | Bidirectional, per-item emission |
| UNDERSIGN | `=` | Unidirectional (reversed), per-item emission |
| CANONIZE | `=>` | Unidirectional, all items aggregated into single kline |
| CONNOTATE | `>` | Unidirectional, per-item emission |

### 2.3 Lexing Rules

1. **Multi-character operators** (`==`, `=>`) are matched before single-character operators (`=`, `>`).
2. **Identifiers** `[A-Z][A-Z0-9]*` are classified as `SIGNATURE` if all characters are uppercase alpha. Non-uppercase identifiers raise a `LexerError`.
3. **BPE Annotations** `(...)` support nested parentheses and may span multiple lines.
4. **Inline annotations** after an identifier are preserved as `ANNOTATION` tokens attached to the preceding signature.
5. **Indentation** uses Python-style INDENT/DEDENT tokens based on leading whitespace.
6. **Unknown characters** (not matching any rule, including `<`) raise a `LexerError`.

### 2.4 Indentation

Indentation is counted as the number of leading spaces/tabs. At each line start, the indent level is compared to the stack:

- Greater → emit `INDENT`, push new level
- Less → emit one or more `DEDENT` tokens, pop to matching level
- Equal → no structural token

At EOF, remaining levels are closed with `DEDENT` tokens.

---

## 3. Scope Model

Scope is the central organising principle of KScript compilation. Scope determines both kline structure (what becomes a signature, what becomes a node) and binding resolution (how BPE annotations map characters to words).

### 3.1 Scope Rules

**Rule S1 — Scope is operator-delimited.** Each operator (`==`, `=`, `>`, `=>`) creates a scope boundary.

**Rule S2 — Preceding identifier is the signature.** The identifier immediately preceding an operator is that scope's signature.

**Rule S3 — Succeeding identifiers are nodes.** The identifiers immediately succeeding the operator are nodes in that scope. The last node becomes the signature for the next operator's scope.

**Rule S4 — Indent creates child scope.** Indented lines extend the current operator's scope as a child. The signature carries forward from the parent operator.

**Rule S5 — DEDENT returns to parent scope.** Indentation decreases close the child scope and resume the parent.

**Rule S6 — CANONIZE aggregates.** All nodes in a `=>` scope form a single kline's node list.

**Rule S7 — Other operators emit per-item.** Each node in a non-CANONIZE scope produces its own kline (with operator-specific directionality).

### 3.2 Scope Examples

#### Simple chain

```
A == B > C = D
```

| Scope | Operator | Signature | Nodes | Compiled |
|-------|----------|-----------|-------|----------|
| 1 | `==` | A | B | `{A: B}, {B: A}` |
| 2 | `>` | B | C | `{B: C}` |
| 3 | `=` | C | D | `{D: C}` |

#### CANONIZE aggregation

```
A => B C D
```

| Scope | Operator | Signature | Nodes | Compiled |
|-------|----------|-----------|-------|----------|
| 1 | `=>` | A | B, C, D | `{A: [B, C, D]}` |

#### Mixed operators with CANONIZE

```
A == B => C D
```

| Scope | Operator | Signature | Nodes | Compiled |
|-------|----------|-----------|-------|----------|
| 1 | `==` | A | B | `{A: B}, {B: A}` |
| 2 | `=>` | B | C, D | `{B: [C, D]}` |

#### CANONIZE followed by inline

```
A => B == C
```

| Scope | Operator | Signature | Nodes | Compiled |
|-------|----------|-----------|-------|----------|
| 1 | `=>` | A | B | `{A: B}` |
| 2 | `==` | B | C | `{B: C}, {C: B}` |

#### Indented child scope

```
A =>
  B = C
  D > E
```

| Scope | Operator | Signature | Nodes | Compiled |
|-------|----------|-----------|-------|----------|
| 1 | `=>` | A | B, D (from child scope) | `{A: [B, D]}` |
| 1a (child) | `=` | B | C | `{C: B}` |
| 1b (child) | `>` | D | E | `{D: E}` |

Child-scope items B and D are the CANONIZE nodes for scope 1. Each child item is also compiled recursively.

#### Non-CANONIZE with indent

```
A == B
  C
  D
```

| Scope | Operator | Signature | Nodes | Compiled |
|-------|----------|-----------|-------|----------|
| 1 | `==` | A | B, C, D (from child scope) | `{A: B}, {B: A}, {A: C}, {C: A}, {A: D}, {D: A}` |

Indentation extends the enclosing operator's scope. The signature carries forward.

---

## 4. Grammar

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

### 4.1 Key Constraints

1. Only SIGNATUREs can be identifiers (signatures or nodes).
2. NEWLINE tokens are insignificant — skipped between constructs.
3. ANNOTATION tokens are preserved as AST nodes for binding resolution.
4. Empty source produces empty script (no error).

---

## 5. AST Structure

```
KScriptFile
  └── scripts: [Script]
        └── constructs: [ConstructItem]

ConstructItem = Annotation | OperatorScope

Annotation
  ├── text: str         # raw annotation text including parentheses
  ├── line: int
  └── column: int

OperatorScope
  ├── sig: Signature           # the identifier preceding the operator
  ├── op: TokenType | None     # None = bare signature (unsigned)
  ├── items: [ConstructItem]   # nodes and child constructs
  └── child_block: Block | None  # indented child scope

Block
  └── constructs: [ConstructItem]

Signature
  ├── id: str           # uppercase
  ├── line: int
  └── column: int
```

### 5.1 Inline Annotations

Inline annotations appear immediately after a SIGNATURE token:

- **Sig-side**: `S(ubject) = M` — annotation attached to the signature `S`
- **Node-side**: `A = D(et)` — annotation attached to the node `D`

The parser attaches inline annotations to the nearest `Signature` node.

---

## 6. Compilation Semantics

### 6.1 Entry Model

Each compiled entry is a **KLine**:

```
CompiledEntry:
    signature: uint64              # encoded identity key
    nodes: list[uint64]             # encoded node values (always a list, may be empty)
```

**No singleton rule.** Nodes are always a list. An unsigned entry has an empty list. A single-node entry has a one-element list. A multi-node entry has multiple elements.

### 6.2 Significance Level Assignment

Each emitted entry is tagged with a significance level based on the operator that produced it:

| Operator | Level | Meaning |
|----------|-------|---------|
| COUNTERSIGN (`==`) | S1 | Mutual / bidirectional |
| UNDERSIGN (`=`) | S1 | Unconditional |
| CANONIZE (`=>`) | S2 | Canonical |
| CONNOTATE (`>`) | S3 | Connotative |
| UNSIGNED (bare) | S4 | Identity only |

Significance bits are not encoded into the token IDs. The level is carried as metadata on the compiled entry.

---

## 7. Operator Compilation Rules

### 7.1 Unsigned (bare signature)

```
A       → {A: []}
```

Plus MCS expansion if multi-character (§8).

### 7.2 COUNTERSIGN (`==`) — bidirectional, per-item

```
A == B       → {A: B}, {B: A}
A == B C D   → {A: B}, {B: A}, {A: C}, {C: A}, {A: D}, {D: A}
```

Each node produces a bidirectional pair.

### 7.3 UNDERSIGN (`=`) — unidirectional reversed, per-item

```
A = B       → {B: A}
A = A       → {A: []}              (self-identity, UNSIGNED)
A = B C D   → {B: A}, {C: A}, {D: A}
```

Direction is reversed: the node becomes the signature, the original signature becomes the node. Self-identity collapses to unsigned.

### 7.4 CONNOTATE (`>`) — unidirectional, per-item

```
A > B       → {A: B}
A > B C D   → {A: B}, {A: C}, {A: D}
```

Direction is forward: signature retains its role.

### 7.5 CANONIZE (`=>`) — aggregated

```
A => B          → {A: B}
A => B C D      → {A: [B, C, D]}
```

All items in scope form a single kline's node list.

### 7.6 Subscript Blocks

A subscript block (indented constructs after any operator) is flattened to extract all items for the parent scope. Items are then compiled recursively.

```
A =>
  B
  C = D
```

Items in child scope: B, C. CANONIZE aggregates → `{A: [B, C]}`. Recursive compilation of children: `{D: [C]}`, `{B: []}`, `{C: []}`, `{D: []}`.

---

## 8. MCS (Multi-Character Signature) Expansion

When a signature has **more than one character**, the ASTEmitter automatically emits:

1. **Component identities:** One unsigned entry per constituent character (resolved via BindingScope).
2. **MCS canonization:** One entry mapping the compound to its resolved components.

```
ABC  →  {A: []}, {B: []}, {C: []}, {ABC: [A, B, C]}, {ABC: []}
```

The compound's own unsigned identity (`{ABC: []}`) is emitted by the normal unsigned path.

Single-character signatures do NOT trigger MCS expansion.

MCS applies to any multi-character identifier wherever it appears — signature side or node side, any operator. There is no position-dependent rule.

### 8.1 Character Resolution

Each constituent character is resolved via the BindingScope before emitting. If the character has a binding (e.g., `M` → "Mary"), the resolved word is used instead of the raw character. If unbound, the raw character is used.

### 8.2 Node Count Invariant

The number of nodes in an MCS canonization entry always equals the number of characters in the compound identifier. Each character resolves to exactly one node, regardless of how many BPE tokens the resolved word produces at encoding time.

### 8.3 MCS Deduplication

An MCS canonization entry is a CANONIZE entry mapping a compound to its components. If the compilation rules would produce another CANONIZE entry with the same signature and nodes (e.g., a CANONIZE scope that aggregates the same components), the duplicate is silently dropped. Deduplication applies only to this specific MCS-vs-CANONIZE overlap — it is not a general deduplication mechanism.

---

## 9. BPE Annotations

### 9.1 Definition

A BPE annotation is a parenthesised expression in KScript source that provides word text for BPE token encoding. Two syntactic forms:

**Block annotation**: `(word1 word2 ... wordN)` — a parenthesised list of whitespace-separated words. Appears as an AST node in the construct sequence.

**Inline annotation**: `S(ubject)` — a SIGNATURE token immediately followed by a parenthesised suffix. The first character comes from the SIGNATURE token, the remainder from the annotation content stripped of parentheses. Case is preserved: `S` + `ubject` → `"Subject"`.

### 9.2 When Annotations Are Used

Annotations drive word→BPE-token resolution via BindingScope. The BindingScope is always active during compilation.

### 9.3 Inert Annotations

Annotations where no word's first letter matches any encountered character are inert — they have no effect on compilation. Surplus words are simply never matched. There is no all-or-nothing rule: an annotation can partially match, binding some characters while surplus words are ignored.

Orphan annotations (annotations with no following identifiers) and annotations at end of script are also inert.

---

## 10. NLP Binding Resolution

Binding resolution maps single-character identifiers to NLP words via BPE annotations. Resolution happens inline during the ASTEmitter's single walk via a BindingScope.

### 10.1 Binding Rules

Four rules govern resolution:

**Rule B1 — Binding.** A binding maps a single character to a word. Once bound in a scope, it cannot be re-bound within that scope.

**Rule B2 — Characters Seek Bindings.** When the emitter encounters a single-character signature, resolution proceeds in this order:
1. Inline annotation on this position → bind immediately (Rule B4).
2. Block annotations → search current scope most-recent-first, then parent scopes upward (Rule B3).

**Rule B3 — First-Letter Matching.** Block annotations match by first letter, case-insensitive: `word[0].lower() == char.lower()`. An occurrence counter per scope per character handles disambiguation:
- **Single match** (unambiguous): bind the word. Counter does NOT increment.
- **Multiple matches** (ambiguous): bind the Nth word where N = current counter value, then increment counter by 1.
- **Counter exceeds matches**: no match in this annotation — continue to next annotation or outer scope.
- **No matches in any scope**: character is unbound.

The counter is per-scope-per-character, keyed on the lowercase character value. Each new scope starts at zero.

**Rule B4 — Inline Override.** An inline annotation `S(ubject)` binds immediately, bypassing the occurrence counter. Additionally, it retroactively patches the matching character in the parent scope's MCS CANONIZE entry. Only the immediate parent scope is patched — no propagation beyond one level. If the character is not found in the parent's MCS entry, the override is a safe no-op.

### 10.2 Binding Scope

Scopes are created by operator boundaries (§3). Each scope holds:
- **Annotations**: ordered collection of block annotations, searched most-recent-first.
- **Occurrence counters**: per-character disambiguation counters, starting at zero.

Characters seek from the current (innermost) scope first, then parent scopes upward.

### 10.3 BindingScope API

The BindingScope is a lightweight scope stack:

| Method | Description |
|--------|-------------|
| `push_scope()` | Push a new scope onto the stack |
| `pop_scope()` | Pop the top scope |
| `add_words(words)` | Append a word list to the current scope |
| `resolve(char) → str \| None` | Walk the scope stack, first-letter matching, occurrence counter |

### 10.4 Unbound Characters

When a single-character identifier cannot be resolved through any mechanism, it remains unbound. Unbound characters are encoded using Mod32 bit-packed encoding as a fallback. This produces mixed NLP/Mod32 klines within the same knowledge graph.

---

## 11. Encoding

### 11.1 Encoding-Agnostic Principle

The ASTEmitter produces symbolic entries (strings). The TokenEncoder converts strings to opaque `uint64` values. The compiler treats encoded values as opaque integers — no inspection, no masking.

### 11.2 NLP-BPE Encoding

Identifiers are encoded as typed BPE tokens carrying linguistic annotations (POS + DEP + MORPH):

```
encode("HELLO") → (nlp_type32 << 32) | bpe_token_id
```

### 11.3 Mod32 Fallback for Unbound Characters

When a character cannot be resolved through any binding mechanism (§10.4), it is encoded using Mod32 bit-packed encoding: characters are packed via bitwise-OR into a single `uint64` with bit 0 clear. Lossy: order and multiplicity are lost.

```
encode_unbound("A") → bit_A
```

This produces mixed NLP/Mod32 klines within the same graph.

### 11.4 Multi-Token Words

When a resolved word BPE-encodes to multiple tokens (e.g., "Mary" → `[mar, y]`), the TokenEncoder runs the full MCS process at the BPE-token level:

1. **Component identities:** One unsigned entry per BPE subword token.
2. **MCS canonization:** One CANONIZE entry mapping the packed signature to all component tokens.
3. **Packed signature:** The OR-reduction of all component tokens becomes the single `uint64` node used in the parent kline.

```
"Mary" → tokens [mar, y]
  emits: {mar: []}, {y: []}, {mar|y: [mar, y]}
  parent kline uses (mar|y) as its single node
```

This is structurally identical to §8 character-level MCS, applied at the BPE subword level. The node-count invariant (§8.2) is maintained: one character = one node, regardless of BPE token count. The consumer of the knowledge graph sees one node per character; the BPE decomposition is recorded in the graph for downstream use.

### 11.5 Signature Construction

Signatures are constructed via plain OR-reduction of raw, unmasked node values:

```
make_signature([node_A, node_B]) → node_A | node_B
```

The compiler does not inspect or mask node values — they are opaque `uint64` integers.

### 11.6 Design Tension: Annotations and Encoding Opacity

BPE annotations are the mechanism by which the BPE token ID component of a node is determined. This creates a tension: the ASTEmitter must be aware of encoding semantics to resolve annotations, even though nodes are otherwise treated as opaque `uint64` values. The BindingScope is the single point where this leak occurs — it resolves character→word mappings that determine BPE token IDs. All other compilation stages are encoding-agnostic.

---

## 12. Implementation Components

### 12.1 Module Structure

```
kscript/
├── __init__.py         # KScript class (public API)
├── token.py            # TokenType enum, Token dataclass
├── lexer.py            # Lexer (source → tokens)
├── ast.py              # AST node dataclasses
├── parser.py           # Parser (tokens → AST)
├── ast_emitter.py      # ASTEmitter (AST → symbolic entries)
├── token_encoder.py    # TokenEncoder (symbolic → encoded entries)
├── binding_scope.py    # BindingScope (NLP binding resolution)
└── compiler.py         # Compiler (orchestrator)
```

### 12.2 Compiler

The Compiler orchestrates the pipeline. It creates the BindingScope (NLP mode only), passes it to the ASTEmitter, and passes the symbolic output to the TokenEncoder. No encoding logic lives in the Compiler itself.

### 12.3 Dependencies

| Dependency | Used By |
|------------|---------|
| `KLine` | CompiledEntry base class |
| `KTokenizer` | TokenEncoder encoding/decoding |
| `make_signature()` | Signature construction |

---

## 13. Public API

### 13.1 Python API

```python
from kscript import KScript

# Compile from source string
entries = KScript("A == B").entries

# Compile with specific tokenizer
entries = KScript("A == B", tokenizer=NLPTokenizer()).entries
```

The `entries` property returns a list of `CompiledEntry` objects.

---

## 14. Worked Examples

### 14.1 Minimal Unsigned

```
A
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|----|-------|
| 1 | A | [] | UNSIGNED | S4 |

### 14.2 Bidirectional Link

```
A == B
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|----|-------|
| 1 | A | [B] | COUNTERSIGN | S1 |
| 2 | B | [A] | COUNTERSIGN | S1 |

### 14.3 Undersign (Reversed)

```
A = B
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|-------|-------|
| 1 | B | [A] | UNDERSIGN | S1 |

### 14.4 Connotate (Forward)

```
A > B
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|-------|-------|
| 1 | A | [B] | CONNOTATE | S3 |

### 14.5 Self-Identity

```
A = A
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|-------|-------|
| 1 | A | [] | UNSIGNED | S4 |

### 14.6 MCS Expansion

```
ABC
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|---------|-------|
| 1 | A | [] | UNSIGNED | S4 |
| 2 | B | [] | UNSIGNED | S4 |
| 3 | C | [] | UNSIGNED | S4 |
| 4 | ABC | [A, B, C] | CANONIZE | S2 |
| 5 | ABC | [] | UNSIGNED | S4 |

### 14.7 Operator Chain

```
A == B > C = D
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|-------------|-------|
| 1 | A | [B] | COUNTERSIGN | S1 |
| 2 | B | [A] | COUNTERSIGN | S1 |
| 3 | B | [C] | CONNOTATE | S3 |
| 4 | D | [C] | UNDERSIGN | S1 |

### 14.8 CANONIZE with Subscript Block

```
A =>
  B
  C = D
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|---------|-------|
| 1 | A | [B, C] | CANONIZE | S2 |
| 2 | D | C | UNDERSIGN | S1 |
| 3 | B | [] | UNSIGNED | S4 |
| 4 | C | [] | UNSIGNED | S4 |
| 5 | D | [] | UNSIGNED | S4 |

### 14.9 Chained CANONIZE

```
A => B => C
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|---------|-------|
| 1 | A | [B] | CANONIZE | S2 |
| 2 | B | [C] | CANONIZE | S2 |
| 3 | C | [] | UNSIGNED | S4 |

### 14.10 Non-CANONIZE with Indent

```
A == B
  C
  D
```

Compiled:

| Entry | Signature | Nodes | Op | Level |
|-------|-----------|-------|-------------|-------|
| 1 | A | [B] | COUNTERSIGN | S1 |
| 2 | B | [A] | COUNTERSIGN | S1 |
| 3 | A | [C] | COUNTERSIGN | S1 |
| 4 | C | [A] | COUNTERSIGN | S1 |
| 5 | A | [D] | COUNTERSIGN | S1 |
| 6 | D | [A] | COUNTERSIGN | S1 |

### 14.11 Complex Nested (Full)

```
MHALL == SVO =>
  S = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O
```

Compiled:

| # | Entry | Signature | Nodes | Op | Level |
|---|-------|-----------|-------|-------------|-------|
| 1 | MCS M | M | [] | UNSIGNED | S4 |
| 2 | MCS H | H | [] | UNSIGNED | S4 |
| 3 | MCS A | A | [] | UNSIGNED | S4 |
| 4 | MCS L | L | [] | UNSIGNED | S4 |
| 5 | MCS MHALL canonize | MHALL | [M, H, A, L, L] | CANONIZE | S2 |
| 6 | MCS MHALL unsigned | MHALL | [] | UNSIGNED | S4 |
| 7 | MCS S | S | [] | UNSIGNED | S4 |
| 8 | MCS V | V | [] | UNSIGNED | S4 |
| 9 | MCS O | O | [] | UNSIGNED | S4 |
| 10 | MCS SVO canonize | SVO | [S, V, O] | CANONIZE | S2 |
| 11 | MCS SVO unsigned | SVO | [] | UNSIGNED | S4 |
| 12 | Countersign | MHALL | [SVO] | COUNTERSIGN | S1 |
| 13 | Countersign reverse | SVO | [MHALL] | COUNTERSIGN | S1 |
| — | SVO canonize subscript | — | — | — | Dropped (MCS dedup: identical to entry 10) |
| 14 | Undersign S | M | [S] | UNDERSIGN | S1 |
| 15 | Undersign V | H | [V] | UNDERSIGN | S1 |
| — | MCS ALL A | — | — | — | Dropped (MCS dedup: {A:[]} identical to entry 3) |
| — | MCS ALL L | — | — | — | Dropped (MCS dedup: {L:[]} identical to entry 4) |
| 16 | MCS ALL canonize | ALL | [A, L, L] | CANONIZE | S2 |
| 17 | MCS ALL unsigned | ALL | [] | UNSIGNED | S4 |
| 18 | Undersign O | ALL | [O] | UNDERSIGN | S1 |
| — | ALL canonize subscript | — | — | — | Dropped (MCS dedup: identical to entry 16) |
| 19 | Undersign D | D | [A] | UNDERSIGN | S1 |
| 20 | Undersign M | M | [L] | UNDERSIGN | S1 |
| 21 | Connotate | L | [O] | CONNOTATE | S3 |

> **MCS deduplication in action:** Four entries are silently dropped because their `(signature, nodes)` pairs were already emitted by MCS expansion. Only MCS canonization entries are deduplicated — other duplicate patterns are emitted as-is.

### 14.12 NLP-Bound Example

```
(Mary Had A Little Lamb)
MHALL == SVO =>
  S(ubject) = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O
```

Binding resolution:
- Block annotation `(Mary Had A Little Lamb)` provides words for MHALL's characters.
- `M` → "Mary", `H` → "Had", `A` → "A" (first-letter match), `L` → "Little" (counter 0), `L` → "Lamb" (counter 1, ambiguous).
- `S(ubject)` → inline binding, overrides `S` → "Subject" in parent MCS.

MCS for MHALL (resolved):
```
{Mary: []}, {Had: []}, {A: []}, {Little: []}, {Lamb: []}
{MHALL: [Mary, Had, A, Little, Lamb]}
```

With Rule B4 override, the parent SVO canonize entry becomes:
```
{SVO: [Subject, V, O]}    (S patched to "Subject")
```

All other entries follow the same operator rules as §14.11, with resolved words replacing raw characters where bindings exist. Unbound characters (V, O, D) use Mod32 fallback encoding (§11.3).

---

## 15. Test Matrix

| ID | Criterion | Category |
|----|-----------|----------|
| **Lexer** | | |
| KS-1 | All token types recognized: SIGNATURE, COUNTERSIGN, CANONIZE, CONNOTATE, UNDERSIGN, ANNOTATION, NEWLINE, INDENT, DEDENT, EOF | Lexer |
| KS-2 | Multi-char operator priority: `==`, `=>` matched before `=`, `>` | Lexer |
| KS-3 | BPE annotations: `(...)` with nested parens preserved as ANNOTATION tokens | Lexer |
| KS-4 | Indent/dedent tracking: Python-style INDENT/DEDENT tokens | Lexer |
| KS-5 | Edge cases: empty input, whitespace-only, unknown characters raise LexerError | Lexer |
| **Parser** | | |
| KS-6 | AST structure reflects scope model: OperatorScope nodes with sig, op, items, child_block | Parser |
| KS-7 | Block parsing: INDENT/DEDENT creates Block nodes | Parser |
| KS-8 | Annotations preserved as AST nodes (not discarded) | Parser |
| KS-9 | Inline annotation attachment: sig-side and node-side | Parser |
| KS-10 | Empty source produces empty script (no error) | Parser |
| **Scope & Operators** | | |
| KS-11 | COUNTERSIGN per-item: `A == B C` → `{A:B}, {B:A}, {A:C}, {C:A}` | Scope |
| KS-12 | UNDERSIGN per-item reversed: `A = B C` → `{B:A}, {C:A}` | Scope |
| KS-13 | CONNOTATE per-item: `A > B C` → `{A:B}, {A:C}` | Scope |
| KS-14 | CANONIZE aggregates: `A => B C D` → `{A:[B,C,D]}` | Scope |
| KS-15 | Operator chain: `A == B > C = D` → correct signatures per scope | Scope |
| KS-16 | Indent extends scope: items in child block belong to parent operator | Scope |
| KS-17 | DEDENT returns to parent scope | Scope |
| KS-18 | Non-CANONIZE with indent: per-item emission extends into child block | Scope |
| **MCS** | | |
| KS-19 | MCS expansion: multi-char identifier produces components + canonization + unsigned | MCS |
| KS-20 | No MCS for single-char identifiers | MCS |
| KS-21 | MCS on node side: `A == MHALL` triggers MCS for MHALL | MCS |
| KS-22 | Node count invariant: MCS node count equals character count | MCS |
| **Binding** | | |
| KS-23 | Block annotation first-letter matching: `(Mary Had A Little Lamb)` + `MHALL` | Binding |
| KS-24 | Occurrence counter: duplicate letters resolved to different words | Binding |
| KS-25 | Inline binding: `S(ubject)` resolves immediately, bypasses counter | Binding |
| KS-26 | Rule B4 override: inline binding patches parent MCS CANONIZE entry | Binding |
| KS-27 | Scope inheritance: characters seek from inner to outer scope | Binding |
| KS-28 | Scope shadowing: inner scope binding shadows outer for same character | Binding |
| KS-29 | Counter reset: each new scope starts counters at zero | Binding |
| KS-30 | Unbound characters: fallback to Mod32 encoding | Binding |
| KS-31 | Inert annotation: no matching characters → no effect | Binding |
| KS-32 | Unbound characters use Mod32 fallback encoding | Encoding |
| **Self-Identity** | | |
| KS-33 | Self-identity: `A = A` → `{A: []}` with op=UNSIGNED | Operators |
| **Structure** | | |
| KS-34 | Nodes always a list: `A => B` → `{A: [B]}`, `A` → `{A: []}` | Structure |
| **Integration** | | |
| KS-35 | Complex nested example (§14.11) produces correct complete entry list | Integration |
| KS-36 | NLP-bound example (§14.12) produces correct resolved entries | Integration |
| KS-37 | Mixed NLP/Mod32 klines: bound characters encoded via NLP-BPE, unbound characters via Mod32 fallback | Integration |
