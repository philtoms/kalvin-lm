# KScript Language Specification

**Version:** 3.0  
**Date:** 2026-06-16  
**Status:** Authoritative

---

## 1. Overview

KScript is a domain-specific language for constructing **klines** — ordered sequences of identified nodes (each called a **KLine**). It compiles declarative scripts into a list of compiled entries that can be loaded into a Kalvin Agent for rationalisation.

### 1.1 Compilation Pipeline

```
Source (.ks) → Lexer → Parser → AST → ASTEmitter → TokenEncoder → [CompiledEntry]
```

| Stage            | Responsibility                                                                      |
| ---------------- | ----------------------------------------------------------------------------------- |
| **Lexer**        | Source text → token stream                                                          |
| **Parser**       | Token stream → AST (structural)                                                     |
| **ASTEmitter**   | AST → symbolic entries (strings). Resolves BPE annotations inline via BindingScope. |
| **TokenEncoder** | Symbolic entries → encoded entries (uint64 values). Encoding-agnostic.              |

The pipeline is strictly one-directional. The ASTEmitter operates on symbolic strings — no encoding happens there. The TokenEncoder converts strings to opaque `uint64` values via a pluggable tokenizer.

### 1.2 Core Concepts

| Concept            | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| **Node**           | An opaque `uint64` value — the universal atom                                       |
| **KLine**          | An identified, ordered sequence of zero or more nodes: `{signature, nodes[]}`       |
| **Signature**      | A `uint64` identity key (constructed from node values via the @signifier)           |
| **Significance**   | A four-level classification (S1–S4) of how strongly one KLine relates to another    |
| **Identifier**     | An uppercase alpha string like `ABC` — the lexical form of a signature or node      |
| **BPE Annotation** | A parenthesised annotation `(...)` providing word text for BPE token encoding       |
| **Scope**          | An operator-delimited region that determines kline structure and binding resolution |

---

## 2. Lexical Structure

### 2.1 Token Types

| Token         | Pattern                    | Category       |
| ------------- | -------------------------- | -------------- |
| `SIGNATURE`   | `[A-Z]+`                   | Identifier     |
| `COUNTERSIGN` | `==`                       | Operator       |
| `CANONIZE`    | `=>`                       | Operator       |
| `CONNOTATE`   | `>`                        | Operator       |
| `UNDERSIGN`   | `=`                        | Operator       |
| `ANNOTATION`  | `(...)` with nested parens | BPE Annotation |
| `NEWLINE`     | `\n`                       | Insignificant  |
| `INDENT`      | Increased indentation      | Structural     |
| `DEDENT`      | Decreased indentation      | Structural     |
| `EOF`         | End of file                | Sentinel       |

### 2.2 Operator Classification

| Operator    | Symbol | Behaviour                                              |
| ----------- | ------ | ------------------------------------------------------ |
| COUNTERSIGN | `==`   | Bidirectional, per-item emission                       |
| UNDERSIGN   | `=`    | Unidirectional (reversed), per-item emission           |
| CANONIZE    | `=>`   | Unidirectional, all items aggregated into single kline |
| CONNOTATE   | `>`    | Unidirectional, per-item emission                      |

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

| Scope | Operator | Signature | Nodes | Compiled         |
| ----- | -------- | --------- | ----- | ---------------- |
| 1     | `==`     | A         | B     | `{A: B}, {B: A}` |
| 2     | `>`      | B         | C     | `{B: C}`         |
| 3     | `=`      | C         | D     | `{D: C}`         |

#### CANONIZE aggregation

```
A => B C D
```

| Scope | Operator | Signature | Nodes   | Compiled         |
| ----- | -------- | --------- | ------- | ---------------- |
| 1     | `=>`     | A         | B, C, D | `{A: [B, C, D]}` |

#### Mixed operators with CANONIZE

```
A == B => C D
```

| Scope | Operator | Signature | Nodes | Compiled         |
| ----- | -------- | --------- | ----- | ---------------- |
| 1     | `==`     | A         | B     | `{A: B}, {B: A}` |
| 2     | `=>`     | B         | C, D  | `{B: [C, D]}`    |

#### CANONIZE followed by inline

```
A => B == C
```

| Scope | Operator | Signature | Nodes | Compiled         |
| ----- | -------- | --------- | ----- | ---------------- |
| 1     | `=>`     | A         | B     | `{A: B}`         |
| 2     | `==`     | B         | C     | `{B: C}, {C: B}` |

#### Indented child scope

```
A =>
  B = C
  D > E
```

| Scope      | Operator | Signature | Nodes                   | Compiled      |
| ---------- | -------- | --------- | ----------------------- | ------------- |
| 1          | `=>`     | A         | B, D (from child scope) | `{A: [B, D]}` |
| 1a (child) | `=`      | B         | C                       | `{C: B}`      |
| 1b (child) | `>`      | D         | E                       | `{D: E}`      |

Child-scope items B and D are the CANONIZE nodes for scope 1. Each child item is also compiled recursively.

#### Non-CANONIZE with indent

```
A == B
  C
  D
```

| Scope | Operator | Signature | Nodes                      | Compiled                                         |
| ----- | -------- | --------- | -------------------------- | ------------------------------------------------ |
| 1     | `==`     | A         | B, C, D (from child scope) | `{A: B}, {B: A}, {A: C}, {C: A}, {A: D}, {D: A}` |

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
  ├── op: TokenType | None     # None = bare signature (identity)
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

Each compiled entry is a **KLine** (the same `KLine` type used throughout Kalvin; there is no separate `CompiledEntry` subclass):

```
KLine:
    signature: uint64              # encoded identity key
    nodes: list[uint64]             # encoded node values (always a list, may be empty)
    dbg: KDbg                       # op (structural state) + diagnostic fields
```

**No singleton rule.** Nodes are always a list. An identity entry has an empty list. A single-node entry has a one-element list. A multi-node entry has multiple elements.

### 6.2 Significance Level Assignment

Each emitted entry is tagged with a significance level based on its structural state (the resulting signature↔nodes relationship, recorded in the `op` field):

| State                | Level | Meaning                                |
| -------------------- | ----- | -------------------------------------- |
| COUNTERSIGNED (`==`) | S1    | Mutual / bidirectional                 |
| UNDERSIGNED (`=`)    | S3    | Unidirectional reversed                |
| CANONIZED (`=>`)     | S2    | Canonical                              |
| CONNOTED (`>`)       | S3    | Connotative                            |
| IDENTITY (bare)      | S4    | Identity — bare node, no relationships |

Significance bits are not encoded into the token IDs. The level is carried as metadata on the compiled entry.

---

## 7. Operator Compilation Rules

### 7.1 Identity (bare signature)

```
A       → {A: []}
```

Plus MTS expansion if multi-character (§8).

### 7.2 COUNTERSIGN (`==`) — bidirectional, per-item

```
A == B       → {A: B}, {B: A}
A == B C D   → {A: B}, {B: A}, {A: C}, {C: A}, {A: D}, {D: A}
```

Each node produces a bidirectional pair.

### 7.3 UNDERSIGN (`=`) — unidirectional reversed, per-item

```
A = B       → {B: A}
A = A       → {A: []}              (self-identity, IDENTITY)
A = B C D   → {B: A}, {C: A}, {D: A}
```

Direction is reversed: the node becomes the signature, the original signature becomes the node. Self-identity collapses to identity.

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

A subscript block (indented constructs after an operator) is flattened to extract all items for the parent scope. Items are then compiled recursively.

```
A =>
  B
  C = D
```

Items in child scope: B, C. CANONIZE aggregates → `{A: [B, C]}`. Recursive compilation of children: `{D: [C]}`, `{B: []}`, `{C: []}`, `{D: []}`.

**Subscript identity rule.** In a CANONIZE subscript block, every identifier that appears in the block must be the signature of at least one emitted entry. Some identifiers never produce such an entry on their own:

- A **leaf Signature item** (a bare node like `B` above) emits no operator entry, so it has no entry whose signature is `B`.
- An **UNDERSIGN scope sig** (`C = D` produces `{D: [C]}` — the scope's own sig `C` is never a signature).

For these, an IDENTITY entry `{sig: []}` is emitted to fill the gap. Identity emission is deduplicated: if the identifier already has an IDENTITY entry, or was already introduced as an MTS component, or is a compound already introduced by its CANONIZE entry, no new IDENTITY is emitted.

**MTS-expanded sig suppression.** This identity filling applies only when the CANONIZE scope's sig did **not** trigger MTS — i.e. it is a single-character sig. A multi-character CANONIZE sig already receives component identities from its own MTS expansion (§8), so subscript identity would only produce spurious duplicates and is suppressed. (This is why §14.11 emits no identity for `ALL`'s subscript children `A`, `L` — they were already introduced by `MHALL`'s MTS expansion.)

The identity-filling flag does not propagate between CANONIZE scopes; only the immediate CANONIZE subscript block is affected. Non-CANONIZE operators with indented child blocks do **not** trigger identity filling — their child items are already collected as the parent operator's nodes (§3, §14.10).

---

## 8. MTS (Multi-Token Signature) Expansion

When a signature is an **all-uppercase multi-character identifier** (a compound, e.g. `MHALL`, `SVO`, `ALL`), the ASTEmitter automatically emits:

1. **Component identities:** One identity entry per constituent character (resolved via BindingScope).
2. **MTS canonization:** One entry mapping the compound to its resolved components.

```
ABC  →  {A: []}, {B: []}, {C: []}, {ABC: [A, B, C]}
```

Single-character signatures do NOT trigger MTS expansion.

**Compounds vs words.** MTS character-expansion applies only to all-uppercase identifiers. A lowercase or mixed-case multi-character identifier (e.g. `had`, `did`, `all`) is a **single word** — one token — not a compound; it is emitted as its own IDENTITY and is never decomposed into per-character entries. Case is the discriminator that separates a compound from a word; both are admitted by the case-insensitive SIGNATURE rule (§2). (Historically every multi-character identifier was uppercase, so the case guard was implicit; the SIGNATURE relaxation made it explicit.)

MTS applies to compounds wherever they appear — signature side or node side, any operator. There is no position-dependent rule.

### 8.1 Character Resolution

Each constituent character is resolved via the BindingScope before emitting. If the character has a binding (e.g., `M` → "Mary"), the resolved word is used instead of the raw character. If the character has no binding, the raw character is used.

### 8.2 Node Count Invariant

The number of nodes in an MTS canonization entry always equals the number of characters in the compound identifier. Each character resolves to exactly one node, regardless of how many BPE tokens the resolved word produces at encoding time.

### 8.3 MTS Deduplication

MTS deduplication prevents duplicate entries when the same identifier appears in multiple MTS expansions. Two categories are deduplicated:

**Component identity dedup.** Each constituent character of an MTS expansion produces one IDENTITY (S4) entry. If a character was already emitted by a previous MTS expansion, the duplicate is silently dropped. Intra-expansion dedup also prevents duplicate emission when the same character appears multiple times in a compound (e.g., the second L in MHALL).

**Canonization dedup.** An MTS canonization entry is a CANONIZE (S2) entry mapping a compound to its components. If another CANONIZE entry with the same signature and nodes would be produced (e.g., a CANONIZE scope that aggregates the same components), the duplicate is silently dropped.

A compound identifier does not receive its own IDENTITY entry. An identity requires a single-token signature; a compound's signature is built from multiple token IDs (@signifier) and cannot form one (CONTEXT.md "Identity" glossary).

Deduplication applies only to MTS-produced entries — it is not a general deduplication mechanism. Operator-produced entries (COUNTERSIGN, UNDERSIGN, CONNOTATE) and non-MTS IDENTITY entries are always emitted.

**Canonical resolution.** An identifier's MTS component list is computed once, on first expansion, and reused by every subsequent reference (node-side or signature-side, any operator). The occurrence counter (§10.1) disambiguates characters within a single expansion (e.g. the two L's in `MHALL`); it does not advance between expansions of the same identifier.

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

## 10. Word Binding Resolution

Binding resolution maps single-character identifiers to words via BPE annotations. Resolution happens inline during the ASTEmitter's single walk via a BindingScope.

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
- **No matches in any scope**: the character is not resolved (no binding); it is encoded as its own raw BPE token (§11.2).

The counter is per-scope-per-character, keyed on the lowercase character value. Each new scope starts at zero. **Pushing a new scope clears the occurrence counters in all existing (parent) scopes**, so when resolution falls through from a child scope to a parent, the parent's counter restarts fresh rather than retaining stale state from before the child was entered.

**Rule B4 — Inline Override.** An inline annotation `S(ubject)` binds immediately, bypassing the occurrence counter. Additionally, it retroactively patches the matching character in the parent scope's MTS CANONIZE entry. Only the immediate parent scope is patched — no propagation beyond one level. If the character is not found in the parent's MTS entry, the override is a safe no-op.

### 10.2 Binding Scope

Scopes are created by operator boundaries (§3). Each scope holds:

- **Annotations**: ordered collection of block annotations, searched most-recent-first.
- **Occurrence counters**: per-character disambiguation counters, starting at zero.

Characters seek from the current (innermost) scope first, then parent scopes upward.

### 10.3 BindingScope API

The BindingScope is a lightweight scope stack:

| Method                        | Description                                                     |
| ----------------------------- | --------------------------------------------------------------- |
| `push_scope()`                | Push a new scope onto the stack                                 |
| `pop_scope()`                 | Pop the top scope                                               |
| `add_words(words)`            | Append a word list to the current scope                         |
| `resolve(char) → str \| None` | Walk the scope stack, first-letter matching, occurrence counter |

**Resolution failure.** When `BindingScope.resolve(char)` returns `None`, the identifier is encoded as its own raw BPE token — the same encoding path as any resolved identifier, minus the word binding. There is no named resolution-outcome state and no fallback encoding scheme.

---

## 11. Encoding

### 11.1 Encoding-Agnostic Principle

The ASTEmitter produces symbolic entries (strings). The TokenEncoder converts strings to opaque `uint64` values. The compiler treats encoded values as opaque integers — no inspection, no masking.

### 11.2 Typed-Node Encoding

Identifiers are encoded as typed BPE tokens carrying a sig word in the upper 32 bits (its meaning — e.g. POS + DEP + MORPH — is a deployment concern; see the @nlp_tokenizer spec):

```
encode("HELLO") → (sig_word << 32) | bpe_token_id
```

### 11.3 Multi-Token Words

When a resolved word BPE-encodes to multiple tokens (e.g., "Mary" → `[mar, y]`), the TokenEncoder runs the full MTS process at the BPE-token level:

1. **Component identities:** One identity entry per BPE subword token.
2. **MTS canonization:** One CANONIZE entry mapping the packed signature to all component tokens.
3. **Packed signature:** The signature of all component tokens becomes the single `uint64` node used in the parent kline.

```
"Mary" → tokens [mar, y]
  emits: {mar: []}, {y: []}, {mar|y: [mar, y]}
  parent kline uses (mar|y) as its single node
```

This is structurally identical to §8 character-level MTS, applied at the BPE subword level. The node-count invariant (§8.2) is maintained: one character = one node, regardless of BPE token count. The consumer of the model sees one node per character; the BPE decomposition is recorded in the model for downstream use.

§11.3 applies only to resolved words, not to compound identifiers (§8): a compound's decomposition is its CANONIZED entry, so re-encoding its literal string is prohibited. A packed signature — whether a §11.3 multi-token word's packed signature or a §11.4 compound — is the signature of multiple token IDs (@signifier) and cannot head an IDENTITY kline (CONTEXT.md "Identity"); the decomposition above (component identities + canonize) is a multi-token word's sole representation, and no standalone IDENTITY kline is emitted at the packed signature.

### 11.4 Signature Construction

Signatures are constructed from node values via the Signifier (@signifier).

The compiler does not inspect or mask node values — they are opaque `uint64` integers.

**Canonical encoding.** A compound identifier's signature is computed once — at its CANONIZED definition, via `make_signature` over its resolved component node values (@signifier) — and reused by every referencing entry (as signature or node), so all klines referring to the same compound share one uint64. (Operator entries where the compound is the signature but the node is a different identifier — e.g. `COUNTERSIGNED MHALL [SVO]` — have `signature ≠ make_signature(nodes)` by design: the signature is a registry lookup, not a reduction of that entry's own nodes.)

### 11.5 Design Tension: Annotations and Encoding Opacity

BPE annotations are the mechanism by which the BPE token ID component of a node is determined. This creates a tension: the ASTEmitter must be aware of encoding semantics to resolve annotations, even though nodes are otherwise treated as opaque `uint64` values. The BindingScope is the single point where this leak occurs — it resolves character→word mappings that determine BPE token IDs. All other compilation stages are encoding-agnostic.

### 11.6 Output Ordering

The compiled entry list emits **compiled source before any MTS entries.** Compiled source is the set of klines the script directly writes: operator-produced klines (COUNTERSIGNED, UNDERSIGNED, CONNOTED), single-character CANONIZE aggregates (§7.5), and subscript identities (§7.6). MTS entries — both §8 character-level expansions and §11.3 BPE-subword decompositions — follow.

The partition is **stable**: relative order is preserved within each group. Encoding still runs in definition-before-reference order internally (a compound's canonical signature is registered before any reference is encoded); source-before-MTS is a stable re-partition of the finished output, not a change to per-entry encoding or deduplication. When a script emits only MTS (e.g. a bare compound `ABC`, §14.6) or only source (e.g. `A == B`, §14.2), the ordering is a no-op.

---

## 12. Implementation Components

### 12.1 Module Structure

```
ks/
├── __init__.py         # KScript class (public API)
└── ...                 # (token.py, lexer.py, ast.py, parser.py,
                        #  ast_emitter.py, token_encoder.py,
                        #  binding_scope.py, compiler.py)
```

> **Note:** The module package is `ks` (imported as `from ks import KScript`). The package lives under `src/ks/`.

### 12.2 Compiler

The Compiler orchestrates the pipeline. It always creates a BindingScope and pushes a root scope, passes it to the ASTEmitter, and passes the symbolic output to the TokenEncoder. There is no tokenizer "mode" switch — tokenizer data is mandatory and the BindingScope is always active (§10). No encoding logic lives in the Compiler itself.

### 12.3 Dependencies

| Dependency         | Used By                                                    |
| ------------------ | ---------------------------------------------------------- |
| `KLine`            | Compiled entry type (no separate `CompiledEntry` subclass) |
| `KTokenizer`       | TokenEncoder encoding/decoding                             |
| `make_signature()` | Signature construction                                     |

---

## 13. Public API

### 13.1 Python API

```python
from ks import KScript

# Compile from source string
entries = KScript("A == B").entries

# Compile with a specific tokenizer
entries = KScript("A == B", tokenizer=NLPTokenizer()).entries
```

The `entries` property returns a list of `KLine` objects, ordered compiled-source-first (§11.6). The tokenizer defaults to the production NLP tokenizer (`NLPTokenizer()`) — tokenizer data is mandatory, so a tokenizer is always in effect.

---

## 14. Worked Examples

### 14.1 Minimal Unsigned

```
A
```

Compiled:

| Entry | Signature | Nodes | Op       | Level |
| ----- | --------- | ----- | -------- | ----- |
| 1     | A         | []    | IDENTITY | S4    |

### 14.2 Bidirectional Link

```
A == B
```

Compiled:

| Entry | Signature | Nodes | Op            | Level |
| ----- | --------- | ----- | ------------- | ----- |
| 1     | A         | [B]   | COUNTERSIGNED | S1    |
| 2     | B         | [A]   | COUNTERSIGNED | S1    |

### 14.3 Undersign (Reversed)

```
A = B
```

Compiled:

| Entry | Signature | Nodes | Op          | Level |
| ----- | --------- | ----- | ----------- | ----- |
| 1     | B         | [A]   | UNDERSIGNED | S3    |

### 14.4 Connotate (Forward)

```
A > B
```

Compiled:

| Entry | Signature | Nodes | Op       | Level |
| ----- | --------- | ----- | -------- | ----- |
| 1     | A         | [B]   | CONNOTED | S3    |

### 14.5 Self-Identity

```
A = A
```

Compiled:

| Entry | Signature | Nodes | Op       | Level |
| ----- | --------- | ----- | -------- | ----- |
| 1     | A         | []    | IDENTITY | S4    |

### 14.6 MTS Expansion

```
ABC
```

Compiled:

| Entry | Signature | Nodes     | Op        | Level |
| ----- | --------- | --------- | --------- | ----- |
| 1     | A         | []        | IDENTITY  | S4    |
| 2     | B         | []        | IDENTITY  | S4    |
| 3     | C         | []        | IDENTITY  | S4    |
| 4     | ABC       | [A, B, C] | CANONIZED | S2    |

### 14.7 Operator Chain

```
A == B > C = D
```

Compiled:

| Entry | Signature | Nodes | Op            | Level |
| ----- | --------- | ----- | ------------- | ----- |
| 1     | A         | [B]   | COUNTERSIGNED | S1    |
| 2     | B         | [A]   | COUNTERSIGNED | S1    |
| 3     | B         | [C]   | CONNOTED      | S3    |
| 4     | D         | [C]   | UNDERSIGNED   | S3    |

### 14.8 CANONIZE with Subscript Block

```
A =>
  B
  C = D
```

Compiled:

| Entry | Signature | Nodes  | Op          | Level |
| ----- | --------- | ------ | ----------- | ----- |
| 1     | A         | [B, C] | CANONIZED   | S2    |
| 2     | D         | [C]    | UNDERSIGNED | S3    |
| 3     | B         | []     | IDENTITY    | S4    |
| 4     | C         | []     | IDENTITY    | S4    |
| 5     | D         | []     | IDENTITY    | S4    |

### 14.9 Chained CANONIZE

```
A => B => C
```

Compiled:

| Entry | Signature | Nodes | Op        | Level |
| ----- | --------- | ----- | --------- | ----- |
| 1     | A         | [B]   | CANONIZED | S2    |
| 2     | B         | [C]   | CANONIZED | S2    |
| 3     | C         | []    | IDENTITY  | S4    |

### 14.10 Non-CANONIZE with Indent

```
A == B
  C
  D
```

Compiled:

| Entry | Signature | Nodes | Op            | Level |
| ----- | --------- | ----- | ------------- | ----- |
| 1     | A         | [B]   | COUNTERSIGNED | S1    |
| 2     | B         | [A]   | COUNTERSIGNED | S1    |
| 3     | A         | [C]   | COUNTERSIGNED | S1    |
| 4     | C         | [A]   | COUNTERSIGNED | S1    |
| 5     | A         | [D]   | COUNTERSIGNED | S1    |
| 6     | D         | [A]   | COUNTERSIGNED | S1    |

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

Compiled (source-first — §11.6):

| #   | Entry                  | Signature | Nodes           | Op            | Level                                                 |
| --- | ---------------------- | --------- | --------------- | ------------- | ----------------------------------------------------- |
| 1   | Countersign            | MHALL     | [SVO]           | COUNTERSIGNED | S1                                                    |
| 2   | Countersign reverse    | SVO       | [MHALL]         | COUNTERSIGNED | S1                                                    |
| 3   | Undersign S            | M         | [S]             | UNDERSIGNED   | S3                                                    |
| 4   | Undersign V            | H         | [V]             | UNDERSIGNED   | S3                                                    |
| 5   | Undersign O            | ALL       | [O]             | UNDERSIGNED   | S3                                                    |
| 6   | Undersign D            | D         | [A]             | UNDERSIGNED   | S3                                                    |
| 7   | Undersign M            | M         | [L]             | UNDERSIGNED   | S3                                                    |
| 8   | Connotate              | L         | [O]             | CONNOTED      | S3                                                    |
| 9   | MTS M                  | M         | []              | IDENTITY      | S4                                                    |
| 10  | MTS H                  | H         | []              | IDENTITY      | S4                                                    |
| 11  | MTS A                  | A         | []              | IDENTITY      | S4                                                    |
| 12  | MTS L                  | L         | []              | IDENTITY      | S4                                                    |
| 13  | MTS MHALL canonize     | MHALL     | [M, H, A, L, L] | CANONIZED     | S2                                                    |
| 14  | MTS S                  | S         | []              | IDENTITY      | S4                                                    |
| 15  | MTS V                  | V         | []              | IDENTITY      | S4                                                    |
| 16  | MTS O                  | O         | []              | IDENTITY      | S4                                                    |
| 17  | MTS SVO canonize       | SVO       | [S, V, O]       | CANONIZED     | S2                                                    |
| —   | SVO canonize subscript | —         | —               | —             | Dropped (canonize dedup: identical to entry 17)       |
| —   | MTS ALL A              | —         | —               | —             | Dropped (identity dedup: {A:[]} identical to entry 11) |
| —   | MTS ALL L              | —         | —               | —             | Dropped (identity dedup: {L:[]} identical to entry 12) |
| 18  | MTS ALL canonize       | ALL       | [A, L, L]       | CANONIZED     | S2                                                    |
| —   | ALL canonize subscript | —         | —               | —             | Dropped (canonize dedup: identical to entry 18)       |

> **MTS deduplication in action:** Four entries are silently dropped because they duplicate already-emitted MTS entries. Two component identity entries (MTS ALL component A and L) are dropped because MHALL's expansion already provided them. Two canonization entries (SVO subscript and ALL subscript) are dropped because their MTS canonization counterparts already exist. Compound identifiers receive no IDENTITY of their own (an identity requires a single-token signature), so there is nothing to drop for those. Only MTS-produced entries (component identity and canonization) are deduplicated — operator-produced duplicates are emitted as-is.

### 14.12 Word-Bound Example

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
- `S(ubject)` → inline binding, overrides `S` → "Subject" in parent MTS.

MTS for MHALL (resolved):

```
{Mary: []}, {Had: []}, {A: []}, {Little: []}, {Lamb: []}
{MHALL: [Mary, Had, A, Little, Lamb]}
```

With Rule B4 override, the parent SVO canonize entry becomes:

```
{SVO: [Subject, V, O]}    (S patched to "Subject")
```

Compiled (resolved-word level; source-first — §11.6). MHALL has five distinct resolved components — no intra-expansion dedup — so this table has 19 entries versus §14.11's 18:

| #   | Signature | Nodes                        | Op            | Level |
| --- | --------- | ---------------------------- | ------------- | ----- |
| 1   | MHALL     | [SVO]                        | COUNTERSIGNED | S1    |
| 2   | SVO       | [MHALL]                      | COUNTERSIGNED | S1    |
| 3   | Mary      | [Subject]                    | UNDERSIGNED   | S3    |
| 4   | Had       | [V]                          | UNDERSIGNED   | S3    |
| 5   | ALL       | [O]                          | UNDERSIGNED   | S3    |
| 6   | D         | [A]                          | UNDERSIGNED   | S3    |
| 7   | Mary      | [Little]                     | UNDERSIGNED   | S3    |
| 8   | Lamb      | [O]                          | CONNOTED      | S3    |
| 9   | Mary      | []                           | IDENTITY      | S4    |
| 10  | Had       | []                           | IDENTITY      | S4    |
| 11  | A         | []                           | IDENTITY      | S4    |
| 12  | Little    | []                           | IDENTITY      | S4    |
| 13  | Lamb      | []                           | IDENTITY      | S4    |
| 14  | MHALL     | [Mary, Had, A, Little, Lamb] | CANONIZED     | S2    |
| 15  | S         | []                           | IDENTITY      | S4    |
| 16  | V         | []                           | IDENTITY      | S4    |
| 17  | O         | []                           | IDENTITY      | S4    |
| 18  | SVO       | [Subject, V, O]              | CANONIZED     | S2    |
| 19  | ALL       | [A, Little, Lamb]            | CANONIZED     | S2    |

SVO and ALL subscript canonizations are dropped by §8.3 dedup; MTS ALL component identities (A, Little, Lamb) are dropped by identity dedup. `V`, `O`, `D` have no word binding and encode to their own raw typed nodes (§10 resolution-failure clause) — the same encoding path as any resolved character, minus the word.

---

## 15. Test Matrix

| ID                    | Criterion                                                                                                                                                                | Category    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| **Lexer**             |                                                                                                                                                                          |             |
| KS-1                  | All token types recognized: SIGNATURE, COUNTERSIGN, CANONIZE, CONNOTATE, UNDERSIGN, ANNOTATION, NEWLINE, INDENT, DEDENT, EOF                                             | Lexer       |
| KS-2                  | Multi-char operator priority: `==`, `=>` matched before `=`, `>`                                                                                                         | Lexer       |
| KS-3                  | BPE annotations: `(...)` with nested parens preserved as ANNOTATION tokens                                                                                               | Lexer       |
| KS-4                  | Indent/dedent tracking: Python-style INDENT/DEDENT tokens                                                                                                                | Lexer       |
| KS-5                  | Edge cases: empty input, whitespace-only, unknown characters raise LexerError                                                                                            | Lexer       |
| **Parser**            |                                                                                                                                                                          |             |
| KS-6                  | AST structure reflects scope model: OperatorScope nodes with sig, op, items, child_block                                                                                 | Parser      |
| KS-7                  | Block parsing: INDENT/DEDENT creates Block nodes                                                                                                                         | Parser      |
| KS-8                  | Annotations preserved as AST nodes (not discarded)                                                                                                                       | Parser      |
| KS-9                  | Inline annotation attachment: sig-side and node-side                                                                                                                     | Parser      |
| KS-10                 | Empty source produces empty script (no error)                                                                                                                            | Parser      |
| **Scope & Operators** |                                                                                                                                                                          |             |
| KS-11                 | COUNTERSIGN per-item: `A == B C` → `{A:B}, {B:A}, {A:C}, {C:A}`                                                                                                          | Scope       |
| KS-12                 | UNDERSIGN per-item reversed: `A = B C` → `{B:A}, {C:A}`                                                                                                                  | Scope       |
| KS-13                 | CONNOTATE per-item: `A > B C` → `{A:B}, {A:C}`                                                                                                                           | Scope       |
| KS-14                 | CANONIZE aggregates: `A => B C D` → `{A:[B,C,D]}`                                                                                                                        | Scope       |
| KS-15                 | Operator chain: `A == B > C = D` → correct signatures per scope                                                                                                          | Scope       |
| KS-16                 | Indent extends scope: items in child block belong to parent operator                                                                                                     | Scope       |
| KS-17                 | DEDENT returns to parent scope                                                                                                                                           | Scope       |
| KS-18                 | Non-CANONIZE with indent: per-item emission extends into child block                                                                                                     | Scope       |
| **MTS**               |                                                                                                                                                                          |             |
| KS-19                 | MTS expansion: all-uppercase multi-char compound produces component identities + canonization                                                                            | MTS         |
| KS-20                 | No MTS for single-char identifiers                                                                                                                                       | MTS         |
| KS-20b                | No MTS for lowercase/mixed-case words (`had`, `Hello`) — single-token, not decomposed                                                                                   | MTS         |
| KS-21                 | MTS on node side: `A == MHALL` triggers MTS for MHALL                                                                                                                    | MTS         |
| KS-22                 | Node count invariant: MTS node count equals character count                                                                                                              | MTS         |
| **Binding**           |                                                                                                                                                                          |             |
| KS-23                 | Block annotation first-letter matching: `(Mary Had A Little Lamb)` + `MHALL`                                                                                             | Binding     |
| KS-24                 | Occurrence counter: duplicate letters resolved to different words                                                                                                        | Binding     |
| KS-25                 | Inline binding: `S(ubject)` resolves immediately, bypasses counter                                                                                                       | Binding     |
| KS-26                 | Rule B4 override: inline binding patches parent MTS CANONIZE entry                                                                                                       | Binding     |
| KS-27                 | Scope inheritance: characters seek from inner to outer scope                                                                                                             | Binding     |
| KS-28                 | Scope shadowing: inner scope binding shadows outer for same character                                                                                                    | Binding     |
| KS-29                 | Counter reset: each new scope starts counters at zero                                                                                                                    | Binding     |
| KS-30                 | Unresolved identifier (BindingScope returns None) is encoded as its own raw BPE token — no special fallback state                                                        | Binding     |
| KS-31                 | Inert annotation: no matching characters → no effect                                                                                                                     | Binding     |
| KS-32                 | An unresolved single character (e.g. `Z`) encodes to a single typed uint64 node — the same encoding path as any resolved character                                     | Encoding    |
| **Self-Identity**     |                                                                                                                                                                          |             |
| KS-33                 | Self-identity: `A = A` → `{A: []}` with op=IDENTITY                                                                                                                      | Operators   |
| **Structure**         |                                                                                                                                                                          |             |
| KS-34                 | Nodes always a list: `A => B` → `{A: [B]}`, `A` → `{A: []}`                                                                                                              | Structure   |
| **Integration**       |                                                                                                                                                                          |             |
| KS-35                 | Complex nested example (§14.11) produces correct complete entry list                                                                                                     | Integration |
| KS-36                 | Word-bound example (§14.12) produces correct resolved entries                                                                                                            | Integration |
| KS-37                 | Uniform-tokenizer integration: all characters (bound and unresolved) produce valid typed nodes                                                                            | Integration |
| **MTS Deduplication** |                                                                                                                                                                          |             |
| KS-38                 | Component identity dedup: overlapping MTS expansions silently drop duplicate character identities (S4)                                                                   | MTS Dedup   |
| KS-39                 | Intra-expansion dedup: repeated characters in one compound emit only one identity (e.g., second L in MHALL)                                                              | MTS Dedup   |
| KS-40                 | Canonization dedup: CANONIZE entries with same (sig, nodes) silently dropped across MTS and subscript                                                                    | MTS Dedup   |
| KS-41                 | Canonical resolution (§8.3): an identifier's MTS components are identical wherever it appears (node-side and signature-side), even under an ambiguous occurrence counter | MTS         |
| KS-42                 | Canonical encoding (§11.3/§11.4): exactly one CANONIZED kline per compound identifier; identity klines carry single-token signatures only (CONTEXT.md "Identity")        | Encoding    |
| **Output Ordering**   |                                                                                                                                                                          |             |
| KS-43                 | Output ordering (§11.6): compiled source (operators, subscript identities, single-char CANONIZE) precedes all MTS entries (§8 character-level, §11.3 BPE-subword); partition is stable | Output      |
