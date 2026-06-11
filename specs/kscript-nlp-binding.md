# KScript NLP Binding Specification

**Version:** 2.0  
**Date:** 2026-06-10  
**Status:** Implemented  

---

## 1. Overview

NLP binding resolves single-character KScript signatures to NLP words via comment word lists in the source. When a signature is bound, its encoded value carries grammatically meaningful NLP type bits (POS + DEP + MORPH) rather than Mod32 character bits. The pass is integrated into the AST emitter's single walk, resolving characters inline via a `BindingScope`.

Without NLP binding, a signature like `M` encoded under NLP-BPE receives the NLP type of the BPE token for the letter "M" — uninformative. With binding, `M` resolves to "Mary" and receives the NLP type of the BPE token for "Mary" — grammatically rich.

### 1.1 Compilation Pipeline

```
Source (.ks) → Lexer → Token Stream → Parser → AST → ASTEmitter (+ BindingScope inline) → TokenEncoder → [CompiledEntry]
```

The ASTEmitter resolves bindings inline during its single walk of the AST. There is no separate resolution pass. No mapping artefact is produced between stages. The `BindingScope` is a lightweight object created by the `Compiler` and passed to the `ASTEmitter`, which resolves characters as it walks.

### 1.2 Mode Selection

Inline binding resolution is only active when an NLP tokenizer is selected. Mod32 compilation skips binding entirely — comments remain inert, no `BindingScope` is created, and all encoding follows existing Mod32 behaviour. See @kscript §5 and @kscript-nlp §5.

---

## 2. Dependencies

- **@kscript** — KScript language specification. Lexer, parser, grammar, operators, and compilation semantics are defined there.
- **@kscript-nlp** — NLP-BPE encoding mode. Node format `(nlp_type32 << 32) | bpe_token_id`, BPE decomposition, signature masking.
- **@signature** — `make_signature()` with `NLP_TYPE_MASK`, `is_nlp_node()`, `is_literal_node()`.

---

## 3. NLP Word Lists

### 3.1 Definition

An NLP word list is a KScript comment interpreted as a sequence of words for signature binding. Two syntactic forms:

**Block comment**: `(word1 word2 ... wordN)` — a parenthesised list of whitespace-separated words. Becomes an AST node in the construct sequence.

**Inline comment**: `S(ubject)` — a SIGNATURE token immediately followed by a comment. The first character comes from the SIGNATURE token, the remainder from the comment content stripped of parentheses. Case is preserved: `S` + `ubject` → `"Subject"`.

### 3.2 Word List Semantics

A word list is a sequence of N words. Each word is a string that will be encoded through the NLP tokenizer to produce an NLP-BPE token (or tokens). The binding scope stores raw word strings — no encoding happens during binding. Encoding occurs during the TokenEncoder stage.

### 3.3 Word List Matching

Block word lists serve characters that follow them in the AST. A character matches a word whose first letter equals the character (**case-sensitive**: `word[0] == char`). When multiple words in the same word list start with the same letter, an occurrence counter disambiguates: the first occurrence of the character resolves to the first matching word, the second occurrence resolves to the second matching word, and so on.

Example: `(Mary Had A Little Lamb)` followed by `MHALL`:
- `M` → "Mary" (first-letter match: `M` == `M`)
- `H` → "Had" (first-letter match: `H` == `H`)
- `A` → "A" (first-letter match: `A` == `A`)
- `L` → "Little" (first `L` match, counter = 0)
- `L` → "Lamb" (second `L` match, counter increments to 1)

> **Note on case-sensitivity:** Matching is case-sensitive: `word[0] == char`. A character `A` matches "Alice" but not "alpha". A character `a` matches "alpha" but not "Alice".

### 3.4 Inert Comments

Comments where no word's first letter matches any encountered character are inert — they have no effect on compilation. Surplus words (words not matched by any character) are simply never matched. There is no all-or-nothing rule: a word list can partially match, binding some characters while surplus words are ignored.

Orphan comments (comments with no following characters) and comments at end of script are also inert.

---

## 4. NLP Binding Rules

Four rules govern binding resolution, processed inline during the ASTEmitter's single walk.

### Rule 1 — Binding

A binding maps a single character to a word. Once bound in a scope, it cannot be re-bound except by inline override (Rule 4).

### Rule 2 — Characters Seek Bindings

When the emitter encounters a single-character signature, resolution proceeds in this order:

1. **Inline comment on this position** → bind immediately (Rule 4).
2. **Word lists** → search current scope most-recent-first, then parent scopes upward (Rule 3).

### Rule 3 — Word List Matching (First-Letter)

First-letter matching is **case-sensitive**: a word matches a character when `word[0] == char` (no case folding).

An occurrence counter per scope per character handles disambiguation:

- **Single match** (unambiguous): bind the word. Counter does **not** increment.
- **Multiple matches** (ambiguous): bind the Nth word where N = current counter value, then increment counter by 1.
- **Counter exceeds matches**: no match in this word list — continue to next (older) word list or outer scope.
- **No matches in any scope**: character is unbound.

The counter is per-scope-per-character, keyed on the raw character value. Each new scope starts at zero. The counter only increments on ambiguous matches (when multiple words in the same list start with the same letter).

### Rule 4 — Inline Binding

An inline comment `S(ubject)` binds the signature character immediately:

1. **Bind**: the character is resolved to the inline word, bypassing the word-list counter entirely. The occurrence counter is not incremented.
2. **Override**: the inline word retroactively patches the matching character in the immediate parent kline's MCS CANONIZE entry only. No propagation beyond one level. If the character is not found in the parent kline, the override is a safe no-op.

Example: `SVO => Block([S(ubject) = M])`
- Before override: `CANONIZE("SVO", ["S","V","O"])`
- After override: `CANONIZE("SVO", ["Subject","V","O"])` — `S` patched at index 0.

### Scope

Scopes are created by `chain_right` (`=>`) boundaries. Each scope holds:

- **Word lists**: ordered collection of block word lists, searched most-recent-first.
- **Occurrence counters**: per-character disambiguation counters, starting at zero.

Characters seek from the current (innermost) scope first, then parent scopes upward. Each new scope starts with all counters at zero.

### Architecture (Single-Pass)

The emitter walks the AST once, resolving bindings inline. There is no separate resolution pass. No mapping artefact is produced between stages. The `BindingScope` is a lightweight object with four operations:

- `push_scope()` — push a new scope onto the stack.
- `pop_scope()` — pop the top scope.
- `add_word_list(words)` — append a word list to the current scope.
- `resolve(char) → str | None` — walk the scope stack, first-letter matching, occurrence counter.

The `Compiler` creates a `BindingScope` as a local variable in `compile()` (no `scope` property on `Compiler`), initialises its root scope, and passes it to the `ASTEmitter`. In Mod32 mode, no `BindingScope` is created.

### Unbound Characters

When a single-character signature cannot be resolved through any mechanism (no inline comment, no matching word in any scope), it remains unbound. Unbound characters fall back to Mod32 encoding. This produces mixed NLP/Mod32 klines within the same graph.

Multi-character signatures (MCS) are composed from their constituent single-character bindings. If any constituent character is unbound, the MCS signature includes Mod32 bits for that character alongside NLP bits for bound characters.

---

## 5. Parser Changes

### 5.1 Comments as AST Nodes

The parser currently discards COMMENT tokens as insignificant whitespace (@kscript §3.1). Under this spec, the parser preserves comments as first-class AST nodes in the construct sequence.

A new AST node type `Comment` is added:

```
Comment:
    text: str       # Raw comment text including parentheses
    line: int
    column: int
```

`Comment` is a `ConstructItem` — it appears in the construct sequence alongside `PrimaryConstruct` and `Literal`.

### 5.2 Parser Behaviour

The parser's `_skip_insignificant` method is modified: NEWLINE tokens are still skipped, but COMMENT tokens are no longer skipped. Instead, when the parser encounters a COMMENT token in construct position, it emits a `Comment` AST node.

Inline comments can appear in two positions within a primary construct:

1. **Sig-side**: immediately after the SIGNATURE token, before any operator. Example: `S(ubject) = M` — the comment `(ubject)` attaches to the signature side.
2. **Node-side**: immediately after the node token (right side of operator). Example: `A = D(et)` — the comment `(et)` attaches to the node side.

Sig-side comments are attached to the `PrimaryConstruct` via its `inline_comment` field. Node-side comments are attached via its `node_inline_comment` field. The lexer already produces SIGNATURE+COMMENT (or IDENT+COMMENT) token pairs for both positions; the parser handles attachment based on position.

### 5.3 No Grammar Changes

The KScript grammar (@kscript §3) is unchanged. The parser's grammar rules remain the same. The only change is that COMMENT tokens are now preserved in the AST rather than discarded.

---

## 6. Encoding with NLP Bindings

### 6.1 Signature Encoding

When the BindingScope has a binding for a single-character signature, the TokenEncoder encodes the resolved word through the NLP tokenizer and uses the resulting NLP type bits as the signature:

```
Binding: S → "Subject"
encode("Subject") → [(nlp_type32 << 32) | bpe_id]
Signature of S = nlp_type32 (high 32 bits only, BPE ID masked)
```

This follows existing `make_signature()` behaviour (@signature) — NLP-BPE nodes contribute only their NLP type bits.

When no binding exists, the character is encoded using Mod32 bit-packed encoding as a fallback.

### 6.2 BindingScope API

The `BindingScope` class (`src/kscript/binding_scope.py`) provides the binding resolution interface:

**Public methods:**

| Method | Description |
|--------|-------------|
| `push_scope()` | Push a new scope onto the stack. Each scope starts with empty word lists and zero counters. |
| `pop_scope()` | Pop the top scope off the stack. |
| `add_word_list(words)` | Append a word list to the current (top) scope. Multiple calls accumulate — a scope may have many word lists, searched most-recent-first. |
| `resolve(char) → str \| None` | Walk scope stack innermost-first, searching each scope's word lists most-recent-first. Returns matched word or `None` if unbound. |

**Internal `_Scope` dataclass:**

| Field | Type | Description |
|-------|------|-------------|
| `word_lists` | `list[list[str]]` | Ordered collection of word lists in this scope. |
| `counters` | `dict[str, int]` | Per-character occurrence counter for disambiguation. Keyed on raw character value. |

**Resolution algorithm** (inside `resolve(char)`):

1. Walk scope stack from innermost to outermost.
2. For each scope, iterate word lists in reverse order (most-recent-first).
3. For each word list, collect words whose first letter equals `char` (case-sensitive: `word[0] == char`).
4. If matches found:
   - Read current counter for `char` in this scope.
   - If counter < number of matches: return the word at that index. If ambiguous (multiple matches), increment counter. If unambiguous (single match), counter unchanged.
   - If counter ≥ number of matches: continue to next word list in this scope.
5. No match in any scope: return `None`.

### 6.3 MCS Signature Encoding

Multi-character signatures are composed via OR-reduction of their constituent single-character signatures, exactly as in Mod32 mode:

```
MHALL = sig("Mary") | sig("Had") | sig("A") | sig("Little") | sig("Lamb")
```

If some characters are NLP-bound and others are Mod32, the OR-reduction mixes both bit types. The resulting signature has NLP type bits for bound characters and Mod32 character bits for unbound characters.

### 6.4 Node Encoding

Nodes carry full NLP-BPE tokens — the BPE token ID is retained for decoding:

```
Binding: S → "Subject"
Node for S = (nlp_type32 << 32) | bpe_id    (full 64-bit NLP-BPE node)
```

Multi-word bindings produce a node list (one NLP-BPE token per BPE subword). The TokenEncoder already handles this for NLP-BPE mode (@kscript-nlp §5.2).

Unbound node positions use Mod32 encoding.

### 6.5 TokenEncoder Integration

The TokenEncoder's `_encode_sig` and `_encode_node` methods already handle NLP-BPE encoding (@kscript-nlp §6). No changes are needed to the TokenEncoder itself — it receives the resolved word string and encodes it through the tokenizer as usual. The inline resolution via BindingScope happens upstream in the ASTEmitter; the encoder remains agnostic.

---

## 7. Decompilation

### 7.1 NLP Signature Recovery

NLP-bound signatures are OR-reduced NLP type bits. Without a source map, the decompiler cannot recover the original KScript identifier or NLP word from the signature alone. The decompiler decodes what it can:

- **NLP-BPE nodes**: decoded via `tokenizer.decode([node])` to recover the word text.
- **NLP type signatures**: decoded to a description of set NLP type bits (e.g., `"<PROPN|VERB|DET|ADJ|NOUN>"`).
- **Mod32 signatures**: decoded normally via existing Mod32 decompilation.

### 7.2 Diagnostic Value

The Trainer's diagnostic reporting benefits from:
1. **BPE-decoded nodes**: the node side of each kline is fully decodable, providing readable output.
2. **LLM agent interpretation**: the Trainer has access to an LLM agent that can reason about signature composition from training context.
3. **Readability as signal**: when Kalvin's responses produce readable (NLP-bound) output, it indicates good learning. Garbled output (unbound Mod32 characters) indicates underfitting.

### 7.3 Source Map (Deferred)

A source map — a separate artefact mapping compiled entries back to KScript identifiers and NLP words — is deferred to a future spec. Current diagnostic needs are served by the mechanisms above.

---

## 8. Impact on Existing Code

### 8.1 Unchanged Components

| Component | Impact |
|-----------|--------|
| Lexer | None — token types unchanged |
| TokenEncoder | None — receives resolved words, encodes as usual |
| KLine format | None — same uint64 signatures and nodes |
| `make_signature()` | None — OR-reduction unchanged |
| Signature construction | None — NLP_TYPE_MASK behaviour unchanged |
| Mod32 compilation | None — BindingScope is not created, comments are inert |

### 8.2 Modified Components

| Component | Change |
|-----------|--------|
| Parser | Preserve COMMENT tokens as AST nodes instead of discarding |
| AST | New `Comment` node type; `PrimaryConstruct` gains optional inline comment field |
| ASTEmitter | Inline resolution via BindingScope: resolves single-character signatures during its single AST walk; handles Rule 4 inline override patching |
| Compiler | Creates BindingScope as local variable in `compile()` (no `scope` property on `Compiler`); passes to ASTEmitter |
| Decompiler | NLP-aware signature decoding (type bit descriptions) |

### 8.3 Eliminated Components

| Eliminated Concept | Replacement |
|-------------------|-------------|
| Separate resolution pass | Inline resolution in ASTEmitter's single walk |
| NLP mapping artefact | BindingScope — lightweight scope stack, no separate artefact |
| Scope and Binding dataclasses | Folded into BindingScope's internal `_Scope` |
| Binding consumption flag | Word lists are immutable; occurrence counting replaces consumption |
| Positional word-list matching | First-letter matching with case-sensitive comparison |
| Word count mismatch rule | Surplus words are inert — no all-or-nothing rule |
| Repeated walk API | Single pass eliminates need for repeated walking |

---

## 9. Test Matrix

| ID | Criterion | Category |
| -- | --------- | -------- |
| NB-1 | Inline binding: `S(ubject)` binds S to "Subject" (case preserved) | Binding |
| NB-2 | Inline binding: `V(erb)` binds V to "Verb" | Binding |
| NB-3 | Inline binding: `D(et)` binds D to "Det" | Binding |
| NB-4 | Block word list matching: `(Mary Had A Little Lamb)` followed by `MHALL` binds M→Mary, H→Had, A→A, L→Little, L→Lamb via first-letter matching with occurrence counter for duplicate L | Binding |
| NB-5 | Word list inert: `(one two three)` followed by `AB` — no words start with A or B, so both characters remain unbound | Binding |
| NB-6 | Orphan comment: `(note)` with no following signature is inert | Binding |
| NB-7 | Multiple pending comments: only the most recent unclaimed comment is available for matching | Binding |
| NB-8 | Scope inheritance: `M` in `S = M` resolves to "Mary" from enclosing MHALL scope via scope stack walk | Binding |
| NB-9 | Inline resolution in subscript: `S` in SVO resolves to "Subject" via `S(ubject)` in subscript | Binding |
| NB-10 | Occurrence counter: in `(Alice Alpha)`, first A resolves to "Alice" (counter 0), second A resolves to "Alpha" (counter increments on ambiguous match) | Binding |
| NB-11 | Duplicate character disambiguation: `L#0` in ALL subscript binds to "little", `L#1` binds to "lamb" | Binding |
| NB-12 | Lexical scoping: inline binding `M(od)` shadows outer M→"Mary" within its subscript block | Binding |
| NB-13 | Scope restoration: after exiting a subscript block that shadowed M, M reverts to outer binding | Binding |
| NB-14 | Unbound signature: character with no binding in any scope falls back to Mod32 encoding | Encoding |
| NB-15 | NLP-bound signature: carries NLP type bits only (BPE ID masked) — same as @kscript-nlp §5.4 | Encoding |
| NB-16 | NLP-bound node: carries full NLP-BPE token (BPE ID retained) — same as @kscript-nlp §5.2 | Encoding |
| NB-17 | Mixed MCS: MHALL signature has NLP bits for bound chars, Mod32 bits for unbound chars | Encoding |
| NB-18 | Mod32 compilation unchanged: BindingScope is not created, comments are inert, all encoding is Mod32 | Compatibility | ✅ `TestNB18Mod32Unchanged` |
| NB-19 | Same `.ks` source compiles under both Mod32 and NLP without modification | Compatibility | ✅ `TestNB19SameSourceBothModes` |
| NB-20 | Parser preserves comments as AST nodes: COMMENT tokens appear in construct sequence | Parser |
| NB-21 | Inline comment attachment: `S(ubject)` produces PrimaryConstruct with attached comment metadata | Parser |
| NB-22 | No grammar changes: parser grammar rules are identical to @kscript §3 | Parser |
| NB-23 | Full example: the MHALL script produces correct NLP-bound entries for all 11 bindings | Integration |
| NB-24 | Decompiler decodes NLP-BPE nodes to readable words | Decompilation |
| NB-25 | Decompiler represents NLP-type-only signatures as type bit descriptions | Decompilation |
| NB-26 | Significance routing is tokenizer-agnostic: same S1/S2/S3/S4 mechanism works with NLP-bound klines | Rationalisation | ✅ `TestNB26SignificanceRouting` |
| NB-27 | Occurrence counter only increments on ambiguous (multiple match) binding | Binding |
| NB-28 | Counter reset: each new scope starts at zero | Binding |
| NB-29 | Inline binding bypasses counter: does not increment occurrence counter | Binding |
| NB-30 | Single match does not increment counter | Binding |
| NB-31 | Counter exceeds matches: character unbound for that occurrence | Binding |
| NB-32 | Forward-only word list: block comment only serves characters after it | Binding |
| NB-33 | Inline override no-match: bind succeeds, override does nothing | Binding |

---

## 10. Out of Scope

- **Source map for decompilation.** Deferred — see §7.3.
- **Mixed NLP/Mod32 rationalisation.** Unbound signatures produce Mod32 klines that coexist with NLP klines. How rationalisation handles this is a separate concern.
- **New KScript syntax for word lists.** Comments serve as word lists for MVP. A distinct syntactic form may be introduced later if ambiguity arises.
- **NLP grammar training.** How the BPE vocabulary and grammar dictionary are trained is defined in @tokenizer.
- **BPE multi-token decomposition.** Already handled by @kscript-nlp §5.3. This spec does not modify that behaviour.
- **Positional word-list claiming.** Eliminated in v2.0. First-letter matching replaces positional claiming entirely.
- **Separate resolution pass.** Eliminated in v2.0. Inline resolution in the emitter replaces it.
- **NLP mapping artefact.** Eliminated in v2.0. BindingScope is a lightweight scope stack, not a separate artefact.
- **Repeated walk API.** Eliminated in v2.0. Single-pass architecture removes the need for repeated walking.
