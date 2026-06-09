# KScript NLP Binding Specification

**Version:** 1.0  
**Date:** 2026-06-09  
**Status:** Design

---

## 1. Overview

NLP binding is a compilation pass that resolves single-character KScript signatures to NLP words via comment word lists in the source. When a signature is bound, its encoded value carries grammatically meaningful NLP type bits (POS + DEP + MORPH) rather than Mod32 character bits. The pass sits between the parser and the AST emitter, producing a symbol table that the encoding layer consumes.

Without NLP binding, a signature like `M` encoded under NLP-BPE receives the NLP type of the BPE token for the letter "M" — uninformative. With binding, `M` resolves to "Mary" and receives the NLP type of the BPE token for "Mary" — grammatically rich.

### 1.1 Compilation Pipeline

```
Source (.ks) → Lexer → Token Stream → Parser → AST → Binding Resolver → ASTEmitter → TokenEncoder → [CompiledEntry]
                                                          ↓
                                                   NLP Symbol Table
```

The binding resolver is a new stage between parser and ASTEmitter. It walks the AST, processes NLP word lists from comments, and builds a symbol table mapping single-character signatures to NLP words. The ASTEmitter and TokenEncoder consult this table during encoding.

### 1.2 Mode Selection

The binding resolver is only active when an NLP tokenizer is selected. Mod32 compilation skips the binding resolver entirely — comments remain inert, no symbol table is built, and all encoding follows existing Mod32 behaviour. See @kscript §5 and @kscript-nlp §5.

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

A word list is a sequence of N words. Each word is a string that will be encoded through the NLP tokenizer to produce an NLP-BPE token (or tokens). The binding resolver does not perform encoding — it stores the raw word strings. Encoding happens during the TokenEncoder stage.

### 3.3 Claiming Rule

A word list *claims* the immediately following signature in the AST when word count equals character count. When claimed, a positional zip binds each character to its corresponding word: character at position i binds to word at position i.

When word count does not match character count, the word list is not claimed. It has no effect on compilation.

When multiple comments precede a signature without intervening claims, only the most recent pending comment is available for claiming. Earlier unclaimed comments are discarded.

### 3.4 Inert Comments

Comments that are not claimed by any signature (mismatched word count, orphaned at end of script, etc.) have no effect on compilation. They do not produce entries, bindings, or errors.

---

## 4. NLP Binding Mechanisms

Three mechanisms resolve a single-character signature to an NLP word, in order of precedence:

### 4.1 Inline Binding

An inline comment `S(ubject)` immediately binds the signature `S` to the word `"Subject"`. This is the tightest binding — no traversal required. Inline bindings are processed first during the binding resolver's AST walk.

### 4.2 Upward Traversal

An unbound signature looks up the AST to the nearest enclosing scope for a binding with the same character. If found, the signature is bound to that word.

Example: `M` in `S = M` (within SVO's subscript) looks upward, finds M bound to "Mary" in MHALL's scope, and binds to "Mary".

### 4.3 Downward Traversal

An unbound signature looks down the AST into its subscript block for a binding with the same character. If found, the signature is bound to that word.

Example: `S` in SVO looks downward into its subscript, finds `S(ubject)`, and binds S to "Subject".

### 4.4 Binding Order

Bindings are consumed in order. Within a scope, when character position i has been claimed, subsequent bindings for the same character position advance to position i+1. This is relevant when a character appears multiple times in a word list: in `(Alice Alpha)`, the AA decomposition binds A#0 → "Alice" (claimed), then A#1 → "Alpha".

### 4.5 Lexical Scoping

Inline bindings follow standard lexical scoping. The scope of an inline binding is its containing subscript block. Inner blocks inherit outer bindings and can shadow them. When the block exits, the outer binding is restored.

Example: if M → "Mary" is bound in an outer scope and `M(od)` appears in a subscript block, M → "Mod" shadows "Mary" within that block only. Outside the block, M → "Mary" is restored.

### 4.6 Unbound Signatures

When a single-character signature cannot be resolved through any binding mechanism (no inline, no upward match, no downward match), it remains unbound. Unbound signatures are encoded using standard Mod32 bit-packed encoding instead of NLP encoding. This produces mixed NLP/Mod32 klines within the same graph.

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

Inline comments (those immediately following a SIGNATURE token, before any operator) are attached to the preceding signature's `PrimaryConstruct` as metadata. The lexer already separates these into two tokens (SIGNATURE then COMMENT).

### 5.3 No Grammar Changes

The KScript grammar (@kscript §3) is unchanged. The parser's grammar rules remain the same. The only change is that COMMENT tokens are now preserved in the AST rather than discarded.

---

## 6. Binding Resolver

### 6.1 Interface

The binding resolver accepts a `KScriptFile` AST and an NLP tokenizer, and produces an `NLPSymbolTable`:

```
BindingResolver:
    resolve(file: KScriptFile, tokenizer: NLPTokenizer) -> NLPSymbolTable
```

The symbol table is a mapping from `(signature_character, scope)` to NLP word string. The exact data structure is an implementation concern.

### 6.2 Algorithm

The binding resolver performs a single recursive walk of the AST:

1. Maintain a scope stack. Each scope contains: the pending word list (if any), the set of character bindings with consumption counters, and a reference to the parent scope.

2. For each construct in the AST:
   - **Comment**: set as the pending word list for the current scope.
   - **PrimaryConstruct with inline comment**: extract the NLP word, bind the signature character in the current scope. Clear any pending word list.
   - **PrimaryConstruct with SIGNATURE** (no inline comment): attempt to claim the pending word list if word count matches character count. Bind each character positionally. Clear pending word list.
   - **Subscript block (=>)**: push a new scope (inheriting parent bindings). Recurse into the block. Pop scope on exit.
   - **Single-character signature in operator position** (`S = M`, `L > O`): resolve the character using the current scope's bindings (inline first, then inherited). If not found, mark as unbound.

3. On scope exit, discard the scope's bindings. Parent bindings are restored.

### 6.3 Inline Comment Extraction

For `S(ubject)`, the resolver:
1. Reads the SIGNATURE token value: `"S"`
2. Reads the attached COMMENT token value: `"(ubject)"`
3. Strips parentheses: `"ubject"`
4. Prepends the signature character: `"S" + "ubject"` → `"Subject"`
5. Binds `"S"` → `"Subject"` in the current scope

Case is preserved. The resulting word is encoded as-is through the NLP tokenizer.

### 6.4 Output

The `NLPSymbolTable` maps each single-character signature to its resolved NLP word (or marks it as unbound). This table is consumed by the ASTEmitter and TokenEncoder.

---

## 7. Encoding with NLP Bindings

### 7.1 Signature Encoding

When the symbol table has a binding for a single-character signature, the TokenEncoder encodes the bound word through the NLP tokenizer and uses the resulting NLP type bits as the signature:

```
Binding: S → "Subject"
encode("Subject") → [(nlp_type32 << 32) | bpe_id]
Signature of S = nlp_type32 (high 32 bits only, BPE ID masked)
```

This follows existing `make_signature()` behaviour (@signature) — NLP-BPE nodes contribute only their NLP type bits.

When no binding exists, the character is encoded using Mod32 bit-packed encoding as a fallback.

### 7.2 MCS Signature Encoding

Multi-character signatures are composed via OR-reduction of their constituent single-character signatures, exactly as in Mod32 mode:

```
MHALL = sig("Mary") | sig("had") | sig("a") | sig("little") | sig("lamb")
```

If some characters are NLP-bound and others are Mod32, the OR-reduction mixes both bit types. The resulting signature has NLP type bits for bound characters and Mod32 character bits for unbound characters.

### 7.3 Node Encoding

Nodes carry full NLP-BPE tokens — the BPE token ID is retained for decoding:

```
Binding: S → "Subject"
Node for S = (nlp_type32 << 32) | bpe_id    (full 64-bit NLP-BPE node)
```

Multi-word bindings produce a node list (one NLP-BPE token per BPE subword). The TokenEncoder already handles this for NLP-BPE mode (@kscript-nlp §5.2).

Unbound node positions use Mod32 encoding.

### 7.4 ASTEmitter Integration

The ASTEmitter's `_emit_entry` method is extended to consult the NLP symbol table. When a binding exists for the signature character, the emitter passes the bound word (not the raw character) to the TokenEncoder. The symbolic entry's `sig` field carries the NLP word when bound, the raw character when unbound.

### 7.5 TokenEncoder Integration

The TokenEncoder's `_encode_sig` and `_encode_node` methods already handle NLP-BPE encoding (@kscript-nlp §6). No changes are needed to the TokenEncoder itself — it receives the bound word string and encodes it through the tokenizer as usual. The binding resolver and ASTEmitter handle the resolution; the encoder remains agnostic.

---

## 8. Decompilation

### 8.1 NLP Signature Recovery

NLP-bound signatures are OR-reduced NLP type bits. Without a source map, the decompiler cannot recover the original KScript identifier or NLP word from the signature alone. The decompiler decodes what it can:

- **NLP-BPE nodes**: decoded via `tokenizer.decode([node])` to recover the word text.
- **NLP type signatures**: decoded to a description of set NLP type bits (e.g., `"<PROPN|VERB|DET|ADJ|NOUN>"`).
- **Mod32 signatures**: decoded normally via existing Mod32 decompilation.

### 8.2 Diagnostic Value

The Trainer's diagnostic reporting benefits from:
1. **BPE-decoded nodes**: the node side of each kline is fully decodable, providing readable output.
2. **LLM agent interpretation**: the Trainer has access to an LLM agent that can reason about signature composition from training context.
3. **Readability as signal**: when Kalvin's responses produce readable (NLP-bound) output, it indicates good learning. Garbled output (unbound Mod32 characters) indicates underfitting.

### 8.3 Source Map (Deferred)

A source map — a separate artefact mapping compiled entries back to KScript identifiers and NLP words — is deferred to a future spec. Current diagnostic needs are served by the mechanisms above.

---

## 9. Impact on Existing Code

### 9.1 Unchanged Components

| Component | Impact |
|-----------|--------|
| Lexer | None — token types unchanged |
| TokenEncoder | None — receives bound words, encodes as usual |
| KLine format | None — same uint64 signatures and nodes |
| `make_signature()` | None — OR-reduction unchanged |
| Signature construction | None — NLP_TYPE_MASK behaviour unchanged |
| Mod32 compilation | None — binding resolver skipped entirely |

### 9.2 Modified Components

| Component | Change |
|-----------|--------|
| Parser | Preserve COMMENT tokens as AST nodes instead of discarding |
| AST | New `Comment` node type; `PrimaryConstruct` gains optional inline comment field |
| ASTEmitter | Consult NLP symbol table; pass bound words to encoder |
| Decompiler | NLP-aware signature decoding (type bit descriptions) |

### 9.3 New Components

| Component | Responsibility |
|-----------|---------------|
| BindingResolver | Walk AST, process word lists, build symbol table |
| NLPSymbolTable | Map signature characters to NLP words, scoped |

---

## 10. Test Matrix

| ID | Criterion | Category |
| -- | --------- | -------- |
| NB-1 | Inline binding: `S(ubject)` binds S to "Subject" (case preserved) | Binding |
| NB-2 | Inline binding: `V(erb)` binds V to "Verb" | Binding |
| NB-3 | Inline binding: `D(et)` binds D to "Det" | Binding |
| NB-4 | Block word list claiming: `(Mary had a little lamb)` followed by `MHALL` binds M→Mary, H→had, A→a, L→little, L→lamb positionally | Binding |
| NB-5 | Word list mismatch: `(one two three)` followed by `AB` does not bind — comment is inert | Binding |
| NB-6 | Orphan comment: `(note)` with no following signature is inert | Binding |
| NB-7 | Multiple pending comments: only the most recent unclaimed comment is available for claiming | Binding |
| NB-8 | Upward traversal: `M` in `S = M` resolves to "Mary" from enclosing MHALL scope | Binding |
| NB-9 | Downward traversal: `S` in SVO resolves to "Subject" via `S(ubject)` in subscript | Binding |
| NB-10 | Binding consumption: in `(Alice Alpha)`, A#0 binds to "Alice" (claimed), A#1 binds to "Alpha" | Binding |
| NB-11 | Duplicate character disambiguation: `L#0` in ALL subscript binds to "little", `L#1` binds to "lamb" | Binding |
| NB-12 | Lexical scoping: inline binding `M(od)` shadows outer M→"Mary" within its subscript block | Binding |
| NB-13 | Scope restoration: after exiting a subscript block that shadowed M, M reverts to outer binding | Binding |
| NB-14 | Unbound signature: character with no binding in any scope falls back to Mod32 encoding | Encoding |
| NB-15 | NLP-bound signature: carries NLP type bits only (BPE ID masked) — same as @kscript-nlp §5.4 | Encoding |
| NB-16 | NLP-bound node: carries full NLP-BPE token (BPE ID retained) — same as @kscript-nlp §5.2 | Encoding |
| NB-17 | Mixed MCS: MHALL signature has NLP bits for bound chars, Mod32 bits for unbound chars | Encoding |
| NB-18 | Mod32 compilation unchanged: binding resolver is skipped, comments are inert, all encoding is Mod32 | Compatibility | ✅ `TestNB18Mod32Unchanged` |
| NB-19 | Same `.ks` source compiles under both Mod32 and NLP without modification | Compatibility | ✅ `TestNB19SameSourceBothModes` |
| NB-20 | Parser preserves comments as AST nodes: COMMENT tokens appear in construct sequence | Parser |
| NB-21 | Inline comment attachment: `S(ubject)` produces PrimaryConstruct with attached comment metadata | Parser |
| NB-22 | No grammar changes: parser grammar rules are identical to @kscript §3 | Parser |
| NB-23 | Full example: the MHALL script produces correct NLP-bound entries for all 11 bindings | Integration |
| NB-24 | Decompiler decodes NLP-BPE nodes to readable words | Decompilation |
| NB-25 | Decompiler represents NLP-type-only signatures as type bit descriptions | Decompilation |
| NB-26 | Significance routing is tokenizer-agnostic: same S1/S2/S3/S4 mechanism works with NLP-bound klines | Rationalisation | ✅ `TestNB26SignificanceRouting` |

---

## 11. Out of Scope

- **Source map for decompilation.** Deferred — see §8.3.
- **Mixed NLP/Mod32 rationalisation.** Unbound signatures produce Mod32 klines that coexist with NLP klines. How rationalisation handles this is a separate concern.
- **New KScript syntax for word lists.** Comments serve as word lists for MVP. A distinct syntactic form may be introduced later if ambiguity arises.
- **NLP grammar training.** How the BPE vocabulary and grammar dictionary are trained is defined in @tokenizer.
- **TokenEncoder changes.** The encoder already handles NLP-BPE encoding. No changes needed.
- **BPE multi-token decomposition.** Already handled by @kscript-nlp §5.3. This spec does not modify that behaviour.
