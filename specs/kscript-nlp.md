# KScript NLP-BPE Mode Specification

**Version:** 1.1  
**Date:** 2026-06-11  
**Status:** Design

---

## 1. Overview

KScript NLP-BPE is an alternative encoding mode for the KScript language. Where the default (Mod32) mode packs uppercase identifiers as bitwise-OR character compositions, NLP-BPE mode encodes identifiers as typed BPE tokens carrying linguistic annotations (POS + DEP + MORPH). The language's declarative structure — operators, grammar, AST — is unchanged. Only the encoding layer differs.

NLP-BPE KScript enables knowledge graphs whose signature similarity reflects grammatical structure rather than character overlap. Two synonyms like "run" and "sprint" share no BPE token IDs but may share NLP type bits, producing overlapping signatures suitable for candidate retrieval.

### 1.1 Compilation Pipeline

```
Source (.ks) → Lexer → Token Stream → Parser → AST → Compiler → [CompiledEntry]
                                                                        ↓
                                                            NLP-BPE-encoded KLines
```

Same pipeline as @kscript §1.1. The lexer, parser, and AST emitter are encoding-independent. The token encoder swaps Mod32 for NLP-BPE at compile time.

### 1.2 Mode Selection

NLP-BPE KScript is a compile-time mode, not a source-level construct. The same `.ks` source can be compiled under either encoding by choosing the tokenizer. No syntax changes, no mode declarations in source.

---

## 2. Dependencies

- **@kscript** — KScript language specification. NLP-BPE mode is defined as a delta against this baseline. All sections not explicitly overridden remain in force.
- **@tokenizer** — NLP-BPE node format `(nlp_type32 << 32) | bpe_token_id`, encoding/decoding rules, grammar dictionary.
- **@signature** — Plain OR-reduce `make_signature()` with no masking. NLP-BPE nodes contribute their full value (both NLP type and BPE ID bits) to signatures.

---

## 3. Impact Analysis: Mod32 → NLP-BPE

This section documents every KScript feature that changes or breaks when identifiers are NLP-BPE encoded instead of Mod32 packed.

### 3.1 Feature Classification

| Feature | Classification | Rationale |
| ------- | -------------- | --------- |
| Signature encoding | **Adapted** | `encode("ABC")` produces `(nlp_type32 << 32) \| bpe_id` — a single NLP-typed token, not a bitmask of character bits. |
| MCS expansion | **Replaced** | Mod32 MCS decomposes `MHALL` into per-character entries. NLP-BPE uses word-level MCS instead: multi-token resolved words get S4+S2 MCS entries; single-token words get S4 identity only. See §5.3. |
| Bitwise AND matching | **Adapted** | `(sig_a & sig_b) != 0` still works but tests shared bits across both NLP type and BPE ID dimensions. |
| Level inference | **Adapted** | Decompiler's `(sig & node) != 0` tests full bit overlap. Still a useful heuristic but with different false-positive profile. |
| Deduplication | **Preserved** | Dedup is on `(sig_id, nodes_tuple)` — works identically regardless of encoding. Different token IDs produce different dedup keys, which is correct. |
| Singleton unwrapping | **Preserved** | Structural rule — no encoding dependency. |
| Operator semantics | **Preserved** | COUNTERSIGN, CANONIZE, CONNOTATE, UNDERSIGN emit the same entry structures. Only the encoded values differ. |
| Lexer | **Preserved** | Lexical rules (uppercase identifiers, operators, comments, indentation) are encoding-independent. |
| Parser | **Preserved** | Grammar is encoding-independent. |
| AST emitter | **Preserved** | Symbolic entries operate on strings, not encoded values. |

### 3.2 Detailed Analysis

#### 3.2.1 Signature Encoding

**Mod32**: `encode("ABC") → [bit_A | bit_B | bit_C]` — characters OR'd into a single value where each character occupies a distinct bit position.

**NLP-BPE**: `encode("ABC") → [(nlp_type32_ABC << 32) | bpe_id_ABC]` — BPE segments "ABC" into one or more subword tokens, each annotated with NLP type. A short string like "ABC" typically becomes a single BPE token.

**Impact**: The encoded value is fundamentally different. Where Mod32 `ABC = bit_A | bit_B | bit_C` shares individual bits with `A`, `B`, and `C`, NLP-BPE `ABC` shares only NLP type bits with other tokens that have the same linguistic classification.

#### 3.2.2 MCS Expansion

**Mod32**: Multi-character signatures like `MHALL` are decomposed into per-character unsigned entries `{M: None}, {H: None}, {A: None}, {L: None}` plus a canonization `{MHALL: [M, H, A, L, L]}`. This enables name recovery in the decompiler.

**NLP-BPE**: `MHALL` is a single BPE token. There is no character-level decomposition. Instead, word-level MCS provides identity and decomposition entries at the BPE subword level (see §5.3).

**Impact**: This is the most significant change. MCS expansion operates at the BPE subword level instead of the character level.

#### 3.2.3 Bitwise AND Matching

**Mod32**: `signifies(A, AB)` is `True` because `bit_A` is set in both `A` and `AB`. This tests character overlap.

**NLP-BPE**: `signifies(A, AB)` depends on whether the full node values of token `A` and token `AB` share any bits. Since NLP-BPE nodes include both NLP type bits (high 32) and BPE ID bits (low 32), matching may occur on either dimension.

**Impact**: The matching model shifts from character-based to a combination of NLP type and BPE ID overlap. This produces a different (and for NLP use cases, richer) candidate set.

#### 3.2.4 Level Inference

**Mod32**: `(sig & node) != 0` distinguishes S2 (canonize) from S3 (connotate). If a canonized compound shares bits with its components, bit overlap detects the relationship.

**NLP-BPE**: `(sig & node) != 0` tests full bit overlap (both NLP type and BPE ID). A canonize like `{noun_A: [noun_B, verb_C]}` would produce overlap on shared bits between signature and first node, correctly inferring S2. A connotate like `{noun_A: verb_C}` with no shared bits would correctly infer S3.

**Impact**: Level inference remains a useful heuristic but with a different error profile. Cross-type relationships (e.g., a noun canonizing verbs) may be misclassified, just as cross-character Mod32 relationships sometimes are.

---

## 4. Encoding Model

### 4.1 Identifier Encoding

Under NLP-BPE, identifiers are encoded as typed BPE tokens:

```
encode("HELLO") → [(nlp_type32 << 32) | bpe_token_id]
```

The NLP tokenizer BPE-encodes the text, then looks up each token's `nlp_type32` in the grammar dictionary. Tokens without dictionary entries receive `POS_X = 65536` as their `nlp_type32`.

Unlike Mod32 (where multi-character uppercase strings are always packed to a single node), NLP-BPE may produce **multiple nodes** for a single identifier if the BPE vocabulary segments it into multiple subword tokens. This affects compilation semantics (see §5.3).

### 4.2 Multi-Token Identifiers

When a BPE encoding produces multiple tokens, the identifier is represented as a node list, not a single node:

```
encode("XYZZY") → [(type1 << 32) | id1, (type2 << 32) | id2]
```

For KScript signatures, which must be a single `uint64`, the **first token** is used as the signature node. If multiple tokens are produced, a CANONIZE-like entry maps the first token to all tokens (see §5.3).

### 4.3 Worked Example: Encoding

```
Identifier: "TEA"

BPE encode: token 12465
Grammar lookup: nlp_type32 = 131200 (POS_PROPN | DEP_SUBJ | MORPH_SING)
Node: (131200 << 32) | 12465 = 563499709247665

Identifier: "BREW"

BPE encode: token 4964
Grammar lookup: nlp_type32 = 8421376 (POS_VERB | DEP_ROOT | MORPH_PAST | MORPH_SING)
Node: (8421376 << 32) | 4964 = 36169534507324260
```

---

## 5. Core Semantics

### 5.1 Design Decisions

**Decision 1: One KScript, two encoding modes.** KScript does not have separate "symbolic" and "natural" dialects. The source language is identical. The encoding mode is selected at compile time by choosing the tokenizer. Same source, different compiled output.

**Decision 2: Uppercase-only identifiers unchanged.** The lexer accepts only `[A-Z]+` as SIGNATURE tokens. Under NLP-BPE, these are encoded as NLP-typed BPE tokens. Arbitrary strings (lowercase, mixed-case) are NOT valid identifiers — use quoted strings. Rationale: changing the lexer would be a language-level change, not an encoding change. NLP-BPE affects what the tokens mean, not what the source looks like.

**Decision 3: Word-level MCS replaces character-level MCS.** MCS expansion decomposes multi-character signatures into per-character entries. Under NLP-BPE, identifiers are opaque tokens — character decomposition is meaningless. Instead, NLP-BPE uses word-level MCS: multi-token resolved words get S4+S2 MCS entries; single-token words get S4 identity only (see §5.3).

**Decision 4: Operators are encoding-independent.** COUNTERSIGN, CANONIZE, CONNOTATE, UNDERSIGN produce the same entry structures regardless of encoding. The operator semantics depend on the AST structure, not the encoded values.

### 5.2 Operator Semantics Under NLP-BPE

All operators produce the same entry structures as @kscript §5.5. The compiled entries contain NLP-BPE node values instead of Mod32 packed values, but the structure is identical.

#### Comparison Table

| Operator | Mod32 Example | Mod32 Compiled | NLP-BPE Compiled |
| -------- | ------------- | -------------- | ---------------- |
| UNSIGNED | `A` | `{packed_A: None}` | `{nlp_A: None}` |
| COUNTERSIGN | `A == B` | `{packed_A: packed_B}, {packed_B: packed_A}` | `{nlp_A: nlp_B}, {nlp_B: nlp_A}` |
| CANONIZE | `A => B C` | `{packed_A: [packed_B, packed_C]}` | `{nlp_A: [nlp_B, nlp_C]}` |
| CONNOTATE | `A > B` | `{packed_A: packed_B}` | `{nlp_A: nlp_B}` |
| UNDERSIGN | `A = B` | `{packed_B: packed_A}` | `{nlp_B: nlp_A}` |

In every case, the entry structure is the same. Only the encoded values differ.

#### COUNTERSIGN (`==`)

```
A == B  →  {nlp_A: nlp_B}, {nlp_B: nlp_A}
```

No change in semantics. The bidirectional pair is emitted as in Mod32. If `B` is a signature (uppercase), word-level MCS is applied instead of character-level MCS (see §5.3).

#### CANONIZE (`=>`)

```
A => B C  →  {nlp_A: [nlp_B, nlp_C]}
```

No change in semantics. The owner is the last primary's node (or its signature), and the right-hand items form the node list.

#### CONNOTATE (`>`)

```
A > B  →  {nlp_A: nlp_B}
```

No change.

#### UNDERSIGN (`=`)

```
A = B  →  {nlp_B: nlp_A}    (reversed)
A = A  →  {nlp_A: None}     (self-identity collapsed)
```

No change.

### 5.3 Word-Level MCS

Character-level MCS expansion does not apply under NLP-BPE. Mod32 MCS decomposes multi-character signatures into per-character entries because characters are individually meaningful bit positions. Under NLP-BPE, identifiers are opaque BPE tokens — character decomposition is meaningless.

**Replacement: Word-level MCS with lazy emission.**

Word-level MCS operates at the BPE subword level. Resolved NLP words (identifiers that BPE-encode to one or more tokens) are handled as follows:

**Single-token words** (the common case for short identifiers):

```
Identifier "TEA" BPE-encodes to [tok_tea]  (single token)

Compiled:
  {nlp_tea: None}                         (unsigned identity, S4 — no decomposition)
```

No decomposition entries. The single token serves as both the identity and the signature node. This parallels the single-character Mod32 rule.

**Multi-token words** (identifiers that BPE-segment into multiple subword tokens):

```
Identifier "XYZZY" BPE-encodes to [tok_x, tok_yz, tok_zy]

Compiled:
  {nlp_x: None}                           (unsigned identity for first token, S4)
  {nlp_x: [nlp_x, nlp_yz, nlp_zy]}       (decomposition, S2)
```

Multi-token resolved words produce:
1. **S4 identity entry** for the first (signature) token.
2. **S2 decomposition entry** mapping the first token to all component tokens.

This mirrors the Mod32 MCS pattern but operates at the BPE subword level instead of the character level. No per-character unsigned entries are emitted for the sub-component tokens — only the first token's identity is recorded, plus the decomposition.

**Lazy emission.** Word-level MCS entries are emitted on first reference — when a word is first compiled in any construct. A dedup set tracks which word-level MCS entries have been emitted. Subsequent references to the same word produce no additional MCS entries. This prevents duplicate identity and decomposition entries when the same identifier appears multiple times in a script.

### 5.4 Signature Construction

Signature construction follows @signature: a plain OR-reduce of raw unmasked node values. NLP-BPE nodes contribute their full value (both NLP type bits in the high 32 and BPE token ID in the low 32):

```
make_signature([nlp_A, nlp_B]) → nlp_A | nlp_B
```

Both NLP type and BPE ID bits contribute to the signature. This makes signatures vocabulary-dependent — two synonyms with different BPE IDs produce different signatures.

**Implication for `signifies()`**: The `signifies()` function is unchanged — it remains `(a & b) != 0`. Candidate retrieval using `signifies()` now filters by overlap in any bit dimension (NLP type, BPE ID, or both). This produces a different (and for NLP use cases, richer) candidate set than pure NLP-type masking.

### 5.5 Lexical Structure

**The lexer is unchanged.** Token types, patterns, indentation rules, and operator classification are all identical to @kscript §2. The only difference is what the SIGNATURE tokens mean after encoding.

| Token | Mod32 Meaning | NLP-BPE Meaning |
| ----- | ------------- | ---------------- |
| `SIGNATURE` (`[A-Z]+`) | Bit-packed character composition | NLP-typed BPE token |

Identifiers remain uppercase-only `[A-Z]+`. Arbitrary strings are not valid identifiers — they must be quoted. This is a deliberate choice: the lexer defines the language surface; the encoder defines the compiled representation.

### 5.6 Decompilation

#### 5.6.1 Name Recovery

**Mod32**: MCS name recovery detects packed single-char nodes and reconstructs multi-character names by concatenating characters whose OR equals the signature.

**NLP-BPE**: MCS name recovery does not apply. Instead, names are recovered by decoding NLP-BPE tokens:

1. **Single-token identifiers**: `tokenizer.decode([node])` recovers the text directly.
2. **Multi-token identifiers**: A decomposition entry `{first_token: [all_tokens]}` provides all component tokens. Decoding each component and concatenating recovers the original text (with BPE subword boundary handling).

The decompiler's MCS name-building logic is replaced by a two-pass approach:

- **Pass 1**: Scan for decomposition entries (entries where `signature == first_token` and nodes is a multi-element list of NLP-BPE tokens). These provide the full token sequence for multi-token identifiers.
- **Pass 2**: For all other entries, decode the signature directly via `tokenizer.decode([sig])`.

No packed single-char detection is needed. The BPE vocabulary's `decode()` handles token-to-text mapping.

#### 5.6.2 Level Inference

Level inference uses the same bit-overlap heuristic as Mod32, but the overlap semantics differ:

| Condition | Mod32 Interpretation | NLP-BPE Interpretation |
| --------- | -------------------- | ---------------------- |
| `(sig & node) != 0` | Shared character bits → S2 (canonize) | Shared bits (NLP type or BPE ID) → S2 (canonize) |
| `(sig & node) == 0` | No shared characters → S3 (connotate) | No shared bits → S3 (connotate) |

The heuristic remains useful. A noun canonizing other nouns overlaps on shared NLP type bits. A noun connotating a verb with no shared bits would correctly infer S3. False positives occur when unrelated tokens happen to share bits in either dimension.

**S1/S3 ambiguity** remains as in Mod32: countersigned pairs with no shared bits are indistinguishable from connotate pairs.

---

## 6. Compilation Semantics — Complete Reference

### 6.1 Entry Model

Identical to @kscript §5.1. Each compiled entry is a KLine with a signature (uint64), nodes (uint64, list[uint64], or None), and debug metadata.

### 6.2 Encoding

Strings are encoded via the NLP tokenizer:

- **Signatures** (uppercase alpha): NLP-BPE encoding — one or more nodes of the form `(nlp_type32 << 32) | bpe_token_id`. If a single token, used directly as the signature node. If multiple tokens, first token is the signature, and a decomposition entry is emitted (see §5.3).

### 6.3 Construct Compilation Rules

All rules from @kscript §5.5 apply. Operator emission is identical. Only encoding differs.

#### Unsigned (bare signature)

```
A       → {nlp_A: None}     (plus word-level MCS if multi-token)
```

#### COUNTERSIGN (`==`)

```
A == B  → {nlp_A: nlp_B}, {nlp_B: nlp_A}   (bidirectional)
```

No BPE decomposition for the node side — only the signature side triggers word-level MCS.

#### UNDERSIGN (`=`)

```
A = B   → {nlp_B: nlp_A}           (unidirectional, reversed)
A = A   → {nlp_A: None}            (self-identity collapsed)
```

#### CONNOTATE (`>`)

```
A > B   → {nlp_A: nlp_B}           (unidirectional)
```

#### CANONIZE chain (`=>`)

```
A => B C       → {nlp_A: [nlp_B, nlp_C]}              (single entry, all items)
A > X => B C   → {nlp_A: nlp_X}, {nlp_X: [nlp_B, nlp_C]}
A = X => B C   → {nlp_X: nlp_A}, {nlp_X: [nlp_B, nlp_C]}
```

### 6.4 Subscript Blocks

Identical to @kscript §5.6. Block flattening, item extraction, and recursive compilation are encoding-independent.

### 6.5 Significance Level Assignment

Identical to @kscript §5.7. Levels are assigned by operator, not by encoding.

### 6.6 Deduplication

Identical to @kscript §5.4. Dedup is on `(sig_id, nodes_tuple)` encoded values. Different encodings produce different dedup keys — correct behavior.

---

## 7. Backward Compatibility

Mod32 KScript is **unchanged**. NLP-BPE KScript is a separate encoding mode selected at compile time.

| Aspect | Mod32 Mode | NLP-BPE Mode |
| ------ | ---------- | ------------ |
| Source language | Identical | Identical |
| Lexer | Identical | Identical |
| Parser | Identical | Identical |
| AST | Identical | Identical |
| Token encoder | Mod32Tokenizer | NLPTokenizer |
| MCS expansion | Per-character decomposition | Word-level BPE decomposition |
| Signature bits | Character positions | NLP type + BPE ID (full unmasked) |
| Decompiler name recovery | MCS packed-char detection | BPE token decoding |

A `.ks` file is encoding-agnostic. The same file compiles under either mode without modification.

---

## 8. Worked Examples

### 8.1 Minimal Unsigned

```
A
```

**Assumptions**: `A` BPE-encodes to a single token (common for single characters).

```
BPE encode("A") → [token 36]     (example token ID)
Grammar lookup → nlp_type32 = 131072  (POS_NOUN)
NLP node: (131072 << 32) | 36 = 562949953421348

Compiled:
  {562949953421348: None}         (unsigned, S4)
```

Decompiled: `A` (via `tokenizer.decode([36])` → `"A"`)

### 8.2 Bidirectional Link

```
TEA == BREW
```

**Assumptions**:
- `TEA` → BPE token 12465, nlp_type32 = 131200 (POS_PROPN | DEP_SUBJ | MORPH_SING)
- `BREW` → BPE token 4964, nlp_type32 = 8421376 (POS_VERB | DEP_ROOT | MORPH_PAST | MORPH_SING)

```
nlp_TEA = (131200 << 32) | 12465 = 563499709247665
nlp_BREW = (8421376 << 32) | 4964 = 36169534507324260

Compiled:
  {563499709247665: 36169534507324260}      (countersign, S1)
  {36169534507324260: 563499709247665}      (countersign reverse, S1)

Signatures:
  make_signature([nlp_TEA]) = 563499709247665       (full value, unmasked)
  make_signature([nlp_BREW]) = 36169534507324260     (full value, unmasked)
  signifies(sig_TEA, sig_BREW) → depends on bit overlap in full values
```

Decompiled:
```
TEA == BREW
BREW == TEA
```

### 8.3 Canonization with NLP Types

```
SVO => S V O
```

**Assumptions**:
- `SVO` → single BPE token, nlp_type32 = 131200 (POS_PROPN)
- `S` → single BPE token, nlp_type32 = 65536 (POS_X, unknown)
- `V` → single BPE token, nlp_type32 = 65536 (POS_X)
- `O` → single BPE token, nlp_type32 = 65536 (POS_X)

```
Compiled:
  {nlp_SVO: [nlp_S, nlp_V, nlp_O]}    (canonize, S2)
  {nlp_S: None}                         (unsigned S)
  {nlp_V: None}                         (unsigned V)
  {nlp_O: None}                         (unsigned O)

Level inference for {nlp_SVO: [nlp_S, nlp_V, nlp_O]}:
  nodes_sig = make_signature([nlp_S, nlp_V, nlp_O]) = OR of their full values
  (nlp_SVO.signature & nodes_sig) != 0  → depends on shared bits
```

### 8.4 Complex Example — Parallel to @kscript §9.3

```
MHALL == SVO =>
  S = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O
```

**Key difference from Mod32**: No per-character MCS expansion for `MHALL`. `MHALL` is a single NLP-BPE token (or a word-level MCS decomposed multi-token). Under single-token BPE:

```
Compiled:
  {nlp_MHALL: nlp_SVO}                   (countersign, S1)
  {nlp_SVO: nlp_MHALL}                   (countersign reverse, S1)
  {nlp_M: nlp_S}                         (undersign, S1)
  {nlp_H: nlp_V}                         (undersign, S1)
  {nlp_ALL: nlp_O}                       (undersign, S1)
  {nlp_D: nlp_A}                         (undersign, S1)
  {nlp_M: nlp_L}                         (undersign, S1 — note: deduplicated if same as earlier {nlp_M: ...})
  {nlp_L: nlp_O}                         (connotate, S3)
  {nlp_S: None}                          (unsigned S)
  {nlp_V: None}                          (unsigned V)
  {nlp_O: None}                          (unsigned O)
  {nlp_M: None}                          (unsigned M)
  {nlp_H: None}                          (unsigned H)
  {nlp_A: None}                          (unsigned A)
  {nlp_L: None}                          (unsigned L)
  {nlp_D: None}                          (unsigned D)
  {nlp_MHALL: None}                      (unsigned MHALL)
  {nlp_SVO: None}                        (unsigned SVO)
  {nlp_ALL: None}                        (unsigned ALL)
```

If `ALL` BPE-encodes to multiple tokens `[tok_A, tok_LL]`:

```
Additional entries (word-level MCS, lazy emission):
  {nlp_A_tok: None}                      (unsigned identity for first token, S4)
  {nlp_A_tok: [nlp_A_tok, nlp_LL_tok]}  (decomposition, S2)
```

### 8.5 Canonization with Multiple Items

```
A =>
  B
  C = D
```

```
Compiled:
  {nlp_A: [nlp_B, nlp_C]}              (canonize, single entry with all items, S2)
  {nlp_D: nlp_C}                       (undersign, S1 — from recursive compilation of C = D)
  {nlp_B: None}                         (unsigned B, from recursive compilation)
  {nlp_C: None}                         (unsigned C, from recursive compilation)
  {nlp_D: None}                         (unsigned D, from recursive compilation)
```

---

## 9. Test Matrix

| ID | Criterion | Category |
| -- | --------- | -------- |
| KSN-1 | NLP-BPE encode produces correct node format: `(nlp_type32 << 32) \| bpe_token_id` for each identifier | Encoding |
| KSN-2 | Single-token identifiers produce single-node signatures with no decomposition | Encoding |
| KSN-3 | Multi-token identifiers produce word-level MCS: S4 identity + S2 decomposition entry | Encoding |
| KSN-6 | COUNTERSIGN under NLP-BPE: `A == B` → `{nlp_A: nlp_B}, {nlp_B: nlp_A}` | Operators |
| KSN-7 | CANONIZE under NLP-BPE: `A => B C` → `{nlp_A: [nlp_B, nlp_C]}` | Operators |
| KSN-8 | CONNOTATE under NLP-BPE: `A > B` → `{nlp_A: nlp_B}` | Operators |
| KSN-9 | UNDERSIGN under NLP-BPE: `A = B` → `{nlp_B: nlp_A}` | Operators |
| KSN-10 | Self-identity under NLP-BPE: `A = A` → `{nlp_A: None}` | Operators |
| KSN-11 | No per-character MCS expansion for NLP-BPE identifiers | MCS |
| KSN-12 | Word-level MCS for multi-token identifiers: S4 identity + S2 decomposition entry | MCS |
| KSN-15 | Deduplication works on NLP-BPE encoded values: same `(sig_id, nodes_tuple)` dedup as Mod32 | Dedup |
| KSN-16 | Singleton unwrapping works for NLP-BPE nodes: single-element node list → single node | Structure |
| KSN-17 | Decompiler name recovery: single-token NLP identifiers decoded via `tokenizer.decode()` | Decompilation |
| KSN-18 | Decompiler name recovery: multi-token identifiers recovered via decomposition entries | Decompilation |
| KSN-19 | Level inference: `(sig & node) != 0` tests full bit overlap under NLP-BPE | Decompilation |
| KSN-20 | Lexer produces identical token stream for same source regardless of encoding mode | Lexer |
| KSN-21 | Parser produces identical AST for same source regardless of encoding mode | Parser |
| KSN-22 | AST emitter produces identical symbolic entries regardless of encoding mode | AST |
| KSN-23 | Same `.ks` source compiles under both Mod32 and NLP-BPE without modification | Compatibility |
| KSN-24 | Mod32 compilation is unchanged when NLP-BPE mode exists in the codebase | Compatibility |
| KSN-25 | Complex example (§8.4) produces correct entry structure with all operators | Integration |
| KSN-27 | CANONIZE with subscript block flattens correctly under NLP-BPE | Structure |
| KSN-28 | Chained constructs (`A => B => C`) work identically under NLP-BPE | Structure |
| KSN-29 | Word-level MCS lazy emission: first reference emits MCS entries, subsequent references skip | MCS |
| KSN-30 | `signifies()` tests bit overlap in full unmasked node values (NLP type + BPE ID) | Signatures |

---

## 10. Out of Scope

The following are explicitly **not covered** by this specification:

- **Migration of existing `.ks` files.** No conversion needed — the same source compiles under both modes.
- **Mixed Mod32/NLP compilation.** A single compilation uses one tokenizer. Mixed-mode compilation (some entries Mod32, others NLP-BPE) is not supported.
- **NLP grammar training.** How the BPE vocabulary and grammar dictionary are trained is defined in @tokenizer.
- **Lexer changes for arbitrary identifiers.** Allowing lowercase or mixed-case identifiers is a language-level change that would require a new spec. This spec keeps `[A-Z]+` as the only identifier syntax.
- **Signature format changes.** The 64-bit signature format and `make_signature()` algorithm are defined in @signature. This spec does not modify them.
- **Significance level encoding.** Significance bits are not encoded into token IDs during compilation (as in Mod32). This spec does not change that.
- **Implementation details.** File paths, class names, and function signatures belong in the implementation plan, not this spec.

---

## 11. Design Rationale

### 11.1 Why Keep Uppercase-Only Identifiers?

Changing the lexer to accept arbitrary strings (lowercase, mixed-case) would be a language-level change, not an encoding change. The lexer defines what KScript source looks like; the encoder defines what the compiled output means. Keeping the surface language unchanged ensures:

1. Existing `.ks` files compile under NLP-BPE without modification.
2. The parser and AST emitter need no changes.
3. The design is purely an encoding concern, which is the correct scope.

If arbitrary identifiers are desired in the future, that should be a separate spec that defines new token types and lexer rules, applicable to both encoding modes.

### 11.2 Why Word-Level MCS Instead of No Expansion?

Without any decomposition, multi-token identifiers would lose information — the relationship between the full identifier and its BPE subword components would not be recorded in the knowledge graph. Word-level MCS is the natural analogue of character-level MCS: where Mod32 MCS decomposes characters, word-level MCS decomposes BPE subword tokens.

For single-token identifiers (the common case), word-level MCS produces no extra entries — the identifier is simply an unsigned identity, just like a single-character Mod32 signature.

Lazy emission prevents duplicate entries when the same identifier appears multiple times.

### 11.3 Why Unmasked Signatures

Using full unmasked node values in signatures means BPE token IDs contribute to the signature. This makes signatures vocabulary-dependent — two synonyms with different BPE IDs produce different signatures. However:

1. The tokenizer's BPE vocabulary is stable within a session.
2. `signifies()` still works for candidate retrieval — it tests overlap in all bits, including NLP type.
3. Masking was a simplification that loses BPE identity. Full values preserve all information.

### 11.4 Why Bitwise AND Still Works

The `signifies(a, b) → (a & b) != 0` function is encoding-agnostic. Under Mod32, shared bits mean shared characters. Under NLP-BPE, shared bits may come from either NLP type or BPE ID overlap. The mechanism is the same; the semantics differ. This is acceptable because:

1. NLP-BPE is chosen specifically for linguistically-informed matching.
2. The candidate set filtered by bit overlap is different from (and for NLP use cases, richer than) the character-overlap filter.
3. Significance computation downstream still handles the final relevance determination.

### 11.5 Why the Decompiler Needed Changes

Mod32 decompiler MCS name recovery relies on detecting packed single-char nodes. Under NLP-BPE:

1. Nodes are never single characters in the Mod32 sense — they're BPE tokens.
2. The packed single-char test would fail for all NLP-BPE nodes.
3. A different recovery mechanism (direct token decoding + decomposition entry scanning) is needed.

The replacement is simpler: NLP-BPE tokens can be decoded directly via the BPE vocabulary. No bit-level character detection needed.

### 11.6 Why Level Inference Remains Heuristic

Under NLP-BPE, `(sig & node) != 0` tests full bit overlap. This is a different but equally valid heuristic:

- **True positive**: A noun canonizing other nouns overlaps on shared NLP type bits → correctly S2.
- **False positive**: A noun connotating a verb that happens to share a BPE ID → incorrectly S2.
- **True negative**: A noun connotating a verb with no shared bits → correctly S3.

The false-positive rate is comparable to Mod32 (where unrelated characters sharing bit positions produce similar errors). Significance bits in the compiled entries would eliminate this heuristic, but that's a cross-cutting change outside this spec's scope.
