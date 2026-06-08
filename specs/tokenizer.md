# Tokenizer Specification

## Overview

The tokenizer converts between text and nodes. It is the sole authority
for how text becomes nodes.

Three tokenizer types are defined, all conforming to the same interface:

- **BPE** — byte-pair encoding. Vocabulary learned from a training corpus.
  Tokens are sequential vocabulary indices, combined with type prefixes to
  form typed nodes.
- **NLP** — hybrid BPE + NLP type encoding. BPE subword tokens are combined
  with NLP type information (POS + DEP + MORPH) in a single 64-bit node.
  The tokenizer owns the grammar dictionary that maps BPE tokens to NLP types.
- **Mod** — modular bit-packed encoding. Vocabulary is a fixed character set.
  Tokens are bit positions with bitwise OR/AND semantics.

All types ultimately produce the same kind of output: typed nodes
suitable for signature construction (defined in the @signature spec).

## Dependencies

### Kline (@kline spec)

- **node**: a 64-bit unsigned integer. The tokenizer produces nodes.

The tokenizer does not interpret nodes beyond its own encoding.

### Signature (@signature spec)

- Signature creation (`make_signature`) depends on the standalone
  `is_literal` function to determine which nodes contribute.
- The tokenizer does not create signatures itself. See the @signature spec.

## Interface

All tokenizer types implement:

### `vocab_size → int ≥ 0`

The number of distinct tokens the vocabulary defines.

### `encode(text: str) → list[node]`

Convert a string to an ordered sequence of nodes.

- Empty string → empty list.
- All-uppercase-alpha strings → **packed** encoding (single node, bit 0 clear).
- All other strings → **literal** encoding (one node per character, literal mask).
- The encoding mode is determined automatically from the input content.
- Each node carries enough information to reconstruct the original text
  via `decode`.

### `decode(nodes: list[node]) → str`

Convert a sequence of nodes back to a string.

- Empty list → empty string.
- `decode(encode(text)) == text` for any string the tokenizer can represent.
- Auto-detects literal vs packed from the literal mask on each node.

### `is_literal(node: int) → bool` *(deprecated — moved to node encoding)*

> **Note:** `is_literal` is now a standalone function defined by the node
> encoding layer, not the tokenizer. Tokenizers no longer implement this
> method. It is documented here for historical reference only. See the
> @kline spec for the current definition.

Returns whether the node represents a literal token.

- Literal tokens preserve character identity and order.
- Non-literal (typed) tokens carry structural or type information and
  participate in bitwise OR signature construction.

## Vocabulary

A vocabulary is the ordered set of symbols the tokenizer can encode.

All three tokenizer types provide a default vocabulary. The default may be
overridden at initialisation.

| Type | Default                                               |
| ---- | ----------------------------------------------------- |
| BPE  | 4096 entries (learned from corpus)                    |
| NLP  | 17,392 BPE entries + 12,871 grammar dictionary entries |
| Mod  | 95 printable ASCII characters (codes 32–126), ordered |

## BPE Tokenizer

### Overview

A BPE tokenizer learns subword token vocabulary from a training corpus.
Tokens are sequential integer IDs assigned during training.

BPE is suitable for general-purpose text encoding where subword
segmentation and large vocabularies are needed.

### Vocabulary

- Learned from a text corpus via training.
- Default target size: 4096.
- Configurable at training time.

### Training

```
tokenizer.train(texts: Iterator[str], vocab_size: int, pattern: str | None) → void
```

- `texts` — iterator over training strings.
- `vocab_size` — target vocabulary size (minimum 256).
- `pattern` — optional regex pattern for pre-tokenization.

Training is required before `encode`/`decode` can be used.

### Persistence

```
tokenizer.save_to_directory(path, name) → void
tokenizer.from_directory(path, name) → Tokenizer
```

Save and load trained tokenizer state. Format is implementation-defined.

### Type Prefixes

BPE token IDs are sequential integers without inherent type information.
To produce typed nodes suitable for signature construction, each BPE token
is combined with a **type prefix** — a bit pattern encoding linguistic
properties (part-of-speech, dependency, morphology, etc.).

Type prefixes are external to the tokenizer. They are provided by the
agent layer via a dictionary lookup at encode time:

```
for token in bpe_tokens:
    type_prefix = dictionary.lookup(token)   # linguistic type bits
    node = type_prefix | token               # typed node
```

Unrecognised tokens receive a default type prefix (e.g. `POS_X`).
Whitespace tokens may pass through as-is (no type prefix).

### Encoding

BPE segments text into subword units, each mapped to a vocabulary index:

```
encode("hello") → [15496]
encode("hello world") → [15496, 995]
```

The returned values are raw BPE token IDs. Type prefix combination
happens at the agent layer.

### Literal Test

BPE tokens are never literal. The standalone `is_literal` function
(defined in the @kline spec) returns `False` for all BPE nodes because
their lower 32 bits are never all set (`0xFFFFFFFF`).

### Worked Examples

#### Encoding with type prefixes

```
BPE encode("the air") → [257, 500]

Agent applies type prefixes from dictionary:
  257 "the" → POS_DET | 257 = 4194337
  500 "air" → POS_NOUN | DEP_OBJ | 500 = 262788

Typed nodes: [4194337, 262788]
```

#### Signature construction

```
nodes = [4194337, 262788]

make_signature(nodes) → 4194337 | 262788 = 4456397

Signature captures: token IDs 257 and 500 are present,
with POS_DET and POS_NOUN | DEP_OBJ type information.
```

> **Note.** `make_signature` is defined in the @signature spec, not here.
> This example illustrates how typed BPE nodes feed into it.

## NLP Tokenizer

### Overview

An NLP tokenizer is a hybrid that combines BPE subword encoding with NLP
type information (POS + DEP + MORPH) in a single 64-bit node. Every word
becomes one or more NLP-BPE tokens. There is no packed mode (unlike Mod).

If a word BPE-encodes into multiple subword tokens (e.g. "unhappiness" →
`[un, ##happiness]`), each subword token gets its own grammar lookup.
Subwords without dictionary entries fall back to `POS_X`.

NLP is suitable for text encoding where linguistic type information must
be embedded directly in each node, enabling signature-based similarity
matching that reflects grammatical structure.

### Node Format

Each NLP-BPE node is a 64-bit unsigned integer:

```
node = (nlp_type32 << 32) | bpe_token_id
```

Bit layout:

```
┌──────────────────────────────────────────────────────────────────┐
│ High 32 bits (32–63) │ nlp_type32 — NLP type encoding            │
│                      │   Bits 0–16:  17 POS tags                  │
│                      │   Bits 17–24: 8 DEP groups                 │
│                      │   Bits 25–31: 7 MORPH features             │
│ Low 32 bits (0–31)  │ BPE token ID (0–17391)                     │
└──────────────────────────────────────────────────────────────────┘
```

The `nlp_type32` value is a 32-bit bitmask encoding 32 linguistic
dimensions (17 POS + 8 DEP + 7 MORPH = 32 flags), sourced from the
32-bit NLP type legend. The BPE token ID occupies the low 32 bits.

### Literal Node Semantics

NLP-BPE nodes never trigger `is_literal()` because their low 32 bits
are BPE IDs (0–17391), never `0xFFFFFFFF`. The standalone `is_literal`
function (defined in the @kline spec) always returns `False` for NLP
nodes.

Character-level fallback for unknown or rare words still uses the
existing `(codepoint << 32) | 0xFFFFFFFF` literal pattern defined in
the Mod spec.

### Fallback for Unknown BPE Tokens

BPE tokens without a grammar dictionary entry receive `POS_X = 65536
= 1 << 16` as their `nlp_type32`. This ensures unknown words still have
a valid NLP type, preserving the node format invariant.

### Vocabulary

- **BPE vocabulary**: 17,392 tokens (from `tokenizer-32768.json`, where
  32,768 was the training target; actual trained vocab is 17,392).
- **Grammar dictionary**: 12,871 BPE→NLP entries mapping BPE token IDs
  to `nlp_type32` values (from `simplestories-1_grammar.json`).

Source files:

| File | Purpose |
| ---- | ------- |
| `data/tokenizer/tokenizer-32768.json` | BPE vocabulary |
| `data/tokenizer/simplestories-1_grammar.json` | Grammar dictionary (BPE→NLP) |
| `data/tokenizer/simplestories-1_nlp_type32.json` | 32-bit NLP type legend |

### Encoding

The encode process:

1. **BPE-encode** the text into subword token IDs.
2. **Grammar lookup** — for each token ID, look up the `nlp_type32` in
   the grammar dictionary. Tokens without entries default to `POS_X`.
3. **Construct nodes** — assemble each node as
   `(nlp_type32 << 32) | bpe_token_id`.

#### Worked Example

```
encode("Tea brewed softly")

Step 1 — BPE encode:
  "Tea"     → token 12465
  "brewed"  → token 4964
  "softly"  → token 977

Step 2 — Grammar lookup (nlp_type32 from dictionary):
  12465 "Tea"     → nlp_type32 = 131200
  4964  "brewed"  → nlp_type32 = 8421376
  977   "softly"  → nlp_type32 = 2097156

Step 3 — Node construction:
  "Tea"     → (131200  << 32) | 12465 = 563499709247665
  "brewed"  → (8421376 << 32) | 4964  = 36169534507324260
  "softly"  → (2097156 << 32) | 977   = 9007216434611153

Result: [563499709247665, 36169534507324260, 9007216434611153]
```

### Decoding

Extract the BPE token ID from the low 32 bits and decode via the BPE
vocabulary:

```
bpe_token_id = node & 0xFFFFFFFF
text = bpe_vocab.decode(bpe_token_id)
```

NLP type bits (high 32) are not needed for text reconstruction.

### Dimension Count

The `nlp_type32` encoding provides **32 dimensions**:

- 17 POS tags (bits 0–16)
- 8 DEP groups (bits 17–24)
- 7 MORPH features (bits 25–31)

Comparison with other tokenizers:

| Tokenizer | Dimensions | Notes |
| --------- | ---------- | ----- |
| Mod32     | 31         | Bits 1–31 |
| NLP       | 32         | 17 POS + 8 DEP + 7 MORPH |
| Target    | ~35        | From CONTEXT.md |

### Signature Behavior

`make_signature()` for NLP nodes OR-reduces only the NLP type bits
(high 32), excluding BPE IDs. The masking is handled internally by
`make_signature()` — it detects NLP-BPE nodes via `is_nlp_node()` and
applies `NLP_TYPE_MASK` (`0xFFFFFFFF00000000`) before OR-reducing.
The tokenizer passes raw (unmasked) NLP-BPE nodes to `make_signature()`
as it would any other node sequence.

**Rationale**: BPE IDs are vocabulary-specific indices without semantic
meaning — two synonyms have unrelated BPE IDs but may share NLP type
bits. Only the NLP type bits carry structural/linguistic information
relevant to similarity matching.


### Worked Examples

#### Encoding with grammar lookup

```
NLP encode("Tea brewed softly")

BPE: [12465, 4964, 977]

Grammar dictionary lookups:
  12465 → nlp_type32 = 131200   (POS_PROPN | DEP_SUBJ | MORPH_SING)
  4964  → nlp_type32 = 8421376  (POS_VERB  | DEP_ROOT | MORPH_PAST | MORPH_SING)
  977   → nlp_type32 = 2097156  (POS_ADV   | DEP_ADVMOD)

Nodes: [563499709247665, 36169534507324260, 9007216434611153]
```

#### Round-trip

```
encode("Tea brewed softly") → [563499709247665, 36169534507324260, 9007216434611153]
decode([563499709247665, 36169534507324260, 9007216434611153]) → "Tea brewed softly"
```

#### Signature construction

```
Nodes: [563499709247665, 36169534507324260, 9007216434611153]

make_signature(nodes) →
  Each NLP node is masked internally by make_signature:
    563499709247665  & 0xFFFFFFFF00000000  = 563499709235200
    36169534507324260 & 0xFFFFFFFF00000000 = 36169534507319296
    9007216434611153 & 0xFFFFFFFF00000000  = 9007216434610176
  OR-reduce: 563499709235200 | 36169534507319296 | 9007216434610176

Signature captures: POS_PROPN, POS_VERB, POS_ADV, DEP_SUBJ, DEP_ROOT,
DEP_ADVMOD, MORPH_PAST, MORPH_SING — the grammatical profile of the
input, independent of which specific words were used.
```

> **Note.** `make_signature` is defined in the @signature spec, not here.
> The NLP-aware masking (detecting NLP-BPE nodes and applying `NLP_TYPE_MASK`)
> is built into `make_signature` itself — the tokenizer passes raw nodes.

## Mod Tokenizer

### Overview

A Mod tokenizer maps characters to bit positions within a node. Strings
are encoded as the bitwise OR of constituent character bits, producing a
single node. Individual characters can also be encoded as literal tokens
that preserve order and identity.

Two encoding modes produce two token categories:

| Mode    | Tokens   | Bit 0 | Order preserved | Use                   |
| ------- | -------- | ----- | --------------- | --------------------- |
| Packed  | Single   | 0     | No              | Signatures, AND match |
| Literal | Per-char | 1     | Yes             | Exact text, sequence  |

### Bit Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Bit 0       │ LITERAL flag: 0 = packed, 1 = literal              │
│ Bits 1–N    │ Character bits (N determined by variant)           │
│ Bits N+1–63 │ Unused                                             │
└──────────────────────────────────────────────────────────────────┘
```

### Variants

| Variant | Character bits | Bit range | Fits uint64 |
| ------- | -------------- | --------- | ----------- |
| Mod32   | 31             | Bits 1–31 | Yes         |
| Mod64   | 63             | Bits 1–63 | Yes         |

### Vocabulary

The vocabulary is an ordered string of characters. Each character maps to
a single bit position (bit 1, bit 2, …, bit N). When the character count
exceeds N, positions wrap.

Default vocabulary:

```
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_=
```

All printable ASCII characters (codes 32–126) must be mappable.
Characters not in the explicit vocabulary are assigned the next available
bit position (wrapping).

### Encoding

The encoding mode is determined automatically from the input text:

- All-uppercase-alpha strings → **packed** (single node, bit 0 clear)
- All other strings → **literal** (one node per character, literal mask set)

No `pack` parameter is needed — the tokenizer works it out internally.

```
encode("ABC")     → [CHAR_BIT['A'] | CHAR_BIT['B'] | CHAR_BIT['C']]   # packed
encode("hello")   → [(104<<32)|0xFFFFFFFF, (101<<32)|0xFFFFFFFF, ...]   # literal
encode("123")     → [(49<<32)|0xFFFFFFFF, (50<<32)|0xFFFFFFFF, ...]    # literal
encode('"hi"')    → [(34<<32)|0xFFFFFFFF, ...]                          # literal
```

Multi-character strings are OR-ed into a single node. Bit 0 is clear.

```
encode("ABC") → [CHAR_BIT['A'] | CHAR_BIT['B'] | CHAR_BIT['C']]
```

#### Packed properties

Applies to all-uppercase-alpha strings automatically:
- Order is lost: `"AB"` and `"BA"` produce the same node.
- Multiplicity is lost: `"AA"` and `"A"` produce the same node.
- Suitable for signature construction and bitwise AND matching.

#### Literal properties

Applies to all non-uppercase-alpha strings automatically:
- Order is preserved.
- Identity is preserved (distinct characters → distinct tokens).
- Bypasses the vocabulary: any character with a valid code point can be
  encoded, including characters not in the vocabulary.
- The literal mask (`0xFFFFFFFF`) occupies the lower 32 bits, keeping the
  character code point in the upper 32 bits. This avoids collision with
  packed nodes (which use bits 1–31 or 1–63) and with signatures (where
  bit 0 is the literal-content flag, not a full 32-bit mask).

### Decoding

Auto-detects encoding mode from the literal mask:

```
if (node & 0xFFFFFFFF) == 0xFFFFFFFF  →  literal: chr(node >> 32)
else                                   →  packed: find all set bits, map each to character
```

Packed decode returns characters in bit-position order (lowest bit first),
not in the original text order.

### Literal Test

The standalone `is_literal` function (defined in the @kline spec) returns
`True` for literal Mod nodes because the lower 32 bits are the literal mask
`0xFFFFFFFF`. Packed Mod nodes have bit 0 clear, so `is_literal` returns
`False` for them.

### Worked Examples

#### Packed encoding (Mod32)

```
Vocabulary: "ABC" → bit 1, bit 2, bit 3

encode("A")       → [0b10]                  = [2]
encode("B")       → [0b100]                 = [4]
encode("AB")      → [0b10 | 0b100]          = [6]
encode("ABC")     → [0b10 | 0b100 | 0b1000] = [14]
```

#### Literal encoding (Mod32)

```
encode("12") → [(49 << 32) | 0xFFFFFFFF, (50 << 32) | 0xFFFFFFFF]
               → [4294967345, 4294967346]

decode([4294967345, 4294967346]) → "12"

is_literal(4294967345) → True
is_literal(2)          → False
```

#### Bitwise properties of packed nodes

Packed (non-literal) nodes can be combined with bitwise OR and tested
with bitwise AND. These properties are used by signature construction
(@signature spec) and candidate retrieval (@model spec):

```
2 | 4 = 6              # 'A' | 'B' = combined
(6 & 2) != 0 → True   # combined contains 'A'
```

## Test Matrix

| ID    | Criterion                                                                  | Origin ref |
| ----- | -------------------------------------------------------------------------- | ---------- |
| TOK-1 | Packed encode single char: `encode("A") == [bit_A]`                          | — |
| TOK-2 | Packed encode multi-char: `encode("AB") == [bit_A \| bit_B]`                  | — |
| TOK-3 | Packed round-trip: `decode(encode("ABC"))` contains A, B, C (order may differ) | — |
| TOK-4 | Literal encode: `encode("123")` produces nodes with literal mask            | — |
| TOK-5 | Literal round-trip: `decode(encode("123")) == "123"` (order preserved)      | — |
| TOK-6 | Auto-detection: `encode("A")` → packed; `encode("1")` → literal            | — |
| TOK-7 | Empty string: `encode("") == []`, `decode([]) == ""`                        | — |
| TOK-8 | `is_literal` on literal node: `is_literal((65<<32)\|0xFFFFFFFF) == True`    | — |
| TOK-9 | `is_literal` on packed node: `is_literal(6) == False`                      | — |
| TOK-10 | `is_literal` on zero: `is_literal(0) == False`                             | — |
| TOK-11 | Vocab size matches number of unique characters in alphabet                  | — |
| TOK-12 | Characters not in vocab are still encodable (assigned next bit)            | — |
| TOK-NLP-1 | NLP encode produces correct node format: `(nlp_type32 << 32) \| bpe_token_id` for each token | NLP |
| TOK-NLP-2 | NLP encode with unknown BPE token uses `POS_X = 65536` as `nlp_type32`    | NLP |
| TOK-NLP-3 | NLP nodes are never literal: `is_literal((nlp_type32 << 32) \| bpe_id) == False` for any valid `bpe_id` ≤ 17391 | NLP |
| TOK-NLP-4 | NLP round-trip: `decode(encode("Tea brewed softly")) == "Tea brewed softly"` | NLP |
| TOK-NLP-5 | NLP signature masking: signature OR-reduces only high 32 bits (NLP type), excluding BPE IDs | NLP |
| TOK-NLP-6 | NLP literal fallback: unknown/rare words produce `(codepoint << 32) \| 0xFFFFFFFF` literal nodes | NLP |
| TOK-NLP-7 | Vocabulary sizes: BPE vocab = 17,392 tokens, grammar dictionary = 12,871 entries | NLP |
| TOK-NLP-8 | Dimension count: `nlp_type32` provides 32 dimensions (17 POS + 8 DEP + 7 MORPH) | NLP |

## What a Tokenizer is Not

The following are explicitly **out of scope** for this spec:

- **Signature creation.** Signature construction (`make_signature`) is
  defined in the @signature spec. The `is_literal` function is defined
  in the @kline spec, not by the tokenizer.
- **Significance computation.** Significance is defined in the
  @significance spec. The tokenizer does not compute or store significance.
- **Type prefix assignment (BPE).** How linguistic types are assigned to
  BPE tokens is an agent concern. The BPE tokenizer provides raw token IDs.
  **Note:** For the NLP tokenizer, grammar lookups are intrinsic — the
  tokenizer owns the dictionary. This distinction is by design: BPE type
  prefixes are external (agent layer), while NLP grammar lookups are
  internal (tokenizer layer).
- **Model operations.** Storing, retrieving, and querying klines are model
  concerns.
- **Training data format.** How training data is sourced and preprocessed
  for BPE training is an implementation concern.

## Referenced By

- **Signature** (@signature spec) — signature creation depends on the
  standalone `is_literal` function (defined in the @kline spec).
- **Kline** (@kline spec) — klines contain nodes produced by the tokenizer.
  Defines `is_literal` as a standalone function.
- **Agent** (@agent spec) — uses the tokenizer to encode input, apply type
  prefixes, and construct klines.
- **Significance** (@significance spec) — consumes nodes produced by the
  tokenizer (nodes are opaque to significance).
