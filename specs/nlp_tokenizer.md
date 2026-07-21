# NLP Tokenizer Specification

## Overview

This spec defines the **NLP production tokenizer** — the sole concrete
`KTokenizer` (see @tokenizer) used by the shipped data. It combines a
**BPE** subword base with an **NLP type word** to form 64-bit typed nodes:

```
node = (sig_word << 32) | bpe_token_id
```

`sig_word` is a single-bit type selector occupying the upper 32 bits; the
BPE token ID is the lower 32 bits. Each BPE token's `sig_word` is set by
`dev/nlp/tag_vocab.py` from a frequency-ranked fine-type legend (see
§NLP Sig-Word Layout) — exactly one bit, the residue of the rarest matched
NLP type.

Everything the interface does not specify — the node layout, the type
dictionary, encoding, decoding, and the BPE-engine foundation — **lives
here**. The base `KTokenizer` interface (see @tokenizer) is layout- and
type-agnostic; a base `Tokenizer` class wraps the BPE engine and is not
itself a `KTokenizer`. This specialisation is the production tokenizer.

The signature algebra over these nodes — reduction (`make_signature`) and
overlap matching (`signifies`) — is owned by the @signifier spec. The NLP
deployment bundles this tokenizer with `NLPSignifier`, the production
concrete Signifier, as two sibling NLP specialisations: the tokenizer
fixes the type-word _values_, the signifier owns the _bit algebra_ over
them.

## NLP Sig-Word Layout

The value packed into a node's upper 32 bits is the entry's **`sig_word`** —
a single-bit type selector produced by `dev/nlp/tag_vocab.py`. From a token's
grammar entry (`pos`, `pos_fine`, `dep`, `morph`), `tag_vocab.py` matches
each field against a frequency-ranked fine-type legend
(`*_nlp_fine_types.json`), selects the rarest matched type (the most
informative), and emits `1 << (position % 32)` — exactly one bit set. When
nothing matches, it falls back to the legend's lowest-value (most frequent)
bit. The scheme is intentionally sparse: two tokens share a `sig_word` bit
only when they reduce to the same rarest type, which tightens the
@signifier's `signifies` overlap test.

The grammar also retains **`nlp_type32`** — a legacy multi-bit encoding of
32 linguistic dimensions — as metadata (not packed into nodes):

```
┌──────────────────────────────────────────────────────────────────┐
│ Bits 0–16  │ 17 coarse part-of-speech tags (POS)                  │
│ Bits 17–24 │ 8 simplified dependency groups (DEP)                 │
│ Bits 25–31 │ 7 simplified morphological features (MORPH)          │
└──────────────────────────────────────────────────────────────────┘
```

17 POS + 8 DEP + 7 MORPH = 32 flags, sourced from the 32-bit NLP type
legend (`data/tokenizer/simplestories-1_nlp_type32.json`).

### Fallback for Unknown BPE Tokens

At encode time, BPE tokens without a type-dictionary entry receive `POS_X =
65536 = 1 << 16` as their `sig_word`. This ensures unknown words still carry
a valid NLP type bit, preserving the node-format invariant. (The shipped
data tags every BPE token, so this path is rarely exercised.)

### Compound Marker Token (`COMPOUND_TOKEN`)

A reserved node value the compiler appends to a §11.3 **compound-word**
kline's nodes — a single word (e.g. `Mary`) the external tokenizer splits
into multiple BPE subwords (`[M, ary]`) — so the kline is built as
`Mary: [COMPOUND_TOKEN, M, ary]`. The token is a pure structural marker
(carries no BPE component): its type word is `0x0002_0000` (bit 17 — the
sole free slot in the `sig_word` space; every other bit is assigned by
`dev/nlp/tag_vocab.py`) and its BPE id is 0, giving the full value
`0x0002_0000_0000_0000`.

The token participates in the signature algebra like any other node, so a
compound's signature _encodes_ the marker (`signature ==
make_signature([COMPOUND_TOKEN, M, ary])`) with no bit masking. Detection is
`COMPOUND_TOKEN in kline.nodes` (@kline spec §Structural Predicates). The
marker is confined to the kalvin↔NLP boundary: defined here, appended by
`ks/token_encoder.py`, read by `kalvin/kline.py`; it is a compiler/NLP
concern and does not appear in Kalvin's semantic layer.

## Dependencies

### Tokenizer (@tokenizer spec)

- The `KTokenizer` interface (`encode`, `decode`, `vocab_size`) — this
  specialisation is its sole concrete implementation.
- The base `Tokenizer` BPE-engine wrapper — the foundation this
  specialisation builds on (see §BPE Engine Foundation below).

### Signifier (@signifier spec)

- The NLP deployment pairs this tokenizer with `NLPSignifier`. The two
  share the node-packing agreement (type word in the upper 32 bits); this
  is a property of the NLP bundle, not of either interface.

## Interface

This specialisation implements the `KTokenizer` interface (role: text↔nodes,
see @tokenizer) and additionally owns the node layout, the type
dictionary, and the BPE-engine foundation.

### `encode(text: str, pad_ws: bool = False) → list[node]`

Convert a string to typed nodes:

1. **BPE-encode** the text into subword token IDs.
2. **Type lookup** — for each token ID, look up the `sig_word` in the
   type dictionary. Tokens without entries default to `POS_X`.
3. **Construct nodes** — assemble each node as
   `(sig_word << 32) | bpe_token_id`.

If a word BPE-encodes into multiple subword tokens (e.g. "unhappiness" →
`[un, ##happiness]`), each subword token gets its own type lookup.
Subwords without dictionary entries fall back to `POS_X`.

```
encode("the air")
  BPE → [257, 500]
  257 "the" → sig_word = T_the
  500 "air" → sig_word = T_air
  → [(T_the << 32) | 257, (T_air << 32) | 500]
```

### `decode(nodes: list[node]) → str`

Convert typed nodes back to text. The BPE token ID is taken from the low
32 bits of each node; the type bits (high 32) are not needed for
reconstruction. `decode(encode(text)) == text`.

### `vocab_size → int ≥ 0`

The size of the BPE vocabulary.

## Type Dictionary

The tokenizer owns a **type dictionary** mapping BPE token IDs to their
`sig_word` (a single-bit type selector; see §NLP Sig-Word Layout), plus rich
NLP labels (`pos`, `pos_fine`, `dep`, `morph`, `nlp_type32`) carried as
opaque metadata, and `sig_type` — the legend flag name of the type that
won the `sig_word` selection (e.g. `POS_FINE_NNP`), or `""` when the
fallback bit was used (no type matched). Entries
are loaded from a tagged-grammar file
(`{tokenizer_name}_tagged_grammar.json`) generated by `dev/nlp/tag_vocab.py`,
which tags every BPE token — including sub-words — and writes the result
under the `sig_word` key.

### Lookup

- `lookup_type(token_id) → int | None` — the `sig_word` for a BPE token ID.
- `lookup_type_entry(token_id) → dict | None` — the raw dictionary entry.

These operate on **BPE token IDs**, not on nodes. Callers holding a node
must unpack it themselves or use a node-taking accessor; the high-32/low-32
layout is owned by this tokenizer and not assumed by Kalvin core.

## BPE Engine Foundation

The NLP tokenizer is built on the base `Tokenizer` BPE-engine wrapper
(see @tokenizer §BPE Engine). It inherits the engine-management surface
(`train`, `save_to_directory`, `load_from_directory`, `encode_bpe`, raw
BPE↔text operations) unchanged. The engine produces raw vocabulary
indices; this specialisation adds the type lookup and node packing to turn
them into nodes.

## Vocabulary

The shipped NLP data tags every BPE token, so the BPE vocabulary and the
NLP type dictionary have the same size:

- **BPE vocabulary**: 25,007 tokens (from `tokenizer-32768`, trained toward
  a 32,768 target).
- **NLP type dictionary**: 25,007 entries mapping BPE token IDs to
  `sig_word` (single-bit type selector), with rich NLP labels.

Source files:

| File                                                 | Purpose                                           |
| ---------------------------------------------------- | ------------------------------------------------- |
| `data/tokenizer/tokenizer-32768.json`                | BPE vocabulary metadata                           |
| `data/tokenizer/tokenizer-32768.bin`                 | BPE mergeable ranks                               |
| `data/tokenizer/tokenizer-32768_tagged_grammar.json` | Type dictionary (BPE→`sig_word`, with NLP labels) |
| `data/tokenizer/simplestories-1_grammar.json`        | NLP grammar (input to tagging; uses `nlp_type32`) |
| `data/tokenizer/simplestories-1_nlp_type32.json`     | 32-bit NLP type legend                            |

## Construction

```
NLPTokenizer(tokenizer_path=None, tokenizer_name="tokenizer-32768") → NLPTokenizer
```

Constructing `NLPTokenizer()` loads the BPE engine (via the base
`Tokenizer` engine machinery) and the NLP type dictionary
(`{tokenizer_name}_tagged_grammar.json`) from `tokenizer_path`. When
`tokenizer_path` is omitted it is resolved via `kalvin.paths.tokenizer_dir()`.

This is the production factory: the sole concrete `KTokenizer`. Loading
the NLP-tagged grammar from disk is an NLP concern, so it lives in this
specialisation rather than the base engine wrapper.

## Dimension Count

The `nlp_type32` encoding provides **32 dimensions**:

- 17 POS tags (bits 0–16)
- 8 DEP groups (bits 17–24)
- 7 MORPH features (bits 25–31)

| Tokenizer | Dimensions | Notes                    |
| --------- | ---------- | ------------------------ |
| NLP       | 32         | 17 POS + 8 DEP + 7 MORPH |
| Target    | ~35        | From the vision          |

## Curriculum Compatibility (Bare Signatures)

All existing curricula compile and train correctly under the NLP
specialisation without modification. A "bare" signature carries no
parenthetical annotation.

- A bare single-character signature (e.g. `M`, `H`, `A`) encodes to a
  single 64-bit typed node whose upper 32 bits carry the type word and
  lower 32 bits carry the BPE token ID. The same character always
  produces the same node value.
- A bare multi-token signature (e.g. `MHALL`, `SVO`) decomposes into
  individual identity entries plus a canonize (S2) entry mapping the first
  token to all tokens.
- The binding resolver operates correctly with an empty symbol table —
  bare signatures compile and produce valid graph nodes.
- Comments are optional for curricula using abstract uppercase letters
  (A–Z). Comments are required only when semantic word resolution is
  desired (e.g. `M(ary)` → the bound token for "Mary").

## Test Matrix

| ID         | Criterion                                                                                                                                                     | Origin ref |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| TOK-NLP-1  | NLP encode produces the typed-node format `(sig_word << 32) \| bpe_token_id` for each token                                                                   | NLP        |
| TOK-8      | Typed-node format: every node is `(sig_word << 32) \| bpe_token_id` (production layout)                                                                       | @tokenizer |
| TOK-NLP-2  | NLP encode with unknown BPE token uses `POS_X = 65536` as the sig word                                                                                        | NLP        |
| TOK-10     | Type dictionary: `lookup_type(id)` returns the entry's `sig_word`                                                                                             | @tokenizer |
| TOK-11     | Unknown-token fallback: tokens absent from the dictionary get `POS_X`                                                                                         | @tokenizer |
| TOK-12     | The base `Tokenizer` is a BPE-engine wrapper, not a `KTokenizer` (only `NLPTokenizer` is a concrete `KTokenizer`)                                             | @tokenizer |
| TOK-NLP-4  | NLP round-trip: `decode(encode("Tea brewed softly")) == "Tea brewed softly"`                                                                                  | NLP        |
| TOK-NLP-7  | Vocabulary sizes: BPE vocab = 25,007 tokens, type dictionary = 25,007 entries                                                                                 | NLP        |
| TOK-NLP-8  | Dimension count: `nlp_type32` provides 32 dimensions (17 POS + 8 DEP + 7 MORPH)                                                                               | NLP        |
| TOK-NLP-9  | Bare single-character signatures (e.g. `M`, `H`, `A`) produce consistent typed nodes; the same character always yields the same node value                    | NLP        |
| TOK-NLP-10 | Bare multi-token signatures (e.g. `MHALL`, `SVO`) decompose into individual identity entries plus a canonize (S2) entry mapping the first token to all tokens | NLP        |
| TOK-NLP-11 | The binding resolver operates correctly with an empty symbol table (bare sigs compile without annotation)                                                     | NLP        |
| TOK-NLP-12 | Curricula using abstract uppercase letters (A–Z) require no parenthetical comments; comments are required only when semantic word resolution is desired       | NLP        |
| TOK-NLP-13 | `NLPTokenizer()` loads the BPE engine and NLP type dictionary from standard paths                                                                             | NLP        |
| TOK-NLP-14 | `NLPTokenizer` is the sole concrete `KTokenizer`; the base `Tokenizer` is a BPE-engine wrapper, not a `KTokenizer`                                            | NLP        |

## Referenced By

- **Tokenizer** (@tokenizer spec) — defines the layout-agnostic
  `KTokenizer` interface that this specialisation implements.
- **Signifier** (@signifier spec) — owns the signature algebra; the NLP
  deployment pairs this tokenizer with `NLPSignifier`.
- **Signature** (@signature spec) — the signature value concept, consumed
  unchanged.
