# NLP-BPE Pipeline Plan

**Purpose:** Implementation plan for the NLP-BPE tokenizer pipeline — data
preparation utilities and node-to-signature conversion across the codebase.

**Specs:** @tokenizer (NLP section), @signature (node_to_sig, NLP masking),
@model (resolve with node_to_sig), @kscript-nlp.

---

## 1. Overview

The NLP-BPE pipeline enables Kalvin to encode text as grammatically-typed
BPE tokens instead of Mod32 character bitmasks. This plan covers the
development-time utilities that prepare the grammar data and the runtime
integration of `node_to_sig()` across the knowledge graph.

### Pipeline Stages

```
┌─ Data Preparation (dev-time) ──────────────────────────────────────┐
│                                                                     │
│  1. BPE Training                                                    │
│     tokenizer.train(corpus, vocab_size)                             │
│     → tokenizer-32768.bin, tokenizer-32768.json                     │
│                                                                     │
│  2. NLP Analysis                                                    │
│     nlp_analyzer.py (spaCy)                                         │
│     → simplestories-1_grammar.json                                  │
│     (BPE token ID → NLP type mapping)                               │
│                                                                     │
│  3. Grammar Expansion (expand_grammar.py)                           │
│     --bpe-rekey for new BPE tokenizer alignment                     │
│     → expanded grammar with subword inheritance                     │
│                                                                     │
│  4. Vocabulary Tagging (tag_vocab.py)                               │
│     Every BPE token gets NLP type from grammar                      │
│     → tagged grammar dict (100% vocab coverage)                     │
│                                                                     │
├─ Runtime ───────────────────────────────────────────────────────────┤
│                                                                     │
│  5. NLPTokenizer.encode()                                           │
│     BPE encode → grammar lookup → (nlp_type32 << 32) | bpe_id      │
│                                                                     │
│  6. node_to_sig() integration                                       │
│     All model.find(node) → model.find(node_to_sig(node))            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Spec References

| Component | Spec | Status |
|-----------|------|--------|
| NLP node format | @tokenizer §NLP Tokenizer | ✅ Specified |
| NLP-aware signature | @signature §Node-to-Signature Conversion | ✅ Specified |
| node_to_sig function | @signature §node_to_sig | ✅ Specified |
| Resolve with node_to_sig | @model §Resolve | ✅ Specified |
| Graph traversal node conversion | @model §Graph Traversal | ✅ Specified |
| KScript NLP-BPE mode | @kscript-nlp | ✅ Specified |

---

## 3. Implementation Tasks

### 3.1 BPE Re-keying in expand_grammar.py

**Status:** ✅ Complete  
**File:** `dev/nlp/expand_grammar.py`  
**Spec ref:** @tokenizer §NLP Tokenizer (grammar dictionary)

When the grammar was built using a different BPE tokenizer than the target
one, `rekey_from_bpe()` decomposes each grammar word through the current
BPE tokenizer and re-keys entries by new token IDs.

- `rekey_from_bpe(grammar, bpe_tokenizer, fine_legend_reverse)` — BPE-decomposes
  each grammar word, builds new dict keyed by new BPE token IDs. Subword tokens
  inherit NLP types from parent words (highest-count parent wins on conflict).
- `_add_inherited_entry()` — shared helper for creating inherited entries.
- `--bpe-rekey` CLI flag activates BPE decomposition in the pipeline.
- Pipeline order: load → merge → (optional BPE rekey) → special-token annotation
  → subword inheritance → manual annotations → save.

**Tests:** Manual verification with `--dry-run`. 100% coverage on original
tokenizer, 99.5% on freshly trained tokenizer.

### 3.2 Vocabulary Tagging — tag_vocab.py

**Status:** ✅ Complete  
**File:** `dev/nlp/tag_vocab.py`  
**Spec ref:** @tokenizer §NLP Tokenizer (grammar dictionary)

Takes a trained BPE tokenizer and an NLP grammar dictionary, produces a
tagged grammar dict where **every** BPE token (including subwords) is
annotated with NLP type information. Guarantees 100% vocab coverage.

Matching algorithm (vocab → grammar, in priority order):

1. **Exact match** — token text matches grammar entry's text (case-insensitive).
2. **Space-stripped exact match** — `" the"` matches `"the"` for space-prefixed
   BPE tokens.
3. **BPE decomposition parent** — grammar words are pre-decomposed; subword
   tokens inherit from highest-count parent.
4. **Substring fallback** — alphabetic tokens match grammar text containing
   them as substring.
5. **Special-token rules** — deterministic classification for whitespace,
   punctuation, digits, control characters.
6. **POS_X fallback** — unknown tokens get `nlp_type32 = 65536`.

`GrammarIndex` pre-builds all lookup structures for O(1) exact/BPE matching
and O(n) substring fallback.

**Tests:** Manual verification. 99.9% coverage on original tokenizer
(22 POS_X including control characters), 99.9% on freshly trained tokenizer
(24 POS_X for genuinely novel words).

### 3.3 node_to_sig Integration

**Status:** ✅ Complete  
**Spec ref:** @signature §node_to_sig, @model §Resolve, §Graph Traversal

Added `node_to_sig()` function and applied it at all call sites where raw
node values are used as model lookup keys. For Mod32 packed nodes, this is
identity. For NLP-BPE nodes, the BPE token ID (low 32 bits) is masked out.

**Files changed:**

| File | Changes |
|------|---------|
| `src/kalvin/signature.py` | Added `node_to_sig()`, `BPE_TOKEN_MASK`. Refactored `make_signature()` to use `node_to_sig()` internally. |
| `src/kalvin/agent.py` | `model.find(node)` → `model.find(node_to_sig(node))` at 3 sites. |
| `src/kalvin/expand.py` | Same conversion at 5 sites + `edge_hops()` initial conversion. |
| `src/kalvin/model.py` | `resolve()`, `_query_expand_inner()`, `_descendants_inner()` use `node_to_sig()`. |
| `src/kalvin/misfit.py` | `_split_excess()` compares `node_to_sig(n) & excess` instead of `n & excess`. |

**Tests:** All 332 existing tests pass. No new tests needed — the change is
an internal refactoring that preserves existing Mod32 behavior while enabling
correct NLP-BPE behavior.

### 3.4 tokenizer.py Bug Fix

**Status:** ✅ Complete  
**File:** `src/kalvin/tokenizer.py`

Fixed `train_fromIterator` → `train_from_iterator` (correct rustbpe API).

---

## 4. Test Mapping

| Spec ID | Test | Status |
|---------|------|--------|
| SIG-11 | `test_signature.py` — `is_nlp_node` for NLP-BPE nodes | ✅ |
| SIG-12 | `test_signature.py` — NLP masking: same NLP type, different BPE IDs → identical signatures | ✅ |
| SIG-13 | `test_signature.py` — mixed NLP + literal: bit 0 + NLP bits | ✅ |
| SIG-14 | `test_signature.py` — Mod32 backward compatibility | ✅ |
| SIG-15 | `test_signature.py` — `node_to_sig(literal) == 1` | ✅ |
| SIG-16 | `test_signature.py` — `node_to_sig(nlp_node) == nlp_node & NLP_TYPE_MASK` | ✅ |
| SIG-17 | `test_signature.py` — `node_to_sig(mod32_packed) == mod32_packed` | ✅ |
| SIG-18 | Refactored: `make_signature` uses `node_to_sig` internally | ✅ |
| TOK-NLP-1..8 | `test_tokenizer.py` — NLP tokenizer test suite | ✅ |

---

## 5. Design Decisions

### D1: node_to_sig as explicit conversion

**Decision:** Added `node_to_sig()` as a standalone function rather than
modifying `model.find()` to accept both nodes and signatures.

**Rationale:** Model.find() accepts a signature — this is its contract.
Making it "smart" about node values would blur the signature/node distinction
and require the model to depend on tokenizer-specific knowledge. Keeping
the conversion at call sites is explicit, testable, and keeps the model
signature-agnostic.

### D2: tag_vocab works from vocab → grammar

**Decision:** `tag_vocab.py` iterates over the BPE vocabulary (not the
grammar entries) and finds the nearest grammar match for each token.

**Rationale:** The guarantee is that every encoded token has a grammar entry.
Working from vocab ensures this by construction. The reverse (grammar → vocab)
could miss tokens that have no grammar parent.

### D3: Space-prefixed BPE tokens get exact-match treatment

**Decision:** BPE tokens with leading spaces (e.g. `" the"`) are matched
against the space-stripped grammar text via exact match before falling back
to substring matching.

**Rationale:** Standard BPE training produces space-prefixed tokens for
common words. Stripping the space and doing exact match is fast and accurate —
`" the"` should inherit from the grammar entry for `"the"`, not from a
substring match against `"theater"`.

### D4: Substring fallback accepts space-prefixed tokens

**Decision:** The `isalpha()` guard on substring fallback was widened to
also accept space-prefixed alphabetic tokens (via `_is_space_prefixed_alpha`).

**Rationale:** BPE subwords like `" de"` (from `"development"`) would
otherwise fall through to POS_X. Stripping the space and doing substring
match resolves these correctly.

---

## 6. Status

| Task | Status | Commit |
|------|--------|--------|
| BPE re-keying in expand_grammar.py | ✅ Complete | `4e39434` |
| Vocabulary tagging (tag_vocab.py) | ✅ Complete | `51f743a` |
| node_to_sig integration | ✅ Complete | `833428e` |
| tokenizer.py bug fix | ✅ Complete | `4e39434` |
| Spec updates (signature, model) | ✅ Complete | This commit |
| Plan document | ✅ Complete | This commit |
