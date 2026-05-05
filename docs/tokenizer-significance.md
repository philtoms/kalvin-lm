# Tokenizers and Significance

## Why Tokenizers Matter to Significance

Kalvin does not treat tokenizers as opaque text-to-ID converters. Significance — the system's core metric for agency — is computed at the **bit level** between signatures, and signatures are built directly from the token values that the tokenizer produces. This means the internal encoding strategy of every tokenizer must conform to the same structural constraints: every token must be a `uint64` value whose bit pattern is compatible with bitwise OR (signature construction), bitwise AND (candidate retrieval), and the literal/non-literal distinction (routing).

A tokenizer is not free to encode knowledge however it likes. It is free to choose a *vocabulary* and an *encode/decode strategy*, but once encoded, every token must participate in the same bit-level algebra that drives significance. This is the fundamental contract:

> **All tokenizers produce the same kind of output — `uint64` nodes that participate in signature construction via bitwise OR and candidate retrieval via bitwise AND.**

The vocabularies from which tokens are generated may be completely incompatible (ASCII characters vs. BPE subwords vs. linguistic type tags), but the encoded bit structures must interoperate.

## The Structural Constraints

Regardless of tokenizer choice, every node must satisfy:

1. **Signature compatibility.** Non-literal nodes contribute their full value to `make_signature` via bitwise OR. Literal nodes contribute bit 0 only. The resulting signature must be a valid `uint64` usable for bitwise AND matching.

2. **Literal discrimination.** The system must be able to distinguish literal from non-literal nodes. This is a **Kalvin-level concern**, not a tokenizer concern — the literal mask (`0xFFFFFFFF` in the lower 32 bits) is a standardized bit pattern that Kalvin tests directly via the standalone `is_literal()` function. No tokenizer-specific function is needed.

3. **Packed node bit 0 clear.** Non-literal (packed) nodes have bit 0 clear (`node & 1 == 0`). This ensures that the literal-content flag in signatures is set exclusively by literal nodes.

4. **No external semantics.** Nodes are opaque `uint64` values. The significance pipeline does not interpret them — it operates purely on bit patterns. Tokenizers encode dimensionality, not meaning.

## Token Granularity as Dimensionality

The key difference between tokenizer schemes is **granularity** — which we can understand as *dimensionality*. Each set bit in a non-literal node represents a dimension. A tokenizer determines what those dimensions correspond to and how densely they are packed.

### Coarse Dimensionality: Mod(N)Tokenizer

The `Mod(N)Tokenizer` maps individual ASCII characters to bit positions. Each character occupies one dimension (one bit) in the token. For a Mod32 tokenizer, there are 31 usable bit positions (bits 1–31); for Mod64, there are 63 (bits 1–63).

This is a deliberately coarse encoding:

- **One dimension per character.** The bit for `'A'` means "the character A is present." No more, no less.
- **Collisions are possible.** When the vocabulary exceeds the bit range, characters share bit positions via modular wrapping. `'A'` and the 32nd character in the vocabulary map to the same bit.
- **Order is lost.** Packed encoding is commutative OR — `"AB"` and `"BA"` produce the same node.
- **Multiplicity is lost.** `"AA"` and `"A"` produce the same node.

The vocabulary is tiny (at most 95 printable ASCII characters, fewer usable bits), but that is precisely the point. The Mod tokenizer is not trying to compress information — it is assigning orthogonal bit dimensions to a small, observable character set so that the resulting knowledge graph can be inspected, reasoned about, and even hand-crafted.

### Fine Dimensionality: BPE Tokenizer

The BPE Tokenizer (under development) operates on a much larger vocabulary learned from training data — typically thousands of entries. But crucially, it is **not a standard BPE tokenizer**. A standard BPE tokenizer encodes text as sequential integer IDs with no bit-level semantics. In Kalvin, that approach would be useless for signature construction.

Instead, the BPE Tokenizer encodes token bit structures that operate the same way as Mod tokens do. Its freedom lies in *how those bit structures are populated*: it can derive them from trained BPE datasets, combining vocabulary indices with **type prefixes** — bit patterns encoding linguistic properties such as part-of-speech, dependency relations, and morphology:

```
node = type_prefix | token_id
```

The type prefix is not arbitrary metadata bolted onto the token. It is the dimensional payload. Where the Mod tokenizer assigns one dimension to `'A'`, the BPE tokenizer assigns dimensions to linguistic categories: `POS_DET`, `POS_NOUN`, `DEP_OBJ`, and so on. Each of these is a bit (or set of bits) in the node value, and each participates in bitwise AND matching exactly as a character bit would.

The granularity difference is therefore one of **what the dimensions represent**, not how they function:

| Aspect | Mod(N)Tokenizer | BPE Tokenizer |
|--------|-----------------|---------------|
| Dimension source | ASCII character identity | Trained NLP part-of-speech encoding |
| Vocabulary size | ≤ 95 (printable ASCII) | Thousands (trained) |
| Bits per token | 1 per character (with wrapping) | Multiple per token (type prefix + ID) |
| Encoding strategy | Single character → single bit | Subword sequence → combined type + ID |
| Compression | None (wasteful: one dimension per character) | Moderate (linguistic categories pack denser meaning per bit) |
| `is_literal` | Always `False` for packed; literal mask for per-char | Always `False` |

Both schemes produce the same kind of thing: `uint64` values whose set bits represent dimensions that Kalvin can use to construct knowledge as a graph.

## The Literal Escape Mechanism

The Mod tokenizer provides a dedicated mechanism for encoding exact text: **literal nodes**. When `pack=False`, each character is encoded as a separate node with the lower 32 bits set to `0xFFFFFFFF` (the literal mask) and the character's Unicode code point stored in the upper 32 bits:

```
literal_node = (char_codepoint << 32) | 0xFFFFFFFF
```

Literal nodes bypass the vocabulary entirely. Any character with a valid code point can be encoded, regardless of whether it appears in the alphabet. This is important because:

1. **Signatures can carry literal content.** Literal nodes contribute bit 0 (the literal-content flag) to signatures, ensuring that klines containing exact text are still discoverable via bitwise AND matching.

2. **Order and identity are preserved.** Unlike packed nodes, literal nodes maintain character sequence — critical for representing exact strings like `"Mary"` or `"lamb"`.

3. **Kalvin understands the convention directly.** The significance pipeline calls `is_literal()` — a standalone function that tests the literal mask bit pattern. This is not a tokenizer-vendored function; it is a Kalvin-level test based on the standardized node format. Literal nodes are a recognised category that the system handles consistently without any tokenizer dependency.

The BPE tokenizer does not use literal encoding — its tokens never carry the literal mask pattern (`0xFFFFFFFF` in the lower 32 bits), so `is_literal()` always returns `False` for BPE nodes. All content is encoded through the type-prefix + token-ID scheme. This is a trade-off: the BPE tokenizer gains denser dimensional representation but loses the ability to embed raw character values directly into nodes.

## The Mod(N)Tokenizer and KScript

The Mod tokenizer is designed to work directly with **KScript** — Kalvin's domain-specific language for constructing knowledge graphs. Its purpose is **research**: to provide compact, observable, accessible graphs that can be reasoned with and even hand-crafted during Kalvin development.

Consider **MHALL**, a common entry point for rationalisation. It consists of just 4 tokens — `M`, `H`, `A`, `L`, `L` — which pack into a single signature:

```
encode("MHALL") → [bit_M | bit_H | bit_A | bit_L]
```

Note: because `'L'` maps to a single bit position, the two L's collapse. The resulting signature contains 4 distinct set bits (one each for M, H, A, L), fitting neatly into a single `uint64`.

From this accessible starting point we can explore how to build Kalvin's agency: composing KScript declarations like `MHALL == SVO` to establish countersigned relationships, then drilling into `S = M`, `V = H`, `O = ALL` to ground each component. The Mod tokenizer makes this process transparent — you can read the bits, trace the signatures, and verify the significance calculations by hand.

## Significance Routing Across Tokenizers

Significance routing — the determination of S1/S2/S3/S4 — is **tokenizer-agnostic**. It operates on node membership tests:

```
route(Q, C):
  match_count = |{n ∈ Q.nodes : n ∈ C.nodes}|
  if match_count == len(Q.nodes):  return "S1"
  if match_count > 0:              return "S2"
  else:                            return "S3"
```

This works regardless of whether the nodes are Mod-packed character bits or BPE-composed type prefixes. The routing logic does not care what the bits *mean* — it only cares whether a node value from the query appears verbatim in the candidate. This is why all tokenizers must produce compatible `uint64` nodes: the routing, the distance accumulation, the significance inversion, and the boundary classification all operate on the same bit-level algebra.

The literal test is not a tokenizer-vendored function. Kalvin handles it directly via the standalone `is_literal()` function — a single bit-level check based on the standardized node format. This works regardless of tokenizer because the literal mask is a Kalvin convention, not a tokenizer convention. The signature builder calls `is_literal()` internally, and the routing, distance accumulation, significance inversion, and boundary classification all operate on the same bit-level algebra without any tokenizer dependency.

## Summary

| Property | Mod(N)Tokenizer | BPE Tokenizer |
|----------|-----------------|---------------|
| Purpose | Research, KScript, hand-crafting | Production, large-scale graphs |
| Vocabulary | Fixed ASCII character set | Trained from corpus (thousands) |
| Dimensionality | Coarse: 1 bit per character | Fine: linguistic categories per token |
| Literal support | Yes (escape mechanism) | No (always non-literal) |
| Encoding | Character → bit position (modular wrapping) | Subword → type_prefix \| token_id |
| Signature construction | Same (bitwise OR) | Same (bitwise OR) |
| Candidate retrieval | Same (bitwise AND) | Same (bitwise AND) |
| Routing | Same (node membership) | Same (node membership) |
| Compression | None (intentionally wasteful) | Moderate (denser dimensions per bit) |

The thesis is straightforward: **tokenizers do not encode knowledge — they encode dimensionality.** Kalvin constructs knowledge as a graph by composing dimensional nodes into klines, and significance emerges from the bit-level relationships between those compositions. The Mod tokenizer provides a research-grade sandbox with transparent, hand-verifiable dimensions. The BPE tokenizer provides a production-grade path with denser, linguistically-informed dimensions. Both feed the same engine.
