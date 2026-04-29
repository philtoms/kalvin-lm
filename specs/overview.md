# The Kalvin System: An Overview

## 1. The Core Premise

Kalvin is an **agent** — a system with the capacity to make autonomous choices about how it receives, evaluates, and integrates new information. This capacity for choice is called **Agency**, and it is the defining property of the system. Agency is not bolted on; it arises from the structure itself, through a mechanism called **Significance**.

When confronted with a new idea, Kalvin performs an internal rationalisation — a graduated assessment of how that idea relates to what it already knows. This assessment ranges from full comprehension ("I fully understand this idea") through partial recognition ("I understand some of the idea", "Aspects of this idea are reminiscent") to complete novelty ("This idea is completely new to me"). Each of these outcomes — S1, S2, S3, S4 — is not merely a label but a mathematically derived consequence of the graph's topology, and each carries a distinct structural signature that determines what the agent does next. Agency is thus expressed through **Rationalisation**: the process of determining significance and acting on it.

The system operates on a fundamental distinction: **nodes** (uint64 values) are the atoms — static, opaque integers. **KLines** are the structures — dynamic arrangements of nodes into identified, ordered sequences. The model grows not by creating new kinds of matter, but by composing existing matter into increasingly complex hierarchies, and by refining how those compositions relate to each other.

## 2. The Ontological Foundation

The system is built upon a small, precise set of concepts:

### Nodes

A **node** is a 64-bit unsigned integer. Nodes are opaque — the system does not inspect or interpret them beyond bitwise operations. They come in two flavours:

- **Literal nodes** — carry a 32-bit literal mask (`0xFFFFFFFF`) in the
  lower bits, with the character code point in the upper 32 bits. These
  represent exact, atomic tokens (individual characters or raw values). They
  preserve identity and order.
- **Non-literal (packed) nodes** — bit 0 is clear (`node & 1 == 0`). These
  carry structural or type information.

Both node types participate in signature construction via `make_signature`.
The distinction is an encoding concern — it affects *how* nodes contribute
(literal nodes contribute bit 0; non-literal nodes contribute their full
value), not *whether* they contribute.

### KLines

A **KLine** is the fundamental unit of the knowledge graph. It consists of:

| Field     | Type                   | Description                                 |
| --------- | ---------------------- | ------------------------------------------- |
| signature | uint64                 | Identity key. The "name" of this structure. |
| nodes     | ordered list of uint64 | Zero or more child nodes. Order matters.    |

KLines form a directed graph: when a node value equals the signature of another KLine, that's an **edge**. A KLine's nodes _reference_ other KLines, creating compositional hierarchies.

KLine equality is defined as: same signature **and** same node sequence (same length, same order, same values).

A KLine is **literal** when all of its nodes are literal tokens (per `tokenizer.is_literal`). This is a computed property, not stored. Literal KLines represent exact, atomic token sequences; non-literal KLines are composed structures whose nodes reference other KLines.

### Signatures

A **signature** is a 64-bit unsigned integer. Bit 0 is the **literal-content flag** — set when the kline contains literal nodes. Signatures serve two roles: as kline heads (identifying a kline) and as kline nodes (referencing other klines, forming graph edges). Signatures are created by `make_signature` — an OR-reduction over all nodes. See the @signature spec for full definition.

### The Model (Knowledge Graph)

The **Model** is the collective graph of all KLines. It provides storage, indexing, deduplication, lookup by signature, graph traversal, and the comparison functions consumed by the significance pipeline.

The model has a **three-tier memory architecture**, invisible to callers:

```
STM → Frame → Base
```

| Tier  | Purpose               | Bounded   | Lifetime       | Written by add |
| ----- | --------------------- | --------- | -------------- | -------------- |
| STM   | Transitive grounding  | Yes (256) | Rolling window | Yes            |
| Frame | Session write surface | No        | Per-session    | Yes            |
| Base  | Long-term knowledge   | No        | Persistent     | No (promotion) |

- **STM** (Short-Term Memory) is a bounded, dual-keyed index over the most recently added KLines. It indexes each KLine by both its signature _and_ its nodes signature, enabling **transitive grounding** — finding KLines that share node structure even when their signatures differ. When the bound is exceeded, the oldest entries are evicted (they remain in the frame).
- **Frame** is the primary write surface for the current session. All non-rejected KLines are added here. Lookups that miss in STM fall through to the frame.
- **Base** is an optional long-term knowledge store. It is **read-only** during a session — Klines reach the base through **promotion**, a separate mechanism triggered by the agent on significant results (S1 and S4). Each session layers a fresh frame over a shared base, giving isolated writes with shared knowledge.

Lookups search tiers in order: STM → Frame → Base. Callers see a single unified Model API; tiering is managed internally.

## 3. Agency

Kalvin has **agency** — the capacity to make choices about how new information is received and integrated. This agency operates through **Significance**: a mathematical distance function that evaluates how a new idea relates to what is already known. The result of this computation is not merely a number — it leaves traces in the structure of the model that can be usefully rationalised as an internal dialogue:

| Level | Internal Rationalisation                | Label(s)                    |
| ----- | --------------------------------------- | --------------------------- |
| S1    | "I fully understand this idea."         | Canonical or Countersigned  |
| S2    | "I understand some of the idea."        | Underfitting or Overfitting |
| S3    | "Aspects of this idea are reminiscent." | Connotational               |
| S4    | "This idea is completely new to me."    | Unsigned                    |

Each significance level corresponds to a distinct structural relationship between a KLine and the model. These are not arbitrary labels — they are the observable consequences of the distance function applied to graph topology:

### S1 — Canonical or Countersigned

An S1 KLine is either fully **canonical**, or it is **countersigned** by another KLine. Both represent complete comprehension — the agent has full structural account of the idea.

- A **canonical KLine** is one whose signature fully represents its nodes — the `make_signature` reduction of its nodes produces a signature identical to the KLine's own signature. The structure is self-consistent: nothing is missing and nothing is extraneous. This includes all-literal klines: because `make_signature` sets bit 0 for each literal node, an all-literal kline produces signature `1`, which is a valid canonical signature.
- A **countersigned KLine** is one whose nodes reference another KLine whose nodes reciprocally reference the first. This mutual cross-reference establishes a grounded equivalence — each structure vouches for the other. Countersignature is a **latent** structural relationship: it is not visible from a single KLine's perspective and is typically discovered through **cogitation** (see §6). An S2 result during initial rationalisation may, upon deeper graph traversal, reveal a countersigned relationship and be promoted to S1.

### S2 — Underfitting or Overfitting

An S2 KLine would be S1 but for a mismatch between signature and node content. It falls into one of two sub-categories:

- An **underfitting KLine** has a signature that contains more information than its nodes warrant. The signature promises structure that the nodes don't deliver — the KLine over-represents its content.
- An **overfitting KLine** has a signature that contains less information than its nodes provide. The nodes carry structure that the signature doesn't capture — the KLine under-represents its content.

### S3 — Connotational

An S3 KLine is **connotational**: its nodes are unrelated to its signature. There is no direct structural alignment — the connection, if any, is by association rather than by composition. The KLine evokes rather than describes.

### S4 — Unsigned

An S4 KLine is **unsigned**: it does not have any nodes of its own. Without nodes, there is nothing to ground, nothing to compare, and nothing to fit. It is pure potential — a named absence awaiting content.

## 4. The Engine of Agency: Significance

Agency in Kalvin is driven by **Significance** — a 64-bit unsigned integer that measures how well a candidate KLine answers a query KLine.

Significance is the **bitwise NOT of a distance** in a metric space:

```
significance = ~distance
```

Higher significance = closer match = less work needed. The ordering is strict: **S1 > S2 > S3 > S4** by unsigned integer comparison.

### The Significance Spectrum

| State | Condition                            | Distance          | Significance     | Agent Action         |
| ----- | ------------------------------------ | ----------------- | ---------------- | -------------------- |
| S1    | All nodes match perfectly            | 0                 | MAX (all bits 1) | Confirm. Promote.    |
| S2    | Some nodes match, some don't         | [1, D_boundary)   | High             | Queue for cogitation |
| S3    | No nodes match, but candidates exist | [D_boundary, MAX) | Low              | Queue for cogitation |
| S4    | No candidates found at all           | MAX               | 0 (all bits 0)   | Novel. Promote.      |

`D_boundary` (default `0x8000_0000_0000_0000`, the midpoint) is a hyperparameter that separates S2 and S3 distance ranges.

Key insight: **S1 and S4 are "significants"** — the KLine is either confirmed or entirely novel. No further processing needed. **S2 and S3 are "rationals"** — partial relationships that require deeper investigation through **cogitation**.

### Per-Node Significance Routing

Significance is computed through per-node testing, not holistic comparison:

1. For each node in the query KLine, test: does this node achieve S1 against the candidate? (via `model.is_s1(node, candidate)`).
2. Count the S1 nodes and **route**:
   - All nodes S1 → distance = 0 → **S1**
   - Some nodes S1 → `model.s2_distance(query, candidate)` → **S2**
   - No nodes S1 → `model.s3_distance(query, candidate)` → **S3**
   - No candidates → distance = MAX → **S4**

This routing is **pessimistic**: the presence of any node that doesn't achieve S1 pulls the overall level down. Only S2 and S3 routes call model distance functions.

### Candidate Retrieval

Candidates are found via **bitwise AND matching**:

```
candidates = model.where(k => (k.signature & query.signature) != 0)
```

A KLine whose signature shares _any_ set bit with the query's signature is a candidate. This is a necessary but not sufficient condition — it pre-filters the model before the more expensive significance pipeline runs.

## 5. The Process of Rationalisation

Rationalisation is the agent's core loop: determining how a new KLine relates to existing knowledge and deciding what action to take. It proceeds in six phases:

```
┌─────────────────────────────────────────────────┐
│ 1. PREPARE                                      │
│    Assign signature if missing.                  │
├─────────────────────────────────────────────────┤
│ 2. GROUND CHECK                                 │
│    Does Q already exist in the model?            │
│    → Yes: emit "ground" event, done.             │
├─────────────────────────────────────────────────┤
│ 3. ASSESS                                       │
│    Evaluate Q's structural grounding:            │
│    → Unsigned (no nodes): S4, done.              │
│    → All-literal: S1, done.                      │
│    → Self-grounded (all nodes resolve): S1, done.│
│    → Otherwise: proceed to retrieval.            │
├─────────────────────────────────────────────────┤
│ 4. RETRIEVE CANDIDATES                           │
│    Find KLines with signature AND overlap.       │
│    → No candidates: S4 (novel), done.            │
├─────────────────────────────────────────────────┤
│ 5. COMPUTE SIGNIFICANCE                         │
│    Per-node S1 test → route → distance → invert. │
├─────────────────────────────────────────────────┤
│ 6. INTEGRATE                                    │
│    Add Q to model. Act on best result.           │
│    → S1/S4: promote to base.                    │
│    → S2/S3: queue for cogitation.               │
└─────────────────────────────────────────────────┘
```

### Phase 1: Prepare

If the KLine's signature is 0, compute it: `Q.signature = make_signature(Q.nodes)` as defined in the @signature spec.

### Phase 2: Ground Check

Test whether an equal KLine (same signature, same node sequence) already exists in any tier. If grounded, emit a `"ground"` event and stop. This prevents infinite recursion and avoids re-processing known knowledge.

### Phase 3: Assess

Structural fast-paths that bypass the full significance pipeline. Each corresponds to an agency category:

- **Unsigned** (S4): zero nodes → no information content, no signature.
- **Canonical — all-literal** (S1): every node is literal → `make_signature` produces `1`, a canonical signature representing the presence of literal content.
- **Canonical — self-grounded** (S1): every non-literal node resolves to an existing KLine in the model, and the kline's signature equals its `make_signature` → fully grounded composition. (Literal nodes always contribute to the canonical test via bit 0; only non-literal nodes require resolution.)
- **Ground** (S1): exact duplicate already in the model (handled in Phase 2).

If none apply, proceed to candidate retrieval.

### Phase 4: Retrieve Candidates

Find all KLines in the model whose signatures share at least one bit with the query's signature (bitwise AND ≠ 0). If no candidates are found, the result is S4 (novel) — the KLine is added and promoted.

### Phase 5: Compute Significance

For each candidate, run the significance pipeline: per-node S1 testing, routing, distance calculation, inversion. Collect all `(candidate, significance, level)` triples.

### Phase 6: Integrate

Add the KLine to the model (both STM and frame). Select the best result (highest significance value). If S1 or S4, promote to the base model. If S2 or S3, queue for cogitation.

## 6. Cogitation

Cogitation is background processing of rational KLines (S2/S3). It performs deeper graph traversal to find candidates that the initial bitwise-AND retrieval may have missed, and specifically to discover **latent countersignature** relationships:

```
Cogitate(Q):
  1. Expand Q's graph context:
     candidates = model.query(Q.signature, depth=D_cogitate)

  2. For each candidate Cᵢ:
     a. Run significance pipeline.
     b. Test for countersignature:
        Q's nodes reference Cᵢ's signature, AND
        Cᵢ's nodes reference Q's signature.
     c. If countersigned → upgrade to S1.

  3. If any candidate achieves S1 (via significance or countersignature):
     - Add candidate to model.

  4. Re-rationalise Q.
```

Countersignature discovery is the primary mechanism by which cogitation promotes S2 to S1. A KLine that appears to be underfitting or overfitting during initial rationalisation may, upon deeper traversal, turn out to be part of a mutual cross-reference that the initial bitwise AND retrieval could not detect. The countersignature test is structural: it examines whether two KLines reference each other through their nodes. (Literal tokens cannot match a signature — their 32-bit literal mask in the lower bits is distinct from any signature value — so the test naturally considers only non-literal matches, but this is enforced by the encoding, not by an explicit filter.)

`D_cogitate` (default 2) controls traversal depth. If Q has been cogitated more than a configurable maximum (default 3 passes) without reaching a significant result, it is abandoned. The cogitation thread runs asynchronously and emits a `"done"` event when the backlog has been empty for a timeout (default 2 seconds).

## 7. Tokenizers

Two tokenizer types produce the same output kind — typed nodes:

- **Mod (Modular)** — maps characters to bit positions. Multi-character strings are OR-ed into a single packed node (lossy: order and multiplicity lost). Individual characters can also be encoded as literal nodes (preserving identity and order). Variants: Mod32 (31 character bits), Mod64 (63 character bits).
- **BPE (Byte-Pair Encoding)** — learns subword vocabulary from a training corpus. Tokens are sequential IDs combined with **type prefixes** (linguistic properties like part-of-speech) at the agent layer: `node = type_prefix | token_id`. BPE tokens are never literal.

Signatures are constructed from tokenizer output via `make_signature`, defined in the @signature spec.

## 8. Supporting Mechanisms

### Deduplication

When a **literal** KLine is added, the model checks whether an equal KLine (same signature, same node sequence) already exists in _any_ tier. If so, `add` returns `false`. Non-literal KLines are never deduplicated — composed structures are always accepted.

### Promotion

```
model.promote(kline) → bool
model.promote_all() → int
```

Promotion moves KLines from the frame to the base model, making them persistent across sessions. The agent triggers promotion on significant results (S1 = confirmed knowledge, S4 = novel knowledge).

### Events

The agent publishes events during rationalisation for observers:

| Kind     | Trigger                              | Significance |
| -------- | ------------------------------------ | ------------ |
| `ground` | KLine already exists in model        | S1           |
| `frame`  | KLine integrated (new or confirmed)  | S1–S4        |
| `done`   | Cogitation backlog empty for timeout | 0            |

Subscribers receive events synchronously in publication order.

## 9. The Governing Properties

### Pessimism

Significance can only drop below the per-node ideal. Even if most nodes achieve S1, the presence of any node that doesn't pulls the overall level down. The system is conservative in its assessments.

### Inverted Metric

Significance = ~distance. This means ordering falls out naturally: S1 > S2 > S3 > S4 by unsigned integer comparison. No separate level encoding needed — the numeric value _is_ the level.

### Tiered Memory as Temporal Bias

The three-tier architecture provides the system's temporal structure:

- **STM** (bounded, rolling) keeps the agent focused on the most recent context — the "frontier" of activity.
- **Frame** (session-scoped) captures the current session's work without polluting the long-term store.
- **Base** (persistent) grounds future sessions in accumulated knowledge.

Promotion is the bridge: only S1 (confirmed) and S4 (novel) KLines are promoted, ensuring the base model grows only through validated knowledge.

### Bitwise AND as Necessary Condition

Candidate retrieval uses bitwise AND on signatures as a fast pre-filter. This is lossy — it can miss relevant candidates (false negatives) and include irrelevant ones (false positives). The significance pipeline handles the precision; the AND filter handles the recall.

## 10. Open Questions

The following have TBD semantics in the current specs:

1. **`model.is_s1(node, candidate)`** — What does "perfect match" mean for a single node against a candidate? The agency framing suggests: a canonical match where the node's contribution to the query's signature is fully represented in the candidate's signature (`node & candidate.signature == node`), or a countersigned match (see `model.is_countersigned`). The routing depends on this, but the full comparison semantics are not yet defined.
2. **`model.s2_distance(query, candidate)`** — How is partial-alignment distance computed? The agency framing characterises this as the distance between an underfitting or overfitting KLine and S1: the degree of misalignment between signature information and node content. Must return a value in `[1, D_boundary)`.
3. **`model.s3_distance(query, candidate)`** — How is weak-recognition distance computed? The agency framing characterises this as connotational distance: how far the query's nodes are from the candidate's structure, absent direct alignment. Must return a value in `[D_boundary, MAX)`.
4. **Candidate retrieval efficiency** — `model.where(predicate)` performs a linear scan. A dedicated `candidates_for(signature)` method with an inverted bit-to-signature index may be needed for large models.
5. **Mod literal encoding capacity** — The new literal encoding `(char << 32) | 0xFFFFFFFF` places the character code point in the upper 32 bits, limiting code points to the range [0, 2³²−1]. This covers the entire Unicode range (max code point 0x10FFFF) with substantial headroom.
