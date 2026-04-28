# The Kalvin System: An Overview

## 1. The Core Premise

Kalvin is a rationalizing agent that integrates new information into a growing knowledge graph. It manages the tension between incoming ideas and established knowledge through a process called **Rationalisation** — determining how a new structure relates to what is already known, and deciding what to do about it.

The system operates on a fundamental distinction: **nodes** (uint64 values) are the atoms — static, opaque integers. **KLines** are the structures — dynamic arrangements of nodes into identified, ordered sequences. The model grows not by creating new kinds of matter, but by composing existing matter into increasingly complex hierarchies, and by refining how those compositions relate to each other.

## 2. The Ontological Foundation

The system is built upon a small, precise set of concepts:

### Nodes

A **node** is a 64-bit unsigned integer. Nodes are opaque — the system does not inspect or interpret them beyond bitwise operations. They come in two flavours:

- **Literal nodes** — bit 0 is set (`node & 1 == 1`). These represent exact, atomic tokens (individual characters or raw values). They preserve identity and order. Literal nodes _do not_ participate in signature construction.
- **Non-literal (packed) nodes** — bit 0 is clear (`node & 1 == 0`). These carry structural or type information. They _do_ participate in signature construction via bitwise OR.

### KLines

A **KLine** is the fundamental unit of the knowledge graph. It consists of:

| Field     | Type                   | Description                                 |
| --------- | ---------------------- | ------------------------------------------- |
| signature | uint64                 | Identity key. The "name" of this structure. |
| nodes     | ordered list of uint64 | Zero or more child nodes. Order matters.    |
| literal   | bool                   | Whether this represents an exact token.     |

KLines form a directed graph: when a node value equals the signature of another KLine, that's an **edge**. A KLine's nodes _reference_ other KLines, creating compositional hierarchies.

KLine equality is defined as: same signature **and** same node sequence (same length, same order, same values). The `literal` flag does not participate in equality.

### Signatures

A **signature** is a 64-bit unsigned integer whose bit 0 is clear (`node & 1 == 0`). Signatures serve two roles: as kline heads (identifying a kline) and as kline nodes (referencing other klines, forming graph edges). Signatures are created by OR-reduction of non-literal nodes — see the @signature spec for full definition.

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

## 3. The Engine of Agency: Significance

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

## 4. The Process of Rationalisation

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

Structural fast-paths that bypass the full significance pipeline:

- **Unsigned**: zero nodes → S4, no information content.
- **All-literal**: every node is literal → S1, pure token sequence.
- **Self-grounded**: every non-literal node resolves to an existing KLine in the model → S1, fully grounded composition.

If none apply, proceed to candidate retrieval.

### Phase 4: Retrieve Candidates

Find all KLines in the model whose signatures share at least one bit with the query's signature (bitwise AND ≠ 0). If no candidates are found, the result is S4 (novel) — the KLine is added and promoted.

### Phase 5: Compute Significance

For each candidate, run the significance pipeline: per-node S1 testing, routing, distance calculation, inversion. Collect all `(candidate, significance, level)` triples.

### Phase 6: Integrate

Add the KLine to the model (both STM and frame). Select the best result (highest significance value). If S1 or S4, promote to the base model. If S2 or S3, queue for cogitation.

## 5. Cogitation

Cogitation is background processing of rational KLines (S2/S3). It performs deeper graph traversal to find candidates that the initial bitwise-AND retrieval may have missed:

```
Cogitate(Q):
  1. Expand Q's graph context:
     candidates = model.query(Q.signature, depth=D_cogitate)

  2. For each candidate, run significance pipeline.

  3. If any candidate achieves S1, add it to the model.

  4. Re-rationalise Q.
```

`D_cogitate` (default 2) controls traversal depth. If Q has been cogitated more than a configurable maximum (default 3 passes) without reaching a significant result, it is abandoned. The cogitation thread runs asynchronously and emits a `"done"` event when the backlog has been empty for a timeout (default 2 seconds).

## 6. Tokenizers

Two tokenizer types produce the same output kind — typed nodes:

- **Mod (Modular)** — maps characters to bit positions. Multi-character strings are OR-ed into a single packed node (lossy: order and multiplicity lost). Individual characters can also be encoded as literal nodes (preserving identity and order). Variants: Mod32 (31 character bits), Mod64 (63 character bits).
- **BPE (Byte-Pair Encoding)** — learns subword vocabulary from a training corpus. Tokens are sequential IDs combined with **type prefixes** (linguistic properties like part-of-speech) at the agent layer: `node = type_prefix | token_id`. BPE tokens are never literal.

Signatures are constructed from tokenizer output via `make_signature`, defined in the @signature spec.

## 7. Supporting Mechanisms

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

## 8. The Governing Properties

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

## 9. Open Questions

The following have TBD semantics in the current specs:

1. **`model.is_s1(node, candidate)`** — What does "perfect match" mean for a single node against a candidate? The routing depends on this, but the comparison semantics are undefined.
2. **`model.s2_distance(query, candidate)`** — How is partial-alignment distance computed? Must return a value in `[1, D_boundary)`.
3. **`model.s3_distance(query, candidate)`** — How is weak-recognition distance computed? Must return a value in `[D_boundary, MAX)`.
4. **Candidate retrieval efficiency** — `model.where(predicate)` performs a linear scan. A dedicated `candidates_for(signature)` method with an inverted bit-to-signature index may be needed for large models.

These gaps mean the significance pipeline's routing framework is fully specified, but the actual comparison mechanics inside the model are not yet defined.
