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
The distinction is an encoding concern — it affects _how_ nodes contribute
(literal nodes contribute bit 0; non-literal nodes contribute their full
value), not _whether_ they contribute.

### KLines

A **KLine** is the fundamental unit of the knowledge graph. It consists of:

| Field     | Type                   | Description                                 |
| --------- | ---------------------- | ------------------------------------------- |
| signature | uint64                 | Identity key. The "name" of this structure. |
| nodes     | ordered list of uint64 | Zero or more child nodes. Order matters.    |

KLines form a directed graph: when a node value equals the signature of another KLine, that's an **edge**. A KLine's nodes _reference_ other KLines, creating compositional hierarchies.

KLine equality is defined as: same signature **and** same node sequence (same length, same order, same values).

A KLine is **literal** when all of its nodes are literal tokens (per `is_literal`,
defined in the @kline spec as a standalone bit-layout test). This is a computed
property, not stored. Literal KLines represent exact, atomic token sequences;
non-literal KLines are composed structures whose nodes reference other KLines.

### Signatures

A **signature** is a 64-bit unsigned integer. Bit 0 is the **literal-content flag** — set when the kline contains literal nodes. Signatures serve two roles: as kline heads (identifying a kline) and as kline nodes (referencing other klines, forming graph edges). Signatures are created by `make_signature` — an OR-reduction over all nodes. See the @signature spec for full definition.

### The Model (Knowledge Graph)

The **Model** is the collective graph of all KLines. It provides storage, indexing, deduplication, lookup by signature, graph traversal, and the comparison functions consumed by the significance pipeline.

The model has a **three-tier memory architecture**, invisible to callers:

```
STM → Frame → Base
```

| Tier  | Purpose               | Bounded   | Lifetime       | Written by add | Written by promote |
| ----- | --------------------- | --------- | -------------- | -------------- | ------------------ |
| STM   | Transitive grounding  | Yes (256) | Rolling window | Yes            | No                 |
| Frame | Session write surface | No        | Per-session    | No             | Yes (from STM)     |
| Base  | Long-term knowledge   | No        | Persistent     | No             | No                  |

- **STM** (Short-Term Memory) is a bounded, dual-keyed index over the most recently added KLines. It indexes each KLine by both its signature _and_ its nodes signature, enabling **transitive grounding** — finding KLines that share node structure even when their signatures differ. When the bound is exceeded, the oldest entries are evicted. See the @stm spec for full definition.
- **Frame** is populated by **promotion** from STM. When the agent determines a KLine is significant (S1 or S4), it promotes the KLine from STM to the frame. Lookups that miss in STM fall through to the frame.
- **Base** is an optional long-term knowledge store. It is **read-only** during a session — established at model instantiation and never modified by `add` or `promote`. Each session layers a fresh model over a shared base, giving isolated writes with shared knowledge.

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
| S2    | Some nodes match, some don't         | Low (hop count)   | High             | Queue for cogitation |
| S3    | No nodes match, but candidates exist | High (biased)     | Low              | Queue for cogitation |
| S4    | No candidates found at all           | MAX               | 0 (all bits 0)   | Novel. Promote.      |

Distance is a single integer accumulated from graph hops. S3 connotation hops
are packed via `_pack(hop_count + _S3_BIAS)` (quadratic with tier bias),
ensuring S3 distances moderately exceed S2 distances while keeping both
tiers close enough for bridging via graph expansion. The `_pack` function (d²)
compresses small distances together and spreads large distances apart.

Key insight: **S1 and S4 are "significants"** — the KLine is either confirmed or entirely novel. No further processing needed. **S2 and S3 are "rationals"** — partial relationships that require deeper investigation through **cogitation**.

### Candidate Retrieval

Candidates are found via **bitwise AND matching**:

```
candidates = model.where(k => (k.signature & query.signature) != 0)
```

A KLine whose signature shares _any_ set bit with the query's signature is a candidate. This is a necessary but not sufficient condition — it pre-filters the model before the more expensive significance pipeline runs.

### Significance Routing

Before a distance calculation can be applied to establish significance between two KLines, a simple routing algorithm is applied.

1. For each node in the query KLine, test: does this node exist in the candidate?.
2. Count the matched nodes and **route**:
   - All nodes match → distance = 0 → **S1**
   - Some nodes match → `model.expand(query, candidate)` → **S2**
   - No nodes match → `model.expand(query, candidate)` → **S3**

This routing is **pessimistic**: the presence of any node in either KLine that doesn't exist in the other pulls the overall level down. Only S2 and S3 routes invoke `model.expand()`.

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
│    → Countersigned: S1, done.                    │
│    → Otherwise: proceed to retrieval.            │
├─────────────────────────────────────────────────┤
│ 4. RETRIEVE CANDIDATES                           │
│    Find KLines with signature AND overlap.       │
│    → No candidates: S4 (novel), done.            │
├─────────────────────────────────────────────────┤
│ 5. COMPUTE SIGNIFICANCE                         │
│    Route → expand → invert.                       │
├─────────────────────────────────────────────────┤
│ 6. INTEGRATE                                    │
│    Add Q to model. Act on best result.           │
│    → S1/S4: promote to frame.                   │
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
- **Countersigned — ratified** (S1): another kline in the model vouches for this one structurally — its signature equals `make_signature(nodes)` and its sole node equals the kline's signature. Ratification is checked in the fast lane before candidates are retrieved.
- **Ground** (S1): exact duplicate already in the model (handled in Phase 2).

If none apply, proceed to candidate retrieval.

### Phase 4: Retrieve Candidates

Find all KLines in the model whose signatures share at least one bit with the query's signature (bitwise AND ≠ 0). If no candidates are found, the result is S4 (novel) — the KLine is added and promoted.

### Phase 5: Compute Significance

For each candidate, run the significance pipeline: routing, graph expansion, distance extraction, inversion. Collect all `(candidate, significance)` triples.

### Phase 6: Integrate

Add the KLine to the model (STM only). Select the best result (highest significance value). If S1 or S4, promote to the frame. If S2 or S3, queue for cogitation.

## 6. Cogitation

Cogitation is background processing of rational KLines (S2/S3). The Cogitator
receives pre-routed work items, expands each through `model.expand()` to
discover intermediate connotations, and classifies each result against
significance boundaries. Expansion yields are consumed as a **stream** —
all yields from `model.expand()` are processed without filtering.

### Response Processing

The Cogitator processes all yields from the `expand()` stream without
filtering. Every connotation discovered during graph expansion is evaluated,
and proposals can be emitted at any significance level. S1 and S4 are
"significants" — confirmed or novel — and are acted on immediately.
S2 and S3 are "rationals" — partial relationships that undergo expansion.

#### Significance Boundaries

Three fixed boundaries classify yielded significance values:

```
D_MAX ── S1|S2 ──────── S2|S3 ──────────── S3|S4 ── 0
```

| Boundary | Position                    | Meaning                        |
| -------- | --------------------------- | ------------------------------ |
| S1\|S2   | `D_MAX - 1`                 | Only exact S1 qualifies as S1  |
| S2\|S3   | `~_S2_S3_DISTANCE`         | Packed distance threshold (100)  |
| S3\|S4   | `0`                          | Only zero-significance is S4   |

Classification is a cascade:

```
  sig >= S1|S2  →  S1
  sig >= S2|S3  →  S2
  sig >= S3|S4  →  S3
  else          →  S4
```

Raw significance values are never mutated. Boundaries are fixed by the
distance algorithm and are not parameterized.

#### All Yields Processed

The Cogitator processes all connotations yielded by `expand()`. There is no
truncation or early stopping — every discovered relationship is evaluated for
countersignature and expansion. This ensures the teacher receives the fullest
possible set of proposals for each work item.

### Streaming Pipeline

```
run_work_item(WorkItem(query, candidate)):
  for qc in model.expand(query, candidate):
    if qc.significance >= s12:
      on_s1(query, candidate)     # S1: promote immediately
    else:
      process(qc)                 # S2/S3: expansion check
```

### Countersignature

Countersignature (ratification) is checked in the fast lane during
`rationalise()` Phase 3 (Assess), before candidates are selected. A kline
is countersigned if its `nodes_signature` exists as another kline in the
model with one node — the countersigned kline's signature.

The Cogitator's `process()` handles only S2 expansion: reshaping misfit
klines toward canonical status and emitting proposals for teacher ratification.

### S2 Expansion

The Cogitator **expands** each S2/S3 candidate kline toward canonical
status by reshaping its nodes to match its signature. This is the mechanism
for self-directed study.

For a candidate with signature `S` and nodes signature `N`:

- **Underfitting** (`S & ~N != 0`): search the model for klines whose
  signatures contribute to the gap `S & ~N`, add their nodes.
- **Overfitting** (`N & ~S != 0`): remove excess nodes, verify the removed
  group's signature exists in the model.
- **Dual misfit**: both may apply to the same candidate.

Every signature generated during expansion must already exist in the
model (no invention, no data loss, ratifiability guaranteed). All expansion
proposals are emitted as `frame` events requiring teacher ratification.

### Reentrant Rationalisation

When `process()` discovers an S1 via boundary classification and calls
`on_s1()` → `agent.rationalise(query)`, that re-rationalisation may produce
new S2/S3 candidates that are appended to the backlog. Countersignature
(ratification) is checked during re-rationalisation in the fast lane
(Phase 3: Assess). The streaming approach handles this naturally: each
`expand()` call operates on the model state at the time of the yield, and
re-rationalisation adds new work items processed in their own turn.

The cogitation thread runs asynchronously and emits a `"done"` event when
the backlog has been empty for a timeout (default 2 seconds).

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

Promotion moves KLines from STM to the frame, persisting them for the
session. The agent triggers promotion on significant results (S1 =
confirmed knowledge, S4 = novel knowledge). The base model is never
modified — it is established at model instantiation and remains read-only.

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

Promotion is the bridge: only S1 (confirmed) and S4 (novel) KLines are promoted from STM to the frame, ensuring the frame grows only through validated knowledge.

### Bitwise AND as Necessary Condition

Candidate retrieval uses bitwise AND on signatures as a fast pre-filter. This is lossy — it can miss relevant candidates (false negatives) and include irrelevant ones (false positives). The significance pipeline handles the precision; the AND filter handles the recall.

## 10. Open Questions

The following have TBD semantics in the current specs:

1. ~~**`model.is_s1(node)`**~~ — Resolved. Renamed to `model.is_s1(kline)`: structural grounding check — a kline is S1 if canonical (`make_signature(nodes) == signature`) or countersigned.
2. ~~**`model.s2_distance(query, candidate)`**~~ — Resolved. Replaced by `model.expand(query, candidate)` with per-node hop-distance and connotation bridging. See @model spec.
3. ~~**`model.s3_distance(query, candidate)`**~~ — Resolved. Replaced by `model.expand(query, candidate)` with connotation bridging for indirect node connections. See @model spec.
4. **Candidate retrieval efficiency** — `model.where(predicate)` performs a linear scan. A dedicated `candidates_for(signature)` method with an inverted bit-to-signature index may be needed for large models.
5. **Mod literal encoding capacity** — The new literal encoding `(char << 32) | 0xFFFFFFFF` places the character code point in the upper 32 bits, limiting code points to the range [0, 2³²−1]. This covers the entire Unicode range (max code point 0x10FFFF) with substantial headroom.
