# Kalvin — High-Level Overview

## What It Is

Kalvin is an **autonomous knowledge agent**. It receives new pieces of information, evaluates how each relates to what it already knows, and decides on its own what to do next. That capacity for autonomous decision-making is called **Agency**, and it is the defining property of the system.

Kalvin is not a language model, a database, or a search engine. It is a self-contained system whose intelligence emerges from the structure of its knowledge graph and a mathematical evaluation of similarity.

---

## The Problem It Solves

Most knowledge systems fall into one of two traps:

1. **Exact-match systems** (databases, hash tables) only find what they already have. Anything even slightly different is treated as completely novel, even if it is structurally almost identical to existing knowledge.

2. **Statistical systems** (neural networks, vector databases) find fuzzy matches but have no principled way to distinguish "I already know this" from "this reminds me of something" from "this is genuinely new." They blur the distinction.

Kalvin provides **graduated comprehension**. When confronted with a new idea, the system places it on a precise spectrum:

| Level | Internal Voice | What Happens |
|-------|---------------|--------------|
| **S1** | "I fully understand this." | Confirmed. Integrated permanently. |
| **S2** | "I understand some of it." | Investigated further in the background. |
| **S3** | "This is reminiscent." | Investigated further in the background. |
| **S4** | "This is completely new to me." | Novel. Integrated permanently. |

These are not arbitrary labels or heuristic thresholds. They are the direct mathematical consequence of comparing the structure of a new idea against the topology of the existing knowledge graph.

---

## How It Works

### The Core Ideas

**Nodes** are the atoms — opaque 64-bit numbers. The system never inspects what they "mean." It only compares them.

**KLines** are the structures — identified, ordered sequences of nodes. A KLine is the fundamental unit of knowledge. When a node in one KLine equals the identity of another KLine, that is an edge in the knowledge graph. KLines compose hierarchically: structures reference other structures, building up from simple to complex.

**Signatures** are the identity keys. Every KLine has a signature — a single number derived from its nodes through bitwise OR. Signatures serve double duty: they identify a KLine (like a name) and they enable fast approximate matching (like a fingerprint). Two signatures that share any set bit are potential matches.

### The Knowledge Graph

The **Model** is a three-tier knowledge store:

- **Short-Term Memory (STM):** A bounded rolling window of the most recently seen KLines. This is the frontier of activity — everything new enters here first.
- **Frame:** The session's confirmed knowledge. KLines are promoted here from STM when they are deemed significant (S1 or S4).
- **Base:** Long-term accumulated knowledge, shared across sessions. Read-only during a session.

Lookups cascade: STM → Frame → Base. The caller sees a single unified collection.

### The Rationalisation Pipeline

When a new KLine arrives, the agent runs it through a six-phase pipeline:

1. **Prepare** — assign a signature if the KLine doesn't have one yet.
2. **Ground check** — is this already known? If so, acknowledge and stop.
3. **Assess** — can we classify this without searching? Empty KLines are S4. All-literal KLines are S1. Self-grounded KLines (every node resolves to existing knowledge) are S1.
4. **Retrieve candidates** — find existing KLines whose signatures overlap with the new one.
5. **Compute significance** — for each candidate, measure how well it matches by traversing the graph and accumulating distance.
6. **Integrate** — add the KLine. S1 and S4 are promoted to the frame immediately. S2 and S3 are queued for deeper investigation.

This pipeline has a **fast path** (phases 1–4, plus S1/S4 in phase 5) and a **slow path** (S2/S3 candidates sent to background processing).

### How Distance Becomes Significance

The matching process works by measuring the graph distance between two KLines — specifically, how many hops through the knowledge graph it takes to connect mismatched nodes. Shorter distance means closer match.

Distance is then inverted into **significance**: `significance = NOT distance`. This is not a cosmetic transformation — it means the natural ordering of unsigned integers directly reflects the agent's comprehension:

```
S1 (full comprehension) = highest number
S4 (complete novelty)   = zero
```

The significance spectrum has three boundaries separating S1 from S2, S2 from S3, and S3 from S4. These boundaries can be shifted by a **temperature** parameter:

- **High temperature** lowers the bar for comprehension — S2 results may be promoted to S1, S3 to S2. The agent is more permissive, more exploratory.
- **Low temperature** raises the bar — only very close matches qualify. The agent is more conservative.

Two additional sampling parameters control how deeply the agent investigates each ambiguous candidate:

- **Top-k** limits the number of intermediate results explored per candidate (the exploration budget).
- **Top-p** stops exploration when accumulated evidence crosses a threshold (the evidence quality gate).

### Cogitation

The **Cogitator** is a background processor that handles the S2/S3 work items. For each item, it expands the graph around the query-candidate pair, discovers intermediate relationships (called *connotations*), and checks for a special structural relationship called **countersignature** — when two KLines mutually reference each other through their nodes.

Countersignature is the primary mechanism by which cogitation resolves ambiguity: a KLine that appeared to be a partial match (S2) may, upon deeper graph traversal, turn out to be part of a mutual cross-reference and be promoted to full comprehension (S1).

When cogitation discovers new relationships, it re-rationalises the original query, which may produce new work items. This creates a self-reinforcing cycle of understanding that eventually settles (the cogitator emits a "done" event when its backlog has been empty for a timeout period).

### The KScript Language

**KScript** is a domain-specific language for constructing knowledge graphs declaratively. It compiles human-readable scripts into the KLine entries that populate the model.

KScript provides operators that map directly to significance levels:

| Operator | Meaning | Level |
|----------|---------|-------|
| `==` | Mutual bidirectional link | S1 |
| `=` | Unconditional link | S1 |
| `=>` | Canonical forward link | S2 |
| `<=` | Canonical backward link | S2 |
| `>` | Connotative forward link | S3 |
| `<` | Connotative backward link | S3 |
| (bare name) | Identity only | S4 |

For example, a simple knowledge graph about sentence structure:

```
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A = D
     L = M
     L > O
```

This declares that MHALL and SVO mutually reference each other (S1), that S/V/O are linked to their components (S1), and that the ALL structure canonically contains A and L (S2), with L connotatively relating to O (S3).

### Tokenization

Text enters the system through **tokenizers** that convert characters into the 64-bit nodes the graph operates on. Two approaches are supported:

- **Mod (Modular):** Maps characters to bit positions. Multi-character strings are bitwise-OR'd into a single node (lossy — order is lost, but fast for matching). Individual characters can also be encoded as *literal* nodes that preserve exact identity and order.
- **BPE (Byte-Pair Encoding):** Learns subword vocabulary from training data. Tokens carry type information (part-of-speech, etc.) through bit prefixes assigned at the agent layer.

Both produce the same kind of output — 64-bit nodes — and both feed into the same signature and significance pipeline.

---

## Key Design Properties

**Pessimism.** The system is conservative in its assessments. Even if most of a new KLine matches perfectly, the presence of any single mismatched element pulls the overall significance down. It takes full alignment to achieve S1.

**Structure, not statistics.** Significance is computed purely from graph topology — how many hops, how direct the connections, how well the structures compose. There are no learned weights, no embeddings, no probabilistic models.

**Inverted metric.** Significance is the bitwise NOT of distance. This makes the four levels fall out naturally from unsigned integer comparison. No separate level encoding is needed.

**Agency as emergence.** The agent's capacity to choose — to confirm, to investigate, to treat as novel — is not programmed as a set of rules. It arises from the interaction of signatures, graph topology, and the distance function. The system's behaviour is a consequence of its structure.

**Temporal awareness.** The three-tier memory (STM → Frame → Base) gives the system a natural sense of time. Recent activity is prioritized in STM. Confirmed knowledge persists in the Frame. Accumulated wisdom lives in the Base. Promotion is the bridge: only validated knowledge (S1 confirmed, S4 novel) moves from transient to persistent.

---

## What It Is Not

- **Not a language model.** Kalvin does not generate text. It evaluates and integrates structural knowledge.
- **Not a neural network.** There are no learned parameters, no training loop, no gradient descent. Intelligence comes from graph structure, not statistical pattern matching.
- **Not a database.** While the model stores and retrieves data, its purpose is evaluation, not persistence. The significance pipeline is the core, not the storage layer.
- **Not a chatbot.** There is no conversational interface. The system is a reasoning engine that can be embedded in larger applications.

---

## Component Map

The system is built from a small set of components, each with a clean responsibility:

| Component | Role |
|-----------|------|
| **KLine** | The fundamental unit of knowledge — an identified, ordered sequence of nodes. |
| **Signature** | OR-reduction of nodes into a single identity key with a literal-content flag. |
| **Tokenizer** (Mod / BPE) | Converts text to nodes and back. |
| **STM** | Bounded rolling window for recent KLines, indexed by signature and nodes-signature. |
| **Model** | Three-tier knowledge graph (STM → Frame → Base) with graph traversal and significance computation. |
| **Agent** | Orchestrates the rationalisation pipeline — receives KLines, evaluates significance, integrates results. |
| **Cogitator** | Background processor for ambiguous (S2/S3) results, discovering deeper relationships. |
| **KScript** | Domain-specific language for declaratively constructing knowledge graphs. |
| **Events** | Pub/sub mechanism allowing observers to react to rationalisation outcomes. |
