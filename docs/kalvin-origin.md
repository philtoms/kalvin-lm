# Kalvin — Origin

This is the origin document for the Kalvin system. It defines what Kalvin is, how it thinks, and how it is taught. Every other document in this project takes its understanding of Kalvin from here. If another document conflicts with this one, this document is authoritative.

---

Kalvin is a rationalising system that accepts, thinks, and talks in klines. It receives kline structures — compiled from KScript by an agent — and attempts to understand each one in terms of what it already knows. Understanding, for Kalvin, is not a matter of semantics. It is a matter of shape.

---

## Klines

A kline is a node-like structure. Klines combine to describe the world in which Kalvin operates. There are only two kinds: **identities** and **relationships**. An identity kline is a node with no further links — it simply exists. A relationship kline is a node that links to other klines.

Every relationship kline has a **query** (its head node) and a **value** (the remaining nodes). Training Kalvin to understand some aspect of the world therefore means teaching it to construct that understanding as a graph of kline relationships and identities.

Kalvin's world is built entirely from these two kinds of structures. It has no access to the scripts that produced them, no concept of the characters or words they represent, and no sense of the linguistic or semantic relationships that might be obvious to a human reader. Where we see meaning in the idea of a subject or a verb, Kalvin sees meaning in how one kline connects to another. Where we might say "M stands for Mary," Kalvin sees only that M is linked to S, and S is linked to SVO, and SVO is linked to MHALL. This is not a limitation. It is the point. Meaning for Kalvin is found in shape.

---

## KScript

KScript is a small DSL for compiling scripted kline constructions into training data for a Kalvin system. It targets the Mod32Tokenizer, which supports a limited vocabulary of single-character tokens. The language provides only a small set of operators that describe kline topological structure — it cannot express semantic relationships. This is by design: scripts, constrained by vocabulary and topology alike, keep Kalvin's training focused on recognising and understanding kline relationships.

KScript provides four ways to build kline relationships. For brevity, identity klines are omitted from the table below:

| Syntax | Name | Output |
|--------|------|--------|
| `Q == V` | COUNTERSIGN | `{Q: [V]}, {V: [Q]}` |
| `Q = V` | UNDERSIGN | `{V: [Q]}` |
| `Q > V` | CONNOTATE | `{Q: [V]}` |
| `Q => V₁…Vₙ` | CANONIZE | `{Q: [V₁,…Vₙ]}` |

KScript also supports indented chaining, whereby a value node on one line becomes the query of an indented block that follows. Comments are allowed between brackets.

### A Typical Script

```
    (Mary had a little lamb - anything in brackets is a comment btw)
    MHALL = SVO =>
      M = S(ubject)
      H = V(erb)
      ALL = O(bject) =>
        A > D(et)
        L > M(od)
        L > O
```

The parenthetical annotations serve double duty: they are KScript comments that also convey human-readable meaning. The training value of this script is not to teach Kalvin the nursery rhyme, or even rudimentary linguistics. It is to establish structural relationships that Kalvin can later reconstruct in pursuit of understanding. The granularity of understanding is intentionally subjective. We see a nursery rhyme broken down and explained as SVO. Kalvin sees relationships between opaque numbers. Meaning for us is found in semantics. Meaning for Kalvin is found in shape.

### Compilation

The simplest query-value script defines an acyclic relationship between two identities, Q and V:

```
    Q > V
```

This compiles to three klines:

```
  [
    {Q: [V]},
    {Q: None},
    {V: None},
  ]
```

The first captures the relationship between Q and V as nodes linking to their respective identity klines. The symbols Q and V represent the single-value encodings for those characters. The remaining two klines are the identities.

---

## Significance

When Kalvin receives a kline, it **rationalises** it: it searches its existing knowledge for klines that share structure with the new arrival, measures how closely they relate, and produces a response — a freshly generated kline that represents its best understanding. Attached to this response is a single numerical indicator called **significance**, which expresses how — and how much — the new kline relates to what Kalvin already knows. The higher the significance, the more certainty Kalvin attaches to the relationship it has constructed between the original query, its internal knowledge, and the generated response.

Significance falls on a spectrum. At one end, every node resolves to grounded knowledge — Kalvin fully understands what it has received. At the other end, nothing connects at all — the kline is entirely novel. Between these two extremes lie degrees of partial understanding: some nodes match, some don't, or the connections are associative rather than direct. In every case, significance is not a label Kalvin assigns after the fact. It is a direct consequence of how the new kline fits — or doesn't — into the shape of Kalvin's existing knowledge.

### The Four Levels

In practice, significance falls into four levels, each with a distinct structural meaning:

| Level | Internal Voice | What Kalvin Has Determined |
|-------|---------------|----------------------------|
| **S1** | "I know that I know this." | Every node resolves to grounded, ratified knowledge. |
| **S2** | "I understand some of it." | Some nodes match existing knowledge; others need further resolution. |
| **S3** | "I recognise aspects of it." | No direct node matches, but the graph topology connects the query to existing knowledge. |
| **S4** | "I do not understand this at all." | No candidates found. Completely novel. |

S1 and S4 are **significants** — the kline is either confirmed or entirely novel. No further processing needed. S2 and S3 are **rationals** — partial relationships that Kalvin continues to work on.

---

## Proposals

Kalvin can publish proposals at any significance level. An S2 proposal is not a failed S1 — it is a legitimate expression of partial understanding. An S3 proposal is not noise — it is an associative hint. This is fundamentally different from systems that suppress results below a confidence threshold. In Kalvin, every result is a proposal, and the agent decides what to do with it. The agent may:

- **Accept** a low-significance proposal as sufficient for now.
- **Reject** a high-significance proposal that doesn't match expectations — significance measures structural similarity, not correctness.
- **Instruct** by providing new klines that fill in the gaps.

---

## The Dialog

From Kalvin's perspective, the world is a dialog. Klines arrive, one after another. Some are familiar — Kalvin recognises the shape immediately, responds with high significance, and the exchange is straightforward. Others are partially familiar — Kalvin can trace some of the new kline's nodes through its existing knowledge, but not all. It responds with what it understands and attaches a significance that honestly reflects its confidence. And sometimes a kline arrives that Kalvin cannot connect to anything at all. It says so — low significance, no match, I don't understand. This is not a failure. It is an honest answer, and it is often the most useful one. A low-significance response tells the other side of the dialog exactly where Kalvin's knowledge ends, which is precisely where the next kline should begin.

Consider the exchange:

```
  (What did mary have?)
  WDMH => M H W
```

Kalvin receives this as a compiled kline. It recognises M and H — they connect to S and V, which connect to SVO, which connects to MHALL. Kalvin can trace a path from M and H back to MHALL and responds accordingly. But W does not connect to anything. The gap is visible in the significance: the response is partially grounded but not fully. Kalvin says what it knows and indicates where it doesn't. The next kline to arrive might connect W to something Kalvin already knows — perhaps W to ALL, or W to O. If so, Kalvin integrates the new relationship, the gap closes, and a similar exchange in future would route directly to high significance.

Given the relationships established by the earlier script — `MHALL = SVO`, `M = S`, `H = V`, `ALL = O` — the query `WDMH => M H W` traces M and H back through subject and verb to MHALL, yielding MHALL as the response. The intention is for Kalvin to respond MHALL with a high significance value. However, it is very likely that Kalvin would attach low significance to this kline if it has not found a way to handle the W in WDMH. The agent will clearly see that the response MHALL mismatches the expected response MHW and will be able to construct a new script that captures the relationship between W and ALL. In the long run it matters little whether the agent chooses to teach Kalvin that W UNDERSIGNs ALL (`W = ALL`) or W CONNOTATEs O (`W > O`) — the same object that L(amb) connotates. Both relationships help Kalvin rationalise a highly significant answer to the question: Does Kalvin know that Mary had a little lamb?

---

## Ratification — Knowing That You Know

There is a hard limit to Kalvin's autonomy. **Agency can take significance as far as S2, but no further.** Full S1 — truly knowing — can only be reached through ratification by the agent.

S1 is not merely high confidence. It is confirmed knowledge. The difference between S2 and S1 is the difference between a student who has studied thoroughly and believes they understand, and a student whose understanding has been formally recognised. No amount of autonomous cogitation, study, or graph traversal can promote a kline to S1. Kalvin can bring a kline to the very threshold, but it cannot cross that threshold alone.

Ratification is implemented as **countersignature**: the agent rationalises the reciprocal kline, creating a mutual cross-reference that makes the proposal structurally S1. After ratification, all klines involved in the process are promoted to the frame — not just the kline being ratified, but the supporting identity and partial klines as well. This enriches the model for future rationalisation.

Canonical S1 (all-literal or self-grounded klines) is self-ratifying — no countersignature needed. These promote immediately.

---

## Priming and Querying

The agent has two strategic tools. Both go through `rationalise()`, but they differ in intent:

| | Priming | Querying |
|---|---------|----------|
| **Intent** | "Accept this as known." | "Work this out." |
| **Typical route** | S1 (auto-promote) | S2/S3 (cogitation) |
| **Agent action** | Submit and move on | Listen, evaluate, respond |

**Priming** injects trusted knowledge. A COUNTERSIGN (`==`) compiles to two countersigned klines. Submitted sequentially, the second discovers the first and establishes S1. The agent tells Kalvin: this is known.

**Querying** tests understanding. A CANONIZE (`=>`) compiles to a forward link. If the target is unknown, the result routes to S2 or S3. Cogitation follows. Proposals arrive on the event channel. The agent evaluates and decides whether to ratify, scaffold, or re-submit.

The interplay between priming and querying is the essence of curriculum design.

---

## The Training Relationship

Kalvin rationalises every kline it receives in exactly the same way, regardless of context. There is no training mode, no operational flag, no special behaviour triggered by circumstance. A kline arrives, Kalvin searches its knowledge, computes significance, and responds. Whether that kline came from a deliberate training exercise or a live query, Kalvin's process is identical.

What differs is the **harness** — the code that manages the flow of klines and responses between the agent and Kalvin. A training harness compiles scripts and uses significance to decide what to teach next. An operational harness presents live queries and uses significance to decide how to act on the response. Different harnesses for different jobs; Kalvin is unaware of the distinction. It simply rationalises what it receives and reports how well it understood.

Training a Kalvin system involves three components working in concert: a KScript compiler that translates scripts into kline structures, an agent — human or automated — that evaluates Kalvin's responses and decides what to teach next, and a Kalvin instance that rationalises what it receives. A **harness** coordinates these components, managing the flow of klines and responses between them.

When Kalvin responds with less than full S1, the agent has three options:

1. **Confirm.** The proposal is accepted. Partial understanding is sufficient.
2. **Correct.** The proposal is denied. Kalvin has overreached. The system waits for further input.
3. **Instruct.** The agent provides new klines that Kalvin does not yet have — the missing pieces between what Kalvin already knows and what it needs to understand. This is **scaffolding**.

### Scaffolding

Scaffold klines go through the same rationalisation pipeline as any other kline. The agent does not give Kalvin the answer. It gives Kalvin the next structure that is close enough to what Kalvin already knows that it can be grounded, but which also moves closer to understanding the original query. Each scaffold may itself land at S2 or S3, requiring further scaffolding:

```
Query → rationalise → S3 → scaffold → rationalise → S2 → scaffold → rationalise → S1
```

The system learns structure by structure. Each round adds a piece of understanding. Over successive rounds, the original query's significance ascends — not because standards were relaxed, but because the knowledge graph genuinely grew to support it.

### Convergence

Each training round either increases the query's significance or adds new grounded knowledge to the model. The knowledge base only grows, never shrinks. Even if a query never reaches S1, the scaffolding erected along the way remains in the model, making future learning faster.

---

## The Loop

A training harness coordinates the components described above: compiler, agent, and Kalvin instance. Each round of training follows the same pattern: the agent scripts and compiles new klines, Kalvin rationalises them and responds, and the agent evaluates each response on two axes: a) Kalvin's own confidence — does it need additional support? — and b) how well the response fits the expected relationship generated by the compiler. This evaluation determines the next training step, and the loop continues.

The loop proceeds as follows:

1. The agent compiles a KScript and extracts the top-level query and expectation.
2. The agent submits the query to Kalvin via `rationalise()`.
3. Kalvin rationalises and emits events:
   - **S1/S4** — fast path. The event is emitted immediately.
   - **S2/S3** — slow path. Cogitation runs on a background thread. Events are emitted as proposals are discovered.
4. The agent evaluates each proposal against the expectation:
   - **Match** — countersign to ratify. Structural S1 achieved.
   - **Mismatch** — construct new scaffolding and submit it to the same Kalvin instance.
   - **No response** — re-submit the query to benefit from model enrichment.
5. The loop continues until the expectation is met or the agent determines that current understanding is sufficient.

One agent instance is created per training script. All queries for that script operate within the same model — the same STM, frame, and base. No frame stack, no layering.

### Event Correlation

`rationalise()` returns `True` (fast path, event already emitted) or `False` (slow path, cogitation in progress). The agent uses this return value and the event's signature to correlate events with submitted queries.

### MVP: Exact Match

For the MVP, the agent compares proposals against expectations using kline equality (same signature, same node sequence). Future extension may use reentrant rationalisation for structural equivalence rather than byte-identical matching.

---

## Study

When the dialog pauses, Kalvin continues to think. Any kline that landed at partial significance represents material Kalvin has seen but not fully internalised. During these quiet periods, Kalvin revisits its partial knowledge — retracing paths, discovering connections that were not apparent before, strengthening its own understanding. This study cannot fully confirm a kline — full understanding arrives when the dialog resumes and a new kline routes cleanly to high significance — but it prepares. The next exchange may be immediately recognisable, requiring no further scaffolding, because Kalvin has done the work of connecting what it already knows.

Study is not a separate mechanism. It is the existing cogitator processing its S2/S3 backlog on a background thread. Study can strengthen significance — bringing S3 up to S2, or a weak S2 closer to S1 — but it cannot reach S1 alone. It prepares for ratification; the agent confirms it.

### S2 Expansion

S2 klines are **misfits**: their signature does not match their nodes. During study, the cogitator classifies the mismatch and attempts to resolve it, generating proposals for the agent to ratify.

An **underfit** kline has a signature that promises bits its nodes don't deliver — there are holes to fill. The cogitator searches the model for nodes that would close the gap and proposes the expanded kline.

An **overfit** kline carries nodes whose bits the signature doesn't capture — there is excess to shed. The cogitator proposes a trimmed kline and a **companion kline** formed from the removed nodes, giving each piece an independent existence for the agent to ratify.

A **dual misfit** has both problems. The cogitator proposes a single atomic replacement — swap overfit nodes for ones that fill the underfit gap — plus a companion kline for the removed group.

In every case, expansion proposals go through the same ratification pipeline as any other proposal. The cogitator does not invent — it reshapes what is already there and offers the result to the agent. Two guarantees hold: no kline enters the model without agent ratification, and no removed nodes are discarded without being offered as a companion kline.

This is not accidental behaviour. KScript's `=>` operator deliberately creates S2 misfits. Underfit klines act as **templates** — a known concept with holes to fill, allowing Kalvin to match a question with a question-and-answer structure. Overfit klines act as **sequencers** — step-by-step structure under a single goal signature, allowing Kalvin to rationalise a query in discrete steps. The cogitator fills templates and decomposes sequencers, turning partial understanding into proposals that, once ratified, become grounded S1 knowledge.

### Growth

This is how Kalvin grows: not by adjusting weights or minimising error, but by building a knowledge graph one kline at a time. Each exchange adds structure. Each pause strengthens it. Over time, the graph becomes rich enough that new klines fit into familiar shapes, and Kalvin's responses carry the confidence of a system that has seen enough of the world to understand what it is being shown.

---

## Curriculum

A curriculum is the agent's plan for building a knowledge graph. It determines what to submit, when to prime vs. query, what expectations to hold, and how to respond to proposals. Kalvin has no concept of a curriculum — it exists entirely outside the agent.

A curriculum can range from fully reactive to fully scripted:

| Mode | Description |
|------|-------------|
| **Human-in-the-loop** | A human watches the event channel and creates KScripts in response to proposals. |
| **Scripted reactive** | A program monitors events and selects pre-written KScript responses based on significance and content. |
| **Fully scripted** | A fixed pipeline with deterministic submission order and defined coverage targets. |

All three modes use the same API: `rationalise()` and events.

Completion is a trainer decision. Kalvin does not declare itself trained. For the MVP, completion is coverage: the fraction of expectations that have been matched.

### Mary's World (MW)

Mary's World is the reference training curriculum. Its goal is to train an empty Kalvin instance to understand the nursery rhyme "Mary had a little lamb" — specifically, to answer questions like "What colour was the lamb's fleece?"

MW starts from a blank slate: no klines, no tokens, no vocabulary of inquiry. The curriculum proceeds by introducing the rhyme one line at a time. At each step, the agent submits klines, Kalvin rationalises, and the agent uses the event channel to diagnose gaps and scaffold as needed.

MW will establish:

| Question | Why It Matters |
|----------|---------------|
| How many klines are needed to bootstrap from empty to question-answering? | Sets expectations for curriculum scale. |
| What ordering of priming vs. querying is most efficient? | Informs future scripted curricula. |
| What vocabulary of inquiry must be trained alongside domain content? | Determines curriculum scope. |
| Does order-independence hold in practice? | Validates the core design assumption. |

The patterns that emerge from MW will inform future projects, which can build on previously learned structures rather than starting from scratch.

---

## The Operational Harness

The training loop is one instance of a general pattern: an external coordinator that submits klines to Kalvin and acts on the results. In production, the same pattern becomes an operational harness. The distinction is coordinator intent, not system behaviour. Kalvin rationalises the same way regardless.

Any harness — training or operational — must provide:

1. **Encode/decode** — tokenizer-aware kline serialisation.
2. **Event subscription** — receiving and interpreting significance events.
3. **Agency** — acting on significance levels: S1 (accept), S2/S3 (partial), S4 (novelty).

A pre-trained operational Kalvin may encounter novel information. When this happens, the harness can handle it inline (acting as an inline teacher) or hand off to a dedicated trainer process by serializing the relevant klines.

An agentic harness — one capable of autonomous action based on significance — is the production analogue of the human-in-the-loop trainer. It uses the same API; its sophistication is a matter of implementation.

---

## Component Map

The system is built from a small set of components, each with a clean responsibility:

| Component | Role | Spec |
|-----------|------|------|
| **KLine** | The fundamental unit of knowledge — an identified, ordered sequence of nodes. | `specs/kline.md` |
| **Signature** | OR-reduction of nodes into a single identity key with a literal-content flag. | `specs/signature.md` |
| **Tokenizer** (Mod / BPE) | Converts text to nodes and back. | `specs/tokenizer.md` |
| **STM** | Bounded rolling window for recent KLines, indexed by signature and nodes-signature. | `specs/stm.md` |
| **Model** | Three-tier knowledge graph (STM → Frame → Base) with graph traversal and significance computation. | `specs/model.md` |
| **Agent** | Orchestrates the rationalisation pipeline — receives KLines, evaluates significance, integrates results. | `specs/agent.md` |
| **Cogitator** | Background processor for ambiguous (S2/S3) results, including S2 expansion. | `specs/agent.md` |
| **KScript** | Domain-specific language for declaratively constructing knowledge graphs. | `specs/kscript.md` |
| **Events** | Pub/sub mechanism allowing observers to react to rationalisation outcomes. | `specs/agent.md` |
