# Learning and Training

This document builds on the introductions to KScript and Kalvin. It describes how Kalvin learns, how training drives that learning, and the mechanisms by which an agent teaches a Kalvin instance.

---

## The Four Levels of Significance

The introductions describe significance as a spectrum. In practice, significance falls into four levels, each with a distinct structural meaning:

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

## The Training Relationship

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

## Ratification: Knowing That You Know

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

## The Loop

A training harness coordinates the components described in the KScript introduction: compiler, agent, and Kalvin instance. The loop proceeds as follows:

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

Between interactions, Kalvin can work on its own. This is the study described in the Kalvin introduction — the autonomous process of revisiting partial knowledge, retracing paths, discovering connections. Study is not a separate mechanism. It is the existing cogitator processing its S2/S3 backlog on a background thread.

Study can strengthen significance — bringing S3 up to S2, or a weak S2 closer to S1 — but it cannot reach S1 alone. It prepares for ratification; the agent confirms it.

S2 expansion extends study with the ability to reshape partial understanding: when countersignature fails, the cogitator attempts to add missing nodes or remove redundant ones, generating proposals that move toward canonical status. See `docs/extended-cogitation.md`.

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

---

## Mary's World (MW)

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
| **Cogitator** | Background processor for ambiguous (S2/S3) results, discovering deeper relationships. | `docs/extended-cogitation.md` |
| **KScript** | Domain-specific language for declaratively constructing knowledge graphs. | `docs/kscript-intro.md`, `specs/kscript.md` |
| **Events** | Pub/sub mechanism allowing observers to react to rationalisation outcomes. | `specs/agent.md` |
