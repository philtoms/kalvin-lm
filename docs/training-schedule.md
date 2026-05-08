# Kalvin — The Training Schedule

## Overview

This document describes how a Kalvin instance is trained from a practical perspective: what to teach, in what order, and how training transitions into production operation. It sits above the mechanism described in `docs/training-loop.md` (how the training loop works) and the principles in `docs/learning.md` (why learning works the way it does).

The central thesis: **a training schedule is a trainer's plan, not a Kalvin subsystem.** Kalvin has no concept of a curriculum, a lesson, or a schedule. It receives klines through `rationalise()` and responds on the event channel, the same way it always does. The curriculum exists entirely outside the agent — in the trainer's judgment, in a scripted pipeline, or somewhere in between.

---

## Priming and Querying

The trainer has two strategic tools for building a knowledge graph. Both are structurally identical — they go through `rationalise()` — but they differ in intent:

### Priming: Injecting Trusted Knowledge

```
A == B          ← KScript mutual bidirectional link (S1)
```

The `==` operator compiles to two countersigned klines. When submitted sequentially, the second kline discovers the first via `is_countersigned()`:

```python
# agent.py — rationalise() Phase 3: Assess
if self._model.is_countersigned(kline):
    self._model.add(kline)
    self._model.promote(kline)
    self._publish("frame", kline, kline, D_MAX - 1)  # S1
    return True
```

Both auto-promote to S1. The trainer is telling Kalvin: "Accept this as known."

**Priming is order-sensitive.** The first kline enters as novel (S4 or S2); the second finds the first already in the model and establishes the countersigned relationship. The trainer must submit the two halves in sequence.

### Querying: Testing Understanding

```
A => B          ← KScript canonical forward link (S2)
```

The `=>` operator compiles to a forward link. If B is unknown to Kalvin, the
result routes to S2 or S3. Cogitation follows. The event channel carries
proposals. The trainer evaluates these against expectations and decides
whether to countersign, scaffold, or adjust temperature.

The `=>` operator also produces **intentional S2 misfit**, which the
Cogitator can expand through extended cogitation:

| KScript           | Compiled kline                       | Misfit type  |
| ----------------- | ------------------------------------ | ------------ |
| `AB => C`         | `{AB: [C]}` — sig `A\|B`, nodes `C`  | Underfitting  |
| `A => B C`        | `{A: [B, C]}` — sig `A`, nodes `B\|C` | Overfitting   |
| `WDMH => MHALL`   | `{WDMH: [M, H, A, L, L]}`           | Dual misfit   |

Underfitting klines act as **templates** — they have a known concept
(signature) with holes to fill. Overfitting klines act as **sequencers** —
they carry step-by-step structure under a single goal signature.
See `docs/extended-cogitation.md`.

The trainer is asking Kalvin: "Work this out."

### The Strategic Distinction

| Aspect | Priming (`==`) | Querying (`=>`) |
|--------|----------------|-----------------|
| Trainer intent | "Accept this." | "Work this out." |
| Expected route | S1 (auto-promote) | S2/S3 (cogitation) |
| Trainer action | Submit and move on | Listen, evaluate, respond |
| Risk | None — knowledge is trusted | Proposals may not match expectations |
| Use case | Bootstrapping foundational structures | Testing and extending understanding |

The trainer chooses which to use based on the current state of training — and that state is observable through the event channel. Kalvin reliably signifies what it knows and what it doesn't.

These two approaches are not mutually exclusive. A trainer might prime a set of grammatical structures and then query whether Kalvin can compose them into a sentence. The interplay is the essence of curriculum design.

---

## The Training Curriculum

### What a Curriculum Is

A curriculum is the trainer's plan for building a knowledge graph. It determines:

1. **What klines to submit** — the content of each training step.
2. **When to prime vs. query** — whether to inject trusted knowledge or test understanding.
3. **What expectations to hold** — the target kline for each query, against which proposals are compared.
4. **How to respond to proposals** — scaffolding strategy when Kalvin's response doesn't match expectations.

### Flexibility

The curriculum can range from fully reactive to fully scripted:

| Mode | Description | Curriculum artifact |
|------|-------------|-------------------|
| **Human-in-the-loop** | A human trainer watches the event channel and creates new KScripts in response to Kalvin's proposals. No pre-planned sequence. | The trainer's judgment. |
| **Scripted reactive** | A program (in any suitable language) monitors the event channel and selects pre-written KScript responses based on the significance levels and kline content of proposals. | A decision tree or state machine mapping event patterns to KScript files. |
| **Fully scripted** | A fixed pipeline with specific training goals and a deterministic sequence of submissions. Target knowledge thresholds are defined in advance. | An ordered sequence of KScript files with coverage targets. |

All three modes use the same API: `rationalise()` and events. The difference is in the coordinator's sophistication.

### Ordering

The design intent is that Kalvin's learning should be **order-independent** — significance is a function of graph topology, not submission sequence. Two different curricula that teach the same klines should produce equivalent knowledge graphs.

In practice, an **emergent efficient ordering** is expected, the same way human curricula have prerequisites even though a sufficiently motivated student could learn in any order. Some orderings will require fewer scaffolding steps than others.

Ordering is at the **kline level**. The curriculum's finest grain is the individual kline, not the KScript file.

### Completion: "Enough"

For the MVP, completion is **coverage**: the fraction of expectations that have been matched. A scripted curriculum defines a target value (e.g., "95% of expectations matched"). A human-in-the-loop trainer may judge arbitrarily.

"Enough" is always a trainer decision. Kalvin does not declare itself trained.

---

## Mary's World (MW): The Reference Curriculum

### The Scenario

Mary's World (MW) is the reference implementation of a training curriculum. Its goal is to train an **empty** Kalvin instance to understand the nursery rhyme "Mary had a little Lamb" — specifically, to answer questions like "What colour was the lamb's fleece?"

### Starting Conditions

- **Empty knowledge graph.** No klines. No tokens. No understanding of syntax or semantics.
- **Mod32 tokenizer.** Built into the training harness. Each character token is initially novel.
- **No vocabulary of inquiry.** MW does not contain the tokens or structures needed to parse a question, let alone answer one.

### Expected Training Trajectory

The curriculum proceeds by introducing the nursery rhyme one line at a time. At each step:

1. The trainer submits a kline (prime or query).
2. Kalvin rationalises. Most tokens are novel (S4) or partially recognised (S2/S3).
3. Kalvin emits proposals on the event channel.
4. The trainer evaluates the proposals and decides:
   - **Countersign** if the proposal matches expectations.
   - **Scaffold** by priming or querying the missing structures.
   - **Adjust temperature** to widen or narrow the proposal space.
5. Repeat.

Because MW starts empty, the trainer should expect Kalvin to "ask questions" — to produce S2/S3 proposals that reveal exactly what it doesn't understand. This is the system working as designed. The event channel is the trainer's primary diagnostic tool.

### The Vocabulary of Inquiry

Answering a question requires more than knowing the rhyme. It requires:

- **Tokens for question words** — "what," "colour," "fleece" must exist as klines in the graph.
- **Structures for question patterns** — the relationship between a question and its answer must be grounded.
- **Compositional semantics** — the ability to decompose "the lamb's fleece" into a reference to a previously grounded kline.

The curriculum must teach these alongside the domain content. Whether to prime them first and then query the rhyme, or to let them emerge interactively, is a trainer decision. MW is the research project that will establish what works.

### What MW Will Establish

| Question | Why it matters |
|----------|---------------|
| How many klines are needed to bootstrap from empty to question-answering? | Sets expectations for curriculum scale. |
| What ordering of priming vs. querying is most efficient? | Informs future scripted curricula. |
| What vocabulary of inquiry must be trained alongside domain content? | Determines curriculum scope. |
| How does Kalvin signal gaps via S2/S3 proposals? | Enables automated gap detection. |
| Does order-independence hold in practice? | Validates or refines the core design assumption. |

The patterns that emerge from MW will inform future projects, which can base their models on previously learned syntax and semantics rather than starting from scratch. This is the foundation of domain self-organisation.

---

## The Operational Harness

### From Training Loop to Operational Harness

The training loop described in `docs/training-loop.md` is a specific instance of a more general pattern: an external coordinator that submits klines to Kalvin and acts on the results. In production, the same pattern becomes an **operational harness**.

The distinction between "training" and "operations" is a matter of coordinator intent, not system behaviour. Kalvin rationalises the same way regardless.

### The Harness Contract

Any harness — training or operational — must provide:

1. **Encode/decode** — tokenizer-aware kline serialisation. The harness must be able to convert between external representations (text, structured data) and the klines Kalvin operates on.
2. **Event subscription** — significance routing via the event channel. The harness must be able to receive `frame`, `ground`, and `done` events and interpret their significance levels.
3. **Agency** — acting on significance levels. The harness decides what to do with S1 (accept), S2 (partial understanding), S3 (recognition), and S4 (novelty).

### Handling Novel Information in Production

A pre-trained operational Kalvin may encounter novel information. The harness detects this through significance events:

- **S1** — Kalvin understands. The harness proceeds normally.
- **S2/S3** — Kalvin partially understands. The harness has two options:
  1. **Handle inline** — the harness itself scaffolds and countersigns, acting as an inline teacher.
  2. **Hand off** — serialize the relevant klines to a dedicated trainer process.

### The Trainer Handoff

The handoff exploits the fact that the current implementation accepts a frame as a base. The process:

1. Harness detects a knowledge gap (S2/S3 on a production query).
2. Harness serializes the relevant klines (the query and its partial matches).
3. A dedicated trainer process receives the serialized klines and runs a training session, producing new grounded klines.
4. The trained frame becomes the new base for the production instance.

This is conceptually equivalent to **reentrant ratification** — a Kalvin instance being updated by another process that uses the same rationalisation pipeline. Serialisation relaxes the architectural constraint that training must be synchronous.

**Note:** Restarting a re-trained model is not yet implemented. The conceptual model is clear; the operational procedure is TBD.

### Agentic Harnesses

The harness may itself be agentic — capable of autonomous action based on significance levels. This is very likely the way all Kalvin installations will communicate with the outside world. An agentic harness might:

- Act on S1 by integrating the result into a downstream process.
- Act on S2/S3 by constructing and submitting scaffolding klines autonomously.
- Act on S4 by logging novelty or escalating to a human supervisor.
- Hand off complex training tasks to a dedicated trainer when its own scaffolding is insufficient.

The agentic harness is the production analogue of the human-in-the-loop trainer. It uses the same API; its sophistication is a matter of implementation, not architecture.

### Frames in Production

Each agent has a single frame. In production, most queries will route to S1 through the fast path and accumulate in the frame without requiring scaffolding. The frame grows monotonically as new knowledge is grounded.

Frame consolidation (merging a trained frame back into a base model for reuse) is TBD. Currently, frames grow without bound within an agent's session.

---

## Domain and Self-Organisation

### What Is a Knowledge Domain?

Currently, a specific knowledge domain is synonymous with the set of training scripts used to educate Kalvin. It has no structural relevance to Kalvin's internal organisation or operation. The knowledge graph is a single unified structure — there are no explicit domain boundaries.

### Self-Organisation

The nature of STM → Frame → Base is compatible with the idea of a **super-graph**: a knowledge graph large enough to contain multiple domains, where regional specialisation emerges from topology. Within such a graph, **neighbourhoods** — clusters of densely connected klines — would naturally form around training domains.

The current Kalvin implementation is too restricted to utilise domain self-organisation. This is the pressing development direction after the training loop has been implemented. MW will provide the first empirical data on how knowledge structures form and whether neighbourhoods emerge naturally from training.

---

## Relationship to Other Documents

| Document | Relationship |
|----------|-------------|
| `docs/learning.md` | The *principles* — why learning works, what agency means, why S1 requires ratification. |
| `docs/training-loop.md` | The *mechanism* — how the training loop works in terms of existing APIs and data structures. |
| This document | The *practice* — what to teach, how to plan training, and how training becomes operations. |
| `docs/roadmap.md` | The *implementation plan* — challenges, phasing, and dependencies. |
| `docs/extended-cogitation.md` | The *S2 expansion design* — how the Cogitator reshapes partial understanding into proposals. |

The learning principles guarantee that training works. The training loop provides the API surface. The training schedule is what the trainer does with both.

---

## Implications for Implementation

### What Exists

- `rationalise()` — the single entry point for all training and query operations.
- Event bus — the single output channel for all proposals and results.
- Countersignature — the ratification mechanism (both for priming and for accepting proposals).
- KScript — the language for constructing training data.
- Agent construction — one agent per script, no frame factory needed.

### What MW Needs

| Need | Status | Notes |
|------|--------|-------|
| Structural grounding | Not started | S1 determined by structure; promote all participating klines after ratification. Roadmap Challenge 6. |
| Agent construction | Exists | `Agent(model=Model(base=...))` — one agent per script. |
| Teacher class | Not started | External coordinator. Roadmap Challenge 7. |
| KScript compile → rationalise pipeline | Needs verification | KScript compiles to klines; the teacher submits them via `rationalise()`. |
| Mod32 tokenizer in harness | Needs verification | Tokenizer must be available to both the teacher (for encoding) and Kalvin (for significance computation). |
| Curriculum tooling | Deferred to MW findings | No special format needed for MVP; human trainer creates KScripts ad-hoc. |

### What the Operational Harness Needs

| Need | Status | Notes |
|------|--------|-------|
| Kline serialisation | Exists | Klines are already serializable. |
| Base rebase (trained frame → new base) | Not implemented | Conceptual model is clear. |
| Frame consolidation | TBD | Frames grow within agent session; base rebase is conceptual. |
| Agentic harness infrastructure | Future | Depends on MW findings and production requirements. |

---

## Summary

A training schedule is a trainer's plan for building a knowledge graph. Kalvin provides the mechanisms (`rationalise()`, events, countersignature); the trainer provides the strategy (what to prime, what to query, when to scaffold). The curriculum can be human-driven, scripted, or fully automated — all using the same API surface.

The transition from training to operations is continuous, not discrete. The operational harness is the training loop repurposed: it uses the same coordinator pattern, the same event channel, the same ratification semantics. The difference is that the harness is responding to production stimuli rather than curriculum scripts.

Mary's World is the first concrete curriculum. It will establish the patterns, ordering heuristics, and scale expectations that make future scripted curricula possible. It is both a training exercise and a research programme — the proof that the architecture delivers on its promise of order-independent, constructive, ratifiable learning.
