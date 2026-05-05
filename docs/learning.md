# Kalvin — Learning Through Inference and Feedback

## What Makes Kalvin Trainable

Kalvin is a **trainable knowledge system**. It learns by rationalising new information against what it already knows, receiving feedback on its proposals, and iterating until comprehension is achieved. This cycle — propose, evaluate, receive feedback, re-rationalise — is the mechanism by which Kalvin grows its knowledge base from incomplete understanding to grounded certainty.

Training is not a separate mode or an offline batch process. It happens continuously, inline, during every rationalisation. Every query is a learning opportunity. Every feedback signal is a step towards understanding.

---

## The Learning Goal

When Kalvin rationalises a query kline, its goal is singular: **understand this new information in terms of its grounded knowledge base.** It does this by measuring the significance of the query against the model and then working to raise that significance as far as it can.

Significance is graded on a four-level spectrum:

| Level | Internal Voice | Meaning |
|-------|---------------|---------|
| **S1** | "I know that I know this." | Every node in the query resolves to grounded, ratified knowledge. Confirmed by the trainer. |
| **S2** | "I understand some of it." | Some nodes match; others need further resolution. |
| **S3** | "I recognise aspects of it." | No direct node matches, but the graph topology connects the query to existing knowledge. |
| **S4** | "I do not understand this at all." | No candidates found. Completely novel. |

Kalvin's rationalisation process is fundamentally an **ascent**. Given a query, it climbs from wherever it lands on this spectrum towards S1. The ascent may complete in a single pass (fast path), or it may require multiple rounds of cogitation and feedback (slow path).

### Significance as Entropy

A kline's significance is also a measure of its **structural entropy** — the amount of work required to move it towards S1. A highly significant kline (S1) has low entropy: its signature and nodes are tightly coupled, fully grounded. A low-significance kline (S4) has high entropy: there is no structural relationship between the query and anything the system knows.

This dual interpretation — significance as comprehension and significance as entropy — is not a metaphor. It is a mathematical consequence of the distance function. Significance is the bitwise NOT of distance. High significance means low distance means low entropy means high comprehension. The four levels fall out naturally from unsigned integer comparison against calibrated boundaries.

---

## The Role of Temperature

In many systems, temperature is a sampling dial that controls randomness — higher temperature blurs distinctions, lower temperature sharpens them. **This is not how Kalvin uses temperature.**

Without a feedback loop, temperature would be the only mechanism available to bridge the gap between understanding and not understanding. The system would be forced to lower its standards — blur the line — and call something "understood" when it is merely similar. This is the trap that statistical systems fall into: they conflate "reminds me of" with "I know."

Kalvin does not blur. Instead, it uses temperature as a **training aid**. Temperature shifts the boundaries between significance levels:

- **Higher temperature** lowers the bar — S2 results may be promoted towards S1, S3 towards S2. The system is more permissive in what it proposes as understanding.
- **Lower temperature** raises the bar — only very close matches qualify. The system is more conservative.

Critically, a temperature-shifted significance is a **proposal, not a conclusion**. It represents the system saying: "At this temperature, I believe I partially understand this. Here is my proposal." The feedback system then judges whether that proposal is warranted.

This means temperature does not determine understanding. It determines the **space of proposals** that Kalvin is willing to make. Understanding is confirmed only through feedback.

---

## The Training Relationship

The feedback system is what makes Kalvin a learner rather than a classifier. Without it, Kalvin would rationalise, assign significance, and stop. With it, every rationalisation becomes a teaching moment — an exchange between a learner proposing what it thinks it knows and a trainer guiding it towards genuine understanding.

The dynamic is a familiar one. A child reaches for a word they haven't quite mastered and the parent supplies the correction, not as a judgement but as a gentle nudge that fills in what was missing. A student works through a proof and the teacher points out the missing step, not to mark it wrong but to show what connects what the student already knows to what they are trying to reach. In both cases, the learner does the work of understanding; the teacher provides the scaffold.

Kalvin's training relationship works the same way, and it encompasses the same range of formality — from structured, curriculum-driven instruction to informal, moment-by-moment correction.

### How Training Works

When Kalvin proposes a level of understanding that is less than full S1, the trainer has three responses:

1. **Confirm.** The proposal is accepted. The kline is promoted to the frame at the proposed significance level. This is the trainer acknowledging that partial understanding is good enough for now — the way a teacher might say "close enough" when the core idea is sound even if the detail is incomplete.

2. **Correct.** The proposal is denied. The kline remains at its calculated significance. Kalvin has overreached — the temperature was too high, or the topology was misread. This is the trainer saying "not quite" — firm but not punitive. The system simply returns to its pre-proposal state and waits for further input.

3. **Instruct.** The trainer provides **new information** that Kalvin does not yet have, but which it needs in order to reach higher significance. This is the most important response, because it is how teaching becomes learning. The trainer is not evaluating — it is scaffolding.

### Scaffolding: Building Understanding One Structure at a Time

When the trainer instructs, it supplies new klines — additional context, decompositions, related structures, or the missing pieces that sit between what Kalvin already knows and what it is trying to understand. These new klines go through the **exact same rationalisation pipeline** as the original query:

1. They receive signatures.
2. They are grounded against the model.
3. Their significance is computed.
4. They are integrated into the knowledge graph.

This is the educational principle of **scaffolding** made concrete. The trainer does not give Kalvin the answer. It gives Kalvin the next piece of scaffolding — the next structure that is close enough to what Kalvin already knows that it can be grounded, but which also moves the system closer to understanding the original query.

Each scaffold kline may itself land at S2 or S3, requiring further instruction. This is expected and natural — it is how any learner progresses through material that builds on itself:

```
Query → rationalise → S3 → trainer instructs → scaffold klines → rationalise → S2 → trainer instructs → scaffold klines → ... → S1
```

The system learns **structure by structure**, the way a student learns concept by concept. Each round of instruction adds a piece of understanding. Each piece is grounded, evaluated, and integrated. Over successive rounds, the original query's significance ascends — not because the system relaxed its standards, but because the knowledge graph genuinely grew, scaffold by scaffold, to support it.

### The Ascent Converges

Because each training round either increases the query's significance or adds new grounded knowledge to the model, the ascent is monotonic: significance only increases. The process converges when the original query reaches S1 — full, grounded comprehension — or when the trainer determines that current understanding is sufficient.

This convergence property is important. It means Kalvin's learning is **constructive**: it builds understanding rather than searching for it. Each training round adds permanent structure to the knowledge graph. Even if the current query never reaches S1, the scaffolding that was erected along the way remains in the model, making future learning faster and the graph richer.

---

## Agency: Kalvin's Capacity for Self-Directed Learning

Training is not the whole story. A teacher can bring a student to a high level of understanding — confident, almost there, a strong S2 — but there comes a point where further instruction starts to labour the point. The teacher has done what teaching can do. What remains is for the student to take ownership of what they've learned, to practise it, to discover through their own effort that they already know more than they realise.

This is **agency** — Kalvin's capacity to make choices and pursue understanding on its own. It is the system's core principle of autonomy, and it is always present, always available, regardless of whether a trainer is in the loop.

### Agency Is Learnable

Agency is not a fixed property. It is expressed through significance — the same significance that training cultivates — which means it improves as the knowledge graph grows. A Kalvin with a rich, well-grounded model makes better choices than a Kalvin with a sparse one, not because its decision-making logic has changed, but because its significance calculations return higher, more discriminating values. There is more to be significant *about*.

In this sense, agency is learnable in exactly the way that a student's judgement improves with experience. The choices available to a novice and an expert may look the same from the outside, but the expert chooses better because their understanding of the landscape is richer, more nuanced, more finely graded.

### The S1 Boundary: Knowing That You Know

There is a hard limit to Kalvin's autonomy. **Agency can take significance as far as S2, but no further.** Full S1 significance — the state of truly knowing — can only be reached through ratification by the trainer.

This is not a deficiency in Kalvin's reasoning. It is a deliberate and important constraint. S1 is not merely high confidence; it is confirmed knowledge. The difference between S2 and S1 is the difference between a student who has studied thoroughly and believes they understand, and a student who has sat the exam and had that understanding formally recognised. Both may be equally prepared, but only one has the ratification.

This means that no amount of cogitation, study, or graph traversal can autonomously promote a kline to S1. Kalvin can bring a kline to the very threshold — a strong, well-grounded S2 where every node resolves and the distance is minimal — but it cannot cross that threshold alone. It requires the trainer to present the kline once more, and for that presentation to route directly to S1 without cogitation. Only then does Kalvin know that it knows.

### Study: Preparing for Ratification

Between training sessions, Kalvin exercises its agency through **study** — the autonomous process of working through its own partial understanding without a trainer present. This is the system's background cogitation, and it serves the same purpose as homework or private study in a human learner.

When a training session leaves klines at S2 or S3, those klines represent material that Kalvin has been exposed to but has not yet internalised. The trainer has done the teaching; now Kalvin must do the learning. Study is the process by which Kalvin:

- **Revisits** its own partial knowledge, re-rationalising S2 and S3 klines against a model that may have grown since they were last assessed.
- **Discovers connections** between structures that were not apparent during training — the way a student working through practice problems suddenly sees how two concepts relate.
- **Strengthens** significance through its own cogitation, bringing what was S3 up to S2, or a weak S2 up to a strong one, because the intervening graph traversal has closed the gap.

Study cannot reach S1 — but it can bring a kline to the point where the *next* interaction with the trainer routes directly to S1 without scaffolding. This is the student who has done all the homework and arrives at the exam already knowing the answers. The exam does not teach; it ratifies. But without the homework, ratification would not be possible.

This resolves an apparent tension: study is autonomous, yet S1 requires the trainer. The resolution is that study *prepares* for S1; the trainer *confirms* it. Both are necessary. A student who never studies cannot pass the exam no matter how many times they sit it. A student who studies but never sits the exam has understanding without confirmation — belief without knowledge. Kalvin needs both.

### Agency as Choice

At its core, agency is the ability to choose what to do next. Kalvin exercises this choice at every point in the rationalisation pipeline:

- **What to investigate.** Given a set of S2 and S3 candidates from a query, which does Kalvin pursue first? Significance ranks them, but agency determines the exploration strategy — the allocation of cognitive budget.
- **When to stop.** When is a partial understanding sufficient? Kalvin may continue studying until it has strengthened an S2 as far as autonomy allows — preparing it for ratification — or it may decide that the current S2 is strong enough and move on to other work. It cannot autonomously declare S1, but it can judge when a kline is ready to be presented to the trainer for confirmation.
- **What to propose.** When Kalvin forms a response, it is choosing which parts of its knowledge graph to express. A well-trained Kalvin with strong agency makes proposals that are grounded and relevant. A poorly trained one makes proposals that are speculative or disconnected.

These choices are not random. They are governed by significance — and significance is itself a product of the knowledge graph's structure. Agency, therefore, is an emergent property of a well-trained model. It cannot be programmed directly; it must be cultivated through training and exercised through study.

### The Relationship Between Training and Agency

Training and agency are complementary, and neither is sufficient alone:

- **Training without agency** produces a system that can only learn when taught. It is passive — capable of understanding when guided, but incapable of independent discovery or self-correction.
- **Agency without training** produces a system that is autonomous but undisciplined. It can explore, but its explorations are uninformed, its choices unrefined. It may discover things, but it cannot judge the quality of its own discoveries.

Together, they produce a system that learns when taught *and* learns on its own — one that can be guided by a trainer and then trusted to consolidate and extend that guidance through independent study.

---

## Learning as Rationalisation

The core insight is that **learning and rationalisation are the same process, applied recursively.**

A single rationalisation pass produces a significance level and a set of candidates. If significance is S1 — a direct, ratified match — learning is complete. If not, the trainer provides scaffolding, and rationalisation runs again — but this time, the model is slightly larger, the graph is slightly richer, and the query has more to connect to. Each pass is like a student revisiting a problem with one more tool at their disposal. And between passes, study strengthens what was built, so that the next training interaction has a higher starting point.

There is no separate training loop. There is no loss function. There is no gradient. There is only rationalisation, instruction, and re-rationalisation, until the knowledge graph has grown enough to ground the query.

---

## Summary of Principles

**Understanding is grounded.** A kline is understood (S1) only when every node resolves to grounded knowledge. There is no shortcut — no statistical approximation that substitutes for structural grounding.

**Significance is directional.** Kalvin always attempts to raise significance — to climb from S4 towards S1. It does not accept partial understanding as final unless the feedback system confirms it.

**Temperature proposes; the trainer disposes.** Temperature controls the ambition of proposals. The trainer confirms, corrects, or instructs. Understanding is never declared by temperature alone.

**Learning is recursive rationalisation.** Every piece of scaffolding is rationalised just like the original query. This recursion builds understanding from the bottom up — simple structures first, then composites built on top of them — the way foundational concepts must be mastered before their combinations can be understood.

**Learning is constructive.** Every training round adds permanent structure to the knowledge graph. Even corrected proposals leave behind useful graph topology that accelerates future learning — the way a student who gets the wrong answer still learns something in the process.

**The system learns what it doesn't know.** S4 is not a failure state — it is an expression of readiness. A completely novel query tells Kalvin exactly where its knowledge graph has gaps, the way a student's confusion tells a teacher exactly what to teach next. The trainer fills those gaps, and the model grows.

**Agency must be exercised.** Understanding that is taught but not practised is understanding that may be present but inaccessible — a student who has heard the lesson but not done the homework. Kalvin's study process ensures that trained knowledge becomes grounded, available, and ready for ratification.

**S1 is ratified, not claimed.** No amount of autonomous cogitation can promote a kline to S1. Full understanding requires the trainer's confirmation — the moment when Kalvin rationalises a kline and it routes directly to S1 without scaffolding. This is the system's guarantee that knowing is not merely believing: Kalvin knows that it knows only when the trainer agrees.
