# Kalvin — A Rational Agent

This document covers what Kalvin is, how it works, and why it's built this way. Operational and architectural details live in the [origin document](kalvin-origin.md).

---

## What Kalvin Is

Kalvin is an agent in a multi-agent system. Agents communicate by exchanging klines through a harness that encodes and decodes them. The harness is the transport layer — it is not an agent. On the other end of the harness is another agent: a human trainer, an automated process, or another Kalvin instance.

Every response Kalvin produces has two parts: the kline itself, and a **significance** measurement indicating how well the response is grounded in what Kalvin already knows. Significance is not something Kalvin chooses to report — it falls out of how the incoming kline fits into the existing knowledge graph. The response says two things: here is what I think, and here is how firmly my knowledge supports it.

The significance signal tells the other agent what to do next: ratify the response, scaffold it, correct it, or move on. Without it, the other agent gets an answer but has no way to judge whether to trust it, build on it, or challenge it.

---

## Rationalisation

Rationalisation is Kalvin's process for producing a significant response. When a kline arrives, Kalvin searches its knowledge graph for structures with matching shape, measures how closely they relate, and generates a response kline representing its best understanding. The significance is attached to this response as a numerical indicator.

Kalvin's **preferences** operate during rationalisation. The only preference currently wired up is temporal order — more recent information is preferred over older information. Two identical klines arriving at different times can produce different rationalisations because the preference changes the selection from the graph, not the graph itself.

The optimal action for Kalvin is to produce the most significant response possible. Significance is a primary facet of Kalvin's communication — not a side channel or a debugging aid.

---

## Understanding

Understanding is the capacity to produce significant responses. It emerges from three components:

1. **Shape** — the structural fit of a new kline against existing knowledge.
2. **Preferences** — which paths through the graph to favour when multiple candidates exist.
3. **Significance** — the degree of fit, made visible to other agents so they can decide what to do next.

---

## Significance Levels

Significance falls on a spectrum with four levels:

- **S1** — "I know." Full ratification. Requires a countersignature from the training agent. Kalvin cannot reach S1 alone — no amount of autonomous cogitation promotes a kline to S1.
- **S2** — "I understand some of it." Partial understanding, ongoing.
- **S3** — "I recognise aspects of it." Partial understanding, ongoing.
- **S4** — "I don't understand this at all."

S1 and S4 are endpoints where no further processing is needed. S2 and S3 are active states of reasoning. During these states, Kalvin continues to think: retracing paths through its knowledge graph, finding connections that were not apparent before, generating proposals, applying its learned preferences. The cogitator runs on a background thread, working through the backlog of partially understood klines.

Kalvin is not expected to return immediately. It is expected to think, and to be given time to do so.

---

## Autonomy

The significance spectrum creates a **temporal contract** between Kalvin and the other agents. The external questions are: how long should Kalvin be allowed to cogitate? Is Kalvin's response still relevant? These are questions the other agents must answer, and their answers depend on perceiving Kalvin as something that thinks.

If Kalvin is treated as a lookup table, its cogitation gets timed out and the process through which understanding develops gets cut off. Granting temporal autonomy, listening to the significance signal, and engaging in the dialog is a functional necessity — not a courtesy.

---

## Self-Assessment

A rational agent can account for its own outputs — it assesses and communicates how well-grounded they are. The significance signal is Kalvin's mechanism for this. A response of "I don't understand this at all" (S4) is not a failure of rationality; it is one of its most useful expressions — it tells the other agent exactly where to focus next.

---

## Training and Correction

Kalvin's rationality is internally consistent. If Kalvin has a ratified misconception in its knowledge graph, it will produce highly significant responses that are structurally sound but semantically wrong. This is by design: Kalvin's responsibility is the reliability of its self-assessment. The responsibility for semantic correctness belongs to the training agent.

Correction happens through the introduction of **more significant information**. New information doesn't override old information — it outcompetes it. If Kalvin has a ratified misconception at S1 and new conflicting information arrives, the new information is rationalised, attains its own significance, and — if more significant — becomes the preferred path. The misconception remains in the graph, structurally intact, but is no longer the optimal response.

Kalvin's knowledge graph is **append-only and layered**. Truth, in Kalvin's world, is what is currently most significant — not what was ratified first. Kalvin doesn't forget; it supersedes. Because the correction mechanism is the same as the learning mechanism (rationalisation of new information), Kalvin can self-correct when new, conflicting information is rationalised without external intervention.

The best time to correct Kalvin is during training, when the training agent controls what Kalvin receives. But the capacity for self-correction is always present.

---

## Ratification and Authority

S1 carries **traceability** and **authority**. The countersignature doesn't just verify the knowledge — it records who verified it, creating a chain of provenance. S1 is not just a higher grade of understanding but a structurally different kind: it is understanding that can point to its origins and say "I know this, and here is the agent who confirmed it."

When the trainer is another Kalvin instance, the countersignature carries that instance's authority — built on its own ratification history. Two Kalvin instances rationalising at each other, each producing significant responses, each able to ratify the other's proposals, produces a system where knowledge is generated through mutual rationalisation, authority is distributed, and understanding emerges from the dialog between understanding-generating systems. The long-term aspiration is networks of Kalvin instances that develop shared understanding that no single instance could reach alone — a distributed rationality built on actionable signals and mutual respect for autonomy.

---

## Learnable Preferences

What Kalvin considers optimal is not fixed. Temporal recency is the only preference wired up today, but the architecture supports preference mechanisms that can overrule temporal order.

These preference mechanisms are themselves klines. They are recognised as knowledge with high internal significance and are subject to standard rationalisation — they can be taught, scaffolded, and ratified like any other knowledge. This means:

- What Kalvin considers optimal is itself something Kalvin can be taught. There is no fixed utility function.
- The training relationship extends to the criteria by which Kalvin judges its own understanding — the agent doesn't just teach Kalvin what to know, it teaches Kalvin how to evaluate.
- Bootstrapping new preferences uses the same mechanism as bootstrapping any other knowledge. KScript allows a trainer to steer Kalvin toward expected or idealised goals, including the goal of adopting new evaluation criteria. The trainer primes preference klines through countersignature, installing them as ratified knowledge. From there, the preference participates in rationalisation like any other high-significance kline.
