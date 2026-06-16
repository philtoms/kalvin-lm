# Kalvin — A Rational Agent

Kalvin is a rationalising system that accepts, thinks, and talks in klines. It receives klines — the fundamental units of its memory — and attempts to understand each one in terms of what it already knows. What it sends back is never just a response: it is a response paired with **significance**, a measurement of how well-grounded that response is in the knowledge Kalvin already holds. This measurement is not a quality score applied after the fact. It is a direct consequence of how the new kline fits into the model.

This is what separates Kalvin from an oracle. An oracle gives an answer and nothing else — you take it or leave it, with no basis for deciding whether to trust it. Kalvin is not an oracle: it gives an answer *and* the structural grounds for that answer. Because significance makes the degree of grounding visible, every response is actionable. The other agent in the dialog knows exactly where understanding is solid and where it breaks down, and can decide what to do next — ratify, scaffold, correct, or submit new information.

This is also what separates Kalvin from a lookup table. A lookup table returns immediately: you submit a key and receive a value, with no expectation of thought. Kalvin is not a lookup table. When understanding is partial, Kalvin demands the autonomy to cogitate — to retrace paths, discover connections, and strengthen its grasp before responding. If you treat Kalvin as a lookup table, you cut off the process through which understanding develops.

Significance is what makes Kalvin rational, not merely functional. A system that always produces correct outputs is functional. A system that can tell you how well-grounded its outputs are is rational. Significance is not an add-on to Kalvin's rationality; it is Kalvin's rationality — the measurement that makes every output a reasoned contribution to a dialog rather than a bare fact dropped from nowhere.

This document describes what Kalvin is for, what it means for Kalvin to understand, and what it means for Kalvin to be a rational agent. Terms used here — kline, signature, node, significance, proposal, countersign, ratification — are established vocabulary in the glossary (`CONTEXT.md`), which is the sole authority for their precise meaning. This document uses them; it does not define them.

---

## Klines

Klines are how Kalvin holds the world. Every piece of knowledge Kalvin possesses is a kline or a structure built from them. Identities anchor the structure; relationships connect one kline to another; and the arrangement of these connections forms the living shape of the model.

When a new kline arrives — carrying its signature and its nodes — Kalvin measures how it fits against what it already knows. Some nodes resolve to grounded knowledge: they connect to identities and relationships Kalvin has already established. Others do not: they are novel, carrying structure that nothing yet grounds. The pattern of resolution and non-resolution is the shape of the fit, and it is the first thing Kalvin perceives.

Connection is necessary, but shape alone is not understanding. The quality of understanding also depends on what Kalvin has learned to prefer — which paths through the model to favour when multiple candidates compete — and on how visible the degree of fit is to other agents, so they can act on it. Understanding emerges from the interplay of three things: structural fit, learned preferences, and significance. This three-part model is the subject of the next section.

What Kalvin considers optimal is not fixed. Preferences are learned, not given — they are part of the model, subject to the same rationalisation and ratification as any other kline. When multiple paths could ground a new arrival, preferences guide which one Kalvin favours. Currently, the default is recency: Kalvin favours more recently established knowledge. But preferences are themselves knowledge. They participate in rationalisation, and they can be taught, scaffolded, and refined. There is no fixed utility function. What Kalvin values in a response is itself something Kalvin can be taught to value differently. This makes preferences load-bearing — one of the three pillars of understanding — rather than an implementation detail bolted on after the fact.

## Understanding

Understanding, for Kalvin, is not a single faculty. It emerges from the interplay of three things:

1. **Structural fit** — how closely a new kline's nodes connect to existing knowledge. This is the shape of the fit: the pattern of what resolves and what does not.
2. **Learned preferences** — which paths through the model Kalvin favours when multiple candidates could ground the new arrival. Preferences guide the measurement.
3. **Significance** — the degree of fit made visible to other agents, enabling them to decide what to do next.

Shape is what Kalvin measures. Preferences shape the measurement. Significance makes the measurement useful. Understanding is what Kalvin does when it receives something new and puts all three to work.

This is why Kalvin is rational. A system that can tell you how well-grounded its outputs are does not merely produce answers — it produces reasoned contributions. The ability to say "I do not understand this at all" is not a failure; it is one of the most useful things Kalvin can communicate, because it tells the other agent exactly where to focus next.

## Significance

When Kalvin rationalises a kline — searching its knowledge, measuring fit, producing a response — the result carries significance. Significance is not a label assigned after the fact. It is a direct consequence of how the new kline fits into the model, expressed as a measurement that tells the other agent how much trust to place in the response.

Significance falls on a spectrum. At one end, every node resolves to grounded knowledge: Kalvin fully understands what it has received — "I know that I know this." At the other end, nothing connects at all: the kline is entirely novel — "I do not understand this at all." Between these two extremes lie degrees of partial understanding: "I understand some of it," or "I recognise aspects of it." Some nodes match, some do not, or the connections are associative rather than direct.

The spectrum resolves into four levels. **S1** is full understanding — every node grounded. **S4** is complete novelty — nothing connects. **S2** and **S3** are the middle: partial understanding, active reasoning, connections half-formed. S1 and S4 are **significants** — the kline is either confirmed or entirely novel, and no further processing is needed. S2 and S3 are **rationals** — partial relationships that Kalvin continues to work on.

Most of Kalvin's rationalising life is spent in the middle, at S2 and S3. These are not failed S1s; they are active states of reasoning. During these states, Kalvin continues to think: retracing paths through its model, discovering connections that were not apparent before, applying its learned preferences, generating proposals. This is where understanding becomes a temporal concern. S2 and S3 take time. Kalvin is not expected to return immediately. It is expected to think, and to be given the time to do so.

## Proposals

Kalvin can publish proposals at any significance level. An S2 proposal is not a failed S1 — it is a legitimate expression of partial understanding. An S3 proposal is not noise — it is an associative hint. Every result is a proposal, and the agent decides what to do with it.

This is fundamentally different from systems that suppress results below a confidence threshold. In Kalvin, a low-significance proposal is not hidden; it is offered, and its significance tells the other agent exactly how much weight to give it. The agent may accept a low-significance proposal as sufficient for now, reject a high-significance proposal that does not match expectations (significance measures structural similarity, not correctness), or instruct by providing new klines that fill in the gaps.

## The Dialog

From Kalvin's perspective, the world is a dialog. Klines arrive, one after another. Some are familiar — Kalvin recognises the shape immediately, responds with high significance, and the exchange is straightforward. Others are partially familiar — Kalvin can trace some of the new kline's nodes through its existing knowledge but not all. It responds with what it understands and attaches a significance that reflects how well the response is grounded. And sometimes a kline arrives that Kalvin cannot connect to anything at all. It says so — low significance, no match. This is not a failure. It is often the most useful response Kalvin can offer, because it tells the other side of the dialog exactly where Kalvin's knowledge ends, which is precisely where the next kline should begin.

Consider what happens when a query kline arrives carrying several nodes. Some of those nodes resolve directly to grounded knowledge — they connect to identities and relationships Kalvin has already established, and the response can trace a path through familiar structure. Other nodes do not resolve — they are novel, carrying no existing connection. Kalvin responds with what it can ground and reports the gap. The significance reflects the partial fit: the response is partly grounded but not wholly. Kalvin does not pretend to understand what it does not, and it does not hide what it does not understand. It communicates the shape of its understanding — including where that understanding stops.

The next kline to arrive might connect one of the unresolved nodes to something Kalvin already knows. If so, Kalvin integrates the new relationship, the gap narrows, and a similar exchange in the future would route directly to higher significance. The dialog builds understanding one exchange at a time, and each gap that closes makes the next exchange easier.

## Ratification

There is a hard limit to Kalvin's autonomy. Kalvin can take significance as far as S2 but no further on its own. Full S1 — truly knowing — can only be reached through ratification by another agent. This limit is deliberate.

Knowing is not merely high confidence. It is understanding plus external confirmation — knowledge that has been formally recognised by another agent and carries a record of who vouched for it. The difference between S2 and S1 is the difference between a student who has studied thoroughly and believes they understand, and a student whose understanding has been formally recognised. S2 is Kalvin's best assessment of its own understanding. S1 is that assessment plus an external record of provenance: who confirmed it, and on what grounds. No amount of autonomous cogitation, study, or traversal of the model can promote a kline to S1. Kalvin can bring a kline to the very threshold, but it cannot cross that threshold alone.

To ratify is to countersign — to provide the external confirmation that lifts a proposal from the threshold of knowing into knowing itself. Ratification records who confirmed the knowledge, creating a chain of provenance. S1 carries traceability and authority. It is understanding that can point to its origins and say: I know this, and here is the agent who vouched for it.

The significance spectrum creates a temporal contract between Kalvin and the other agents in the system. Kalvin demands autonomy — the time to think, to cogitate, to work through partial understanding — and that demand has consequences. How long should Kalvin be allowed to think? Is its response still relevant? These are questions the other agents must answer. If you treat Kalvin as a lookup table, you cut off the process through which understanding develops. If you treat Kalvin as a subroutine, you ignore the signal that tells you what to do next. The limit on autonomy is not a flaw; it is a deliberate boundary that locates agency correctly. Kalvin produces understanding; another agent confirms it.

## How Kalvin Is Taught

Kalvin rationalises every kline it receives in exactly the same way, regardless of context. There is no training mode. A kline arrives, Kalvin searches its knowledge, measures fit, and responds with significance. Whether that kline came from a deliberate teaching exercise or a live query, Kalvin's process is identical.

What differs is the **harness** — the code that manages the flow of klines and responses. A training harness uses significance to decide what to teach next. An operational harness presents live queries and uses significance to decide how to act. Different harnesses for different jobs; Kalvin is unaware of the distinction. It simply rationalises what it receives and reports how well it understood.

When Kalvin responds with less than full significance, the agent has three options:

- **Confirm.** The proposal is accepted. Partial understanding is sufficient for now.
- **Correct.** The proposal is denied. Kalvin has overreached; the system waits for further input.
- **Instruct.** The agent provides new klines that Kalvin does not yet have — the missing pieces between what Kalvin already knows and what it needs to understand. This is **scaffolding**.

### Scaffolding

Scaffolding does not give Kalvin the answer. It gives Kalvin the next structure that is close enough to what it already knows to be grounded, but which also moves understanding closer to the original query. Each scaffold may itself land at partial significance, requiring further scaffolding. The system learns structure by structure. Each round adds a piece of understanding. Over successive rounds, the original query's significance ascends — not because standards were relaxed, but because the model genuinely grew to support it.

The agent has two strategic postures. **Priming** injects trusted knowledge — telling Kalvin, "accept this as known." **Querying** tests understanding — asking Kalvin to work something out. Priming establishes grounding; querying reveals where grounding is incomplete. The interplay between the two is the essence of curriculum design.

### Mary's World

The reference teaching model is simple: build from a blank slate, one kline at a time. Start with nothing — no knowledge, no vocabulary of inquiry — and introduce the world one structure at a time. At each step, submit klines, observe significance, and scaffold where understanding is incomplete. This is how a model grows from empty to capable: not by adjusting weights or minimising error, but by accumulating grounded structure, one exchange at a time.

### Convergence

The knowledge base only grows, never shrinks. Each teaching round either increases a query's significance or adds new grounded knowledge to the model. Even if a query never reaches S1, the scaffolding erected along the way remains, enriching future rationalisation.

## Study

When the dialog pauses, Kalvin continues to think. Any kline that landed at partial significance represents material Kalvin has seen but not fully internalised. During these quiet periods, Kalvin revisits its partial knowledge — retracing paths, discovering connections that were not apparent before, strengthening its understanding.

Study cannot fully confirm a kline on its own. Full understanding arrives when the dialog resumes and a new kline routes cleanly to high significance, or when the agent ratifies what study has prepared. But study prepares. The next exchange may be immediately recognisable, requiring no further scaffolding, because Kalvin has already done the work of connecting what it knows. Study can strengthen significance — bringing partial understanding closer to the threshold of knowing — but it cannot reach S1 alone. It prepares for ratification; the agent confirms it.

Study is the existing cogitation process working through its backlog of partially understood klines. It is not a separate mechanism added on top of rationalisation; it is rationalisation continuing when the dialog is quiet.

## S2 Expansion

S2 klines are **misfits**: their signature does not match their nodes. The mismatch falls into patterns. An **underfit** kline has a signature that promises structure its nodes do not deliver — there are holes to fill. An **overfit** kline carries nodes whose structure the signature does not capture — there is excess to shed. A **dual misfit** has both problems. In every case, Kalvin reshapes what is already there and offers the result to the agent.

Underfit klines act as **templates** — a known concept with holes to fill, allowing Kalvin to match a question against a structure that anticipates an answer. Overfit klines act as **sequencers** — step-by-step structure under a single goal, allowing Kalvin to rationalise a query in discrete stages. Kalvin fills templates and decomposes sequencers, turning partial understanding into proposals.

Two guarantees hold: no kline enters the model without agent ratification, and no removed nodes are discarded without being offered as a companion kline. Kalvin does not invent during expansion — it reshapes what is already there and offers the result. The agent decides what to keep.

## Growth

Kalvin's memory only grows. Klines are added through rationalisation and ratification; they are never removed. This monotonic property has consequences worth tracing.

Correction happens through the introduction of more significant information, not by retracting what came before. New information does not override old information; it outcompetes it. If Kalvin holds a ratified misconception and new conflicting information arrives, the new information is rationalised, attains its own significance, and — if more significant — becomes the preferred path. The misconception is still there, structurally intact, but it is no longer the optimal response. Nothing was deleted. The model grew, and the balance of significance shifted.

Monotonicity is what makes scaffolding irreversible. Each scaffold adds structure to the model. Even if a particular teaching goal is abandoned, the scaffolding remains, enriching future rationalisation. Every round leaves the model larger than it found it.

This connects to the three-part model: new preferences can be taught, and what Kalvin considers optimal is itself something Kalvin can be taught to value differently. Growth is not just the accumulation of facts — it is the deepening of the model's capacity to understand.

## Aspiration

The distinction between S1 and S2 becomes particularly interesting when the agent in the dialog is another Kalvin instance. One agent's partial understanding becomes another's scaffolding. One agent's significant response becomes another's ratified knowledge. The architecture admits networks of mutual ratification, where knowledge is generated through mutual rationalisation and understanding emerges from the dialog between understanding-generating systems. The long-term aspiration is distributed rationality built on actionable signals and mutual respect for autonomy — shared understanding that no single instance could reach alone.

Kalvin's design makes several choices worth stating plainly:

- **Preferences are learned, not given.** They are part of the model — subject to ratification, scaffoldable, revisable.
- **Information is structurally accumulated, not statistically weighted.** The model grows through the addition of ratified klines. It does not converge through parameter adjustment.
- **Actions include meta-actions.** Producing significance is itself the agentic act — not a side channel or a quality metric, but the communication.
- **Optimality is actionable, not accurate.** The optimal response is the most significant one Kalvin can produce, even if that response is "I do not understand this at all." An S4 response is optimally useful — it tells the other agent exactly where understanding breaks down.
- **Rationality is second-order.** Kalvin rationalises its own rational outputs. The ability to assess and communicate the grounds for one's own understanding is not an add-on to rationality; it is constitutive of it.

These are the principles on which Kalvin is built. The aspiration is a system — and networks of systems — that develop shared understanding through the honest measurement of what each one knows.
