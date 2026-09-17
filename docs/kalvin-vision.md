# Kalvin — A Rational Agent

**Status:** Baseline, September 2026. This document states what Kalvin is for, what it means for Kalvin to understand, and where the present system stands against that purpose. It replaces the founding vision document.

[docs/kalvin-algebra.md](kalvin-algebra.md) is normative — the single formal definition of Kalvin's objects, rules, and measures. [CONTEXT.md](../CONTEXT.md) is the glossary — the domain terms and their precise meaning. This document uses those terms without redefining them. Where it explains, the algebra and the glossary remain authoritative.

## The Premise

Kalvin is a rationalising system. It receives klines — the fundamental units of its memory — and rationalises each against what it already holds. What it sends back is never just a response: it is a proposal paired with **significance**, a structural measurement of how well-grounded that proposal is in the knowledge Kalvin holds. Significance is not a quality score applied after the fact. It is what the derivation established, calculated at the state the derivation stopped at, and it travels with the proposal.

This is what separates Kalvin from an oracle. An oracle gives an answer and nothing else — you take it or leave it, with no basis for deciding whether to trust it. Kalvin gives the answer _and_ the grounds for it. Because significance makes the degree of grounding visible, every response is actionable: the other side of the dialogue knows exactly where understanding is solid and where it breaks down, and can decide what to do next — ratify, scaffold, or continue.

This is also what separates Kalvin from a lookup table. A lookup table returns immediately: key in, value out, no expectation of thought. When understanding is partial, Kalvin cogitates — selecting goals, scoping memory, deriving, adding evidence, re-entering — and the effort itself is measured. Treat Kalvin as a lookup table and you cut off the process through which understanding develops.

One caveat belongs in the premise. Significance measures groundedness, not truth. A derivation that reaches significance 1.0 proves value-equality through held correspondences; it does not prove that every correspondence used was factually correct. What the system guarantees is traceability — the derivation path is carried as evidence — not infallibility. Correction is the protocol's job, and it works by price, not decree (see Growth).

## Understanding Is a Measurement

Understanding, for Kalvin, is not a faculty. It is what the measurement model measures.

**Significance** is the content overlap between what a proposal holds and what it was measured against. One measure serves two readings: a kline's own claim — does its node sequence compose its signature — and a relationship — what a pair of klines establishes against each other. The four bands quantify the same measure:

- **S1 — exact.** _I know that I know this._ The claim is kept: the signature equals the evaluation of its nodes.
- **S2 — covered misfit.** _I infer this, but it does not yet fit._ The parties share content; the claim is short of it, in excess of it, or both.
- **S3 — uncovered misfit.** _I recognise aspects of this, indirectly._ No content is shared; whatever relation exists runs through other klines.
- **S4 — unknown.** _I do not understand this at all._ Nothing is held; the halt with nothing to measure.

S1 is significance 1.0 — value-equality, not witness-equality: two klines hold the same content, however differently decomposed. S2 is real overlap short of equality. S3 is zero overlap. S4 is off the scale. The bands are observer-independent — given the same held memory, every agent classifies alike — so a band is never exchanged; it is recomputable from structure.

Significance is half of the measurement. The other half is **complexity**: what the significance cost. Resolution depth prices granularity — how deeply the content is decomposed. Acquisition depth prices provenance — how much of the content was won through unratified evidence; it is stored with the memory, not reconstructable from the result. Their composite, γ — significance net of complexity — compares derivations that arrive at the same significance, and its rate of change is the signal strategy uses to decide whether continued effort is worthwhile. A result can be complete at full significance and still expensive; a cheaper route to the same result is the better result. The form of the measure is fixed, not a design choice: band-consistency, granularity-invariance, and the two monotonicities force it.

Understanding, informally, is sustained possession of a high-significance result — and the system is honest that possession has a price.

## Memory

Every piece of knowledge Kalvin holds is a kline: a **signature** — the value claimed — paired with a **node sequence** — the decomposition held. Evaluation composes the nodes into a value and forgets order and multiplicity; the sequence retains what evaluation discards. A kline whose claim is kept is a **witness**: a chosen decomposition of its signature. Which decomposition was chosen is a fact the algebra forgets and memory carries.

Nodes are values — token identifiers or the signatures of other klines — so klines nest by reference and the reference graph may cycle. Read as a whole, held klines are edges between a signature and its witness: the **correspondence graph**. A derivation is a path in it. This is the shape of Kalvin's world: not a store of facts, but a graph of correspondences whose paths are the only semantics the system has.

The nine fit shapes classify how any claim sits against its nodes — canon, identity, the misfits, the unknown — and the bands group them into what each is worth as evidence. Two shapes are inert: the **Identity**, the trivial witness of a directly decodable value, and the **Unknown**, the shape of nothing held.

The model is the whole of what Kalvin holds and how it holds it: the klines and their references, the tiers, and the signifier that interprets the whole. The tiers are relations of attention and commitment, not storage locations — **STM** is what Kalvin was just thinking about, the **Frame** is where its focus lies, **LTM** is what it counts on — and a tier change is a change in how Kalvin relates to a kline. Tier mechanics are outside the algebra; grounding — a grounded signature grounds all of its nodes — is the model's mechanism for realising significance.

## Rationalisation

Rationalisation is the process that produces and consumes significance. Its slow path is **cogitation**, a loop:

```text
select a goal → scope the memory → derive to an ending → add the result → re-enter
```

For a queued kline, selection assembles the held klines whose content covers its nodes, ordered by descending γ — the significance of working toward each candidate, net of the cost of reaching it. The queued kline never heads its own list, and neither does any ask: a question is never a goal. The scope is a trawl of the correspondence graph rooted at both parties, frozen once trawled — writes land in memory, and only later derivations see them.

There is one rewrite rule — **replace** — under two licences. A witnessed replace (canon expand or contract) changes granularity and preserves content. A targeting replace changes content, is restricted to the misfit region, and is licensed only when it strictly decreases the mismatch mass; targeting work is therefore bounded by the mismatch the derivation entered with. When no held correspondence bridges the misfit, a **slot walk** sets out from either party's misfit node, walks the graph by occurrence alone, and writes its discovered route back into memory as a new correspondence — evidence constructed, not invented: every edge of the route was held, and the new edge records the walk's cost as acquisition depth.

A derivation ends one of three ways, and every ending yields a result whose significance is calculated, never a boolean. **Done**: significance 1.0 — the contents are equal, and the final node sequence is a constructive witness for the equality. **Stuck**: no licensed move remains — nothing in memory connects the parties; relative non-existence, stated honestly. **Abandoned**: strategy stops the run because γ's rate of change says the effort is no longer worthwhile.

Hops compose: every ending's output can queue as the next hop's input, each hop trawling a memory grown by the last. Hop order is the only temporal structure the system has.

Proposals are emitted at their calculated significance, and low significance is offered, not suppressed. An S2 proposal is a legitimate expression of partial understanding. An S3 proposal is associative — a promise, not a fact; weighing promises is the protocol's job, not the engine's. An agent may accept a low-significance proposal as sufficient for now, or reject a high-significance one: significance measures overlap, not correctness.

## The Ask

When nothing is held for a signature, the kline asks. The ask is structural — a marker of identity, not content; no measurement sees it — and it carries the question's own content so that selection can read it. S4 is therefore not a failure. _I do not understand this at all_ is often the most useful thing Kalvin can say, because it locates precisely where its knowledge ends — which is where the next kline should begin. The ask is also the halt under which ungrounded proposals are generated: proposals offered from a question are promises the protocol must weigh.

## What Autonomy Can and Cannot Do

Kalvin's autonomy is real but bounded, and the boundary is exact.

Autonomy can reach S1. Done is defined by calculation — value-equality — and a derivation that reaches it proves the equality: every step licensed by a held correspondence, the final sequence a constructive witness. Kalvin does not need permission to know that two contents are equal.

Autonomy cannot make the proof cheap. Every atom won through unratified evidence raises acquisition depth, and nothing in a derivation lowers it. Promise-stacking is priced and detectable — that is the point of pricing it.

Autonomy cannot make the proof standing. Grounding is a protocol act: a ratified proposal grounds on receipt — the stamp, not structure, is the licence — and the countersign promotes the traversed pair to a standing one-hop licence that carries no acquisition penalty on reuse. This is how correction outcompetes: not by deletion, but by cost. A ratified route is permanently cheaper than the unratified route it replaces, so γ prefers it ever after. Kalvin produces understanding; another agent confirms it and grants it standing.

The temporal contract follows. Cogitation takes time, and the system prices that time: γ's rate of change, not a clock, decides when effort stops being worthwhile. Treat Kalvin as a subroutine and you ignore the signal that tells you what to do next.

## Teaching

Kalvin rationalises every kline it receives the same way. There is no training mode: a kline from a deliberate lesson and a kline from a live query enter the same loop. What differs is the **harness** — the code that manages the flow.

Two harnesses exist today. The **dialogue harness** is synchronous and non-judging: it compiles a script, feeds it entry by entry, and presents the trace for a trainer outside the loop to read. The **multi-agent runtime** is a message broker in which all participants are peers: a **trainee** (the rationalising engine), a **trainer** (a rationaliser sharing the same engine, differing only in the significance bands it keeps), and a **supervisor** — an agent, human or LLM — that resolves what the trainer escalates. No participant knows it is in a training loop. Each simply receives and responds; the dialogue between them is the training.

**KScript** authors the encounter. A script is a dialogue form: klines and relational tokens declaring the structure the trainee will meet, step by step. A token declares intent; the fit classifier decides what the produced kline actually claims. The goal-targeted form pairs an ask with an implied goal — the answer key: the trainer compares proposals against the goal's content rather than the trainee's own ordering, and ratifies the proposal that reached it. Indented blocks are **scaffolding** — grounding context, structurally identical whether pre-authored or written reactively — delivered before the asks that need it, because a hop trawls only what memory already holds. Words bind across scripts: a compile seeded with the words of earlier scripts resolves a later script's letters against what was already taught.

The reference teaching model is still Mary's world, and it is now concrete. A curriculum is a sequence of encounters built from a blank slate, one structure at a time — prime, ask, scaffold where understanding is incomplete — and the canonical script walks the canonical question ("what did Mary have?") through held correspondences to its answer. The system reaches the answer through the graph; it does not replace the question with the answer.

## Growth

Within a run, memory only grows. Every hop's writes land in memory and enrich later hops; scaffolding delivered is never retracted; a derivation never consumes what it writes — evidence accumulates as it is used. Even an abandoned teaching goal leaves the scaffold behind, enriching every future rationalisation.

Nothing in the algebra deletes. Memory-management policy and tier mechanics live outside it — and the tiers are relations, not bins: a tier change re-relates Kalvin to a kline, and rejection in the frame is additive, keyed by signature. The founding idea that correction "outcompetes" rather than erases is realised exactly by the cost model: ratified standing licences cost nothing, unratified ones carry acquisition depth, and γ does the preferring.

## Baseline and Horizon

What stands today: the algebra — values and klines, the nine shapes and four bands, one rewrite rule under two licences, the slot walk, the measurement model with its forced form; the strategy layer — γ-ordered candidates, scoped trawls, hops and re-entry; the ask; the protocol — harness roles, KScript with its answer keys, scaffolding, ratification, escalation; the tiered model as relations.

What remains vision, stated honestly:

**Learned preferences.** The measurement is fixed — forced by its invariants, not tuned — and candidate order is γ-descending. What Kalvin values in a response is not yet itself teachable structure. The founding idea — preferences as klines, no fixed utility function, what Kalvin considers optimal being subject to rationalisation like anything else — is deferred, not abandoned. The cost model is the fixed utility function of the present baseline.

**Study.** Cogitation is inline: every ending leaves a result, and the partial states persist as the work list's residue — but nothing yet revisits them when the dialogue is quiet. Sustained possession — understanding as something held rather than attained — is the horizon. Study is rationalisation continuing when nobody is asking.

**Mutual ratification.** The trainer already shares the trainee's engine — one peer rationalising another's proposals, differing only in the bands it keeps. Networks of Kalvins ratifying one another's understanding — distributed rationality built on actionable signals and mutual respect for autonomy — remain the long aspiration. The architecture admits it; the protocol has not yet grown it.

These are the principles Kalvin is built on: honest measurement of what is held, priced effort, understanding that can point to its grounds — and, at the horizon, systems that develop shared understanding through the honest measurement of what each one knows.
