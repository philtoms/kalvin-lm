# Kalvin — Context

This document has two sections. **Operating Notes** contains process instructions and conventions. **Domain Glossary** defines the precise meaning of terms used across specs, plans, and code. Do not mix the two — glossary entries are domain terms only; operating notes are behavioral rules.

---

## Operating Notes

- Commit all work before creating any kb tasks.
- When creating kb tasks for large features, decompose into discrete code tasks with explicit `depends` chains. Each task should cover one coherent piece of work — a single module or a single behavioural change. Do not create monolithic tasks that span multiple modules. Cascade work (specs, plans, vision) follows the `docs/cascade-development.md` flow, not kb.
- Follow the docs/cascade-development.md model strictly.
- CONTEXT.md is a glossary plus operating notes. Keep the two sections separate. Do not add implementation details, spec content, or code to either section.
- **Lesson Labelling Convention.** Lessons are identified by stable labels derived from their headings. Whole-number labels (1, 2, 3) indicate distinct conceptual steps. Sub-labels (2a, 2b) indicate lessons semantically related to their parent — refinements, bridges, or remediations of that concept. If a new lesson is logically subsequent but not semantically related, the document is renumbered instead. The curriculum must always read as a logical and temporal narrative for humans.

---

## Domain Glossary

Kalvin is a rationalising system whose entire world is built from klines. This glossary defines the precise meaning of terms used across specs, plans, and code.

### Structure

**Token ID**:
A value produced by the tokenizer.

**Node**:
A structural slot: a value occupying a position in a kline's nodes list. The structural element over which **OR-reduction** operates. Not a topological concept — topology refers to the signature-nodes relationship (see **Composition**), not to individual nodes.
_Avoid_: child, element (the structural slot is specifically a node)

**Signature**:
The value occupying a kline's head position — the value its **Composition** relationship is assessed against. The same kind of value as a Node: a single Token ID or the OR-reduction of its nodes.

**OR-reduction**:
The structural arithmetic owned by the model by which a kline's signature is derived from its nodes (bitwise OR over the node values). Evaluates the **Composition** relationship (exposed as `signature_of(nodes)`); the arithmetic is structural, the relationship it evaluates is topological.

### Topology

**KLine**:
The fundamental unit of Kalvin's memory: a **signature** (its head node) and a **nodes** list, between which holds a **Composition** relationship. The relationship's value is the kline's **role** — Identity, Canon, or Misfit.
_Avoid_: kvalue (a KValue pairs a KLine with significance)

**Composition**:
The topological primitive: the algebraic relationship between a kline's signature and its nodes. Three values — the kline **roles**: **Identity** (composition is trivial), **Canon** (composition is exact), **Misfit** (composition is partial). Owned at the agent layer alongside **Distance**; the model's structure evaluates it (today, OR-reduction via `signature_of`).

**Distance**:
The topological primitive measuring how far apart two klines are. Owned with **Composition** at the agent layer; evaluated by the model's structure (today, graph hop-counting). **Significance** is the rationalisation-layer projection of Distance.
_Avoid_: hop count (the structural evaluation, not the topological concept), similarity (a consequence of small distance, not the concept)

**Identity**:
A kline **role**: its Composition is trivial — the signature is not algebraically derived from non-trivial nodes. Takes three **structural shapes** (owned by structure, not topology): empty nodes (`{S: []}`), self-referential (`{S: [S]}`), or compound-word (carrying `COMPOUND_TOKEN`). Every kline bottoms out at one or more identities.
_Avoid_: unsigned (implementation term), bare signature (describes syntax, not the role)

**Canon**:
A kline **role**: its Composition is exact — the signature fully composes its nodes (`signature == signature_of(nodes)`). The signature carries no information beyond what its nodes already express. A Canon is recognised (S1) by its own Composition — it accounts for itself, needing no ratification.
_Avoid_: canonical (ambiguous with Relational Tokens), treating `=>` (CANONIZES) as synonymous with being a Canon (the token declares an intent to compose; a CANONIZES kline need not be a Canon), MTS (an example, not the concept)

**Misfit**:
A kline **role**: its Composition is partial — the signature does not fully compose its nodes (`signature ≠ signature_of(nodes)`). An unratified Misfit is contested (S2); ratification promotes it to recognised (S1). A Misfit that is also a **Proposal** is subject to the proposal-layer constraints on origination; the role itself carries no provenance.
_Avoid_: fabrication (informal), conjecture/hypothesis (a misfit is a role, not a distinct emission kind)

### Rationalisation

**Significance**:
A projection of rationalisation — a measurement of how strongly a kline relates to what Kalvin already holds, derived from the topological **Distance** between them (the model's structure evaluates the distance; significance projects it). Classified into four levels — the **significances** — which name qualities of understanding, not mechanisms:
- **S1 — recognised.** Fully accounted for — by its own Composition (a canon) or by ratification. _I know that I know this._
- **S2 — contested.** Relates but diverges: an unratified misfit, an active mismatch. _I infer this, but it does not yet fit._
- **S3 — suggested.** Connects only indirectly, through intermediaries. _I recognise aspects of this, indirectly._
- **S4 — unrecognised.** Shares nothing with what is held; no connection can be drawn. _I do not understand this at all._
Every participant assesses independently. How Kalvin computes its own significance is a model concern (see **Grounding**); the levels themselves are independent of that computation.
_Avoid_: confidence, score, weight, grounded (grounding is the model's implementation of S1, not a synonym for any level)

**Expectation**:
A scripted kline that enters the slow path (S2/S3) during rationalisation and requires a matching proposal to be satisfied.

**Grounding**:
The model's mechanism for realising **S1** (recognised). A kline is **grounded** when the model counts it as S1 — either by its own Composition (a canon self-grounds) or by residing in LTM via ratification (a structural fact the model owns). Grounding is how S1 is _produced_, not what S1 _means_; "recognised" is the significance-level concept.
_Avoid_: self-grounded (legacy; conflates the mechanism with the level), grounded identity (grounding applies to any kline that attains S1, not just identities)

**Frame**:
Recognised working context persisted across sessions. Monotonic.
_Avoid_: session log (Frame is not a log), session

**STM (Short-Term Memory)**:
The lowest tier in the write cascade and Kalvin's event register — every write reaches it. Empty at session start.
_Avoid_: STM caching (too vague), working memory (too vague), context window (implies a passive buffer)

**LTM (Long-Term Memory)**:
Persistent knowledge that survives across sessions. Structurally identical to Frame; the distinction is semantic. A kline residing in LTM is **grounded** (see Grounding).
_Avoid_: persistent store (too vague), knowledge base, LTM frame

**KValue**:
The unit of exchange between participants — a **KLine** (objective structure) paired with a **significance** (the sender's assessment of it).

**Proposal**:
A KLine emitted by the Agent as a subjective response during rationalisation.

**Ratify**:
The action of countersigning a selected proposal. Usually performed by the Trainer during curriculum execution.

**Escalation**:
The Trainer deferring a proposal it cannot auto-ratify to the supervisor for resolution. The boundary between what the Trainer resolves and what the supervisor resolves.

### Runtime

**Harness**:
The multi-agent runtime that loads participants and runs a dialogue loop between them. A message broker — participants send role-addressed messages through the harness and it routes them to all subscribers of that role. Participants never communicate directly.

**Message**:
A unit of inter-participant communication routed by the harness — addressed to a role with an action interpreted by the recipient.

**Role**:
The routing key for inter-participant communication on the harness bus. Three defined roles: **trainee** (Kalvin), **trainer** (Trainer), **supervisor** (TUI, Slack, future AI agents).
_Avoid_: address (legacy), topic (legacy), type (ambiguous with config `type: embedded/client`)

**Dialogue**:
The alternating exchange between participants in the harness loop. No participant is aware it is in a training loop — each simply receives and responds.

### KScript

**Relational Tokens**:
The closed set of written tokens that declare how a kline is produced in KScript — `==` (COUNTERSIGNS), `=>` (CANONIZES), `>` (CONNOTES), `=` (DENOTES), or none (identity). A compiler/provenance concept: the token declares an _intent_ (e.g. CANONIZES declares an intent to compose), which the resulting kline's actual **Composition** (topological) may or may not satisfy. Distinct from Composition despite the surface similarity.

- **COUNTERSIGNS** (`==`) — 1:1 emits a reciprocal pair `{A: [B]}`, `{B: [A]}`. The signature countersigns each other's nodes.
- **CANONIZES** (`=>`) — 1:many `{A: [B, C, D]}`. The signature canonizes its nodes into a single kline; this declares an intent to aggregate, not that the result is a Canon (see Canon).
- **CONNOTES** (`>`) — 1:1 `{A: [B]}`. The signature connotes each node (`A > B` ⇒ A connotes B; subjectively, _A is a B_).
- **DENOTES** (`=`) — 1:1 `{B: [A]}`. The signature denotes each node (`A = B` ⇒ A denotes B; objectively, _B is an A_).
- **IDENTITY** — `{A: []}` or `{A: [A]}` (self-referential) — see Identity.
  _Avoid_: structural relationship (legacy name — collides with the structure layer), relational operator (the token declares provenance, not an operation)

**MTS (Multi-Token Signature)**:
A KScript device for representing a multi-token signature on the LHS in a simpler syntax than would otherwise be required. A compound signature built from more than one Token ID by OR-reduction; the compiler expands a multi-character KScript identifier into its constituent character identities plus one MTS relationship. This expansion is a property of the _signature string_, distinct from any CANONIZES decomposition a script declares for that signature via a block. A CANONIZES scope's nodes are the declared block operands, never the signature's own MTS character expansion.
_Avoid_: decomposition (overloaded — a Canon decomposes into its nodes; an MTS expands a signature into characters), packed signature (the uint64 result, not the expansion)

**Word Binding**:
The association of a single-character KScript signature with a word, resolved through annotations in the source. Bindings are scoped by relational-token boundaries; a character resolves to the most recent matching word in its scope. Two annotation kinds bind with different strength: a **top-level annotation** (on a scope signature) binds only if the character is currently unbound — fill-if-empty, never overriding an outer binding; an **inline annotation** (on an item) binds unconditionally, overriding any outer binding for that occurrence. Each identity occurrence is bound exactly once by the most specific annotation that applies to it, so one character never acquires two competing tokens.
_Avoid_: comment mapping (the binding is a specific compiler artefact, not a general comment feature), rebind (a top-level annotation never overrides; an inline annotation always does — use the specific kind)

### Training

**Trainee**:
The participant under instruction — the rationalising system being trained, and the subject of a training session. Registered on the harness bus with role `trainee`.

**Kalvin**:
The project's name for the trainee.
_Avoid_: Agent (ambiguous), KAgent (that's the implementation class)

**Trainer**:
An agent-in-the-loop that drives the training loop on behalf of a supervisor. Registered on the harness bus with role `trainer`.
_Avoid_: auto-agent, training bot

**Supervisor**:
A participant subscribed to the `supervisor` role that monitors the training session and may intercede when needed. Independent of medium — TUI, Slack, or a future AI agent all share the same capabilities.
_Avoid_: UI (too narrow), human (a supervisor may be an AI agent)

**Curriculum**:
A living structured document owned by the Harness and accessible to all participants. The source of truth for training — never rolled back, only evolved forward. Three sections: **objective** (what it teaches), **approach** (the pedagogical strategy), and **lessons** (ordered KScript entries with human-readable context).
_Avoid_: lesson plan (too narrow — the curriculum is more than its lessons)

**Scaffolding**:
KScript entries that provide grounding context for other entries. Structurally identical regardless of origin; the difference is only when they are created — **pre-compiled** (written into the original script by the curriculum designer) or **reactive** (written by the supervisor when Kalvin's S2/S3 proposals mismatch expectations).

**Auto-Tune**:
A tuning loop where an LLM coding agent runs repeated training sessions against the codebase, observes results, edits code, and re-runs to converge on a goal. Not a training concept — auto-tune improves the _codebase_, not Kalvin's model.
_Avoid_: tuning session (ambiguous with training session), auto-train (it's not training), auto-tune supervisor (the CLI includes the full auto-tune tool, not just the supervisor)
