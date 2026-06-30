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

**Role**:
The routing key for inter-participant communication on the harness bus. Three defined roles: **trainee** (Kalvin), **trainer** (Trainer), **supervisor** (TUI, Slack, future AI agents).
_Avoid_: address (legacy), topic (legacy), type (ambiguous with config `type: embedded/client`)

**Supervisor**:
A participant subscribed to the `supervisor` role that monitors the training session and may intercede when needed. Independent of medium — TUI, Slack, or a future AI agent all share the same capabilities.
_Avoid_: UI (too narrow), human (a supervisor may be an AI agent)

**Trainee**:
The participant under instruction — the rationalising system being trained, and the subject of a training session. Registered on the harness bus with role `trainee`.

**Kalvin**:
The project's name for the trainee.
_Avoid_: Agent (ambiguous), KAgent (that's the implementation class)

**Trainer**:
An agent-in-the-loop that drives the training loop on behalf of a supervisor. Registered on the harness bus with role `trainer`.
_Avoid_: auto-agent, training bot

**Scaffolding**:
KScript entries that provide grounding context for other entries. Two forms — **pre-compiled** (written into the script by the curriculum designer) and **reactive** (written by the supervisor during rationalisation). Structurally identical; the difference is when they are created.

**Pre-compiled Scaffolding**:
Scaffolding written into the original script by the curriculum designer — what they anticipated Kalvin needs to understand the parent line.

**Reactive Scaffolding**:
Scaffolding written by the supervisor when Kalvin's S2/S3 proposals mismatch expectations — targeted klines written to bridge the gap.

**Curriculum**:
A living structured document owned by the Harness and accessible to all participants. The source of truth for training — never rolled back, only evolved forward. Three sections: **objective** (what it teaches), **approach** (the pedagogical strategy), and **lessons** (ordered KScript entries with human-readable context).
_Avoid_: lesson plan (too narrow — the curriculum is more than its lessons)

**Curriculum Amendment**:
A change to a running curriculum — inserting, appending, or modifying lessons. Any participant may request one via the Trainer; applied immediately without human ratification.

**Goal**:
The input that starts a training session — either a natural language request (the Trainer generates a curriculum via LLM agent) or a file path to an existing curriculum. The Trainer resolves either to a curriculum file and begins training.

**Harness**:
The multi-agent runtime that loads participants and runs a dialogue loop between them. A message broker — participants send role-addressed messages through the harness and it routes them to all subscribers of that role. Participants never communicate directly.

**Message**:
A unit of inter-participant communication routed by the harness — addressed to a role with an action interpreted by the recipient.

**Dialogue**:
The alternating exchange between participants in the harness loop. No participant is aware it is in a training loop — each simply receives and responds.

**Participant**:
Any agent loaded into the harness loop (Kalvin, Trainer, supervisor participant, etc.). Each subscribes to a role and receives all messages sent to that role; multiple participants may share a role. All participants are equal — none has special status or privileged access to the harness.

**Auto-Tune**:
A tuning loop where an LLM coding agent runs repeated training sessions against the codebase, observes results, edits code, and re-runs to converge on a goal. Not a training concept — auto-tune improves the _codebase_, not Kalvin's model.
_Avoid_: tuning session (ambiguous with training session), auto-train (it's not training), auto-tune supervisor (the CLI includes the full auto-tune tool, not just the supervisor)

**Token ID**:
A value produced by the tokenizer.

**Node**:
A token value occupying a slot in a kline's nodes list. May be a single Token ID or the OR-reduction of two or more token IDs.

**KLine**:
The fundamental unit of Kalvin's memory: a **signature** and a **nodes** list. Two structural kinds: **identity** (signature is a single Token ID, empty nodes list — see Identity) and **relationship** (one or more nodes). Objective structure only — see **KValue** for the subjective+objective exchange unit.
_Avoid_: kvalue (a KValue pairs a KLine with significance; a KLine is objective-only)

**Signature**:
The value occupying a kline's head position. The same kind of value as a Node — a single Token ID (for an identity) or the OR-reduction of its nodes (for a relationship kline).

**Expectation**:
A scripted kline that enters the slow path (S2/S3) during rationalisation and requires a matching proposal to be satisfied.

**Proposal**:
A KLine emitted by the Agent as a candidate response during rationalisation.

**Significance**:
A participant's subjective assessment of how well a KLine relates to what that participant already knows — classified into four levels: S1 (fully grounded), S2, S3, S4 (completely novel). Every participant assesses independently. Kalvin realises its own assessment by _computing_ it from structure (a 64-bit inverted distance whose bands map onto the structural states); that structural mapping is Kalvin's method, not a definition of the levels. See **Structural State** for the objective structural facts the bands map onto.
_Avoid_: confidence, score, weight

**KValue**:
The unit of exchange between participants — a **KLine** (objective structure) paired with a **significance** (the sender's assessment of it). In flight a KValue carries both; in memory only the KLine is stored, and significance is re-derived on retrieval.
_Avoid_: KLine-with-significance, annotated kline, assessed kline (all describe the wrapping, not the unit)

**Structural State**:
The structural relationship between a kline's signature and its nodes, implied by the written relational token that produces the kline. A closed set of five states, each produced by compiling a written relational token (`==`, `=`, `>`, `=>`) or, for identity, by the absence of one:

- **COUNTERSIGNED** — bidirectional: emits a reciprocal pair `{A: [B]}`, `{B: [A]}`.
- **CANONIZED** — aggregated: `{A: [B, C, D]}`. The structural state produced by the `=>` written token; it declares an intent to aggregate, not that the result is a Canon (see Canon).
- **CONNOTED** — forward: `{A: [B]}`.
- **UNDERSIGNED** — reversed: `{B: [A]}`.
- **IDENTITY** — `{A: []}` or `{A: [A]}` (self-referential) — see Identity.

States describe structure, not operation: no actor operates on a kline; the written token merely declares the relationship, and the state names the resulting structure.
_Avoid_: operator (implies an action); COUNTERSIGN / UNDERSIGN / CONNOTATE / CANONIZE as state names (those are the written-token names)

**Identity**:
A kline that carries no decomposition — either of two forms: empty nodes (`{S: []}`) or self-referential (`{S: [S]}`, whose sole node is its own signature). The self-referential form is identity _by definition_: a value that decomposes into itself carries no further information, and this overrules any CANON classification. Every kline bottoms out at one or more identities.
_Avoid_: unsigned (implementation term), bare signature (describes the syntax, not the structure)

**STM (Short-Term Memory)**:
The lowest tier in the write cascade and Kalvin's event register — every write reaches it. All klines encountered during a session pass through: queries, retrieved candidates, expanded proposals, and ratified results. Empty at session start.
_Avoid_: STM caching (too vague), working memory (too vague), context window (implies a passive buffer)

**Frame**:
Recognised working context — the klines Kalvin has matched, expanded, or ratified during a session. Receives only expanded proposals (S2/S3), ratified klines (S1), and novel klines (S4); the slow-path query reaches STM alone unless cogitation produces a result. Monotonic; persisted across sessions.
_Avoid_: session log (Frame is not a log), session

**LTM (Long-Term Memory)**:
Persistent knowledge that survives across sessions. Structurally identical to Frame; the distinction is semantic. Monotonic; loaded at session start, saved at session end. Kalvin learns continuously (monotonic memory); under S1 ratification it grounds an entry by updating LTM. It does **not** emit a grounding event — grounding is a memory update, not a signal.
_Avoid_: persistent store (too vague), knowledge base, LTM frame

**Escalation**:
The Trainer deferring a proposal it cannot auto-ratify to the supervisor for resolution. The boundary between what the Trainer resolves and what the supervisor resolves.

**Ratify**:
The action of countersigning a selected proposal. Usually performed by the Trainer during curriculum execution.

**Canon**:
A relationship kline whose signature is the OR-reduction of its nodes (`signature == make_signature(nodes)`) — structurally a self-grounded fit (S1), even when compiled at the S2 band. The structural fact; distinct from CANONIZED, which names the written token (`=>`) and the intent to aggregate. A CANONIZED kline need not be a Canon.
_Avoid_: canonical (ambiguous with the Structural State), canonized (the intent, not the property), MTS (an example, not the concept)

**MTS Entry**:
A Canon produced by multi-token signature expansion — the tokenizer encodes a compound signature as the OR-reduction of its constituent token IDs, so the resulting kline is a Canon by construction. An instance of Canon, not the definition; other Canons need not arise from MTS expansion.

<!-- "Learned" was retired: the glossary entry framed learning as a detectable,
band-matched Kalvin event, which is both wrong and redundant. Kalvin learns
continuously (monotonic memory) and grounds under S1 by updating LTM; it emits
no grounding event, so there is no event to match a "declared band" against.
The dialogue-driven-training spec derives run termination from dual cursor
exhaustion (trainer-side loop + stub), not from any "learned ⊇ submitted"
computation. -->

**Word Binding**:
The association of a single-character KScript signature with a word, resolved through BPE annotations in the source. Bindings are scoped by relational-token boundaries; a character resolves to the most recent matching word in its scope.
_Avoid_: comment mapping (the binding is a specific compiler artefact, not a general comment feature)
