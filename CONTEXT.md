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

The objective shape of a kline — its signature and nodes — and the significance that shape _claims_ on its own, with no model and no observer. Structure is the ground truth every participant measures against; it is independent of who is looking.

**KLine**:
The fundamental unit of Kalvin's memory, and the unit Kalvin rationalises. A structure containing a **signature** (its head node) and a **nodes** list, between which holds a relationship Kalvin rationalises as a **Structural Significance**.

**Signature**:
The value occupying a kline's head position — the head value a kline's nodes compose against (see **Structural Significance**). Also a value other klines hold as nodes to evaluate **Rational Significance**.

**Node**:
A structural slot: a value occupying a position in a kline's nodes list. A node is either a **Token Id** or the **signature** of another kline.
_Avoid_: child, element (the structural slot is specifically a node)

**Structural Significance**:
The significance a kline's structure **claims** — an S-level (the same **S1**–**S4** as **Rational Significance**) derived from the signature–nodes relationship alone, without model traversal. Each structure makes its claim: **Unknown** claims **S4** (nothing held for this signature), **Identity** and **Canon** claim **S1** (a known value; a signature that stands for its nodes), **Misfit** claims **S2** (diverges). A claim that **Cogitation** measures against what Kalvin actually holds.

**Terminal**:
A kline whose structure carries no further decomposition — a leaf that tells Kalvin to stop traversing. Three shapes are terminal: empty nodes, self-referential nodes, and the compound-word form. _Avoid_: leaf node (a terminal is a kline, not a node), base case (implementation term), atomic (overloaded)

**Unknown**:
A kline **structure**: a **Terminal** with empty nodes (`{S: []}`). Claims **S4** — _"I don't know this"_ (nothing held for this signature). The structural form of an ask: an S4 proposal that requests an **Identity** ratification.
_Avoid_: empty kline (describes syntax, not the meaning), bare signature (describes syntax, not the structure), identity (the empty form is _not_ an identity — it is the opposite: unknown, not known)

**Identity**:
A kline **structure**: a **Terminal** that is directly decodable — a known value that translates to something in the outside world. Claims **S1** — _"I know this."_ One structural shape:

- self-referential (`{S: [S]}`)
  _Avoid_: unsigned (implementation term), bare signature (describes syntax, not the structure), treating the empty kline as an Identity (it is an **Unknown**)

**Canon**:
A kline **structure**: the signature equals `signature_of(nodes)`. Claims **S1** — the signature stands for its nodes, so it is safe to use the signature in place of them. The signature carries no information beyond what its nodes already express. Structural shape: `{AB: [A, B]}` where `AB` represents a combination of two or more nodes.
_Avoid_: canonical (ambiguous with Relational Tokens), treating `=>` (CANONIZES) as synonymous with being a Canon (the token declares an intent to compose; a CANONIZES statement need not construct a Canon), MTS (an example, not the concept)

**Misfit**:
A kline **structure**: the signature does not equal `signature_of(nodes)`. Structural shapes:

- no-fit (`{AB: [C, D]}`): signature attracts kline substitution. Claims **S2**
- underfit (`{AB: [A]}`): signature attracts kline expansion. Claims **S2**
- overfit (`{A: [A, B]}`): signature attracts kline contraction. Claims **S2**
- connote/denote (`{A: [B]}`): signature attracts association. Claims **S3**
  _Avoid_: fabrication (informal), conjecture/hypothesis (a misfit is a structure, not a distinct emission kind)

### Rationalisation

How a participant tests a kline's structural claim against what Kalvin actually holds — the slow, model-traversing path that arrives at a participant's own significance for a kline. Distinct from Structure (the claim) and from KScript's Target Significance (the authored answer): rationalisation is a participant's private derivation, and the gap between it and the target is what training closes.

**Significance (Rational)**:
The measurement of whether a kline's structural claim holds against what Kalvin holds — refined through model traversal and learned preferences. Classified into four levels of understanding:

- **S1** Fully accounted for — by its own structure (an **Identity** or a **Canon**) or by ratification. _I know that I know this._
- **S2** Relates but diverges: an unratified misfit, an active mismatch. _I infer this, but it does not yet fit._
- **S3** Connects only indirectly, through intermediaries. _I recognise aspects of this, indirectly._
- **S4** Shares nothing with what is held; no connection can be drawn. _I do not understand this at all._
  Every agent assesses independently. How Kalvin computes its own significance is a model concern (see **Grounding**); the levels themselves are independent of that computation.
  _Avoid_: confidence, score, weight, grounded (grounding is the model's implementation of S1, not a synonym for any level)

**Cogitation**:
The slow path of rationalisation — model traversal that tests a kline's structural **claim** against what Kalvin holds. Where **Structural Significance** is derived from the signature–nodes relationship alone, Cogitation expands the kline through the model: retracing paths, discovering connections, classifying each against the **Rational Significance** levels. It drains a backlog of unresolved (S2/S3) klines, emitting **proposals** for ratification; it is the work whose result is a Rationally Significant KLine - A kline that Kalvin understands.
_Avoid_: thinking (informal), background thread (implementation), the cogitator (the implementation class)

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

### KScript

The language that authors training material. A script declares klines and, through its relational tokens and the structures they generate, labels each with a **Target Significance** — the answer a trainee must learn to derive for itself. KScript is a compiler/provenance concern: it produces structures and their target labels, never a participant's lived significance.

**Token ID**:
A value produced by the tokenizer.

**Target Significance**:
The significance a compiled kline is _labelled_ with — the answer the script asserts the trainee should learn to derive.
_Avoid_: compiled significance (describes provenance, not the purpose), the kline's significance (a kline has no significance of its own — participants assign one; the compiled label is the target they are measured against), ground truth (overloaded with Grounding)

**Relational Tokens**:
The closed set of written tokens that declare how a kline is produced in KScript — `==` (COUNTERSIGNS), `=>` (CANONIZES), `>` (CONNOTES), `=` (DENOTES), or none (UNKNOWN). A compiler/provenance concept: the token declares an _intent_ (e.g. CANONIZES declares an intent to compose), which the resulting kline's actual **Structural Significance** may or may not satisfy.

- **COUNTERSIGNS** (`==`) — 1:1 emits a reciprocal pair `{A: [B]}`, `{B: [A]}`. The signature countersigns each other's nodes.
- **CANONIZES** (`=>`) — 1:many `{A: [B, C, D]}`. The signature canonizes its nodes into a single kline; this declares an intent to aggregate, not that the result is a Canon (see Canon).
- **CONNOTES** (`>`) — 1:1 `{A: [B]}`. The signature connotes each node (`A > B` ⇒ A connotes B; subjectively, _A is a B_).
- **DENOTES** (`=`) — 1:1 `{B: [A]}`. The signature denotes each node (`A = B` ⇒ A denotes B; objectively, _B is an A_).
- **UNKNOWN** — a bare, unbound signature. See **Unknown**. A bare signature with no **Word Binding** compiles to the empty Unknown `{A: []}` — the structural form of an ask. A bare signature that is word-bound compiles instead to an **Identity** `{A: [A]}` (see Identity): the binding gives it a decodable value, so the script labels it a known identity rather than an ask. Binding chooses the structure; the structure then determines the **Target Significance**.
  _Avoid_: structural relationship (collides with Structural Significance), relational operator (the token declares provenance, not an operation)

**MTS (Multi-Token Signature)**:
A KScript device for representing a multi-token signature on the LHS in a simpler syntax than would otherwise be required. A compound signature built from more than one Token ID by composition; the compiler expands a multi-character KScript identifier into its constituent character identities plus one MTS relationship. This expansion is a property of the _signature string_, distinct from any CANONIZES decomposition a script declares for that signature via a block. A CANONIZES scope's nodes are the declared block operands, never the signature's own MTS character expansion.
_Avoid_: decomposition (overloaded — a Canon decomposes into its nodes; an MTS expands a signature into characters), packed signature (the uint64 result, not the expansion)

**Word Binding**:
The association of a single-character KScript signature with a word, resolved through annotations in the source. Bindings are scoped by relational-token boundaries; a character resolves to the most recent matching word in its scope. Two annotation kinds bind with different strength: a **top-level annotation** (on a scope signature) binds only if the character is currently unbound — fill-if-empty, never overriding an outer binding; an **inline annotation** (on an item) binds unconditionally, overriding any outer binding for that occurrence. Each identity occurrence is bound exactly once by the most specific annotation that applies to it, so one character never acquires two competing tokens.
_Avoid_: comment mapping (the binding is a specific compiler artefact, not a general comment feature), rebind (a top-level annotation never overrides; an inline annotation always does — use the specific kind)

### Training and Runtime

The multi-agent loop in which authored material becomes understanding. A trainer presents klines carrying **Target Significance**; a trainee rationalises them and is graded on the gap; a supervisor resolves what rationalisation alone cannot. The runtime is where Structure, Rationalisation, and KScript's targets finally meet.

**Harness**:
The multi-agent runtime that loads agents as participants and runs a dialogue loop between them. A message broker — agents send role-addressed messages through the harness and it routes them to all subscribers of that role. Participants never communicate directly.

**Agent**:
A participant that forms its own **significance** on a kline — a subjective grading. Three roles, each registered on the harness bus, determine rationalising strategies: **Trainee** (Cogitation), **Trainer** (Cogitation + Escalation), and **Supervisor** (Oracle, Human-in-the-loop, etc).

**Message**:
A unit of inter-participant communication routed by the harness — addressed to a role with an action interpreted by the recipient.

**Dialogue**:
The alternating exchange between participants in the harness loop. No participant is aware it is in a training loop — each simply receives and responds.

**Trainee**:
The participant under instruction — the rationalising system being trained, and the subject of a training session. Registered on the harness bus with role `trainee`.

**Trainer**:
A rationaliser — the trainer-side peer of the trainee, sharing the same rationalising engine and differing only in the significance bands it keeps (S1 ratifications and S2 proposals). Cogitates over incoming proposals and emits its own; escalates to the supervisor only when its cogitation yields no reply. Registered on the harness bus with role `trainer`.
_Avoid_: auto-agent, training bot, the deterministic ratifier of the earlier path (it now rationalises; see `specs/dialogue-driven-training.md`)

**Supervisor**:
An Agent that resolves the proposals the Trainer escalates — deciding ratify, scaffold, or continue. Independent of medium — TUI, Slack, CLI, or an LLMSupervisor all share the same capabilities; a judgement may be a human decision or an LLM's internal assessment. Registered on the harness bus with role `supervisor`.
_Avoid_: UI (too narrow), human (a supervisor may be an LLMSupervisor)

**Curriculum**:
A living structured document owned by the Harness and accessible to all participants. The source of truth for training — never rolled back, only evolved forward. Three sections: **objective** (what it teaches), **approach** (the pedagogical strategy), and **lessons** (ordered KScript entries with human-readable context).
_Avoid_: lesson plan (too narrow — the curriculum is more than its lessons)

**Scaffolding**:
KScript entries that provide grounding context for other entries. Structurally identical regardless of origin; the difference is only when they are created — **pre-compiled** (written into the original script by the curriculum designer) or **reactive** (written by the supervisor when Kalvin's S2/S3 proposals mismatch expectations).

**Proposal**:
A KLine emitted by an agent during rationalisation.

**Ratify**:
The action of countersigning a selected proposal. Usually performed by the Trainer during curriculum execution.

**Escalation**:
The rationalising trainer deferring a proposal to the supervisor when its cogitation yields no reply. The boundary between what the Trainer resolves by rationalising and what the supervisor resolves.
_Avoid_: auto-ratify failure (the earlier path's trigger — the trainer now escalates on cogitation-empty, not on a failed deterministic countersign)

**Expectation**:
A scripted kline that enters the slow path (S2/S3) during rationalisation and requires a matching proposal to be satisfied.

**Grounding**:
The model's mechanism for realising **S1** (recognised). A kline is **grounded** when the model counts it as S1 — either by its own structure (a canon self-grounds) or by residing in LTM via ratification (a structural fact the model owns). Grounding is how S1 is _produced_, not what S1 _means_; "recognised" is the significance-level concept.
_Avoid_: self-grounded (legacy; conflates the mechanism with the level), grounded identity (grounding applies to any kline that attains S1, not just identities)

**Auto-Tune**:
The project's experimental loop for tuning Kalvin's rationalisation behaviour. An LLM coding agent runs repeated sessions against a curriculum, observes how the reactor/cogitator/rationaliser actually behave, edits the significance-model code (`expand()`, `significance.py`, the rationaliser) together with the owning spec, and re-runs to confirm.
_Avoid_: tuning session (ambiguous with training session), auto-train (it's not training), auto-tune supervisor (the CLI includes the full auto-tune tool, not just the supervisor).
