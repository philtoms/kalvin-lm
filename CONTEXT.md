# Kalvin — Context

This document has three sections. **Operating Notes** contains process instructions and conventions. **Domain Glossary** defines the precise meaning of terms used across code and docs. **Project Navigation** maps the source tree to its concerns — the entry point for finding where a concept lives in code. Do not mix the three: glossary entries are domain terms only; operating notes are behavioral rules; navigation points into source, never restating it.

---

## Operating Notes

- Commit all work before creating any kb tasks.
- When creating kb tasks for large features, decompose into discrete code tasks with explicit `depends` chains. Each task should cover one coherent piece of work — a single module or a single behavioural change. Do not create monolithic tasks that span multiple modules.
- Source is the truth document. The **Project Navigation** section below maps the source tree to its concerns; read the code for behaviour. When a term's precise meaning matters, use this glossary.
- CONTEXT.md is a glossary plus operating notes plus project navigation. Keep the three sections separate. Do not add implementation details or code to the glossary or operating notes; navigation points into source, never restating it.
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
_Avoid_: auto-agent, training bot, the deterministic ratifier of the earlier path (it now rationalises; see `src/dialogue/`)

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

---

## Project Navigation

Source is the truth document. This section maps the source tree to its concerns so a reader can locate where a concept lives. Each entry names the module(s) and the role they play; the code itself is authoritative for behaviour. Filled incrementally as areas are appraised.

### Signature & Token primitives

The value layer — how text becomes opaque uint64 nodes, and the bit-algebra over them.

- `src/kalvin/abstract.py` — the `KTokenizer` (text↔nodes) and `KSignifier` (signature creation, `signifies` overlap, `residual`) interfaces. Layout- and algebra-agnostic.
- `src/kalvin/tokenizer.py` — `Tokenizer`, the BPE-engine wrapper (train/save/load, raw `encode_bpe`/`decode_bpe`). Not a `KTokenizer`; produces raw BPE IDs, not nodes.
- `src/kalvin/nlp_tokenizer.py` — `NLPTokenizer`, the sole concrete `KTokenizer`. Owns the node layout `(sig_word << 32) | bpe_token_id`, the type dictionary, and the `POS_X` (`65536`) fallback for untyped tokens. Builds on `Tokenizer`.
- `src/kalvin/signifier.py` — `NLPSignifier`, the sole concrete `KSignifier`. Owns the NLP bit-algebra: `signature_of` (OR-reduce), `signifies` (upper-32 type-word overlap), `residual` (masked set-difference). The empty node sequence reduces to signature `0`.

### KLine & KValue

The memory unit and the exchange unit.

- `src/kalvin/kline.py` — `KLine` (signature + ordered nodes; equality/hash by both) and the structural predicates: `is_terminal`/`is_unknown`/`is_identity`/`is_canon`/`is_misfit`/`classify_misfit`. `KDbg` carries non-structural provenance (op, label, decoded text) for display only.
- `src/kalvin/kvalue.py` — `KValue`, the unit of exchange: an immutable `KLine` paired with a sender's `significance`. Identity is structural — equality and hashing consider `kline` only, ignoring `significance`.
- `src/kalvin/events.py` — `RationaliseEvent` (carries `query` and `proposal` as two KValues, no top-level significance field) and the `EventBus` pub/sub adapter.

### Model, STM, Significance, Expansion

The memory and the significance algebra. Four cooperating modules; the dependency is strictly one-way: significance (byte algebra) ← expand (graph walk) ← proposals (misfit reshape); all read the Model.

- `src/kalvin/model.py` — `Model`, the four-tier collection (STM → Frame → LTM → Base). The write cascade (`add_to_stm` / `add_to_frame` / `add_to_ltm`), unified cross-tier read API (`find`, `find_all`, `find_by_nodes`, `exists`, `grounded`, `where`, `klines`), graph traversal (`resolve`, `query_expand`, `unpack`, `query`), and `is_countersigned`. `KLineStore` backs Frame and LTM; `_TierChain`/`_TierAdapter` normalise the tiers. S1 recognition is `model.grounded()` (Frame/LTM/Base presence) composed with structural predicates at call sites — there is no standalone `is_s1`.
- `src/kalvin/stm.py` — `STM`, a bounded (default 256) dual-keyed index (signature + nodes-signature) with FIFO eviction and snapshot iterators.
- `src/kalvin/significance.py` — the 8-bit compositional grade: `SIG_MASK`/sentinels, `distance_to_byte`, `BandLayout` (only `S2_S3_BOUNDARY` configurable), the band-representative constants `SIG_S1..SIG_S4`, `band_significance` (production op → Target Significance), and the `Aggregator` bundling layout + `DecayFunction`/`ComposeFunction` seams (`DEFAULT_AGGREGATOR`).
- `src/kalvin/expand.py` — `expand()` (compose-on-return graph expansion yielding connotation `KValue`s + a terminal grade) and `edge_hops()` (bounded non-canonical resolution chain, cycle/dead-end/canon/unknown termination).
- `src/kalvin/proposals.py` — misfit comprehension: `propose_expansions` / `generate_expansions` reshape an underfit/overfit/dual misfit into self-consistent klines plus companions (no invention, no orphan nodes; terminals never emitted).
- `src/kalvin/agent_codec.py` — binary/JSON serialization for Agent persistence (STM/Frame/LTM + activity); storage is objective-only, significance never persisted.
- `src/kalvin/paths.py` — data-directory resolution (`tokenizer_dir`, `agent_bin`, etc.).

### Cogitator & Rationaliser

The rationalisation pipeline: a fast path (routing) and a slow path (background cogitation).

- `src/kalvin/rationaliser.py` — `Rationaliser` (aliased `Agent`), the orchestrator. `rationalise(KValue)` runs the phased pipeline: Phase 1 prepare (asserts signature set), Phase 1b significance-comparison gate (declared-S4 disagreement drops), Phase 2 ground check, Phase 3 assess (Unknown/Identity/canon/countersigned fast-tracks), Phase 4 candidate retrieval, Phase 5 route-and-submit. Also `_route` (node-membership → S2/S3), `_promote_participating` (cascade participating STM klines to LTM on S1), the `CogitationHandler` callbacks (`on_s1`, `on_expansion`), `countersign`, and serialization delegation to `AgentCodec`. S1 recognition = `model.grounded()` + structural predicates — there is no standalone `is_s1`.
- `src/kalvin/cogitator.py` — `Cogitator` (background daemon thread), `WorkItem` (query|candidate|level), and the `CogitationHandler` protocol. Drains the backlog by `expand()`-ing each pair, classifying yields via `BandLayout`, calling `on_s1` on a terminal S1 (then breaking) or `on_expansion` for S2/S3 proposals via `propose_expansions`. Emits `"done"` after an idle timeout (default 2s) without halting; `drain(timeout)` blocks until backlog empty and no work item processing (inter-lesson drain, `_processing` flag guarded).

### KScript

The authoring language. A four-stage pipeline (`src/ks/`) compiles declarative scripts into encoded `KValue`s: source → lexer → parser → ASTEmitter → TokenEncoder. The compiler treats encoded node values as opaque `uint64`.

- `src/ks/token.py` / `src/ks/lexer.py` — `TokenType`/`Token` and the `Lexer`: operators (`==` `=>` `>` `=`), case-insensitive `SIGNATURE` identifiers `[a-zA-Z][a-zA-Z0-9]*`, parenthesised `ANNOTATION`s (nested, multi-line), Python-style INDENT/DEDENT.
- `src/ks/ast.py` / `src/ks/parser.py` — the scope-model AST (`OperatorScope` = sig + op + items + child_block; `Signature` items may carry an inline annotation) and the recursive-descent `Parser`. Scope is operator-delimited; the preceding identifier is the signature, succeeding identifiers are nodes, INDENT extends the scope.
- `src/ks/binding_scope.py` — `BindingScope`, the word-binding stack implementing rules B1–B4: first-letter matching (case-insensitive) with a per-scope-per-character occurrence counter, counter reset on `push_scope`, and inline `bind_override` (binds tighter than word-list).
- `src/ks/ast_emitter.py` — `ASTEmitter`: walks the AST emitting `SymbolicEntry` tuples. Operator rules (COUNTERSIGNS bidirectional per-item, DENOTES reversed, CONNOTES forward, CANONIZES aggregated), **MTS** (a multi-char all-uppercase identifier emits exactly one CANONIZES canon over its resolved characters — no per-component entries), CANONIZES subscript identity filling, CANONIZES dedup, and Rule B4 inline-override patching of the parent MTS canon.
- `src/ks/token_encoder.py` — `TokenEncoder`: encodes symbolic entries to `KValue`s via the tokenizer. Compound-word decomposition (a word the BPE tokenizer splits into ≥2 subwords → one self-referential identity whose signature is the OR-reduction of the subwords), the canonical-signature registry (a declared compound's signature computed once and reused by references), and the source-before-decomposition output partition.
- `src/ks/compiler.py` / `src/ks/__init__.py` — `Compiler` (orchestrator; always creates a BindingScope) and `KScript` (the one-shot public API: `KScript(source).entries → list[KValue]`).

### Training: trainer, reactor, curriculum

The training driver. The `Trainer` submits curriculum lessons to the rationaliser, tracks satisfaction, and routes proposals it cannot auto-resolve to the supervisor.

- `src/training/trainer/curriculum_document.py` — `CurriculumDocument` (markdown parser: three required sections `## Objective`/`## Approach`/`## Lessons`, `### <label>` lessons with stable labels `\d+[a-z]?`, fenced KScript blocks) and `Lesson`. Supports `from_file`/`from_string` and `amend` (insert/append/modify with write-back).
- `src/training/trainer/curriculum.py` — `Curriculum` (ordered lesson container, document- or flat-list-backed) and `CurriculumState` (per-session tracking: entry-level `submitted`/`satisfied`/`pending` `EntryKey` sets **and** label-level `lesson_submitted`/`lesson_satisfied`, JSON persistence with legacy-format compat).
- `src/training/trainer/curriculum_generator.py` — `CurriculumGenerator`: LLM goal→curriculum markdown (one call, one retry on parse failure, slug-derived filename).
- `src/training/trainer/reactor.py` — `Reactor`: the Trainer's mechanical S2/S3 handler. Auto-countersigns structurally matching proposals (kline-only equality), and re-submits intra-lesson recurrences at declared `SIG_S4` (drop signal). Everything else returns `False` for the Trainer to escalate.
- `src/training/trainer/trainer.py` — `Trainer`: the embedded harness participant. Compiles and submits lessons, tracks satisfaction (`satisfied ⊇ submitted`, lesson-completion guarded against late cogitation), emits progress events, handles session lifecycle (goal/file resolution, file polling, amendment), and escalates unresolved proposals to the supervisor with a decision gate. Logging lives throughout (`kline_display`-decompiled event lines).

### Harness & supervisors

The multi-agent runtime and the supervisor participants. The harness is a message broker — participants send role-addressed messages through it and it routes them to all subscribers of that role (fan-out). It is not itself a participant.

- `src/training/harness/bus.py` — `MessageBus`: thread-safe role-based router with a single-dispatch event loop, fan-out to all subscribers of a role, wildcard diagnostic subscribers, and error replies for unknown roles.
- `src/training/harness/message.py` / `constants.py` — `Message` (role/action/message/sender; routed by role only) and the role constants (`trainee`/`trainer`/`supervisor`).
- `src/training/harness/adapter.py` — `RationaliserAdapter`: Kalvin's bridge to the bus. Handles `submit` (compile + rationalise each entry), `countersign` (reciprocal at S1), and `rationalise` (deliver a KValue as-is into the significance-comparison gate); maintains the sender map so callbacks route back to the originator; materialises three payload forms (live KValue, wire dict, legacy KLine).
- `src/training/harness/server.py` / `protocol.py` — `HarnessServer` (YAML/JSON config → embedded-participant registry + WebSocket + bus loop) and `WebSocketProtocol` (registration, bidirectional JSON frames, silent-drop on disconnect).
- `src/training/harness/llm.py` — shared `LLMClient` protocol, `LLMResponse`, `OpenAICompatibleClient` (used by the curriculum generator and the LLMSupervisor).
- `src/training/harness/__main__.py` — CLI entry point (loads config, wires participants, runs the server).
- `src/training/supervisors/commands.py` — the shared command parser mapping free-text to structured commands (`start`/`stop`/`pause`/`resume`/`goal:`/`ratify`/`scaffold:`/file-path/guidance).
- `src/training/supervisors/tui_client.py` / `slack_agent.py` / `cli_supervisor.py` / `llm_supervisor.py` — four client supervisors, all registering as role `supervisor`, sharing one decision contract. The **decision gate** lives in the Trainer (hold-and-replay, lesson-boundary drain window, `ratify`/`scaffold`/`continue` answers); the `LLMSupervisor` resolves `ratify_request`s via its own pipeline (prompt build, `#`-comment sanitisation, LLM call, scaffold extraction).
- `src/training/harness/README.md` — the operator guide for running the harness (survives as a usage doc).

### Dialogue subsystem

The authored-script ↔ real-actor ↔ rules triad. An authored **dialogue script** drives a turn-by-turn exchange between a Trainer (T) and Trainee (K); a **runner** decodes the script and drives two **actors** over the harness bus, tracking how much of the authored exchange the actors traverse. The script is one of three coupled artefacts (script, code, rules) the dialogue work exists to bring into agreement — not a golden master.

- `src/dialogue/decoder.py` — `DialogueScript`/`Turn`/`DecodedTurn`/`RunConfig` and `decode()`: a configuration-time resolver that builds each turn's kline from `source`, attaches significance by band, drops annotation-only turns, and treats `priors` as a sequence of independent runs (not merged). Handles UNKNOWN (`X:[]`), IDENTITY (self-referential or compound-word), and multi-CANONIZES labels.
- `src/dialogue/runner.py` — `run()`: a coverage-tracking wildcard subscriber over the `MessageBus`. A thin driver opens a run by delivering the first row to the opposite role; the bus then drives the exchange. Three terminal conditions (close observed / coverage exhausted / mutual PASS); `on_divergence` governs fail-vs-accept; `RunResult.uncovered` is the **displacement** (rows never emitted). White-box grounding verification via the trainee's `drain_observations`. `PASS` is the no-content sentinel.
- `src/dialogue/actors.py` — `EventSink`/`Actor` protocols and the actors: `ScriptTrainer`/`ScriptTrainee` (content-blind, cursor-advancing, bursts paced all-S1-or-all-non-S1), `SynthesizingTrainer` (derives replies from compiled source, falls back to the table for driving moves), `RationalisingTrainee`/`RationalisingTrainer` (wrap the rationalising engine; the trainee exposes `drain_observations`). Actors take an optional `RationaliserState` for state injection.
- `src/dialogue/rationalise.py` — the rationalising engine: derives one turn from `(state, incoming)` returning `(batch, observations)`. Two cogitation paths — the S3 countersignature path (pair two canons' operands, establish the reciprocal) and the S2 similar-fit-proposal path (recombine grounded klines, no invention) — plus the work-list, frame (dedup/match), and significance-as-structure routing.
- `src/dialogue/synthesize.py` — `synthesize`: the supervisor/engine behind the `SynthesizingTrainer`, deriving a trainer turn from the compiled script.

### Auto-tune

The project's experimental loop for tuning Kalvin's rationalisation behaviour. A CLI tool lets an LLM coding agent (pi) autonomously control training sessions, observe results, modify the significance-model code, and re-run — converging on a behavioural goal.

- `src/training/auto_tune/session.py` — `SessionConfig` (serialisable session config: session name, curriculum path, harness URL, model path, run counter, source branch/commit, worktree path) and `SessionDir` (directory layout, git worktree + branch management, config I/O).
- `src/training/auto_tune/lifecycle.py` — harness and supervisor process lifecycle: start/stop as background processes with PID tracking, readiness polling (fail-fast on spawn crash), orphan-port kill before bind, SIGTERM→SIGKILL escalation.
- `src/training/auto_tune/orchestrate.py` — the file-based protocol pi drives: `send_command`, `read_events`, `step` (write + block-until-next-event), `read_status`, and `summarize` — the **run-summary arbiter** that classifies the run (`completed`/`deadlocked`/`supervisor-stalled`/`stalled`/`crashed`/`incomplete`) from the event stream + trainer state, with the S1–S4 significance histogram and a diagnosis pointer.
- `src/training/auto_tune/snapshots.py` — `snapshot` (capture state/events/model/git metadata to a run directory), `restore`, and `reset`.
- `src/training/auto_tune/cli.py` / `__main__.py` — the 13-subcommand CLI (`init`/`teardown`/`start|stop-harness`/`start|stop-supervisor`/`send`/`events`/`step`/`status`/`summary`/`snapshot`/`restore`/`reset`).
- `src/training/supervisors/cli_supervisor.py` / `cli_events.py` — the CLI supervisor: a headless WebSocket client that blocks per-event (receive → write event → wait for command → process), and the event-enrichment layer (decompiled KScript source, significance breakdown, KLine/Significance display objects).

See **Auto-Tune** in the glossary.

See **Dialogue**, **Trainee**, **Trainer**, **Proposal**, **Ratify**, **Canon**, **Misfit** in the glossary.

See **Harness**, **Agent**, **Message**, **Dialogue**, **Supervisor**, **Trainee**, **Trainer**, **Scaffolding**, **Ratify**, **Escalation** in the glossary.

See **Curriculum**, **Scaffolding**, **Trainee**, **Trainer**, **Proposal**, **Ratify**, **Escalation**, **Expectation** in the glossary.

See **KScript**, **Relational Tokens**, **MTS**, **Word Binding**, **Target Significance** in the glossary.

See **Cogitation**, **Proposal**, **Ratify**, **Escalation**, **Expectation** in the glossary.

See **Frame**, **STM**, **LTM**, **Grounding**, **Significance (Rational)**, **Cogitation**, **S2 Expansion** in the glossary.

See **KLine**, **Signature**, **Node**, **Structural Significance**, **Terminal/Unknown/Identity/Canon/Misfit**, **KValue** in the glossary.
