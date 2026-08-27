# Domain Glossary

Kalvin is a rationalising system whose entire world is built from klines. This glossary defines the precise meaning of terms used across the code.

## Structure

The objective shape of a kline — its signature and nodes — and the significance that shape _claims_ on its own, with no model and no observer. Structure is the ground truth every participant measures against; it is independent of who is looking.

**KLine**:
The fundamental unit of Kalvin's memory, and the unit Kalvin rationalises. A structure containing a **signature** (its head node) and a **nodes** list, between which holds a relationship Kalvin rationalises as a **Structural Significance**.

**Signature**:
The value occupying a kline's head position — the head value a kline's nodes compose against (see **Structural Significance**). Also a value other klines hold as nodes to evaluate **Rational Significance**.

**Node**:
A structural slot: a value occupying a position in a kline's nodes list. A node is either a **Token Id** or the **signature** of another kline. At runtime a node may carry an optional label (the authored word it was encoded from).
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

**Relationship**:
A kline **structure**: the single-node misfit — a non-terminal whose signature associates with exactly one other value (`{A: [B]}`, `A != B`). The connote/denote shape, named in its own right because the engine treats it as a distinct routing class (a candidate for reciprocal grounding / countersignature) separate from multi-node misfits (no-fit/underfit/overfit, which propose rather than associate). A relationship is a kind of **Misfit**; it is not a synonym for "any non-identity" (a canon is also a non-identity, and a multi-node misfit is too).
_Avoid_: link (too vague), association (overloaded with the connote action), any-non-identity (a canon and a multi-node misfit are also non-identities)

## Rationalisation

How a participant tests a kline's structural claim against what Kalvin actually holds — the slow, model-traversing path that arrives at a participant's own significance for a kline. Distinct from Structure (the claim) and from KScript's Target Significance (the authored intent): rationalisation is a participant's private derivation; whether it made sense of the encounter is judged outside the loop.

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

**Model**:
The whole of what Kalvin holds and how it holds it: the klines, their signature/node references, the memory tiers as relations of attention and commitment, and the signifier's compositional interpretation that makes the whole traversable. Cogitation traverses the model through the tiers — conscious of what it just thought (**STM**), of where its focus lies and is shifting (**Frame**), and of what it counts as grounded (**Frame** and **LTM**).
_Avoid_: the learned function (Kalvin has no weights; understanding is traversal over held klines), using model and memory interchangeably (memory is the tiered structure inside the model)

**Memory**:
The tiered structure inside the **Model** — not a substrate beneath it. The tiers are modes of relation to held klines, not storage locations: **STM** is recent attention, **Frame** is current focus and its shift, **LTM** is held knowledge. A tier change (promotion, framing, eviction) is a change in how Kalvin relates to a kline, so tier changes belong to rationalisation, not storage bookkeeping. Untiered klines in a file are a serialisation; they become memory only when loaded into a model that can attend to them.

**STM (Short-Term Memory)**:
What Kalvin was just thinking about — the recency-of-attention relation to held klines. Written by attention: whatever cogitation touches hits STM. This is how traversal is temporally situated and how Kalvin can notice it is revisiting something. Empty at session start.
_Avoid_: STM caching (too vague), working memory (too vague), context window (implies a passive buffer), an index (an implementation detail of the attention relation, not the concept)

**Frame**:
Kalvin's focus of attention and how it is shifting. The active kline in **Cogitation** is held in Frame, and the S1-grounded klines and proposals with their S4 disposition that focused attention produces are registered there. Monotonic and signature-keyed: a signature accumulates a set of klines, so S4 rejection is additive (`Mary:[identity, canon]` → S4 → `Mary:[identity, canon, unknown]`).
_Avoid_: session log (Frame is not a log), session, a bucket of working context (Frame is a relation — where Kalvin's attention currently is — not a location)

**LTM (Long-Term Memory)**:
What Kalvin holds as grounded knowledge. Structurally identical to Frame; the distinction is the relation — LTM is what is counted on, Frame is what is in focus. A kline residing in LTM is **grounded** (see Grounding).
_Avoid_: persistent store (too vague), knowledge base, LTM frame

**Grounding**:
The model's mechanism for realising significance. If a signature is grounded, then Kalvin knows that all of its nodes are grounded also. KLines grounded in a **Frame** are available for cogitation. KLines grounded in _LTM_ are frame promotions that Kalvin deems important enough to remember.

**KValue**:
The unit of exchange between participants — a **KLine** (objective structure) paired with a **significance** (the sender's assessment of it).

## KScript

The language that authors training material. A script is an encounter, authored in dialogue form: klines and relational tokens declaring the structure the trainee will meet, step by step — priming before questioning, asks awaiting answers.

**Token ID**:
A value produced by the tokenizer.

**Relational Tokens**:
The closed set of written tokens that declare how a kline is produced in KScript — `==` (COUNTERSIGNS), `=>` (CANONIZES), `>` (CONNOTES), `=` (DENOTES), or none (UNKNOWN). A compiler/provenance concept: the token declares an _intent_ (e.g. CANONIZES declares an intent to compose), which the resulting kline's actual **Structural Significance** may or may not satisfy.

- **COUNTERSIGNS** (`==`) — 1:1 emits a reciprocal pair `{A: [B]}`, `{B: [A]}`. The signature countersigns each other's nodes.
- **CANONIZES** (`=>`) — 1:many `{A: [B, C, D]}`. The signature canonizes its nodes into a single kline; this declares an intent to aggregate, not that the result is a Canon (see Canon).
- **CONNOTES** (`>`) — 1:1 `{A: [B]}`. The signature connotes each node (`A > B` ⇒ A connotes B; subjectively, _A is a B_).
- **DENOTES** (`=`) — 1:1 `{B: [A]}`. The signature denotes each node (`A = B` ⇒ A denotes B; objectively, _B is an A_).
- **UNKNOWN** — a bare, unbound signature. See **Unknown**. A bare signature with no **Word Binding** compiles to the empty Unknown `{A: []}` — the structural form of an ask. A bare signature that is word-bound compiles instead to an **Identity** `{A: [A]}` (see Identity): the binding gives it a decodable value, so the script labels it a known identity rather than an ask.
  _Avoid_: structural relationship (collides with Structural Significance), relational operator (the token declares provenance, not an operation)

**Annotation**:
The semantic layer of a KScript: parenthetical prose that instructs the agent running the session, in high-level language, what the surrounding structure means — aligning an easily-understood concept with an otherwise opaque structural label. Also resolves **Word Binding**. Annotations exist for the agent, not the trainee: the trainee never sees them and their words are not encoded into klines. A script is valid without annotations; their absence or mismatch with a signature is a missed opportunity for the agent, never a compilation error.
_Avoid_: comment (an annotation is load-bearing for binding and interpretation), trainer rationale (it is instruction for the reading agent)

**MTS (Multi-Token Signature)**:
A KScript device for representing a multi-token signature on the LHS in a simpler syntax than would otherwise be required. A compound signature built from more than one Token ID by composition; the compiler expands a multi-character KScript identifier into one MTS canon relationship (compound → its resolved characters) plus a self-identity (`X:[X]`) for each word-bound token. The identifier may be omitted entirely: a single-line annotation followed directly by an operator (`(did Fred pet a sheep) =>`) synthesizes the MTS signature from the annotation words' initials (`DFPAS`), compiling identically to the explicit form — the capability large texts are chunked through.
_Avoid_: decomposition (overloaded — a Canon decomposes into its nodes; an MTS expands a signature into characters)

**Word Binding**:
The association of a single-character KScript signature with a word, resolved through annotations in the source. Uppercase letters bind to the nearest annotation or resolved binding, in three tiers: a **top-level annotation** (prefix, often on its own line) binds by scope — a prefix annotation at an inner scope binds before one at an outer scope; an **inline annotation** (on an item) is the nearest binding and overrides all others, and additionally binds uppercase letters in its immediate parent scope (the enclosing scope, not beyond) so it outlives scope exit without reaching unrelated outer scopes; finally, a **resolved binding** — the char→word memory of any earlier successful resolution — binds after all annotations, so a character unannotated in a later script still binds to its established word. Within a tier, a character resolves to the most recent matching word in its scope. Each identity occurrence is bound exactly once by the most specific annotation that applies to it, so one character never acquires two competing tokens.
_Avoid_: comment mapping (the binding is a specific compiler artefact, not a general comment feature), rebind (an inline annotation always overrides — use the specific kind)

## Training and Runtime

The multi-agent loop in which an authored encounter becomes understanding. A trainer presents the script's klines — the encounter, in dialogue form; a trainee rationalises them and assigns its own significance, proposing, asking, grounding; a supervisor resolves what rationalisation alone cannot. The runtime is where Structure, Rationalisation, and KScript's authored material finally meet, and where the agent reading the session judges whether the trainee is making sense of the encounter.

**Harness**:
The multi-agent runtime that loads agents as participants and runs a dialogue loop between them. A message broker — agents send role-addressed messages through the harness and it routes them to all subscribers of that role. Participants never communicate directly.

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

**Scaffolding**:
KScript entries that provide grounding context for other entries. Structurally identical regardless of origin; the difference is only when they are created — **pre-compiled** (written into the original script by its author) or **reactive** (written by the supervisor when Kalvin's S2/S3 proposals mismatch expectations).

**Proposal**:
A KLine emitted by a trainee during rationalisation.

**Ratify**:
The action of countersigning a selected proposal. Usually performed by the Trainer while running a script.

**Escalation**:
The rationalising trainer deferring a proposal to the supervisor when its cogitation yields no reply. The boundary between what the Trainer resolves by rationalising and what the supervisor resolves.
_Avoid_: auto-ratify failure (the earlier path's trigger — the trainer now escalates on cogitation-empty, not on a failed deterministic countersign)

**Semantic Evidence**:
The undeclared backbone of a KScript: the canon index (which signatures canonise to what), countersign pairs, and denotation/connotation edges that the entries hold collectively but no single kline declares. Emitted by compilation as derived structure (distinct from each entry's own **Structural Significance**), it carries the script's _intended_ significance — what the klines prove when every declaration is held.
