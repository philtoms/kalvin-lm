# Domain Glossary

Kalvin is a rationalising system whose entire world is built from klines. This glossary defines the precise meaning of terms used across the code.

## Structure

The objective shape of a kline and the significance that shape claims on its own — no model, no observer.

**KLine**:
The fundamental unit of Kalvin's memory and the unit Kalvin rationalises: a **signature** (head) and a **nodes** list, between which holds a relationship Kalvin rationalises as significance.

**Signature**:
The value in a kline's head position — what its nodes compose against, and what other klines hold as nodes to evaluate significance.

**Node**:
A structural slot: a value in a kline's nodes list — either a **Token Id** or the signature of another kline.
_Avoid_: child, element

**Significance**:
The measure of the gap between two klines A and B — Kalvin's formalisation of understanding. Derived structurally by forming the relationship kline `signature_of(A.nodes): B.nodes` and classifying its shape; refined through traversal into a graded distance partitioned into four bands:

- **S1** — exact: the signature covers its nodes. _I know that I know this._
- **S2** — relates but diverges: at least one node covered. _I infer this, but it does not yet fit._
- **S3** — connects only indirectly: no node covered. _I recognise aspects of this, indirectly._
- **S4** — shares nothing: no connection can be drawn. _I do not understand this at all._
  _Avoid_: confidence, score, weight, grounded

The graded distance is fixed, not free: the Jaccard overlap of the two contents, discounted by the mean witness depth at which A's content is held (`γ = J · δ^D̄`).

**Terminal**:
A kline whose structure carries no further decomposition — a leaf that stops traversal. Two shapes: **Unknown** and **Identity**.
_Avoid_: leaf node, base case, atomic

**Unknown**:
`{S: []}` — nothing held for this signature; the structural form of an ask. Claims S4. In cogitation an unknown signature is a halt signal — the event under which Kalvin generates ungrounded proposals.
_Avoid_: empty kline, bare signature, identity

**Identity**:
`{S: [S]}` — a directly decodable known value. Claims S1.
_Avoid_: unsigned, treating the empty kline as an Identity

**Canon**:
The signature equals `signature_of(nodes)` — the signature stands for its nodes and is safe to use in their place. Claims S1.
_Avoid_: canonical; treating `=>` as synonymous (the token declares intent to compose; the result need not be a Canon); MTS (an example, not the concept)

**Misfit**:
A non-terminal whose signature does not equal `signature_of(nodes)`. Claims S2 when at least one node is covered by the signature, S3 when none is.
_Avoid_: fabrication, conjecture

**Relationship**:
The single-node misfit — the connotation/denotation shape — named in its own right as a distinct routing class (a candidate for reciprocal grounding), set apart from multi-node misfits, which propose.
_Avoid_: link, association, any-non-identity

The nine structures, the band each claims, and the rewrite operation each doubles as:

| Structure   | Shape         | Band | Rewrite operation | Scripted form  |
| ----------- | ------------- | ---- | ----------------- | -------------- |
| Canon       | `ABC:[A,B,C]` | S1   | expand nodes      | `ABC => A B C` |
| Identity    | `A:[A]`       | S1   | none — done       | `A = A`        |
| Underfit    | `ABC:[A,C]`   | S2   | rewrite A→B       | `ABC => A C`   |
| Overfit     | `AB:[A,B,C]`  | S2   | rewrite B→A       | `AB => A B C`  |
| Under+over  | `ABC:[B,C,D]` | S2   | rewrite both ways | `ABC => B C D` |
| Denotation  | `AB:[B]`      | S2   | remove nodes      | `A = B`        |
| Connotation | `A:[B]`       | S3   | replace nodes     | `A > B`        |
| No-fit      | `AB:[C,D]`    | S3   | rewrite both ways | `AB => C D`    |
| Unknown     | `A:[]`        | S4   | halt              | `A`            |

**Signature Behaviour**:
Every node is a symbol with signature behaviour — the axioms that constrain how nodes may interact. They are discrete (A AND B is (NOT A) AND (NOT B)); they compose (A OR B → AB); they overlap (AB and BC overlap on B); they decompose (AB → A and B); they relate (A is to B as A is to AB). Every kline construction cogitation performs is constrained by them.

## Rationalisation

How a participant tests a kline's structural claim against what Kalvin actually holds.

**Cogitation**:
The slow path of rationalisation: rewrite operations derived from the nine structures applied stepwise from a new kline A towards klines B already held, bridging the gap through **Candidates**. Significance measured at each step — and its rate of change — feed back into the strategy in real time, telling Kalvin whether its effort is increasingly or decreasingly significant.
_Avoid_: thinking, background thread, the cogitator

**Candidates**:
The klines that fill a rewrite's slots. A is the input or queued kline; B candidates are grounded klines — and a grounded kline holds content, so an Unknown never targets. A B is selectable when its signature occurs as a node of A; the pair's band then routes the derivation — S2 into ordinary targeting, S3 into the progressive path. STM lets S3 relationships evolve stepwise towards S2 overlap through progressive connotation.

**Reentry**:
Rationalisation on more than one axis — space (significance as a metric of distance between klines) and time (projecting A:B into the future at C) — where the sequence output of one axis is rationalised as the input of another: `space(AB) → time(AB'C) → space(B'C)`. In the engine, the reentry arm proposes from a proposal, one hop further out.

**Model**:
The whole of what Kalvin holds and how it holds it: the klines, their signature/node references, the memory tiers as relations of attention and commitment, and the signifier's compositional interpretation that makes the whole traversable.
_Avoid_: the learned function; using model and memory interchangeably

**Memory**:
The tiered structure inside the **Model**. Tiers are modes of relation to held klines, not storage locations; a tier change is a change in how Kalvin relates to a kline, so tier changes belong to rationalisation.

**STM**:
What Kalvin was just thinking about — the recency-of-attention relation, written by whatever cogitation touches. How traversal is temporally situated. Empty at session start.
_Avoid_: working memory, context window, cache

**Frame**:
Where Kalvin's focus lies and how it is shifting — the active kline, the S1-grounded klines and proposals, and their S4 dispositions. Monotonic and signature-keyed: rejection is additive.
_Avoid_: session log, a bucket of working context

**LTM**:
What Kalvin counts on — held, grounded knowledge. Structurally identical to Frame; the distinction is the relation.
_Avoid_: persistent store, knowledge base

**Grounding**:
The model's mechanism for realising significance: if a signature is grounded, all of its nodes are grounded. Frame-grounded klines are available to cogitation; LTM grounding is a frame promotion Kalvin deems important enough to remember.

**KValue**:
The unit of exchange between participants — a KLine paired with a significance (the sender's assessment).

## KScript

The language that authors training material. A script is an encounter in dialogue form: klines and relational tokens declaring the structure the trainee will meet, step by step.

**Token ID**:
A value produced by the tokenizer: `(word_bit << 32) | bpe_token_id`. The word word (upper 32 bits) carries one bit per distinct word, assigned first-encountered at bits 0–30, bit 31 reserved for ASK. A multi-subword word is one word and one bit; a compound (MTS signature, DENOTES concatenation) composes no bit of its own — it is the OR-reduction of its component words' values.

**Relational Tokens**:
The closed set of written tokens that declare how a kline is produced. A token declares an intent; the resulting kline's **Structural Significance** may or may not satisfy it.

- `==` **COUNTERSIGNS** — reciprocal pair `{A:[B]}`, `{B:[A]}`
- `=>` **CANONIZES** — intent to aggregate `{A:[B,C,D]}`; the result need not be a Canon
- `>` / `<` **CONNOTES** — `{A:[B]}`; `A < B` ⇒ `B:[A]` (the identifier reverses to match the reading direction). Self-reference collapses to Identity
- `=` **DENOTES** — the compound-signature shape `A = B` ⇒ `{AB:[B]}`: the signature is the compound of both operands, the node the denoted value. Self-denote collapses to Identity
- none **UNKNOWN** — a bare signature: unbound compiles to `{A:[]}` (the ask); word-bound to Identity `{A:[A]}`
- **ASK** — a bare compound or sigless annotation: keeps its original signature with the ASK bit marking it, so any signature can be an ask. S4

**Comment**:
A leading `#` — the rest of the line is dropped by the lexer and never reaches binding or klines.

**Annotation**:
The semantic layer of KScript: parenthetical prose instructing the agent running the session what the surrounding structure means, and resolving **Word Binding**. Exists for the agent, not the trainee; absence is a missed opportunity, never a compilation error.
_Avoid_: comment

**MTS (Multi-Token Signature)**:
A device for representing a multi-token signature on the LHS: the compiler expands a multi-character identifier into one MTS canon relationship plus a self-identity per word-bound token. An omitted identifier is synthesized from the preceding annotation's word initials — the capability large texts are chunked through.
_Avoid_: decomposition (a Canon decomposes into nodes; an MTS expands a signature into characters)

**Word Binding**:
The association of a single-character signature with a word, resolved through annotations. Precedence: inline annotation (nearest, overrides all others, also binds its immediate parent scope), then top-level annotations by scope, then resolved bindings (the char→word memory of earlier resolutions). Within a tier, the most recent match wins; each identity occurrence binds exactly once.
_Avoid_: comment mapping, rebind

## Training and Runtime

The multi-agent loop in which an authored encounter becomes understanding: a trainer presents the script's klines; a trainee rationalises them and assigns its own significance; a supervisor resolves what rationalisation alone cannot.

**Harness**:
The multi-agent runtime that loads participants and routes role-addressed messages between them. A message broker — participants never communicate directly.

**Message**:
A unit of inter-participant communication — addressed to a role with an action interpreted by the recipient.

**Dialogue**:
The alternating exchange between participants in the harness loop. No participant is aware it is in a training loop.

**Trainee**:
The participant under instruction — the rationalising system being trained. Role `trainee`.

**Trainer**:
A rationaliser — the trainer-side peer of the trainee, sharing the same rationalising engine and differing only in the significance bands it keeps (S1 ratifications, S2 proposals). Cogitates over incoming proposals and emits its own; escalates when cogitation yields no reply. Role `trainer`.
_Avoid_: auto-agent, training bot

**Supervisor**:
An agent that resolves what the Trainer escalates — deciding ratify, scaffold, or continue. Independent of medium (TUI, Slack, CLI, LLM). Role `supervisor`.
_Avoid_: UI, human

**Scaffolding**:
KScript entries that provide grounding context for other entries — structurally identical regardless of origin (pre-compiled by the author, or reactive from the supervisor). Delivery is a harness mode: batch (all before the group's opening entry) or on-demand (released as the trainee asks).

**Proposal**:
A KLine emitted by a trainee during rationalisation.

**Ratify**:
The action of countersigning a selected proposal. Usually performed by the Trainer while running a script.

**Escalation**:
The Trainer deferring a proposal to the supervisor when its cogitation yields no reply.
_Avoid_: auto-ratify failure

**Semantic Evidence**:
The undeclared backbone of a KScript: the canon index, countersign pairs, and denotation/connotation edges the entries hold collectively but no single kline declares. Emitted by compilation as derived structure, it carries the script's intended significance.

**Target Significance**:
The band a KScript production op declares — the answer key a trainee must learn to derive, not a measurement of any one kline.
