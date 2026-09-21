# Domain Glossary

Kalvin is a rationalising system whose entire world is built from klines. This glossary defines the precise meaning of terms used across the code. [docs/kalvin-algebra.md](docs/kalvin-algebra.md) fixes the formal definitions and the terminology (its §12); algebraic terms appear here as brief pointers to their normative definitions — the definitions are not restated. This glossary remains normative for role names, tier mechanics, KScript compilation, and the training protocol.

## Structure

The objective shape of a kline and the claim that shape makes on its own — no model, no observer.

**Atom**:
The indivisible unit of the reference realisation — one element of a finite set; in the engine, one bit per distinct word (Def 2, Appendix).
_Avoid_: token, subword (a Token ID or BPE token is an encoding, not an atom)

**Value**:
What signatures and nodes are made of — opaque content with the five capabilities of the value space (Def 1); a set of atoms in the reference realisation (Def 2).

**Node Sequence**:
A kline's nodes with order and multiplicity retained, no empty nodes (Def 3).

**Evaluation (`signature_of`)**:
The map from a node sequence to the value its nodes compose — forgets exactly order and multiplicity, nothing else (Def 4).

**KLine**:
The fundamental unit of Kalvin's memory and the unit Kalvin rationalises: a nonzero **signature** paired with a **node sequence** — a claim that the signature is the composition of the nodes (Def 5).

**Signature**:
The value in a kline's head position — the claim its nodes compose to, and what other klines hold as a node.
_Avoid_: head (positional name for the same thing)

**Node**:
A value in a kline's node sequence — either a **Token ID** or the signature of another kline; nesting is by reference, and the reference graph may cycle (Def 5, Def 7).
_Avoid_: child, element

**Exact**:
A kline whose signature equals `signature_of(nodes)` — the claim is kept (Def 6).

**Witness**:
An exact, non-empty kline — a chosen decomposition of its signature; the Identity is the trivial one (Def 6).

**Terminal**:
A kline inert as evidence — **Unknown** and **Identity**; targeting-closed, not rule-closed (Def 13, §7).
_Avoid_: leaf node, base case, atomic

**Unknown**:
`{S: []}` — nothing held for this signature; the structural form of the **ask**. Claims S4 (Def 10, §4).
_Avoid_: empty kline, bare signature, identity

**Identity**:
`{S: [S]}` — the trivial witness; a directly decodable known value. Claims S1 (Def 10, case 2).
_Avoid_: unsigned, treating the empty kline as an Identity

**Canon**:
An exact kline whose witness carries decomposition content. Claims S1; as evidence it must be well-founded (Defs 6, 10, 13).
_Avoid_: canonical; treating `=>` as synonymous (the token declares intent to compose; the result need not be a Canon); expansion (an example, not the concept)

**Misfit**:
A non-terminal kline that is not exact — claims S2 when covered, S3 when not (Defs 9–10).
_Avoid_: fabrication, conjecture

**Coverage**:
A node is covered by a value when they share content — overlap, not containment (Def 8).

**Fit**:
The total classifier `fit : V × V* → Shape` — nine shapes, four bands, exactly one per pair; reads a kline's own claim and the relationship `fit(C(A,B))` alike (Defs 10–11).

**Shape**:
One of the nine fit cases (Def 10); Underfit and Overfit also name the misfit quantities (Def 9), and Connotation and Denotation are single-node names of convenience (Def 10, §13).

**Band**:
The fit's tier, ordered S1 > S2 > S3 > S4 — the quantization of significance; observer-independent, so never exchanged (§4, §10).

**Relationship**:
The construction grading two klines against each other: `C(A,B) = signature_of(A.nodes) : B.nodes` — the head is defined, not claimed (Def 11).

The nine structures, their shapes, bands, replace modes, and scripted forms are tabulated at Def 10, Def 13, and §13.

## Measurement

The measures of understanding and of work — significance, complexity, and their composite γ — and the KValue that carries the sender's assessment between participants (§10).

**Significance**:
The measure of rational understanding — the content overlap of the two parties, whose quantization is the band; travels with the proposal (Defs 17, 20, §12).
_Avoid_: confidence, score, weight, grounded; structural significance (redundant — the band is quantized significance)

**Complexity**:
The measure of work — `1 − δ^(D̄ + Ĥ)` over the mean resolution and acquisition depths (Defs 18–20); independent of significance, never selects a band.

**Gamma (γ)**:
Significance net of complexity — `J · δ^(D̄ + Ĥ)`; its rate of change is the signal strategy acts on (Def 20).

**KValue**:
The unit of exchange between participants — a KLine paired with a significance (the sender's assessment). For a proposal that is its significance — J at entry depths; the complexity stays with the kline as its acquisition record.

## Rationalisation

How a participant tests a kline's structural claim against what Kalvin actually holds.

**Rationalisation**:
The process of producing, using, and consolidating the evidence represented in memory (§12).

**Cogitation**:
The slow path of rationalisation — the strategy loop (select → scope → derive → add → re-enter) steered by γ's rate of change (§11).
_Avoid_: thinking, background thread, the cogitator

**Hop**:
The strategy unit of one queued kline: goals from the top of its candidate list in order, each scoped and derived to an ending (Def 21).

**Derivation**:
The rewrite of a queued kline's node sequence against one held goal, `A ⊢_{M,B} A′` — signature fixed, states differ only in nodes (Def 12); its outcome is the significance established at the stopping state (Def 16).

**Replace**:
The only rule: a held correspondence's two sides swap at a multiset-wise occurrence — forward (signature → witness) or reverse; the evidence's own fit fixes the mode (Def 13).
_Avoid_: rewrite rule, mutation

**Canonicalisation**:
The Canon instance of the mirror clause — a survey-based reverse replace contracting exactly-witnessed node groups; not a second rule (Def 13).
_Avoid_: gather, reordering (no arrangement work exists — occurrence is multiset-wise)

**Correspondence**:
A held kline usable as evidence — every held kline but the two terminals; the whole set is the correspondence graph, and a derivation is a path in it (Def 13, §7).
_Avoid_: rule (in prose; a kline is not a rule), candidate (a candidate is a correspondence selected for use)

**Licence**:
What permits a replace — **witnessed** (memory-licensed canon moves, blind to the goal) or **evidenced targeting** (correspondence-licensed, scoped to the misfit region, strictly decreasing misfit mass) (Def 14, §7).

**Misfit Mass**:
The content the two sides disagree on — the unit of progress: every licensed targeting replace strictly decreases it (Def 14, T1).

**Done**:
The ending where significance is 1.0 — value-equality of the two parties, not node-equality (Def 16).

**Stuck**:
The ending where no licensed targeting move remains — no goal held, or no connection across the correspondence graph reaches the misfit (Def 16).

**Abandoned**:
Not an ending the rules produce: a strategy halt or retarget mid-derivation, e.g. on falling γ (Def 16).

**Ask**:
The structural halt condition: the Unknown shape is the ask's form, and a misfit region no held correspondence reaches asks — the event under which ungrounded proposals are generated (§4, §8; the marker: §13).

**Candidates**:
The held klines that may serve as a derivation's goal — the coverage pool ordered by descending γ(A, K) (Def 22).

**Scope**:
A derivation's memory for one hop: a dual-rooted, depth-bounded trawl of the correspondence graph, frozen for the hop's duration (Def 23). Engine realisation: a kline joins the scope when its signature or any node shares a word bit with the reached set — token-id bits carry no correspondence.

**Slot**:
A node carrying misfit content on either party of a relationship — an underfit node of A, an overfit node of B. A slot without a licensed replace is walked: a **meeting** of two **descents** (one from each party, licensed by heading alone) writes a **bridge** — the composed correspondence the main derivation consumes (Def 15).
_Avoid_: subgoal, subroutine, task

**Progressive Path**:
The evidence-building route from S3 to S2 — each hop's writes grow the memory later hops trawl (§11).

**Reentry**:
Derivations compose: a hop's end state queues as the next hop's input, reselecting its candidate list; bounded by the **hop ceiling**. Hop order is the only time the system has (§11).

## Memory

What Kalvin holds and how it holds it — the klines, their references, and the tiers as relations of attention and commitment. Tier mechanics are properties of the memory system rather than the algebra (§14).

**Model**:
The whole of what Kalvin holds and how it holds it: the klines, their signature/node references, the memory tiers as relations of attention and commitment, and the signifier's compositional interpretation that makes the whole traversable.
_Avoid_: the learned function; using model and memory interchangeably

**Memory**:
The tiered structure inside the **Model** — formally a finite set of klines (Def 7); the tiers are relations over it. Tiers are modes of relation to held klines, not storage locations; a tier change is a change in how Kalvin relates to a kline, so tier changes belong to rationalisation.

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

## KScript

The language that authors training material. A script is an encounter in dialogue form: klines and relational tokens declaring the structure the trainee will meet, step by step.

**Token ID**:
A value produced by the tokenizer: `(word_bit << 32) | bpe_token_id`. The word half (upper 32 bits) carries one bit per distinct word — the atoms' engine realisation — assigned first-encountered at bits 0–30; bit 31 is the ASK marker (`ASK_SIG`), reserved — no word ever carries it. A multi-subword word is one word and one bit; a compound (expansion-introduced, or a CONNOTES concatenation) composes no bit of its own — it is the OR-reduction of its component words' values.

**Relational Tokens**:
The closed set of written tokens that declare how a kline is produced. A token declares an intent; the fit classification of the produced kline may or may not satisfy it (§13).

- `==` **COUNTERSIGNS** — goal-targeted training: `A == B => C D` compiles to the queued ask `{A:[]}` (S4, the entry) and the implied goal `{B:[C,D]}` (a Canon or covered misfit per the block's scaffolding). No reciprocal pair is emitted; the engine's own selection is unchanged — the goal is the trainer's answer key. The harness feeds the ask at its subjective significance γ(A, B) (Def 20 — fresh content carries zero depths, so γ = J, the significance of the two contents); the structural S4 shape is never the exchanged byte. The engine's proposals under the goal grade the same way — the proposal's content against the goal's target (its signature value): the proposal that reached its goal is ratified at S1 and grounds on receipt (the stamp, not structure, is the licence); one off the goal grades low and refuses on re-feed
- `=>` **CANONICALISES** — intent to aggregate `{A:[B,C,D]}`; the result need not be a Canon
- `>` / `<` **CONNOTES** — the compound-signature shape `A > B` ⇒ `{AB:[B]}`, `A < B` ⇒ `{BA:[A]}` (reading order, `A < B ≡ B > A`): the signature is the compound of both operands, the node the connoted value. Self-reference collapses to Identity
- `=` **DENOTES** — `{A:[B]}`: the signature denotes each node. Self-denote collapses to Identity
- (none) **ASK** — a bare signature: unbound compiles to `{A|ASK:[A]}` (the ask — the identity shape, marked); word-bound to Identity `{A:[A]}` — except a sigless annotation's own utterance, always the ask with its binding resolving the nodes. The ASK op is the one op for every ask — a sigless annotation compiles to `{ABC|ASK:[a,big,cat]}` (the annotation's words as nodes), a `==` entry to the queued ask, a bare compound to its canon-noded ask. One ask structure: the signature carries the ASK marker OR-ed in and the canon's nodes ride along (`WDMH|ASK_SIG:[what,did,Mary,have]`) — the marker manufactures the ask's distinctiveness from its canon (identity, store keys, and lookups see it; every content measurement — signifies, residual, γ — masks it out) and the nodes are what candidate selection (Def 22) reads. S4

**Comment**:
A leading `#` — the rest of the line is dropped by the lexer and never reaches binding or klines.

**Annotation**:
The semantic layer of KScript: parenthetical prose instructing the agent running the session what the surrounding structure means, and resolving **Word Binding**. Exists for the agent, not the trainee; absence is a missed opportunity, never a compilation error. Brackets are optional when the word carries them in its case: a Capitalized identifier reads exactly as its bracketed form (`Mood` ≡ `M(ood)`) — a Capitalized word declares its binding — while ALL-UPPER stays a compound (`MHALL`) and lowercase-first stays a literal word (`had`). An inline annotation attaches to a single character's tail and witnesses one word: `Word(x)` and a whitespace-bearing tail (`M(ary had)`) are parse errors, not literals or concatenations — phrasal content belongs in a prefix annotation.
_Avoid_: comment

**Compound**:
A signature composed of multiple words: its value is the OR-reduction of its component words' values, so it takes no word bit of its own. Introduced two ways: by expansion — an ALL-UPPER identifier (`MHALL`) resolves each character via Word Binding, the resolved words becoming one canon kline's nodes — or by CONNOTES concatenation (`A > B` ⇒ `AB`). An omitted identifier is synthesized from the preceding annotation's word initials — the capability large texts are chunked through.
_Avoid_: decomposition (a Canon decomposes into nodes; an expansion introduces a compound's words); MTS (superseded name for the expansion)

**Word Binding**:
The association of a single-character signature with a word, resolved through annotations. An authored binding is case-blind — a witness `h(ad)` binds the compound char `H` as readily as `H(ad)` (the sig char's case is typographic; the word's case is the word) — but ambient attraction is sig-case-gated: a bare single character in sig case attracts the nearest scope's most recent word-list word with matching initial (occurrence-counted), while a lowercase single character is the literal word (the article `a`) and never attracts. An identifier's case frames its reading everywhere: ALL-UPPER is a compound (`MHALL`, alnum-only — a punctured identifier is a word), Capitalized carries its own expansion (`Mood` ≡ `M(ood)`), lowercase-first is a literal word (`had`), and so is a caseless first character — a digit or a word-internal punctuation mark (`42`, `3.14`, `don't`): identifiers admit `- . _ '` beyond alphanumerics, while the relational and structural marks (`= > < ( ) #`) are reserved and never part of an identifier, as are `? ! ,` (reserved syntax candidates); a word needing a reserved character is written bracketed — `w(hat?)` → `what?`. Precedence: inline annotation (nearest, overrides all others, also binds its immediate parent scope), then top-level annotations by scope, then resolved bindings (the char→word memory of earlier resolutions). Within a tier, the most recent match wins; each identity occurrence binds exactly once. A compile seeded with the prior state's known words (acquisition order) carries the binding across scripts: the seed is the outermost word list, binding chars the script cannot bind itself — an underfit question script's answer chars resolve to the earlier script's words instead of minting fresh.
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
KScript entries that provide grounding context for other entries — structurally identical regardless of origin (pre-compiled by the author, or reactive from the supervisor). Delivery is a harness mode: batch (all before the group's opening entry) or on-demand (released as the trainee asks). Scaffold groups open before ask groups: the trainer primes K before asking, so the ask's hops trawl the scaffold from memory.

**Proposal**:
A KLine emitted by a trainee during rationalisation. Ungrounded when generated under the ask — S3 evidence is a promise, not a fact; weighing promises is protocol.

**Ratify**:
The action of countersigning a selected proposal — usually performed by the Trainer while running a script. Its effect is a tier relation over memory: a ratified correspondence edge costs nothing in acquisition depth, and ratifying a traversed pair promotes it to a standing one-hop licence (§10–11, §14).

**Escalation**:
The Trainer deferring a proposal to the supervisor when its cogitation yields no reply.
_Avoid_: auto-ratify failure

**Semantic Evidence**:
The correspondences a KScript's entries hold collectively but no single kline declares: the canon index and the connotation/denotation edges. Emitted by compilation as derived structure, it carries the script's intended significance.

**Target Significance**:
The band a KScript production op declares — the answer key a trainee must learn to derive, not a measurement of any one kline.
