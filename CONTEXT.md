# Domain Glossary

Kalvin is a rationalising system whose entire world is built from klines. This glossary defines the precise meaning of terms used across the code. [docs/kalvin-algebra.md](docs/kalvin-algebra.md) fixes the formal definitions and the terminology (its §12); this glossary restates them as domain terms and remains normative for role names, tier mechanics, and the training protocol.

## Structure

The objective shape of a kline and the claim that shape makes on its own — no model, no observer.

**Atom**:
The indivisible unit of the reference realisation — one element of a finite set, only its finiteness load-bearing. In the engine, the word-bit space: one bit per distinct word (Def 2).
_Avoid_: token, subword (a Token ID or BPE token is an encoding, not an atom)

**Value**:
What signatures and nodes are made of — opaque: the algebra exercises composition `v ∨ w`, overlap `v ∧ w`, residue `v ∖ w`, the content measure `|v|`, and equality — nothing more (Def 1). A set of atoms in the reference realisation, where the Boolean laws hold by construction (Def 2).

**Node Sequence**:
A kline's nodes with order and multiplicity retained, no empty nodes (Def 3). Membership, difference, and occurrence are multiset-wise; no rule reads order — arrangement is witness structure alone (Def 12).

**Evaluation (`signature_of`)**:
The map from a node sequence to the value its nodes compose: `signature_of([n₁ … nₖ]) = n₁ ∨ … ∨ nₖ`. Forgets exactly order and multiplicity — nothing else. Node sequences are the terms; values are what they evaluate to (Def 4).

**KLine**:
The fundamental unit of Kalvin's memory and the unit Kalvin rationalises: a nonzero **signature** paired with a **node sequence**. A kline claims its signature as the composition of its nodes (Def 5).

**Signature**:
The value in a kline's head position — the claim its nodes compose to, and what other klines hold as a node.
_Avoid_: head (positional name for the same thing)

**Node**:
A value in a kline's node sequence — either a **Token ID** or the signature of another kline; nesting is by reference, and the reference graph may cycle (Def 5, Def 7).
_Avoid_: child, element

**Exact**:
A kline is exact when its signature equals `signature_of(nodes)` — the claim is kept. Underfit and overfit are both empty exactly then (Def 6).

**Witness**:
An exact, non-empty kline — a chosen decomposition of its signature. The Identity is the trivial witness; every other witness is a real choice, and which choice was made is a fact the algebra forgets and memory carries (Def 6).

**Terminal**:
A kline inert as evidence: an **Unknown** has no second side; an **Identity** replaces a node by itself. Targeting-closed, not rule-closed — an Identity's node may still expand under a held well-founded witness (Def 13, §7).
_Avoid_: leaf node, base case, atomic

**Unknown**:
`{S: []}` — nothing held for this signature; the structural form of the **ask**. Claims S4: the halt signal under which strategy generates ungrounded proposals (§4).
_Avoid_: empty kline, bare signature, identity

**Identity**:
`{S: [S]}` — the trivial witness; a directly decodable known value. Claims S1 (Def 10, case 2).
_Avoid_: unsigned, treating the empty kline as an Identity

**Canon**:
An exact kline whose witness carries decomposition content: the signature stands for its nodes and is safe to use in their place. Claims S1. As evidence a canon must be well-founded — identities and self-containing canons are the two inert witness classes (Def 13).
_Avoid_: canonical; treating `=>` as synonymous (the token declares intent to compose; the result need not be a Canon); MTS (an example, not the concept)

**Misfit**:
A non-terminal kline that is not exact. Claims S2 when at least one node is covered by the signature, S3 when none is (Def 10, cases 4–8).
_Avoid_: fabrication, conjecture

**Coverage**:
A node is covered by a value when they share content — overlap, not containment: a covered node may carry content outside the value (Def 8). The classifier's primary split: covered misfits are S2, uncovered are S3.

**Fit**:
The total classifier `fit : V × V* → Shape` — one function, two readings: a kline's own fit (the claim it makes standing alone) and the relationship fit `fit(C(A,B))` (what a pair establishes). Nine shapes, four bands; every pair matches exactly one (Def 10, §10).

**Shape**:
One of the nine fit cases — Canon, Identity, Underfit, Overfit, Under+over, Connotation, Denotation, No-fit, Unknown. Underfit and Overfit also name the misfit quantities (Def 9): the content the signature claims beyond its nodes, and the content the nodes carry beyond the signature. Connotation (single-node Underfit) and Denotation (uncovered single-node) are names of convenience for KScript; algebraically they are single-node instances of cases 6 and 4 (Def 10).

**Band**:
The quantization of significance — the fit's tier, ordered S1 > S2 > S3 > S4, shapes within a band unordered:

- **S1** — exact: the claim is kept. _I know that I know this._
- **S2** — covered misfit. _I infer this, but it does not yet fit._
- **S3** — uncovered misfit. _I recognise aspects of this, indirectly._
- **S4** — Unknown. _I do not understand this at all._

Observer-independent — given the same held memory, every agent classifies alike — so a band is never exchanged; it is recomputable from structure (§10).

**Relationship**:
The construction that grades two klines against each other: `C(A,B) = signature_of(A.nodes) : B.nodes` — the head is defined, not claimed, so all misfit comes from B's side. `fit(C(A,B))` is the structural relationship of A and B; Canon iff the two klines hold the same value, differently decomposed (Def 11).

The nine structures, the band each claims, and the replace licence each doubles as (Def 13 — mode by the evidence's own shape, direction by arrival):

| Structure   | Shape         | Band | Replace mode        | Scripted form  |
| ----------- | ------------- | ---- | ------------------- | -------------- |
| Canon       | `ABC:[A,B,C]` | S1   | expand / contract   | `ABC => A B C` |
| Identity    | `A:[A]`       | S1   | inert — terminal    | `A = A`        |
| Underfit    | `ABC:[A,C]`   | S2   | shed fwd, adopt rev | `ABC => A C`   |
| Overfit     | `AB:[A,B,C]`  | S2   | adopt fwd, shed rev | `AB => A B C`  |
| Under+over  | `ABC:[B,C,D]` | S2   | shed and adopt      | `ABC => B C D` |
| Connotation | `AB:[B]`      | S2   | shed                | `A > B`        |
| Denotation  | `A:[B]`       | S3   | traverse            | `A = B`        |
| No-fit      | `AB:[C,D]`    | S3   | traverse            | `AB => C D`    |
| Unknown     | `A:[]`        | S4   | inert — the ask     | `A`            |

## Measurement

The measures of understanding and of work — significance, complexity, and their composite γ — and the KValue that carries the sender's assessment between participants (§10).

**Significance**:
The measure of rational understanding — the value rationalisation produces and consumes; understanding, informally, is high significance attained and held (§12). Formally the content overlap `J(σ(ν_A), σ(ν_B))` (Defs 17, 20): path-independent, 1.0 exactly at value-equality (done), 0 at disjointness; γ at entry depths, where the depths vanish and only overlap remains. The bands are its quantization — S1 = 1.0, S2 = overlap short of equality, S3 = zero overlap, S4 the vacuous halt off the scale — so one measure serves a kline's own claim and a relationship alike. Selects the band; travels with the proposal.
_Avoid_: confidence, score, weight, grounded; structural significance (redundant — the band is quantized significance)

**Complexity**:
The measure of work — how much effort arriving at a significance cost: `1 − δ^(D̄ + Ĥ)` (Def 20), the complement of the discount over the mean **resolution depth** at which A's content is held (granularity) and the mean **acquisition depth** of the unratified correspondence edges crossed to win it (provenance). Entry content and ratified standing licences cost nothing: hard-won until it consolidates. Independent of significance — it prices moving between embedded concepts — and never selects a band. The composite `γ = J · δ^(D̄ + Ĥ)`, significance net of complexity, compares derivations of equal significance and steers strategy by its rate of change.

**KValue**:
The unit of exchange between participants — a KLine paired with a significance (the sender's assessment). For a proposal that is its significance — J at entry depths; the complexity stays with the kline as its acquisition record.

## Rationalisation

How a participant tests a kline's structural claim against what Kalvin actually holds.

**Rationalisation**:
The process that produces and consumes significance (§12).

**Cogitation**:
The slow path of rationalisation — the strategy loop over derivations: **select** a goal, **scope** the memory, **derive** to an ending, **add** the result to memory, **reenter** with the output as the next queue's input (§11). Each phase is strategy: the rule system constrains what any of it may do, never what it must. Significance is graded at each state; γ's rate of change — significance net of complexity — feeds back, telling Kalvin whether its effort is increasingly or decreasingly worthwhile.
_Avoid_: thinking, background thread, the cogitator

**Hop**:
The strategy unit of one queued kline: goals taken from the top of its candidate list in order, each scoped and derived to an ending — a hop may run several derivations, one that ends without done yielding the next (Def 21). The queued head is fixed within the hop; each derivation's goal and scope are fixed for its duration (Def 12); writes land in memory for later hops alone. Re-entry changes A, and A reselects candidates for B; hop order is the system's only temporal structure.

**Derivation**:
The rewrite of a queued kline's node sequence against one held goal: `A ⊢_{M,B} A′`. The signature — the claim — never changes; states differ only in nodes (Def 12). The goal is read, never rewritten: it scopes targeting (Def 14), determines the ending (Def 16), and fixes the overfit the meeting walk bridges to (Def 15). Its outcome is the significance established at the stopping state — a calculated level, never a boolean (Def 16, §9).

**Replace**:
The only rule: a held **correspondence** kline's two sides swap at a multiset-wise occurrence in the node sequence — forward (signature → witness) or reverse (witness → signature). The evidence kline's own fit fixes the **mode**: canon — expand/contract, granularity at constant content; covered misfit — shed/adopt, its underfit out and its overfit in; uncovered misfit — traverse, disjoint atoms swap. Direction is not a property of the kline: arrival orients the licence (Def 13).
_Avoid_: rewrite rule, mutation

**Canonicalisation**:
The Canon instance of the mirror clause — the reverse replace engaged position-free: survey the unordered configurations of a node sequence against held witnesses and contract the correctly witnessed ones — the nodes covering the candidate compound from below (coverage), a held canon counter-witnessing exactly them from above. Held witnesses propose the configurations; nothing unwitnessed contracts. Not a second rule (Def 13).
_Avoid_: gather, reordering (no arrangement work exists — occurrence is multiset-wise)

**Correspondence**:
A held kline usable as evidence — every held kline except the two terminals (Def 13). Held klines are edges between a signature and its witness; the whole set is the **correspondence graph**, and a derivation is a path in it (§7).
_Avoid_: rule (in prose; a kline is not a rule), candidate (a candidate is a correspondence selected for use)

**Licence**:
What permits a replace — two kinds on one rule (§7): **witnessed** (canon-mode: expand/contract, licensed by memory alone, blind to any goal) and **evidenced targeting** (licensed by a correspondence and scoped to the misfit region — read on both ends of the move: forward departs the underfit or adopts the overfit, reverse consumes the underfit or lands in the overfit, so an empty underfit bars nothing; targeting-licensed iff it strictly decreases the misfit mass, Def 14).

**Misfit Mass**:
`|signature_of(A.nodes) Δ signature_of(B.nodes)|` — the content the two sides disagree on. The unit of progress: every licensed targeting replace strictly decreases it, and a run from entry is bounded by its initial value (Def 14, T1).

**Done**:
The ending where the calculation saturates: **significance 1.0** — the band S1, which states **value-equality**, `signature_of(A.nodes) = signature_of(B.nodes)` — not node-equality. Done may arrive early; pending nodes are witness structure. The final node sequence is a constructive witness for the equality, every step licensed by a held correspondence (Def 16, §9).

**Stuck**:
The ending where no licensed targeting move remains — not done, and nothing in memory connects. Two conditions, both the **ask**: no goal held, or no connection across the correspondence graph. Relative non-existence — the honest stopping when the bridge is missing. The significance established there — real overlap short of equality — is still the result (Def 16, §9).

**Abandoned**:
Not an ending the rules produce: strategy halts or re-targets a run mid-derivation, e.g. when γ — significance net of complexity — falls (Def 16). The significance established at abandonment is the result.

**Ask**:
The structural halt condition — no atom, mark, or decree involved. The Unknown shape (`S:[]` — nothing held) is the ask's shape, and a misfit region no held correspondence reaches asks. The event under which ungrounded proposals are generated (§4, §8).

**Candidates**:
The held klines that may serve as a derivation's goal, in order. The pool (Def 22): every held kline whose content covers a node of A's node sequence — overlap at the node level (Def 8). The order: descending `γ(A, K)` (Def 20) — the significance of working from A toward the candidate; significance, not band, sets the order. The goal is taken from the top of the list; a derivation ending without done yields the next candidate, so a hop may run several derivations down the list. A new A reselects the list.

**Scope**:
The derivation's memory for one hop (Def 23): a trawl of the correspondence graph rooted at both parties — every correspondence reachable from A's nodes and B's nodes within a fixed depth. An edge is a shared word bit: a kline joins the scope when its signature or any node shares a word bit with the reached set (token-id bits carry no correspondence) — the same content-level coverage the candidate pool reads (Def 22). Dual-rooted, so the misfit edges joining the parties are in scope by construction; depth-bounded and unranked — fast but stupid. Frozen for the hop's duration: writes go to memory, and only later hops' trawls reach them.

**Slot**:
The per-node decomposition of a misfit, carried on both parties: a node of A bearing underfit content, and a node of B bearing overfit content, are each a slot — one notion read on the two parties (an overfit slot of C(A,B) is an underfit slot of C(B,A)). A slot with a licensed replace fires it; a slot without is **walked** — a meeting of two descents, one from each party, licensed by heading alone: a descent step crosses a held kline its current value heads (signature → witness), and nothing else. A's descent departs its underfit slots; B's departs the held value containing the overfit — the compound the overfit composes into. The **meeting** — a value delivered by distinct klines on the two sides — writes the **bridge**: the composed correspondence `slot_a:[slot_b]` (the A-side departure replaced by the B-side departure), acquisition depth the edges both descents crossed, consumed by the main line (Def 15). A misfit with no slot on either side has no descent to meet; the ask is the honest outcome.
_Avoid_: subgoal, subroutine, task

**Progressive Path**:
The evidence-building route from S3 to S2: each hop writes its output to STM, and the written end states are the composed correspondences later hops consume — hops matter because memory grows with them and between them (§11).

**Reentry**:
Derivations compose: re-entry changes A, and A reselects candidates for B — hop k's end state queues as hop k+1's input, and the new A takes its goal from the top of the fresh list (possibly the same kline again). Each derivation trawls its scope from memory as it stands, grown by every earlier hop. The reentry arm proposes from a proposal, one hop further out, bounded by the **hop ceiling**. Hop order is the only time the system has (§11).

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
A value produced by the tokenizer: `(word_bit << 32) | bpe_token_id`. The word half (upper 32 bits) carries one bit per distinct word — the atoms' engine realisation — assigned first-encountered at bits 0–30; bit 31 is the ASK marker (`ASK_SIG`), reserved — no word ever carries it. A multi-subword word is one word and one bit; a compound (MTS signature, CONNOTES concatenation) composes no bit of its own — it is the OR-reduction of its component words' values.

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
The semantic layer of KScript: parenthetical prose instructing the agent running the session what the surrounding structure means, and resolving **Word Binding**. Exists for the agent, not the trainee; absence is a missed opportunity, never a compilation error. Brackets are optional when the word carries them in its case: a Capitalized identifier reads exactly as its bracketed form (`Mood` ≡ `M(ood)`; an explicit annotation suppresses the expansion), while ALL-UPPER stays a compound (`MHALL`) and lowercase-first stays a literal word (`had`).
_Avoid_: comment

**MTS (Multi-Token Signature)**:
A device for representing a multi-token signature on the LHS: the compiler expands a multi-character identifier into one MTS canon relationship plus a self-identity per word-bound token. An omitted identifier is synthesized from the preceding annotation's word initials — the capability large texts are chunked through.
_Avoid_: decomposition (a Canon decomposes into nodes; an MTS expands a signature into characters)

**Word Binding**:
The association of a single-character signature with a word, resolved through annotations. An authored binding is case-blind — a witness `h(ad)` binds the compound char `H` as readily as `H(ad)` (the sig char's case is typographic; the word's case is the word) — but ambient attraction is sig-case-gated: a bare single character in sig case attracts the nearest scope's most recent word-list word with matching initial (occurrence-counted), while a lowercase single character is the literal word (the article `a`) and never attracts. An identifier's case frames its reading everywhere: ALL-UPPER is a compound (`MHALL`, alnum-only — a punctured identifier is a word), Capitalized carries its own expansion (`Mood` ≡ `M(ood)`), lowercase-first is a literal word (`had`), and so is a caseless first character — a digit or a word-internal punctuation mark (`42`, `3.14`, `don't`): identifiers admit `- . _ '` beyond alphanumerics, while the relational and structural marks (`= > < ( ) #`) are reserved and never part of an identifier. Precedence: inline annotation (nearest, overrides all others, also binds its immediate parent scope), then top-level annotations by scope, then resolved bindings (the char→word memory of earlier resolutions). Within a tier, the most recent match wins; each identity occurrence binds exactly once. A compile seeded with the prior state's known words (acquisition order) carries the binding across scripts: the seed is the outermost word list, binding chars the script cannot bind itself — an underfit question script's answer chars resolve to the earlier script's words instead of minting fresh.
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
