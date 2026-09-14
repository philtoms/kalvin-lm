# Domain Glossary

Kalvin is a rationalising system whose entire world is built from klines. This glossary defines the precise meaning of terms used across the code. [docs/kalvin-algebra.md](docs/kalvin-algebra.md) fixes the formal definitions and the terminology (its §12); this glossary restates them as domain terms and remains normative for role names, tier mechanics, and the training protocol.

## Structure

The objective shape of a kline and the significance that shape claims on its own — no model, no observer.

**Atom**:
The indivisible unit of the value space — one element of a finite set, only its finiteness load-bearing. In the engine, the word-bit space: one bit per distinct word (Def 1).
_Avoid_: token, subword (a Token ID or BPE token is an encoding, not an atom)

**Value**:
A set of atoms — what signatures and nodes are made of. Operations: composition `v ∨ w` (union — the whole is the sum of its parts), overlap `v ∧ w` (intersection — what two values share), complement `¬v`. The Boolean laws hold by construction, not as axioms (Def 2).

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
A node is covered by a value when they share at least one atom — overlap, not containment: a covered node may carry atoms outside the value (Def 8). The classifier's primary split: covered misfits are S2, uncovered are S3.

**Fit**:
The total classifier `fit : V × V* → Shape` — one function, two readings: a kline's own fit (the claim it makes standing alone) and the relationship fit `fit(C(A,B))` (what a pair establishes). Nine shapes, four bands; every pair matches exactly one (Def 10, §11).

**Shape**:
One of the nine fit cases — Canon, Identity, Underfit, Overfit, Under+over, Denotation, Connotation, No-fit, Unknown. Underfit and Overfit also name the misfit quantities (Def 9): the atoms the signature claims beyond its nodes, and the atoms the nodes carry beyond the signature. Denotation (single-node Underfit) and Connotation (uncovered single-node) are names of convenience for KScript; algebraically they are single-node instances of cases 6 and 4 (Def 10).

**Band**:
The structural form of significance — the fit's tier, ordered S1 > S2 > S3 > S4, shapes within a band unordered:

- **S1** — exact: the claim is kept. _I know that I know this._
- **S2** — covered misfit. _I infer this, but it does not yet fit._
- **S3** — uncovered misfit. _I recognise aspects of this, indirectly._
- **S4** — Unknown. _I do not understand this at all._

Observer-independent — given the same held memory, every agent classifies alike — so a band is never exchanged; it is recomputable from structure (§11).

**Significance**:
The value rationalisation produces and consumes; understanding, informally, is high significance attained and held (§12). Two forms: **structural** — the band of a fit, derived by forming the relationship kline and classifying its shape; **graded** — the distance `γ = J · δ^(D̄ + Ĥ)`: the Jaccard overlap of the two contents, discounted by two depths — the mean **resolution depth** at which A's content is held (granularity), and the mean **acquisition depth** of the unratified correspondence edges crossed to win it (provenance). Ratified edges cost nothing: hard-won until it consolidates.
_Avoid_: confidence, score, weight, grounded

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
| Denotation  | `AB:[B]`      | S2   | shed                | `A = B`        |
| Connotation | `A:[B]`       | S3   | traverse            | `A > B`        |
| No-fit      | `AB:[C,D]`    | S3   | traverse            | `AB => C D`    |
| Unknown     | `A:[]`        | S4   | inert — the ask     | `A`            |

## Rationalisation

How a participant tests a kline's structural claim against what Kalvin actually holds.

**Rationalisation**:
The process that produces and consumes significance (§12).

**Cogitation**:
The slow path of rationalisation — the strategy loop over derivations: **select** a hop, **derive** to an ending, **add** the result to memory, **reenter** with the output as the next queue's input (§10). Each phase is strategy: the rule system constrains what any of it may do, never what it must. The fit is graded at each state and its rate of change feeds back, telling Kalvin whether its effort is increasingly or decreasingly significant.
_Avoid_: thinking, background thread, the cogitator

**Derivation**:
The rewrite of a queued kline's node sequence against one held goal: `A ⊢_{M,B} A′`. The signature — the claim — never changes; states differ only in nodes (Def 12).

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
What permits a replace — two kinds on one rule (§7): **witnessed** (canon-mode: expand/contract, licensed by memory alone, blind to any goal) and **evidenced targeting** (licensed by a correspondence and scoped to the misfit region — targeting-licensed iff it strictly decreases the misfit mass, Def 14).

**Misfit Mass**:
`|signature_of(A.nodes) Δ signature_of(B.nodes)|` — the atoms the two contents disagree on. The unit of progress: every licensed targeting replace strictly decreases it, and a run from entry is bounded by its initial value (Def 14, T1).

**Done**:
The ending where the relationship reaches S1: **value-equality**, `signature_of(A.nodes) = signature_of(B.nodes)` — not node-equality. Done may arrive early; pending nodes are witness structure. A constructive existence proof within what is held: every step of the witness was licensed by a correspondence (Def 15, §9).

**Stuck**:
The ending where no licensed targeting move remains — not done, and nothing in memory connects. Two conditions, both the **ask**: no goal held, or no connection across the correspondence graph. Relative non-existence — the honest outcome when the bridge is missing (Def 15, §9).

**Abandoned**:
Not an ending the rules produce: strategy halts or re-targets a run mid-derivation, e.g. when graded effort falls (Def 15).

**Ask**:
The structural halt condition — no atom, mark, or decree involved. The Unknown shape (`S:[]` — nothing held) is the ask's shape, and a misfit region no held correspondence reaches asks. The event under which ungrounded proposals are generated (§4, §8).

**Candidates**:
The held correspondences selectable for a derivation. A candidate is selectable when its signature occurs as a node of A — that occurrence is the replace licence's forward side, and each replace's arrival makes new candidates selectable: the path is the guard, not the point. The goal is never selected for replacement: declared (`=>`) or supplied by reentry, it scopes the misfit region, is checked at done (Def 16), and may seed slot walks without being rewritten (Def 17). Content overlap and signature-in-node imply neither the other — selection requires the second; the band routes by the first.

**Slot**:
The per-node decomposition of a misfit, carried on both parties: a node of A bearing an underfit atom, and a node of B bearing an overfit atom, are each a slot — one notion read on the two parties (an overfit slot of C(A,B) is an underfit slot of C(B,A)). A slot with a licensed replace fires it; a slot without is **walked** — a goal-less derivation over the correspondence graph, licensed by occurrence alone (either side of a held kline occurring in the walk's nodes). A walk from A ends at **arrival** in the overfit; a walk from B ends at arrival in A's content, the **anchor**; either may end stuck at the ask. Arrival is not absorption — the walk refines to the consuming resolution (the goal's witness for the overfit, A's nodes for the anchor), each refinement edge counted. The terminal is written into memory as the **composed correspondence** the main line consumes: headed at the A-side end (slot or anchor), its witness holding that end's atoms shared with the goal plus the overfit at the goal's witness resolution (Def 17). The goal is read, never rewritten.
_Avoid_: subgoal, subroutine, task

**Progressive Path**:
The evidence-building route from S3 to S2: each hop writes its output to STM, and the written end states are the composed correspondences the main line consumes — hops matter because memory grows between them (§10).

**Reentry**:
Derivations compose: hop k's end state queues as hop k+1's input, and memory may grow between hops — successive hops are not derivations of one fixed system. The reentry arm proposes from a proposal, one hop further out, bounded by the **hop ceiling**. Hop order is the only time the system has (§10).

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

**KValue**:
The unit of exchange between participants — a KLine paired with a significance (the sender's assessment). The kline carries its acquisition record; the significance is computed against it.

## KScript

The language that authors training material. A script is an encounter in dialogue form: klines and relational tokens declaring the structure the trainee will meet, step by step.

**Token ID**:
A value produced by the tokenizer: `(word_bit << 32) | bpe_token_id`. The word half (upper 32 bits) carries one bit per distinct word — the atoms' engine realisation — assigned first-encountered at bits 0–30, bit 31 reserved for ASK. A multi-subword word is one word and one bit; a compound (MTS signature, DENOTES concatenation) composes no bit of its own — it is the OR-reduction of its component words' values.

**Relational Tokens**:
The closed set of written tokens that declare how a kline is produced. A token declares an intent; the fit classification of the produced kline may or may not satisfy it (§13).

- `==` **COUNTERSIGNS** — reciprocal pair `{A:[B]}`, `{B:[A]}`
- `=>` **CANONICALZES** — intent to aggregate `{A:[B,C,D]}`; the result need not be a Canon
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
A KLine emitted by a trainee during rationalisation. Ungrounded when generated under the ask — S3 evidence is a promise, not a fact; weighing promises is protocol.

**Ratify**:
The action of countersigning a selected proposal — usually performed by the Trainer while running a script. Its effect is a tier relation over memory: a ratified correspondence edge costs nothing in acquisition depth, and ratifying a traversed pair promotes it to a standing one-hop licence (§10, §11, §14).

**Escalation**:
The Trainer deferring a proposal to the supervisor when its cogitation yields no reply.
_Avoid_: auto-ratify failure

**Semantic Evidence**:
The correspondences a KScript's entries hold collectively but no single kline declares: the canon index, countersign pairs, and denotation/connotation edges. Emitted by compilation as derived structure, it carries the script's intended significance.

**Target Significance**:
The band a KScript production op declares — the answer key a trainee must learn to derive, not a measurement of any one kline.
