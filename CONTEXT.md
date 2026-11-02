# Domain Glossary

Kalvin is a rationalising system whose entire world is built from klines. This glossary defines the precise meaning of terms used across the code. [docs/rationalisation.md](docs/rationalisation.md) is the rules document for the model and its guarantees; the retired engine algebra ([docs/attic/kalvin-algebra.md](docs/attic/kalvin-algebra.md)) remains as history. This glossary remains normative for KScript compilation, role names, and the training protocol.

## Structure

The objective shape of a kline and the claim that shape makes on its own — no model, no observer.

**KLine**:
The fundamental unit of the domain and the unit the model rationalises: a nonzero **signature** (the claim) paired with an **adjacency** set of nodes (the construction — references to other klines' signatures); a canon is a claim exactly fulfilled by its construction (docs/rationalisation.md §1).

**Signature**:
The value in a kline's head position — the claim: what the kline asserts itself to be, and what other klines hold as a node. A signature alone is a kline that has not been formally constructed; signatures prime candidates and fit pairings (docs/rationalisation.md §2.3).
_Avoid_: head (positional name for the same thing)

**Node**:
A value in a kline's adjacency — either a **Token ID** or the signature of another kline; nesting is by reference, and the reference graph may cycle.
_Avoid_: child, element

**Identity**:
`{S: [S]}` — the trivial witness; a directly decodable known value. Claims S1.
_Avoid_: unsigned, treating the empty kline as an Identity

**Unknown**:
`{S: []}` — nothing held for this signature; the structural form of the **ask**. Claims S4.
_Avoid_: empty kline, bare signature, identity

**Exact**:
A kline whose signature equals the composition of its nodes — the claim is kept (`src/ks/ast_emitter.py`).

**Canon**:
An exact kline with decomposition content: `sig == can(adj)` — the signature is exactly what its adjacency composes to. A domain's witness for a signature is the first-stored canon holding it (`src/rationalisation/model.py`).
_Avoid_: canonical; treating `=>` as synonymous (the token declares intent to compose; the result need not be a Canon); expansion (an example, not the concept)

**Misfit**:
A kline that is not exact — claims S2 when overlapping, S3 when not (`src/ks/token_encoder.py`).
_Avoid_: fabrication, conjecture

**Fit**:
The classifier reading a kline's own claim and the relationship between two klines alike — the compiler's band claims and the model's F/U/O decomposition are its two live readings (`src/ks/token_encoder.py`).

**Shape**:
A fit case; Underfit and Overfit also name the misfit quantities of a pairing, and Connotation and Denotation are single-node names of convenience (`src/ks/ast_emitter.py`).

**Band**:
The structural classifier, ordered S1 > S2 > S3 > S4: no adjacency → S4; `sig == can(adj)` → S1; overlap → S2; disjoint → S3. Grades and orders; never steers the search (`src/rationalisation/model.py`).
_Avoid_: structural significance (redundant — the band is quantized significance)

**Relationship**:
The construction grading two klines against each other: the head is defined, not claimed (`src/ks/token_encoder.py`).

## Measurement

The measures of understanding and of work, and the KValue that carries the sender's assessment between participants.

**Significance**:
The measure of rational understanding — the content overlap of the two parties, `popcount(sig & can(adj)) / popcount(sig | can(adj))`; its quantization is the band; it travels with the proposal.
_Avoid_: confidence, score, weight, grounded

**Entropy**:
`1 − significance` — the unexplained fraction of a pairing (`src/rationalisation/model.py`).

**Effort**:
`(underfit mass + ½ · overfit mass) / union mass` — entropy biased by repairability; ascending effort orders a run's proposals: least entropy first, ties in generation order (`src/rationalisation/model.py`).

**KValue**:
The unit of exchange between participants — a KLine paired with a significance (the sender's assessment).

## Rationalisation

How the model tests a kline's claim against what the domain holds.

**Rationalisation**:
One run of the model: an entry (with its goal) paired against the held domain, its residuals reentered and discharged until the claim grounds or the run halts (docs/rationalisation.md §2).

**Cogitator**:
The search that runs a rationalisation — the reentry stack, its pairings, frames and resolutions (docs/rationalisation.md §2.2).

**Row**:
One pairing of an entry against a candidate — fit, underfit, overfit, and its fate; an event, never a value.

**Reentry**:
An unconstructed kline — a signature reentered as a fresh entry, remembering the row and side it came from.

**Frame**:
Covered content the run holds evidence for — earned when a candidate's claim covers the entry signature, or joint from an exchange; every frame is a signature, and frames settle the regions they cover.

**Stall**:
A reentry with no candidates at all — the S4 halt; two stalls on one value can exchange.

**Exchange**:
Two stalled reentries on the same value witnessing each other — both resolve, a joint frame records it, nothing is stored.

**Claim**:
The grounded outcome of an ask run — ⟨ask, {goal}⟩, stored when the opening pairing resolves.

**Fingerprint**:
A kline's identity for loop control — `sig | can(adj)`; never content: never fitted against, never minted, never reentered (docs/rationalisation.md §1).

**Domain**:
The insertion-ordered set of held klines — the model's memory; it only grows, and the first-stored canon per signature is its witness index.

**Certificate**:
The justification tree behind a grounded claim — each settled region traced to an empty region, a domain canon, an earned frame, or an exchange pair; checkable without re-running the search (docs/rationalisation.md §3, T3).

## KScript

The language that authors training material. A script is an encounter in dialogue form: klines and relational tokens declaring the structure the trainee will meet, step by step.

**Token ID**:
A value produced by the tokenizer: `(word_bit << 32) | bpe_token_id`, where the word word carries one bit per distinct word (first-encountered at bits 0–30, bit 31 the ASK marker) and a compound composes no bit of its own — layout and encoding rules in `src/ks/token_encoder.py`.

**Relational Tokens**:
The closed set of written tokens declaring how a kline is produced — `==` COUNTERSIGNS, `=>` CANONICALISES, `>`/`<` CONNOTES, `=` DENOTES, none ASK; a token declares an intent the fit classification of the produced kline may or may not satisfy — compilation semantics in `src/ks/ast_emitter.py`, band claims in `src/ks/token_encoder.py`, the `==` ask==goal pairing in `dev/dialogue/harness.py`.

**Comment**:
A leading `#` — the rest of the line is dropped by the lexer and never reaches binding or klines.

**Annotation**:
The semantic layer of KScript — parenthetical prose instructing the agent running the session what the surrounding structure means, and resolving **Word Binding**; it exists for the agent, not the trainee, so absence is a missed opportunity, never a compilation error (case and bracket rules: `src/ks/lexer.py`, `src/ks/parser.py`).
_Avoid_: comment

**Compound**:
A signature composed of multiple words — introduced by expansion (an ALL-UPPER identifier), CONNOTES concatenation, or synthesis from a sigless annotation's initials — whose value is the OR-reduction of its component words' values, taking no word bit of its own (`src/ks/ast_emitter.py`, `src/ks/token_encoder.py`).
_Avoid_: decomposition (a Canon decomposes into nodes; an expansion introduces a compound's words); MTS (superseded name for the expansion)

**Import**:
A compile-time directive — `import <module>` — declaring that the module's encounter precedes the script's own: its compiled entries prepend in import order (depth-first; a module imports once however many paths reach it; cycles are errors), and its word lists seed the importing script's root binding scope exactly as prior-state known_words do (the script's own lists are searched first and always win; a file boundary resets the occurrence counters). Compound canons register once across the boundary, so a reference reuses the module's decomposition and signature value. `import` is reserved at construct position only and must precede all other constructs; module resolution is a `resolver` — module name → source (`path_resolver` searches `<dir>/<name>.ks`; the dialogue harness resolves against the script's own directory) (`src/ks/compiler.py`, `src/ks/parser.py`).

**Word Binding**:
The association of a single-character signature with a word, resolved through annotations — inline annotation over word lists over resolved bindings, ambient attraction sig-case-gated, and acquisition-order seeding across scripts (the full algorithm: `src/ks/binding_scope.py`; case framing: `src/ks/lexer.py`).
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
KScript entries that provide grounding context for other entries — structurally identical whether pre-compiled by the author or reactive from the supervisor, delivered as domain priming before rationalisation (`dev/dialogue/harness.py`).

**Proposal**:
A halted kline read as a request — connect, continue, or acquire — carrying its effort; ordered by ascending effort, ties in generation order. In the training protocol, a trainee-emitted kline generated under an ask: a promise, not a fact, and weighing promises is protocol.

**Ask**:
A kline presented as a question — the ASK-marked `==` entry whose named goal answers it; the harness's ask==goal configuration is the run's opening pairing (`src/ks/token_encoder.py`, `dev/dialogue/harness.py`).

**Ratify**:
The action of countersigning a selected proposal — a ratified correspondence edge costs nothing in acquisition depth.

**Escalation**:
The Trainer deferring a proposal to the supervisor when its cogitation yields no reply.
_Avoid_: auto-ratify failure

**Target Significance**:
The band a KScript production op declares — the answer key a trainee must learn to derive, not a measurement of any one kline (`band_significance` in `src/ks/token_encoder.py`).
