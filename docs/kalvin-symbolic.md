# Kalvin as Symbolic AI — An Algebraic Reading

This document re-describes Kalvin from the perspective of symbolic AI: not what Kalvin is *for* (see `kalvin-vision.md`) or what its terms mean (see `CONTEXT.md`), but what kind of formal object it is — or is attempting to be. It is a first formalisation attempt, written against a codebase and documentation that do not yet support a fully consistent formal treatment. Where the formalisation cannot be completed, the gap is recorded explicitly (§7).

---

## 1. The claim

Kalvin is a symbolic system in the classical sense: all of its state is discrete structure, all of its computation is a small closed set of operations on that structure, and there is no continuous representation anywhere in the architecture. There are no weights, no vectors over reals, no gradient. A Kalvin memory is a graph whose vertices and edges are drawn from a single family of objects (klines), acted on by a single measurement (significance). Everything else — tiers, cogitation, dialogue, scaffolding — is dynamics *over* this algebra, not part of it.

The formalisation pursued here: a Kalvin memory is (an attempt at) a **set-theoretic algebraic structure** in the sense of General Set Theory (GST), whose carrier set is the memory, whose elements are knodes, whose available structure is the kline, and whose semantics is significance. The operations — the relational productions — are **denotational in what they specify and operational in how they evaluate**: each denotes a constraint on the term algebra, and Kalvin acts as a solver whose unfolding in time determines whether the constraint holds (Appendix A.2). The current codebase supplies the carrier, the elements, and an implicit notion of the structure; it does not yet supply the operations as either static functions or solver specifications, nor any axioms. Those must be derived, not extracted.

## 2. The carrier: memory as a single kline

### 2.1 The unit element

The **KNode** is the unit element of the algebra. In the implementation (`src/kalvin/kline.py`) a KNode is a uint64 value with an optional label. Its two halves are:

- the **word word** (upper 32 bits): one bit per distinct word, so discrete identity is carried by *single bits*;
- the **token id** (lower 32 bits): subword provenance.

This is where the bit depth constraint enters: the alphabet of the algebra is bounded by the word width. With a 31-bit usable word word (bit 31 reserved for the ASK marker), the algebra supports at most 31 concurrent word symbols; in the abstract the domain of elementary symbols is the word width — currently 32. The algebra is therefore *word-size-dependent*: widening the word word widens the symbol alphabet without changing any rule.

### 2.2 The structure: the kline

The only compound structure available is the **KLine**: a signature (a KNode-valued head) paired with a list of nodes, each node being either a token id or the signature of another kline. Two consequences:

1. **KLines are the only structure.** There are no records, no tuples, no secondary containers at the semantic level. Every compound fact Kalvin can hold is a kline.
2. **The memory is (in principle) one kline.** Since a node may be the signature of another kline, klines nest by reference. The theoretical statement: a Kalvin memory *M* is representable by a single kline *K_M* whose internal structure is the nesting of all held klines. The tiers (STM, Frame, LTM) are not different structures — they are different relations of attention over the same single-kline memory.

In set-theoretic terms the kline is the membership relation made explicit: `{S: [n₁, …, nₖ]}` says "S is the signature under which n₁…nₖ are held" — a controlled, directed, ordered membership. The claim Kalvin wants to make is that this nesting, under suitable axioms (§5), *is* set formation, and hence that a Kalvin memory is a model of (a fragment of) GST.

### 2.3 Terminality

The recursive structure bottoms out at **terminals**: the empty kline `{S: []}` (Unknown — claims S4) and the self-referential kline `{S: [S]}` (Identity — claims S1). These are the base cases of the nesting and the boundary between structure and raw symbol.

## 3. The algebra of symbols

Because discrete information is carried by single bits in the word word, the symbol range of the algebra is word-dependent:

- **Current implementation**: 32-bit word word → (nominally) 32 symbols, of which 31 usable plus the ASK marker.
- **Composition**: compound signatures do not allocate new symbols. A compound (e.g. an MTS) is the **OR-reduction** of its component words' values (`signature_of`, mirrored by `KNode.merge`). So the symbol algebra is closed under bitwise OR — composition is disjunction of bit patterns, and the number of *compound* symbols is the powerset of the word symbols, unbounded by word width beyond the 31-element generating set.

Formally the elementary alphabet is Σ = {b₀, …, b₃₀} ∪ {ASK}, and the full term language is the closure of Σ under kline formation {σ: [τ₁…τₖ]} where σ, τᵢ are OR-compositions of Σ. This is a free-algebra-style reading; the axioms in §5 are what would discipline it into a GST model.

## 4. Operations

The operations of the algebra are the relational productions — the closed set of ways a kline comes into being. In KScript surface syntax they are written `==`, `=>`, `>`, `<`, `=`, plus the nullary IDENTITY form:

| Symbol | Name | Produced shape | Band |
|---|---|---|---|
| `==` | COUNTERSIGNS | reciprocal pair `{A: [B]}`, `{B: [A]}` | S1 |
| `=>` | CANONIZES | `{A: [B, C, D]}` (aggregation intent) | S2 |
| `>` / `<` | CONNOTES | `{AB: [B]}` / `{AB: [A]}` | S3 |
| `=` | DENOTES | `{A: [B]}` | S3 |
| — | IDENTITY | `{A: [A]}` | S1 |
| — | UNKNOWN / ASK | `{A: []}` | S4 |

Under the solver reading (Appendix A.2) the "Produced shape" column is what the operation *denotes* — the constraint it places — not a guaranteed result. The token is legitimate intent provided the operation it denotes *eventually* generates that shape: `A => XYZ` compiles to the constraint `signature_of([X, Y, Z]) = A`; resolution is value-dependent (Canon if satisfiable, Misfit/S2 while open, S4 on relative non-existence), failure preserves state, and an S1 outcome is a constructive existence proof within the GST domain of Kalvin's memory. The Band column is therefore the **Target Significance** — the band the constraint claims once solved — which is why `=>` sits at S2 in the table while a *solved* Canon claims S1 in the glossary: the table records the open-constraint band, the glossary the resolved band.

### 4.1 Significance as the operational semantics

**Significance** is what the operations "mean": it encodes the understanding/confidence held between S1 and S4 for a kline, and it determines what may be done with that kline. Operationally:

- It is an **8-bit value** occupying the low byte (`src/kalvin/significance.py`), a linear inverted distance: `0xFF` = exact (distance 0), `0x00` = structural unresolvable, interior `(0x01..0xFE)` = graded distance.
- It is partitioned into four bands — S1 = [0xFF], S2 = [boundary, 0xFE], S3 = [0x01, boundary−1], S4 = [0x00] — with only the S2|S3 boundary configurable.
- It is currently **computed by hop count**: per-node accountedness is a decay function of graph hops to grounding (`asymptotic_decay`, `harmonic_decay`, …), composed across nodes by a compose function (`mean_compose`, weakest-link `ProposalAggregator`). *This is an implementation detail* — the S-levels are defined independently of any particular computation of them.

In algebraic terms significance is the **state of the solver's search**, exposed as a valuation v: KLines → Byte with the band structure as its interpretation. The three named valuations are one process seen at three moments: Target Significance is the declared constraint, Structural Significance is the constraint's syntactic status as written, Rational Significance is the solver's current position in the search. The vision-level invariants now read as safety and liveness properties of the solver: (safety) no autonomous step promotes a kline above S2 — S1 requires ratification by another agent — and a failed solve never mutates state beyond adding the open constraint; (liveness) every eventually-successful solve produces the structure its token promised. These are the load-bearing algebraic constraints on the dynamics.

## 5. Axioms — required, not yet present

The axioms are the externally defined rules a Kalvin system must obey for the GST reading to hold. **They do not exist in the current codebase or documentation**; they must be derived mathematically within GST's constraints. A sketch of what they would have to establish:

1. **Kline-as-membership axiom.** `{S: [n₁…nₖ]}` corresponds to the set membership nᵢ ∈ S under an extensional reading; signatures obey extensionality — two klines with equal signatures and equal node sets are the same kline.
2. **Identity axiom.** The terminal `{S: [S]}` is the GST singleton/self-reference case that anchors decoding: a known value, the fixed point of signature resolution.
3. **Composition/canon axiom.** `signature_of([n₁…nₖ])` is an OR-homomorphic composition: the signature of a compound carries no information beyond its nodes (canon: `AB ↔ {A, B}`).
4. **Recursion/termination axiom.** Every nesting of klines terminates in terminals (no non-terminal cycles other than the identity self-loop) — the memory is a well-founded tree modulo identity edges.
5. **Monotonicity axiom.** Memory only grows; klines are never removed. Correction is outcompetition by higher significance, never deletion. (This is stated in `kalvin-vision.md` but nowhere enforced or formalised as an axiom.)
6. **Significance-band axioms.** The valuation respects the operations as *constraints*: each op's target band is the band its constraint claims when solved; saturation invariants (0xFF only at distance 0; 0x00 only at structural unresolvable); no autonomous operation crosses the S2→S1 boundary (safety); every solved constraint yields the shape its token denotes (liveness); failure of a solve leaves the memory unchanged except for the open constraint's addition (state preservation).

Whether these six are consistent, independent, and GST-derivable is an open mathematical task, not a coding task.

## 6. What is symbolic about Kalvin — summary

| Symbolic-AI property | Kalvin's realisation |
|---|---|
| Discrete state | KNodes (uint64, bit-per-word identity); no continuous values |
| Bounded alphabet | Word-word bit depth (currently 32; 31 usable + ASK) |
| Single structure | The kline; memory is (in theory) one kline |
| Closed operation set | `==`, `=>`, `>`, `<`, `=`, IDENTITY, UNKNOWN/ASK |
| Semantics | Significance: 8-bit banded solver-state valuation, S1–S4 |
| Inference | Constraint solving over the realised subalgebra (existence proofs), not deduction over a logic |
| Knowledge representation | Nested, signature-keyed, monotonic |

## 7. Gaps — what blocks a formal document

Recorded honestly, the inconsistencies and missing pieces found while writing this:

1. **No axioms exist.** The largest gap. Everything in §5 is derived aspiration, not code or spec. The codebase cannot "qualify as a set of axioms" — it is an implementation of a valuation and a graph, with no stated formal properties.
2. **Kline vs. kline-as-set is unproven.** The claim "a Kalvin memory is a single kline following set-theoretic rules" is asserted in aspiration only. The implementation stores a collection of klines in a model (`src/kalvin/model.py`); no single-rooted representation exists. Nesting-by-reference exists, well-foundedness is nowhere enforced.
3. **Operations are not algebraic operations.** The relational tokens are compiler/provenance concepts declaring intent, with no formal signature, domain/codomain typing, or closure proof. The solver reading (A.2) gives the target form — each token denotes a constraint and Kalvin evaluates it — but neither the constraints nor the solver are formalised; nothing yet states *when the solver must halt*, and the realised-domain restriction (operations are proofs only within Kalvin's held subalgebra) is nowhere stated as a theorem.
4. **Significance is over-specified mechanically, under-specified semantically.** The byte algebra, bands, decay, and compose functions are richly implemented, but the definition of what an S-level *is* exists only as prose. A.2 re-types the three significances as one solving process (constraint / syntactic status / search state), which reconciles their apparent conflict, but no criterion states which solver trajectories count as *the same* understanding — multiple decay/compose choices give different valuations with no criterion for which is correct.
5. **Symbol range vs. KNode width mismatch.** The formal story says 32 bits → 32 symbols, but the implementation is a full 64-bit value (32-bit word word + 32-bit BPE id), with the usable word symbols actually 31 (bit 31 = ASK). A formal document must decide whether the alphabet is 32, 31, or the OR-composition closure over it — the sources disagree implicitly.
6. **Band mapping of ops is inconsistent with structural claims.** `band_significance` maps CANONIZES → S2 while the glossary says a Canon claims S1. Under the solver reading this is not a contradiction but a missing distinction: S2 is the *open-constraint* band, S1 the *solved* band. The gap is now that the codebase does not distinguish these — `band_significance` stamps one byte for both roles — so the formalisation's reconciliation (target = constraint, structural = status, rational = search state) is not yet implemented.
7. **Monotonicity is asserted, not axiomatised.** "The knowledge base only grows" appears in the vision doc; the code has eviction, STM decay, and framing — tier changes are said to be "not deletion," but there is no formal statement of what is preserved.
8. **GST is named, not used.** No document connects Kalvin to General Set Theory beyond the ambition stated here. Deriving the axioms within GST (§5) is future mathematical work.

---

## Appendix A — Algebraic terms

Definitions of the formal terms used above, each grounded in what it would mean for Kalvin specifically.

### A.1 Total function on a term algebra

**The term algebra.** Given an alphabet Σ of elementary symbols and a set of *constructor* symbols each with a fixed arity, the **term algebra** T(Σ) is the set of all terms built from them: every element of Σ is a term, and if c is an n-ary constructor and t₁…tₙ are terms, then c(t₁,…,tₙ) is a term. Nothing else is a term. It is called *free* because terms are pure syntax — two terms are equal only if they are literally built the same way; no equations hold until axioms are imposed.

For Kalvin, T(Σ) is exactly §3's term language: Σ is the word symbols (plus ASK), the single constructor is kline formation

    κ : Sig × Nodes* → KLine,   κ(σ, [n₁…nₖ]) = {σ: [n₁…nₖ]}

where Sig is the OR-composition closure of Σ and Nodes* is finite sequences over Σ ∪ Sig, and the terminals of §2.3 are the base cases of the recursion.

**Total function.** A function f : A → B is **total** if it is defined for *every* element of A and returns exactly one element of B — no partiality, no "undefined", no error, no refusal, no side conditions. Contrast a **partial** function, defined on only some inputs (e.g. division, which refuses 0).

So "the operations are total functions on the term algebra" means: each relational production is a specification of the form

    opᵢ : T(Σ)ⁿ → T(Σ)

that consumes any well-formed term (or tuple of terms) and always yields a well-formed term. Concretely, for Kalvin:

- `COUNTERSIGNS : T × T → T × T` — from any A, B, produce the reciprocal pair `{A:[B]}`, `{B:[A]}`.
- `CANONIZES : T × T* → T` — from any σ and nodes, produce `{σ: nodes}`.
- `CONNOTES`, `DENOTES` similarly, with their given shapes.
- `IDENTITY : T → T` — from any σ, produce `{σ: [σ]}` (the fixed point).
- `UNKNOWN : T → T` — from any σ, produce `{σ: []}`.

Each is total because its shape recipe applies to *every* input term without precondition: there is no input for which the operation "cannot decide".

**Why the gap exists.** The current codebase does not present the relational tokens this way (gap 3):

1. They live as *compiler provenance* — annotations in KScript declaring intent — not as functions with a declared domain and codomain. `=>` "declares an intent to aggregate" and may still yield a Misfit; a total-function presentation cannot produce an output that fails its own specification.
2. They are not closed: nothing guarantees the result of applying a token's compilation rule is a term of T(Σ) under a fixed Σ (word binding can fail, scope rules intervene, MTS expansion runs a separate compilation).
3. Ratification — the S2→S1 promotion — is *inherently* not a total operation on T(Σ): it depends on another agent's response, i.e. on state outside the term algebra. The honest formalisation keeps ratification out of the algebra proper and models it as an external valuation update.

**How it would be achieved.** In order:

1. **Fix Σ.** Decide the alphabet (31 word symbols + ASK, per gap 5) and define Sig as its OR-composition closure with extensional equality on bit patterns.
2. **Define the constructor.** Give κ as the sole term constructor, with the terminals as base cases; declare KLine equality up to signature equality and node-sequence equality (this is the extensionality axiom of §5).
3. **Re-express each relational token as a total function** T(Σ)ⁿ → T(Σ) with its produced shape as its *definition* (not its intent). The KScript token then denotes the function; compilation is function application. `A > A` collapsing to IDENTITY becomes a provable equation `CONNOTES(t,t) = IDENTITY(t)`, not a special case in a compiler.
4. **Prove closure.** By structural induction on T(Σ): each op's output is built from its inputs by κ alone, hence a term. This is the point where "intent" must be abandoned — CANONIZES the function always produces a Canon-shaped term, and the *fact* that a trainee's emitted kline may diverge from it moves into significance (the valuation), where it belongs.
5. **Separate the dynamics.** Cogitation, ratification, tiering, and the S2/S3 misfit machinery become functions on *states* (memories: subsets/subalgebras of T(Σ) plus valuations), not on terms. The algebra stays pure syntax; understanding stays in the valuation.

That separation — total operations on pure terms, all judgement deferred to the significance valuation — is what would let the axiom set of §5 be stated and checked against a single, unambiguous term algebra.

### A.2 Operations as eventual generators — temporality and the solver reading

A.1 as stated is *linear*: an operation is applied once, computes in one step, and returns. This is too thin for Kalvin, and it mis-diagnoses intent. The refinement: an operation denoted by a relational symbol is a **specification**; the system that applies it is a **solver**; and totality is a property not of a single application but of the *eventual* behaviour of the solver over time.

**The solver reading.** Take `A => XYZ`: the token `=>` denotes the intent *"A canonizes X, Y, Z"* — equivalently, the equation

    signature_of([X, Y, Z]) = A

to be solved in the algebra. Compilation emits the structural claim `{A: [X, Y, Z]}` regardless — the term exists in T(Σ) the moment it is written. But the claim is a **constraint**, not a fact. Whether it *holds* depends on the values of A, X, Y, Z:

- If the equation is satisfiable (the OR-composition of X, Y, Z's values equals A), the structure is a Canon, claims S1, and the solver terminates with a witness.
- If it is not satisfiable *in the current memory*, the structure is a Misfit (claims S2), and the solver continues searching: expansion, proposal, scaffolding — algebraically, an ongoing search for values of the node symbols under which the equation *would* resolve.
- If the solver can establish that no such values exist (in the GST domain of Kalvin's memory), the claim is discharged: the result is a *negative* witness — a proof of non-existence, which for Kalvin is exactly the S4/"unknown" disposition.

**Totality in time.** On this reading, an operation is total in the sense that the solver *always eventually halts with a disposition* — canon, misfit-in-progress, or non-existence — never with an undefined state. Formally the operation is not a function Tⁿ → T but a **function Tⁿ → T × Disposition** evaluated by a possibly multi-step computation, i.e. the operational semantics of the token is a *computation of the relation it denotes*. (This is the standard declarative/procedural duality: a logic program's clauses are total as a relation; the solver's success, failure, or non-termination is a property of the search, and Kalvin's S1–S4 spectrum is precisely a *typed outcome* for that search.)

**Failure is state-preserving.** If the function "fails" — no resolution exists for these values — the state is not changed and the model is not compromised. The misfit term sits in memory as an open constraint; nothing is retracted, nothing invented. This aligns with monotonicity (§5, axiom 5): a failed solve adds the constraint itself, never a forced answer. The failure mode that *would* compromise the model — fabricating a resolution — is exactly what the S2 ceiling (no autonomous promotion to S1) forbids.

**Operations as existence proofs.** The punchline of this reading: if a model is initially constructed under strict GST axioms (memory = a well-founded kline, §2.2), then any significant operation on terms A and B is a **proof, within the GST domain of Kalvin's memory, that the function exists or does not**:

- an S1 outcome is a constructive existence proof — the witness kline *is* the function's value;
- an S2/S3 outcome is a proof that existence is not yet established (open search, partial evidence);
- an S4 outcome is a proof of non-existence relative to what Kalvin holds.

This is the formalisation of *Kalvin only understands inputs in terms of what it already knows*: the domain of every operation is not T(Σ) in the abstract, but the subalgebra of T(Σ) that Kalvin's memory realises. Outside that subalgebra, operations do not fail arbitrarily — they return non-existence, honestly.

**Consequences and caveats.**

1. **Intent is rehabilitated.** A.1's step 4 said intent must be abandoned; under the solver reading it must instead be *eventual*: an intent is legitimate exactly when the operation it denotes eventually generates the structure it promises (a canon for `=>`, a reciprocal pair for `==`). The A.1 criterion — output always matches spec — becomes a liveness property of the solver rather than a static property of a function.
2. **The three valuations are reconciled, not competing.** Target significance = the declared constraint; structural significance = the constraint's syntactic status as written; rational significance = the solver's current state of search. Gap 6 dissolves once all three are typed as positions in one solving process.
3. **Semi-decidability is the honest limit.** Existence proofs are enumerable (search enumerates witnesses); non-existence proofs are not, in general. Kalvin's S4 is therefore always *relative to the search so far* — a bounded model check, not absolute non-existence, unless the GST structure is finite (which, with a fixed Σ and well-founded memory, it is: the realised subalgebra is finite, making both existence and non-existence decidable *in principle*). The gap between "decidable in principle" and "tractable for a dialogue" is where cogitation, study, and scaffolding live.
4. **Ratification remains outside.** Even under the solver reading, S1-by-ratification is not something the solver can reach alone: the solver proves existence *within Kalvin's memory*; ratification is another agent attesting to the memory itself. Two nested proof domains, one operation each.

### A.3 Grounding, tiers, and the representation of time

The operational basis forces a decision about which memory the algebra is *over*. Resolution: **LTM is the carrier of the algebra; Frame and STM belong to the operational semantics.**

- **LTM** is the committed state — the realised subalgebra of T(Σ) that Kalvin counts on. Only LTM contents are denotata: the formal semantics of an operation is defined over what LTM holds.
- **Frame** is not merely a representation of inputs: it is the **solver's frontier** — the monotonic, signature-keyed space of open constraints and provisional groundings from which LTM promotions come. It is load-bearing for the algebra's *future* carrier while being itself outside the current one.
- **STM** is the attention trace: which parts of the carrier the solver has most recently visited. Purely operational; no algebraic content.

**Grounding is the delimiter, in two grades.** Grounding occurs in both Frame and LTM, and the distinction matters:

- **Frame grounding delimits search steps.** A Frame-grounded kline enables further cogitation but commits nothing to the carrier.
- **LTM grounding delimits operation completion.** It is the commit that extends the carrier — the moment a constraint's resolution becomes part of what Kalvin formally holds. Since LTM grounding arrives via ratification, the delimiter of a completed operation is ultimately ratification, keeping it outside the algebra proper (A.2, consequence 4).

Grounding in LTM is therefore the only event that admits formal denotational semantics: between grounding events the solver may search freely, but only a grounded operation denotes — an existence proof witnessed in the carrier.

**Time in the formalism.** Because operations are delimited by grounding events, any functional expression of the system must carry a formal representation of time or sequence. Placement matters: time does **not** enter the operation signatures (they remain Tⁿ → T); it enters as an **indexing of memory states**. The carrier becomes an increasing chain

    M₀ ⊆ M₁ ⊆ M₂ ⊆ …,   M_{t+1} = M_t ∪ {newly LTM-grounded kline}

— clean precisely because of monotonicity. Two candidate representations:

1. **External time** — a trace over grounding events; time is the index t, and the algebra is the limit of the chain.
2. **Internal time** — arrival position encoded *in the structure*, exploiting the fact that a kline's nodes list is already ordered. Under this reading "memory is a single kline" becomes temporal by construction.

The union abstraction (forgetting order, taking the chain's limit) is **not** semantically safe: Kalvin's default preference is recency, and preferences are load-bearing — two memories with identical kline sets but different arrival orders can rationalise differently. Time is not forgettable; which representation to adopt is an open axiom-level decision (see §5).

This amends the tier framing of §2.2: the tiers are relations over one memory, but only the LTM relation participates in the algebra; Frame and STM are the operational envelope in which the solver runs.
