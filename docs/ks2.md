# Kalvin — Symbolic II

Status: draft. Second pass at Layer 1 of the formalisation (the algebra). Layers 2–4 of `kalvin-symbolic.md` — rewrite system, strategy, measurement — are unchanged in intent and remain normative there until restated here. CONTEXT.md remains normative for role names.

---

## 1. Atoms and values

**Definition 1 (atoms).** `A = {a₀ … a₃₀, ask}` — a finite parameter set; only its finiteness is load-bearing. In the engine this is the 32-bit word space: one bit per distinct word, bit 31 reserved for ASK. The u64 packing `(word_bit << 32) | bpe_token_id` is not literally a set of atoms; this algebra is a deliberate tidying of that encoding, and the tidying is what the formalisation builds on.

`ask` is structurally a peer atom: it composes and overlaps like any other. Its specialness is declarative, not algebraic — an ask-marked kline *claims* S4 by decree (KScript annotation), not because its structure derives it: `(ask ∨ a) ∧ b ≠ ∅` whenever `a ∧ b ≠ ∅`.

**Definition 2 (values).** A **value** is a set of atoms: `V = 2^A`. Write `a` for the singleton value `{a}` and `abc` for `{a,b,c}`. The **empty value** is `∅`; the **full value** is `A`. Operations:

- composition `v ∨ w = v ∪ w` — the whole is the sum of its parts;
- overlap `v ∧ w = v ∩ w` — what two values share;
- complement `¬v = A ∖ v`.

The Boolean laws (commutativity, associativity, idempotence, distribution, `a ∧ b = ∅` for distinct atoms) hold by construction. They are consequences of the set definition, not axioms.

## 2. Node sequences — the terms

**Definition 3 (node sequence).** A node sequence `ν = [n₁ … nₖ]` is a member of `(V ∖ {∅})*`: order and multiplicity retained, no empty nodes. `V*` is the free monoid on `V` — the only free object in the system.

**Definition 4 (evaluation).** `signature_of : V* → V`:

> `signature_of([n₁ … nₖ]) = n₁ ∨ … ∨ nₖ`  `signature_of([]) = ∅`

It is a monoid homomorphism that forgets exactly order and multiplicity — nothing else. `V` identifies precisely what `signature_of` identifies. The architecture lives in the gap: **node sequences are the terms; values are what they evaluate to.**

## 3. Klines — claims and witnesses

**Definition 5 (kline).** A **kline** `s:ν` pairs a nonzero **signature** `s ∈ V` with a node sequence `ν`. A kline claims its signature as the composition of its nodes.

**Definition 6 (exactness).** `s:ν` is **exact** when `s = signature_of(ν)`. An exact, non-empty kline is a **witness**: a chosen decomposition of `s`. `signature_of` has no distinguished inverse. The trivial one, `s ↦ [s]`, always exists — its images are the identities, witnesses that carry no decomposition content; every other witness is a real choice. Nothing in `V` reconstructs which choice was made — that is what memory is for.

> **The central claim.** Kalvin's memory is a set of claims over a forgetting map. A kline claims its signature as the composition of its nodes; **Canon** is the claim exactly kept; **Underfit/Overfit** are the two directions a claim can miss while still being answered; the S3 shapes are claims nothing answers; **Unknown** is no claim at all. Significance grades the claim.

**Definition 7 (memory).** A **memory** `M` is a finite set of klines. Two klines may share a signature — distinct claims, or distinct decompositions of the same value. A node may be the signature of another kline: nesting is by reference, and a memory is a DAG of witnesses, not a tree. Which klines are *held*, and in what tier, are relations over `M` defined above the algebra (CONTEXT.md).

A witness may repeat a node (`[l,l]`); the repetition is invisible to `signature_of` but retained — it is part of the chosen decomposition.

## 4. Coverage and the fit classifier

**Definition 8 (coverage).** A node `n` is **covered** by a value `s` when `n ∧ s ≠ ∅` — they share at least one atom. Coverage is overlap, not containment: a covered node may also carry atoms outside `s`.

No new constructors appear beyond §3. The nine shapes are derived predicates — one total function on `(value, sequence)` pairs. A kline is one such pair (its own signature against its own nodes); the relationship kline of §5 is another; one classifier serves both.

**Definition 9 (gap and excess).** For `(s, ν)` with `ν ≠ []`: the **gap** `g = s ∧ ¬signature_of(ν)` — atoms the signature claims beyond its nodes; the **excess** `e = signature_of(ν) ∧ ¬s` — atoms the nodes carry beyond the signature. Note `g = ∅` and `e = ∅` together hold iff `s = signature_of(ν)`.

**Definition 10 (fit).** `fit : V × V* → Shape`. Cases in order; each admissible pair matches exactly one:

| # | Condition                    | Shape       | Band |
| - | ---------------------------- | ----------- | ---- |
| 1 | `ν = []`                     | Unknown     | S4   |
| 2 | `ν = [s]`                    | Identity    | S1   |
| 3 | `s = signature_of(ν)`        | Canon       | S1   |
| 4 | not covered, `\|ν\| = 1`     | Connotation | S3   |
| 5 | not covered, `\|ν\| > 1`     | No-fit      | S3   |
| 6 | covered, `g ≠ ∅`, `e = ∅`    | Underfit    | S2   |
| 7 | covered, `g = ∅`, `e ≠ ∅`    | Overfit     | S2   |
| 8 | covered, `g ≠ ∅`, `e ≠ ∅`    | Under+over  | S2   |

The cases are disjoint by construction. Coverage is the primary split: a pair with no covered node forces `g = s ≠ ∅` and `e = signature_of(ν) ≠ ∅`, so cases 4–5 can never satisfy 6; exactness is caught at case 3, before the covered cases, which require a nonzero gap or excess.

**Species.** **Denotation** is the single-node Underfit (`ab:[b]` — one covered node, gap only); **Connotation** is case 4. These are names of convenience for KScript (`=`, `>`/`<`); algebraically they are single-node instances of cases 6 and 4. Two single-node shapes are unnamed: the single-node Overfit `a:[ab]`, and the single-node Under+over `ab:[bc]` (covered on `b`, gap `a`, excess `c`).

**Bands.** Derived, not asserted: S1 = cases 2–3 (exact), S2 = covered misfits, S3 = uncovered misfits, S4 = Unknown. Readings carry over: S1 — _I know that I know this_; S2 — _I infer this, but it does not yet fit_; S3 — _I recognise aspects of this, indirectly_; S4 — _I do not understand this at all_.

**Invariance.** `fit` is insensitive to node order: it depends on `ν` only through `signature_of(ν)` and the node count. Duplicating a node changes the fit only when it crosses a count boundary (`a:[a]` Identity vs `a:[a,a]` Canon; `a:[b]` Connotation vs `a:[b,b]` No-fit); otherwise the repetition is witness structure invisible to classification.

**Canonical table** (illustration, not definition):

| Structure   | `s:ν`         | gap | excess | Band |
| ----------- | ------------- | --- | ------ | ---- |
| Canon       | `abc:[a,b,c]` | `∅` | `∅`    | S1   |
| Identity    | `a:[a]`       | `∅` | `∅`    | S1   |
| Underfit    | `abc:[a,c]`   | `b` | `∅`    | S2   |
| Overfit     | `ab:[a,b,c]`  | `∅` | `c`    | S2   |
| Under+over  | `abc:[b,c,d]` | `a` | `d`    | S2   |
| Denotation  | `ab:[b]`      | `a` | `∅`    | S2   |
| Connotation | `a:[b]`       | `a` | `b`    | S3   |
| No-fit      | `ab:[c,d]`    | `ab`| `cd`   | S3   |
| Unknown     | `a:[]`        | —   | —      | S4   |

Note the S3 rows: their gap *and* excess are both nonzero, yet they are not Under+over — coverage decides first. That precedence is what keeps the partition disjoint.

## 5. The relationship kline

**Definition 11 (pairwise).** For klines `A = s:ν_A` and `B = t:ν_B`, the **relationship kline** is `C(A,B) = signature_of(ν_A) : ν_B`. Its head is not claimed — it is *defined* as what A's nodes evaluate to — so all of C's misfit-ness comes from B's side. `fit(C(A,B))` is the **structural relationship of A and B**.

`fit` is one function with two readings. Applied to `(s, ν)` it grades a kline's own claim; applied to `(signature_of(ν_A), ν_B)` it grades two klines against each other. Neither reading is a special case of the other construction — both are arguments to the same classifier.

Useful equivalence: `C(A,B)` is Canon iff `signature_of(ν_A) = signature_of(ν_B)` — the two klines hold the same value, differently decomposed.

Canonical relationships (atoms lowercase):

| Relationship | A nodes | B nodes | C(A,B)        | Band |
| ------------ | ------- | ------- | ------------- | ---- |
| Canon        | [a,b,c] | [a,b,c] | `abc:[a,b,c]` | S1   |
| Identity     | [a]     | [a]     | `a:[a]`       | S1   |
| Underfit     | [a,b,c] | [a,c]   | `abc:[a,c]`   | S2   |
| Overfit      | [a,b]   | [a,b,c] | `ab:[a,b,c]`  | S2   |
| Under+over   | [a,b,c] | [b,c,d] | `abc:[b,c,d]` | S2   |
| Denotation   | [a,b]   | [b]     | `ab:[b]`      | S2   |
| Connotation  | [a]     | [b]     | `a:[b]`       | S3   |
| No-fit       | [a,b]   | [c,d]   | `ab:[c,d]`    | S3   |
| Unknown      | [a]     | []      | `a:[]`        | S4   |

## 6. What the algebra does not cover

- **Held, grounded, tiers** — relations over a memory `M` (attention, commitment); defined in CONTEXT.md, consumed by the rewrite system.
- **Graded distance and its rate of change** — strategy-level measurement built on the band predicate; a four-band predicate has no useful derivative.
- **KScript tokens** — surface syntax declaring intent; `fit` may or may not satisfy the declared intent (`=>` declares composition; the result is a Canon only if case 3 fires). See `kalvin-symbolic.md` §5.
- **Countersigning** — a protocol commitment (reciprocal connotation pairs held as ratified); the algebra provides the shape, the protocol the commitment.
