# Kalvin — A Term Algebra and Rewrite System

**Status:** Draft — normative. This document is the single normative definition of Kalvin.

The formal definitions and rewrite rules are normative. Explanatory text is non-normative and is included only to clarify the definitions and their intended use.

Kalvin models memory as a finite set of **klines**. A kline consists of a head value and a node sequence. The head represents the value claimed by the kline; the node sequence represents the decomposition held in memory.

A node sequence is evaluated by set union. Consequently, different node sequences may have the same value. The sequence is therefore significant even when its evaluated value is not: order and multiplicity are retained in the sequence, while evaluation removes both.

A pure classifier maps a value and a node sequence to one of nine fit shapes. These shapes are grouped into four significance bands:

- **S1:** exact
- **S2:** covered but imperfect
- **S3:** uncovered
- **S4:** unknown

A derivation rewrites the node sequence of a queued kline. Rewrites are licensed by held klines and are evaluated relative to a goal kline. A rewrite either changes only granularity or moves the current content toward the goal.

A derivation therefore has two distinct kinds of step:

1. **Witnessed steps**, which preserve the current value and change only its decomposition.
2. **Targeting steps**, which change the current value and strictly reduce the remaining content mismatch with the goal.

A derivation ends when the current value equals the goal value, or when no licensed targeting step remains. Strategy determines which licensed operation to attempt and whether a derivation should be continued, abandoned, or followed by another hop.

The measurement model assigns each derivation a significance value — the understanding achieved — and a complexity value — the work spent achieving it. The two depend on:

- content overlap with the goal;
- the granularity of the current witness; and
- the acquisition depth of the content used.

The formal system is divided as follows:

- **§1–5:** the objects and their relationships;
- **§6–9:** derivation and rewrite rules;
- **§10:** measurement;
- **§11:** strategy;
- **§12:** terminology.

**§13** defines the KScript surface syntax.

**§14** identifies mechanisms that are outside the algebra, including tier mechanics, orchestration, escalation, and ratification.

---

## 0. Notation

### 0.1 Values

The value space (Definition 1) supplies the operations; the right-hand column reads them in the reference realisation (Definition 2).

| Symbol      | Meaning                              | Reference realisation |
| ----------- | ------------------------------------ | --------------------- |
| `v ∨ w`     | composition                          | `v \| w`              |
| `v ∧ w`     | overlap                              | `v & w`               |
| `v ∖ w`     | residue — v beyond w                 | `v & ~w`              |
| `\|v\|`     | μ(v) — the content measure           | `popcount(v)`         |
| `\|x Δ y\|` | misfit mass: `\|x ∨ y\| − \|x ∧ y\|` | `popcount(x ^ y)`     |
| `∅`         | the empty value, composition's unit  | `0`                   |
| `a`, `abc`  | values, written as their atoms       | `{a}`, `{a,b,c}`      |
| `σ(ν)`      | evaluation of node sequence ν        | `fold_or(nu)`         |
| `[n₁ … nₖ]` | node sequence                        | list                  |

The formalisation also uses V\* for sequences of values, and n ∈ ν for a node's multiset membership in a sequence.

The ASK marker is a value tag marking a kline as the question (§13). The
marker is identity, not content: σ, γ, and every measure ignore it; identity,
store keys, and lookups see it.

### 0.2 Klines and derivations

A kline is written:

```text
s:ν
```

where s is its head value and ν is its node sequence.

The derivation relation

```text
A ⊢_{M,B} A′
```

reads: kline A rewrites to A′ in memory M, relative to goal B. The subscript names the parameters and is dropped when clear; A′ is the state after one more step.

A chained sequence

```text
A₀ ⊢ A₁ ⊢ ⋯ ⊢ Aₙ
```

is a derivation run, left to right in time.

The notation

```text
n ⇉ ν_K
```

denotes a replacement of node n by the witness sequence ν_K.

The relationship between two klines is written:

```text
C(A,B)
```

and the fit classifier is:

```text
fit(s, ν).
```

### 0.3 Measurement

```text
γ(A,B)
```

is the composite of working from A toward B: significance net of complexity (Definition 20).

```text
J(x,y) = |x ∧ y| / |x ∨ y|
```

is Jaccard overlap.

δ is a discount factor with 0 < δ < 1.

D̄ is mean resolution depth and Ĥ is mean acquisition depth.

Δ₀ denotes the initial content mismatch:

```text
Δ₀ = |σ(ν_{A₀}) Δ σ(ν_B)|.
```

---

# 1. The Value Space

## Definition 1 — Value space

Kalvin is parameterised over a value space:

```text
(V, ∅, ∨, ∧, ∖, μ)
```

whose elements are **values**. A value is opaque: the algebra assumes nothing
about its inside and exercises five capabilities only.

- **Composition** `v ∨ w` — commutative, associative, idempotent; unit `∅`.
- **Overlap** `v ∧ w` — commutative; the content two values share;
  `v ∧ ∅ = ∅`.
- **Residue** `v ∖ w` — the content `v` carries beyond `w`:
  `(v ∖ w) ∨ (v ∧ w) = v` and `(v ∖ w) ∧ w = ∅`.
- **Measure** `μ(v)`, written `|v|` — how much content a value carries:
  `|∅| = 0`; `|v| = 0` only when `v = ∅`; and `u ≤ v` with `|u| = |v|`
  implies `u = v` (no ghost content).
- **Decidable equality** of values.

Write `u ≤ v` for `u ∨ v = v` (u's content is contained in v's). Two derived
forms are used throughout:

```text
misfit mass   |x Δ y| = |x ∨ y| − |x ∧ y|
Jaccard       J(x,y)  = |x ∧ y| / |x ∨ y|
```

Every result holds for any structure satisfying these laws; realisations are
interchangeable. The measure is exact by requirement: licensing (Definition 14)
and the measurement invariants (§10) read strict inequalities in `|·|`, so a
realisation may scale the universe but must not approximate μ.

Decomposition is not a capability. The algebra never looks inside a value; the
only decomposition it ever sees is a held witness (Definition 6).

## Definition 2 — Reference realisation

The reference realisation is the word-bit space, constructed in detail in the appendix. The atom set A is finite:

```text
A = {a₀ … a₃₀}.
V = 2^A.
```

An atom is the indivisible unit of this realisation — in the engine, one bit
per distinct word. Values are sets of atoms; `∨ ∧ ∖` are union, intersection,
difference; `|v|` is the number of atoms in v; and `¬v = A ∖ v`. The Boolean
laws hold here by construction, because these are ordinary set operations:

```text
ab ∨ ac = abc
ab ∧ ac = a
¬a = bc
```

Two facts of this realisation are often mistaken for the algebra:

- a value is a machine word, so the vocabulary is bounded by its bits
  (31 words);
- a value is its own address — the store key and the content coincide.

A realisation at scale drops both: each distinct word receives an integer id
(first-encountered, unbounded), a value is the exact set of its ids, and the
signifier interns values under a content key, so equal content shares one
address and nesting-by-reference is unchanged. The atoms and their bits are
gloss for this realisation, not primitives of the algebra.

---

# 2. Node Sequences

## Definition 3 — Node sequence

A node sequence is a finite sequence of non-empty values:

```text
ν = [n₁ … nₖ]
```

with each nᵢ ∈ V ∖ {∅}.

Order and multiplicity are retained.

The sequence space is the free monoid

```text
(V ∖ {∅})*.
```

No equations identify sequences at this level. Thus:

```text
[a,b] ≠ [b,a]
[a] ≠ [a,a].
```

The algebra may later evaluate these sequences to the same value.

## Definition 4 — Evaluation

The evaluation function

```text
σ : V* → V
```

is defined by:

```text
σ([n₁ … nₖ]) = n₁ ∨ ⋯ ∨ nₖ
σ([]) = ∅.
```

Evaluation is a monoid homomorphism:

```text
σ(ν₁ · ν₂) = σ(ν₁) ∨ σ(ν₂).
```

Evaluation therefore discards order and multiplicity.

For example:

```text
σ([ab,c])  = abc
σ([c,ab])  = abc
σ([a,b,a]) = ab
σ([abc])   = abc.
```

The distinction is important: the value records **what** is present, while the node sequence records **how that value is represented**.

Consequently, `[abc]` and `[a,b,c]` have the same evaluation but different granularity.

---

# 3. Klines

## Definition 5 — Kline

A kline is a pair

```text
s:ν
```

where s ∈ V ∖ {∅} and ν is a node sequence.

The kline asserts that s is the value represented by ν.

The assertion may be exact or may contain a mismatch.

## Definition 6 — Exactness

A kline s:ν is **exact** when

```text
s = σ(ν).
```

An exact, non-empty kline is a **witness**: it records a particular decomposition of its head value.

Evaluation has no distinguished inverse. In particular, the trivial map s ↦ [s] is always an exact witness, but it contains no decomposition information.

Other witnesses represent explicit decomposition choices. For example, witnesses of abc include:

```text
[a,b,c]
[ab,c]
[a,bc]
[abc]
```

and permutations of these sequences.

A witness may repeat a node, as in `[l,l]`. The repetition is invisible to evaluation but is retained: it is part of the chosen decomposition.

Memory therefore preserves decomposition information that the evaluation function does not.

## Definition 7 — Memory

Memory M is a finite set of klines.

Different klines may have the same signature. A node may also equal the signature of another kline, so klines may reference one another.

The resulting reference graph is unrestricted and may contain cycles, including canon self-reference and reciprocal pairs. Cycles carry no decomposition content.

Tier membership and memory-management policy are outside this algebra.

---

# 4. Coverage and Fit

## Definition 8 — Coverage

A node n is **covered** by a value s when:

```text
n ∧ s ≠ ∅.
```

Coverage is overlap: the node and the value share content. It does not require the node to be contained in the value (n ≤ s).

## Definition 9 — Underfit and overfit

For a pair (s, ν), let:

```text
u = s ∖ σ(ν)
o = σ(ν) ∖ s.
```

The **underfit** u is the content claimed by the head but not supplied by the nodes.

The **overfit** o is the content supplied by the nodes but not claimed by the head.

Therefore:

```text
u = ∅ and o = ∅  iff  s = σ(ν).
```

For example:

```text
abc:[b,c,d]
```

has underfit a and overfit d.

## Definition 10 — Fit

The classifier

```text
fit : V × V* → Shape
```

assigns exactly one shape to every pair. Cases are evaluated in the following order; the first matching case determines the result.

| Case | Condition              | Shape      | Band |
| ---: | ---------------------- | ---------- | ---- |
|    1 | ν = [] or s = ∅        | Unknown    | S4   |
|    2 | ν = [s]                | Identity   | S1   |
|    3 | s = σ(ν)               | Canon      | S1   |
|    4 | not covered, \|ν\| = 1 | Denotation | S3   |
|    5 | not covered, \|ν\| > 1 | No-fit     | S3   |
|    6 | covered, u ≠ ∅, o = ∅  | Underfit   | S2   |
|    7 | covered, u = ∅, o ≠ ∅  | Overfit    | S2   |
|    8 | covered, u ≠ ∅, o ≠ ∅  | Under+over | S2   |

The cases are disjoint because exactness is tested before the misfit cases, and coverage is tested before the S2 cases.

If no node overlaps the head, the pair is necessarily an S3 case. Such a pair must therefore be classified as Denotation or No-fit rather than Under+over.

The classifier is therefore:

```python
def fit(s, nu):
    if nu == [] or s == ∅:
        return Unknown

    if nu == [s]:
        return Identity

    sigma = signature_of(nu)

    if s == sigma:
        return Canon

    underfit = s & ~sigma
    overfit = sigma & ~s
    covered = any(n & s != ∅ for n in nu)

    if not covered:
        return Denotation if len(nu) == 1 else NoFit

    if underfit and not overfit:
        return Underfit

    if overfit and not underfit:
        return Overfit

    return UnderAndOver
```

The fit classifier ignores node order. It uses only evaluation, node count, and coverage.

Duplication can affect the result when it changes node count. For example, `a:[a]` is Identity, while `a:[a,a]` is Canon. Similarly, `a:[b]` is Denotation, while `a:[b,b]` is No-fit.

### Species

Two of the nine shapes have names of convenience:

- **Connotation** is the single-node Underfit: `ab:[b]`, one covered node with an underfit only.
- **Denotation** is case 4: `a:[b]`.

These names are used by the KScript surface syntax (§13); algebraically they are single-node instances of cases 6 and 4.

Two single-node shapes are unnamed: the single-node Overfit `a:[ab]`, and the single-node Under+over `ab:[bc]` (covered on b, underfit a, overfit c).

### Bands

The bands are derived from the shapes:

- **S1:** Identity and Canon
- **S2:** covered misfits
- **S3:** uncovered misfits
- **S4:** Unknown

Informally, the bands represent increasing uncertainty:

- **S1:** exact match — I know that I know this
- **S2:** related but incomplete or excessive — I infer this, but it does not yet fit
- **S3:** indirect or unsupported relation — I recognise aspects of this, indirectly
- **S4:** no represented content — I do not understand this at all

Unknown is also S4's shape of no represented content: the system has no current content on the relevant side. No compiled ask takes it — the ask is the marked kline (§13).

An ask is an unmatched underfit: no goal, no witness. Its entire content is underfit slots, each walkable under Definition 15. The question is held; the arrangement is the work.

---

# 5. Relationship Kline

## Definition 11 — Pairwise relationship

Let

```text
A = s:ν_A   and   B = t:ν_B.
```

The relationship kline is:

```text
C(A,B) = σ(ν_A) : ν_B.
```

The head of C(A,B) is the evaluated content of A; the node sequence is B's witness.

The relationship therefore answers:

> How does the decomposition held by B fit the content currently represented by A?

The same fit classifier is used for individual klines and for relationships. Neither reading is a special case of the other; both are arguments to the same classifier.

For example, if A = abc:[a,b,c] and B = ab:[a,b,c], then:

```text
C(A,B) = abc:[a,b,c].
```

The relationship is Overfit because B's witness contains the additional atom c.

A useful consequence is:

```text
C(A,B) is Canon  iff  σ(ν_A) = σ(ν_B).
```

Thus two klines have a canonical relationship exactly when they hold the same value, regardless of how that value is decomposed.

---

# 6. Derivations

A derivation rewrites the node sequence of a queued kline.

Three klines may be involved:

- **A — queued kline:** the state being rewritten;
- **B — goal kline:** the target used to scope targeting;
- **K — evidence kline:** a held kline that licenses a replacement.

The goal is read, never rewritten: it scopes targeting (Definition 14), determines the ending (Definition 16), and fixes the overfit the meeting walk bridges to (Definition 15).

## Definition 12 — Derivation

For A = s:ν_A and goal B = t:ν_B, a derivation step has the form:

```text
A ⊢_{M,B} A′.
```

The subscript names the derivation's parameters: the memory it reads — its scope, fixed as trawled at entry (Definition 23) — and the goal it works toward, fixed for the derivation's duration. Writes go to memory (Definition 7), not to the scope: a derivation never consumes what it writes.

The head s of A does not change during the derivation. Only its node sequence changes. Nothing in §§6–9 reads s but well-foundedness: licences, endings, bounds, and grades otherwise read only the relationship, whose head σ(ν_A) is exact against ν_A at every state by construction. The queued head rides along inert, mattering only after the hop — in absorption, re-entry (§11), the claim's grounding (§14), and the one licence that bars it from its own witness (Definition 13).

At every state, the current content is:

```text
σ(ν_A).
```

Node order is retained but does not affect licensing. Membership, difference, and occurrence are multiset-wise: a node occurs wherever it occurs, and a witness occurs wherever its nodes occur, in any arrangement. Arrangement is witness structure alone.

## Definition 13 — One-step replacement

Let:

```text
K = n:ν_K ∈ M.
```

A replacement may occur in either direction.

### Forward

An occurrence of n in ν_A is replaced by ν_K:

```text
n ⇉ ν_K.
```

### Reverse

An occurrence of ν_K, treated as an unordered multiset of nodes, is replaced by [n]:

```text
ν_K ⇉ [n].
```

The fit of K determines the interpretation of the replacement:

- **Canon:** change granularity without changing content.
- **Covered S2:** move content according to the underfit and overfit.
- **Uncovered S3:** traverse between otherwise disconnected contents.

Unknown has no witness and therefore cannot license a replacement. Identity is inert because replacing its head with its witness leaves the sequence unchanged. Evidence carrying the queued head s — as its signature or as a node of its witness — is inert for the derivation of s: nothing held licenses writing s into ν_A, so the signature never enters node position in the first place.

A replacement never empties the node sequence: both sides of a correspondence carry content.

States are therefore well-founded with respect to s: a derivation never writes its own head into its witness, and the identity s:[s] is unreachable as a derivation result — the identity is the ask's own shape, a fact delivered by grounding, never a derivation's answer. An occurrence of s at entry belongs to the ask's riding canon; none is added.

### Mirror derivations

A correspondence can be read in either direction. Direction depends on which side is present in the current derivation; the same kline read from the other side licenses the mirror derivation.

The clause is band-general: every band has a mirror — a Canon contracts, a covered misfit undoes its move, an uncovered correspondence traverses back. Canonicalisation, below, names the Canon instance only.

### Canon replacements

A Canon permits:

- **expand:** n → ν_K
- **contract:** ν_K → [n]

Both preserve σ(ν_A).

### Covered misfit replacements

A covered misfit changes content:

- forward sheds the evidence's underfit and adopts its overfit;
- reverse performs the inverse operation.

### Uncovered replacements

An S3 correspondence allows traversal between disjoint content.

Traversal runs in whichever direction arrival supplies: forward enters the witness from the head; reverse, the mirror read, returns from the witness to the head. The worked example of §9 crosses all:[o] in this reverse fashion.

The correspondence itself is the licence. The band alone never licenses a replacement.

### Canonicalisation

Canonicalisation is the Canon instance of the mirror clause: the reverse replace under a held Canon, engaged by survey rather than by arrival.

Because occurrence is multiset-wise, a group of nodes contracts wherever it sits in the sequence: `{d,h}` contracts to `[dh]` out of `[w,d,m,h]` as freely as out of `[d,h,w,m]`. A phrase's discontinuity in the sequence is invisible to the licence.

Held witnesses propose the configurations. A group of nodes is contracted only when it is exactly witnessed: each node covers the candidate compound from below, and a held Canon counter-witnesses exactly them from above. Nothing unwitnessed contracts.

No new rule is involved; canonicalisation is the reverse replacement of this definition under its Canon interpretation.

### Well-founded Canon evidence

A Canon used for expansion must satisfy:

```text
n ∉ ν_K.
```

This prevents self-containing expansion cycles.

Because every node of a Canon is a subset of its head, a self-containing expansion would require the head to contain itself as one of its own witness nodes. Excluding this case ensures that repeated Canon expansion terminates and that resolution depth is finite.

## Definition 14 — Licensing

The relationship C(A,B) determines the region in which targeting may occur. The evidence kline determines the replacement itself.

| Relationship band | Licensed targeting                                           |
| ----------------- | ------------------------------------------------------------ |
| S1                | none; the derivation is done                                 |
| S2                | replacements restricted to the misfit region                 |
| S3                | replacement permitted; all nodes are in the misfit           |
| S4                | none; the unmatched underfit walks its slots (Definition 15) |

The restriction reads on both ends of the move: forward, the departed node carries underfit content or the arriving witness adopts overfit content; reverse, the consumed nodes carry the underfit or the arriving head lands in the overfit. An empty underfit therefore bars nothing — an overfit relationship is worked by adoption, on the arrival clause alone.

A targeting replacement is licensed only when it strictly decreases:

```text
|σ(ν_A) Δ σ(ν_B)|.
```

Thus a replacement that affects only shared content, or increases the mismatch, is not a targeting step.

Canon expansion and contraction are different: they preserve the current content and are licensed by the witness alone.

## Definition 15 — Slot derivation

A targeting relationship can be decomposed into slots.

The misfit is carried on both parties: the underfit by nodes of ν_A, the overfit by nodes of ν_B. A node is a **slot** when it carries misfit content — an underfit slot of ν_A, an overfit slot of ν_B. The notion is one, read on the two parties: an overfit slot of C(A,B) is an underfit slot of C(B,A).

For an underfit slot, strategy first looks for a licensed replacement at the slot. For the overfit, an adoptive replacement is sought by selection, at any node of ν_A. If none exists, the slot is walked: a **meeting** of two walks, one from each party.

A walk is licensed in two step forms. A **descent** crosses a held kline the current value heads, from its signature to its witness — and licenses the kline's whole witness: a multi-node witness cannot be partially consumed, its siblings would dangle. An **ascent** steps into the head of a held kline whose content contains the current value's. Heading licenses the bridge alone — the meeting value is one both parties walk to. No other licence is permitted: occurrence of a witness licenses nothing, and a kline's direction is a property of the walk, not of arrival. An ascent into the head of the kline a party stands on is inert — a walk never re-traverses its own edge. A value with no headed kline and no containing head is a walk's end.

A connotation is the designed walk segment: its head composes the underfit atom with the witness content, so a walk enters from the underfit value — contained in the head — and leaves at the witness: `m → ms → s`. Connotations chain through their witnesses: each arrival may ascend into the next head. B's side reads ν_B's klines as walk material — the goal's own canon enumerates its slots, every node accounted.

A's walk departs its underfit slots — the nodes of ν_A carrying the gap. B's walk departs the held value containing the overfit, the compound the overfit composes into. Both parties walk through held klines until a **shared value is delivered by distinct klines on the two sides** — the meeting. The klines must be distinct: one kline cannot meet itself, and a party's own delivery is not a second witness.

The meeting is written into memory as the **bridge** — a composed correspondence `slot_a:[slot_b]` with acquisition depth equal to the edges both walks crossed, which the main derivation may then consume: the A-side departure value is replaced by the B-side departure value. The goal is never rewritten; ν_B's klines are walk material, read and never changed.

The slot walk is therefore an evidence-construction mechanism: the two parties walk from their misfit locations, and the bridge exists only where held klines genuinely meet — two klines sharing a value, one reachable from each side. Nothing is invented: a value nothing heads is unreachable, a hub's klines replace one another only as the meeting's distinct-kline pair licenses, and a misfit with no slot on either side — a pure overfit against A — has no walk to meet.

---

# 7. Two Licences, One Rewrite Rule

There is only one rewrite operation: **replace**.

It has two distinct licences.

### Witnessed replacement

Canon-mode expansion and contraction:

- preserve current content;
- change granularity;
- require only memory;
- do not depend on the goal.

### Targeting replacement

Denotation- and connotation-based moves:

- change current content;
- reduce the mismatch to the goal;
- require evidence from memory;
- are restricted by the current misfit region.

The distinction is therefore between **what changes** and **what permits the change**, not between two separate rewrite systems.

A derivation may reach the S1 relationship before its node sequence exactly matches the goal's node sequence. This is intentional. Completion is based on value equality, not witness equality.

### The correspondence graph

Read model-theoretically, held klines are edges between a signature and its witness, traversable in either direction from wherever the derivation has arrived. Canon edges join different decompositions of the same content: expand and contract are the two directions of one congruence. Misfit edges join different content.

A derivation is a path in this graph, relativised to what is held. The path is the semantics: witnessed edges never change content, so no amount of granularity change can reduce a content mismatch. Done by alignment alone is unreachable by construction.

### Terminals

Terminals are targeting-closed, not rule-closed. An Identity relationship is done, yet the identity's node may still expand under a held well-founded witness — `s:[s]` expands to `s:[a,b]`, the identity turned Canon. There is no retain move; closure is the S1 row of the licensing table, not a rule.

---

# 8. Endings, Progress and Termination

## Definition 16 — Endings

A derivation stops in one of three ways. Its outcome is not how it stopped but the significance of the state it stops at — a calculated level (Definition 20), never a boolean.

### Done

Done is defined by the calculation:

```text
significance = 1.0.
```

Significance is γ at entry depths — the content overlap J(σ(ν_A), σ(ν_B)) — which is 1.0 exactly when the terminal contents are equal. Each S1 shape of the relationship satisfies that equality: Canon by its case condition, Identity because ν_B = [σ(ν_A)] collapses to the same content. Therefore

```text
fit(C(A,B)) ∈ S1
```

holds at done. The band statement is a proof that done means S1, not the definition.

A proposal carries its significance; one that reaches done is proposed at S1 by calculation.

Node sequences need not be equal.

### Stuck

The derivation is not done and no licensed targeting replacement exists. The current misfit cannot be reduced by any available correspondence, directly or through a meeting walk (Definition 15): no pair of held klines delivers a shared value to the two parties' misfit locations.

A stuck derivation therefore represents the absence of a currently available semantic bridge in memory. Relative non-existence is an honest outcome, reachable at entry and mid-run alike.

### Abandoned

Strategy may stop a derivation before reaching done or stuck, for example because its measured progress has become unsatisfactory.

Abandonment is a strategy decision, not a derivation rule.

Witnessed replacements never make a derivation done because they preserve the current value. They may still be useful because they expose a granularity at which a targeting correspondence can be applied.

### Termination

#### T1 — Targeting bound

For a targeting-only run beginning at A₀:

```text
Δ₀ = |σ(ν_{A₀}) Δ σ(ν_B)|
```

and the run has at most Δ₀ targeting steps.

Each targeting step strictly reduces the mismatch mass, so infinite or cyclic targeting is impossible: a replacement that does not shrink the mismatch is not a targeting move.

A replacement may remove or introduce several units of content in one step. The bound is expressed in content, while the run itself proceeds in evidence-sized steps.

#### T2 — Witnessed and traversal cycles

Witnessed replacements may cycle without changing content. For example:

```text
bc → [b,c] → bc.
```

Traversal may also revisit correspondence states without changing the mismatch.

Therefore termination of a mixed derivation is a strategy property rather than a property of the rewrite relation alone.

A strategy must bound witnessed runs and traversal. Suitable policies forbid expand-after-contract of the same witness, and bound the meeting walk by edges: an ascent admits mutual-coverage cycles (`m → ms → s → ms`), so a walk may revisit values and terminates at the edge bound alone (Definition 15).

### Confluence

Confluence is intentionally not required.

Different derivation paths may ground different correspondences first and may therefore reach different witnesses. Path dependence is part of the system's intended behaviour.

### Decidability

Because the value space may be taken finite and well-founded Canon expansion terminates, the set of reachable derivation states is finite in principle. Reachability and non-reachability are therefore decidable in principle, although an exhaustive search may be impractical.

---

# 9. What a Derivation Establishes

A derivation establishes the significance of the state it stops at (Definition 16). Done, stuck, or abandoned, the stopping is an observation; the established significance is the result, and the calculation proves the band reached.

At 1.0 the band is S1, and S1 states the equality:

```text
σ(ν_A) = σ(ν_B).
```

The final node sequence is a constructive witness for that equality. Every rewrite in the path was licensed by a correspondence held in memory: the path itself is carried as evidence.

Below 1.0 nothing is proven equal. An S2 outcome — real, measured overlap short of equality — may be a useful result, but it is not a proof of equality.

The derivation does **not** prove:

- that the queued kline's original head was correct;
- that every correspondence used was factually correct;
- that an S3 correspondence was grounded.

An S3 kline may represent a promise rather than established knowledge. Ratification and evaluation of such promises are outside the rewrite rules.

Significance — the content overlap of fit(C(Aᵢ, B)) — may be measured at every state; γ, significance net of complexity, is tracked step by step, and its rate of change is the signal strategy uses to decide whether continued effort is worthwhile.

## Worked example — “what did Mary have?”

Suppose memory contains:

```text
mhall:[m,h,a,l,l]     Canon
dh:[d,h]              Canon
all:[a,l,l]           Canon
dh:[h]                Connotation
w:[o]                 Denotation
all:[o]               Denotation
m:[m]                 Identity
```

The queued kline is:

```text
A₀ = wdmh:[w,d,m,h]
```

and the goal is:

```text
B = mhall.
```

The initial relationship is:

```text
C(A₀,B) = wdmh:[m,h,a,l,l]
```

with:

```text
underfit = {w,d}
overfit  = {a,l}
Δ₀       = 4.
```

### Step 1 — Canonicalise the verb

The question arrives as bare word bits — d and h discontinuous around m — while the Connotation `dh:[h]` licenses a replace only at the composed granularity, on a dh node that does not yet exist, and no held kline has signature d. The derivation therefore surveys the unordered configurations of its nodes against held witnesses. The configuration {d,h} is exactly witnessed — d and h each cover dh, and the Canon `dh:[d,h]` counter-witnesses exactly them — so it contracts:

```text
[d,h] ⇉ [dh]
```

giving:

```text
[w,dh,m].
```

The value remains wdhm, so the targeting mismatch is unchanged. The contract is a witnessed move and spends no targeting budget; it exposes the granularity at which Step 2 can apply.

### Step 2 — Apply connotation

The held Connotation:

```text
dh:[h]
```

permits:

```text
dh ⇉ [h]
```

which gives:

```text
[w,h,m].
```

The atom d is removed, so the mismatch decreases from 4 to 3.

### Step 3 — Obtain the object correspondence

No directly held kline maps w to the object content. A slot derivation is therefore used to construct it — a meeting of two descents (Definition 15).

A's descent departs the underfit slot w. B's departs `all`, the held value containing the overfit:

| Descent | Edge    | Correspondence | Licence           |
| ------- | ------- | -------------- | ----------------- |
| A       | w → o   | w:[o]          | w heads w:[o]     |
| B       | all → o | all:[o]        | all heads all:[o] |

The descents meet at o — a value delivered by two distinct klines, one from each side. The meeting writes the bridge:

```text
w:[all]
```

with acquisition depth 2, the edges both descents crossed. The bridge says: w is replaceable by all, licensed at o. Nothing else is written — the klines the descents crossed are already held, and a value nothing heads is unreachable.

The main derivation can then apply:

```text
w ⇉ [all]
```

giving:

```text
[all,h,m]
```

whose value is:

```text
mhall.
```

The relationship is now Canon, so the derivation is done — the adopted overfit arrives as one compound node at acquisition depth 2, so the measurement reads Ĥ = 2/3, γ = 2^(-2/3).

The subject m is never replaced: `m:[m]` is Identity, inert.

Had the Denotations not been held, no descent would leave w and no meeting could form: the derivation would be stuck, and the misfit would ask. Had mhall itself not been held, there would be no goal to check done against.

Read from the answer side, the same klines license mhall's own descent — direction is a property of the walk, fixed by heading, not by arrival.

The important point is that the system reaches the answer through held correspondences and the meetings they permit; it does not simply replace the question with the answer.

---

# 10. Measurement

The significance bands are ordered:

```text
S1 > S2 > S3 > S4.
```

Shapes within a band are not otherwise ordered.

The band predicate is observer-independent: given the same held memory, every agent classifies alike. A band therefore never needs to be exchanged.

There are two relevant bands:

1. the band of a kline considered by itself;
2. the band of the relationship between two klines.

The first describes the claim represented by a kline. The second describes the result of a derivation relative to a goal.

## Definition 17 — Jaccard overlap

For values x and y:

```text
J(x,y) = |x ∧ y| / |x ∨ y|.
```

J is 0 when the contents are disjoint and 1 when they are equal. It is symmetric and therefore measures content overlap independently of derivation direction.

Jaccard is forced rather than chosen. Score each node by its accountedness with the goal,

```text
α(n) = |n ∧ σ(ν_B)| / |n|,
```

and compose content-weighted: the result is A's coverage fraction, |σ(ν_A) ∧ σ(ν_B)| / |σ(ν_A)|. That fraction reads 1 whenever A's content sits wholly inside B's — an Underfit, still S2. A measure that scores perfect on a misfit is band-inconsistent. Weighing B's overfit as well yields the symmetric form, which is Jaccard.

## Definition 18 — Resolution depth

D̄ is the content-weighted mean resolution depth of the current content.

Content held at its own resolution has depth 0. Expansion increases depth; contraction decreases it.

Depth is well-defined because licensed Canon expansion terminates (§6).

## Definition 19 — Acquisition depth

Ĥ is the content-weighted mean acquisition depth of the current content.

Content present at entry has acquisition depth 0.

Content introduced through composed evidence — a correspondence this derivation wrote — records the evidence's depth plus one.

Content introduced through standing evidence — a held kline — inherits the evidence's recorded depth.

Canon-mode rewrites do not change acquisition depth because they do not introduce new content.

Klines written into memory carry their recorded depths; consuming one composes its depth with the current one.

Acquisition depth is stored provenance. It cannot in general be reconstructed from the current node sequence alone, because the sequence does not record the path by which its content was acquired: consumption leaves no trace in ν_A.

## Definition 20 — Significance and complexity

Significance is the measure of rational understanding:

```text
sig(A,B) = J(σ(ν_A), σ(ν_B)).
```

It reads only the two contents: 1.0 exactly at value-equality — done (Definition 16) — and 0 at disjointness. It is γ at entry depths — Definition 19 gives content present at entry depth 0, so the discount vanishes and only the overlap remains. Significance selects the band and travels with the proposal.

Complexity is the measure of work — how much effort arriving at that significance cost:

```text
complexity = 1 − δ^(D̄ + Ĥ),
```

the complement of the discount over both depths (0 < δ < 1; both denominated in edges). Entry content and standing licences cost nothing; deeper decomposition and composed acquisition raise it. Complexity is independent of significance: it prices moving between embedded concepts, and never selects a band.

Their composite is γ:

```text
γ = J(σ(ν_A), σ(ν_B)) · δ^(D̄ + Ĥ) = significance × (1 − complexity)
```

— significance net of the work spent. The form is fixed, not free: four requirements force it.

Consequently:

- deeper decomposition raises complexity;
- composed acquisition raises complexity;
- equal-content states can differ in complexity when reached by different paths.

γ is directional by design: it grades this derivation's effort toward its goal. B's depths are B's own derivation's problem.

### Bands quantize significance

For non-vacuous pairs the band is the quantization of significance: S1 is significance 1.0, S2 is overlap short of equality, S3 is zero overlap. S4 — the unknown — is off the scale: the halt with nothing to measure. The former "structural significance" is this quantization; one measure serves a kline's own claim and a relationship alike.

γ at the recorded depths — significance net of complexity — compares derivations that achieve the same significance: same arrival, different cost. Its rate of change steers strategy. A derivation may arrive at significance 1.0 with γ far below.

### Invariants

Four properties of the measure:

1. **Band-consistency.** Significance is 0 exactly at content-disjointness and 1 exactly at value-equality — the band's two ends. The depths only scale γ down.
2. **Granularity-invariance.** Witnessed moves change complexity only through D̄, never through recomposition. Content-weighted composition is blind to how content is sliced into nodes. An unweighted per-slot mean violates this: expansion alone can raise it at constant content and constant depth.
3. **Granularity-monotonicity.** Expand strictly increases D̄, so strictly increases complexity; contract strictly decreases it. This makes gratuitous expansion detectable.
4. **Provenance-monotonicity.** Composed acquisition strictly increases Ĥ, and nothing in a derivation lowers it — only re-derivation through standing licences does. This makes promise-stacking detectable.

The monotonicities are properties of the intended strategy model, not guarantees for arbitrary mixed derivations. Consuming composed evidence can lower γ even as J rises — the step that wins the answer dips. That dip is the price signal steering strategy toward standing licences.

### Rate of change

γ, unlike the band, changes measurably per step; its rate of change is the signal that cogitation's feedback acts on.

### Exchange

A KValue (CONTEXT.md) carries the sender's assessment — for a proposal, its significance; the complexity stays behind as the acquisition record: depths are part of what memory holds. Bands need not travel, for they are recomputable from structure.

### Example

Suppose the derivation begins with:

```text
J = 2/5
```

and ends with:

```text
J = 1
```

after acquiring three atoms at depth 3.

Then:

```text
Ĥ = (0 + 0 + 3 + 3 + 3) / 5 = 9/5.
```

For δ = 1/2:

```text
γ = 2^(−9/5) ≈ 0.29 — significance 1.0 at complexity 1 − 2^(−9/5) ≈ 0.71.
```

The answer is therefore complete at full significance, high complexity: the work records the cost of reaching the answer through composed evidence.

A standing licence removes that cost: once written into memory, the same correspondence is traversed at its recorded depth, without the composed penalty.

---

# 11. Strategy

Sections 6–9 define what a derivation may do; §10 measures its worth. Strategy determines which permitted operation is attempted and in what order.

The strategy loop is:

```text
select a goal
    ↓
scope the memory
    ↓
derive to an ending
    ↓
add the result to memory
    ↓
re-enter with the resulting kline
```

A hop takes its goals from the top of the candidate list (Definition 22) and works down it: each goal gets its own scope and its own derivation, and a derivation that ends without done yields the next goal. The first four phases repeat within a hop; the last queues its result as the next hop's input. Strategy may also abandon or retarget a derivation.

## Definition 21 — Hop

A hop is the strategy unit of one queued kline: goals are taken from the top of its candidate list (Definition 22) in order, and each goal is scoped (Definition 23) and derived to an ending. A hop may therefore run several derivations; one that ends without done yields the next goal, and the hop ends at done, at the list's exhaustion, or at a bound.

Within a hop, the queued kline's head is fixed, and each derivation's goal and scope are fixed for its duration (Definition 12). The hop's writes land in memory and are available to later hops alone; between hops, re-entry supplies the next queued kline.

## Definition 22 — Selection

For queued kline A, selection assembles the candidate goals, in order.

A held kline

```text
K = t:ν_K ∈ M
```

is a candidate goal for A when its content covers a node of ν_A:

```text
n ∧ σ(ν_K) ≠ ∅, for some node n ∈ ν_A    (Definition 8).
```

The candidates, ordered by descending γ(A, K), are the goal list. γ is the composite (Definition 20): the significance of working from A toward the candidate — content overlap between the two parties — net of the complexity of reaching it. γ, not band, sets the order.

Three klines never join the list. The queued kline itself: C(A, A) is Canon
at entry and proves nothing — a hop that breaks on it never reaches the
real goals down the list. An ask: a question is not held content to derive
toward. And an ask's canon: the marker is invisible to measurement, so the
canon's content is the ask's own — that goal would be A again.

The goal is taken from the front of the list; each derivation within a hop takes the next candidate in turn (Definition 21).

## Definition 23 — Scope

Once A and B are set, the derivation's memory is scoped. A scope is a trawl of the correspondence graph (§7), rooted at both parties: every correspondence reachable from A's nodes and B's nodes within a fixed depth.

The trawl is dual-rooted, so the correspondences that join the parties — the misfit edges a derivation needs — are in scope by construction. It is depth-bounded and unranked: fast but stupid, no lookahead.

The scope is the derivation's M for the hop's duration, read as trawled and never extended mid-hop. What a hop writes goes to memory (Definition 7); later hops' trawls reach it.

## Progressive path

Each hop may add evidence to memory. Later hops therefore operate with:

```text
M_{k+1} ⊇ M_k.
```

The memory a later hop trawls from may therefore contain correspondences generated by earlier hops.

## Bounds

Strategy controls four independent limits:

1. the targeting bound from T1;
2. limits on repeated witnessed transformations and traversal, per T2;
3. the number of goals a hop takes from its list;
4. the maximum re-entry depth.

## Re-entry

Every ending — done, stuck, or abandoned (Definition 16) — leaves a result that may produce the input to a subsequent hop. Re-entry changes A, never B: the resulting A reselects its candidate list (Definition 22), and the top of the new list may be the same kline again.

Hop k is parameterised by:

```text
(M_k, B_k)
```

— the scope trawled from A_k and B_k (Definition 23), and the goal selected from A_k. A hop's writes land in memory as it proceeds, so later hops trawl from a memory grown by every earlier hop.

Hop order is the only temporal structure defined by the system.

---

# 12. Terminology

**Significance** is the numerical value assigned by the measurement model.

**Rationalisation** is the process of producing, using, and consolidating the evidence represented in memory.

**Understanding**, informally, is sustained possession of a high-significance result.

These terms are descriptive rather than additional algebraic primitives.

---

# 13. KScript Surface Syntax

KScript is a surface language for constructing klines and kline pairs.

The syntax specifies an intended structure; the algebra then determines the actual fit shape.

| Token         | Structure                             | Band claim once solved |
| ------------- | ------------------------------------- | ---------------------- | ------------ |
| `a => b c d`  | `a:[b,c,d]`                           | S1 (Canon) or open S2  |
| `a = a`       | `a:[a]`                               | S1                     |
| `a > b`       | `ab:[b]`                              | S2                     |
| `a = b`       | `a:[b]`                               | S3                     |
| `a > a`       | `a:[a]`                               | S1                     |
| `a`           | `a                                    | ASK:[a]`               | S4 — the ask |
| ask-annotated | any signature carrying the ASK marker | S4                     |

`ASK` is the ASK marker (word-word bit 31): it marks identity, never
content — every content measurement masks it out — and the ask's canon
nodes ride along so selection sees the question's content. An ask never
heads a goal list, nor does its canon.

The surface token does not override algebraic classification.

For example:

```text
a => b c d
```

creates:

```text
a:[b,c,d]
```

but that kline is Canon only when:

```text
a = σ([b,c,d]).
```

The `>` form creates:

```text
ab:[b]
```

which is a Connotation — a single-node Underfit — when a and b are distinct.

A bare token:

```text
a
```

creates:

```text
a|ASK:[a]
```

— the identity shape, marked. There is one ask structure: `s|ASK:[nodes]`
— the question with its canon's nodes (a single token's own value) riding
along. The ask is structural — the marker is read from the kline's value,
not decreed — and it is outside the content measure: no measurement weighs it,
and selection (Definition 22) reads the question's content through the
nodes.

The `=>` operator establishes a composition claim and supplies a goal for completion checking. It is not itself a rewrite licence; licences are correspondence klines (§6).

Ratification, tier management, queue policy, and other protocol behaviour remain outside the algebra.

---

# 14. Outside the Algebra

The following mechanisms are not defined by this document:

### Tier mechanics

STM, Frame and LTM membership, promotion rules, and attention management are properties of the memory system rather than the algebra.

### Multi-agent control

Trainer, trainee, supervisor, escalation, and related orchestration belong to the surrounding protocol.

### KScript compilation

Token expansion, word binding, and annotations belong to the implementation and to CONTEXT.md.

---

# Appendix — Worked example: “Mary had”

The §9 example works an under+over relationship: the question holds content the answer lacks (w, d) and lacks content the answer holds (a, l). This example works the pure overfit relationship — the fragment — where the underfit is empty: the case with no A-side slot, and so no meeting.

Suppose memory contains:

```text
mhall:[m,h,all]       Canon
all:[o]               Denotation
all:[a,l,l]           Canon
o:[m]                 Denotation
m:[m]                 Identity
```

The queued kline is:

```text
A₀ = mh:[m,h]
```

and the goal is:

```text
B = mhall.
```

The initial relationship is:

```text
C(A₀,B) = mh:[m,h,all]
```

with:

```text
underfit = ∅
overfit  = {a,l}
Δ₀       = 2.
```

No node of ν_A carries underfit content: the per-party decomposition yields no A-side slot. No held correspondence adopts the overfit at any node of ν_A — `o:[m]` read reverse swaps m for o and worsens the mismatch, `all:[o]` and `all:[a,l,l]` occur nowhere in ν_A, and no exactly-witnessed group contracts. Targeting alone is stuck.

### The meeting cannot form

Under Definition 15's meeting licence the walk needs both parties: A's descent from an underfit slot, B's from the held value containing the overfit. B's side works — `all` heads `all:[o]`, and `o` heads `o:[m]`, so B's descent delivers m, a node of ν_A — but A has no slot to descend from, and a lone descent is not a meeting. The derivation is stuck; the misfit asks.

That `o:[m]` connects the parties is real, but under the meeting licence a connection one party reaches alone licenses no bridge. Whether an **anchor arrival** — B's descent delivering a node of ν_A outright — should count as a meeting is an open licence question: it would answer this fragment, but a descent through the queued kline's own signature would then re-mint its reciprocal (a queued `little:[Mod]` reaching itself backward through `Mod`), the fabrication the meeting licence exists to exclude.

The ask is therefore the honest outcome under the current licence: no meeting, no bridge — even though a route exists on one side. The system reaches answers only where held klines meet from both parties.

---

# Appendix — The word-bit realisation

How the reference realisation (Definition 2) populates the value space (Definition 1): the 1-bit atomic scheme in construction detail, each law checked. Nothing here is algebra — it is one way of making the five capabilities true.

### The bit basis

One bit per distinct word. A multi-subword word (`Mary` → `[mar, y]`) is one word and one bit — the subword token ids OR together into the token half, and the shared bit does the work a decomposition would: no kline is needed to know a word is one.

Bits are assigned on a first-encountered basis at half-positions 0–30, by the ks compiler; a compile seeded with the prior state's known words carries the basis across scripts (word binding; CONTEXT.md). A compound — expansion-introduced or a CONNOTES concatenation — takes no bit of its own: its value is the OR-reduction of its component words' values.

Half-position 31 is reserved: the ASK marker. No word ever carries it.

The basis holds 31 words; the 32nd distinct word is an error — word size overflow. The realisation fails loudly rather than silently widening a word: exhaustion is a property of the scheme, not of the algebra.

### The layout

A value is a 64-bit word:

```text
[63    | 62 … 32     | 31 … 0       ]
[ASK   | word word   | bpe token id ]

node = (word_bit << 32) | bpe_token_id
```

The word word is the value the algebra reads. The token half — the OR of the word's subword token ids — is encoding provenance: inert cargo that rides along and is masked off at every content read. The content mask is the word word without the marker:

```text
MASK = 0x7FFF_FFFF_0000_0000
```

### The capabilities realised

| Definition 1     | realisation                                                 |
| ---------------- | ----------------------------------------------------------- |
| `v ∨ w`          | `a \| b` — full-word OR (token cargo may accumulate; inert) |
| `v ∧ w`          | `(a & b) & MASK`                                            |
| `v ∖ w`          | `(a & ~b) & MASK`                                           |
| `\|v\|`          | `popcount(a & MASK)`                                        |
| content equality | equality under the mask                                     |

Derived forms are exact: `|x Δ y| = popcount((x ^ y) & MASK)`; `J(x,y) = popcount((x ∧ y) & MASK) / popcount((x ∨ y) & MASK)`.

### The laws

Each axiom of Definition 1 is an ordinary fact about bit sets under the mask: OR is commutative, associative, idempotent, with unit 0; the residue laws are set difference; μ is popcount — zero exactly at the empty mask, and a masked subset of equal count is equal (no ghost content). The measure is exact, so the strict inequalities of licensing (Definition 14) and the invariants of §10 hold as written.

### The marker discipline

The ASK marker is OR-ed into an ask's signature at compile time. No node carries it, and a derivation never changes the signature (Definition 12), so the marker rides untouched: only compilation mints it. Every content read masks it out; identity reads — store keys, lookups, kline equality — see it, which is what keeps an ask distinct from its canon at the same content.

### Address resolution

A value is its own address: the store keys klines by the word, and a node references another kline by holding its signature — no indirection, no lookup table, one machine word per reference. This and the 31-word basis are the two engine facts Definition 2 names; a realisation at scale replaces both with integer word ids, exact id sets, and interned content keys.

### Costs

Every capability is one machine-word operation; μ, misfit mass, and J are exact integer arithmetic. The depth accounting of Definitions 18–19 is kept per word bit — the accounting granularity matches the measurement granularity, which is what granularity-invariance expects.
