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

The measurement model assigns each derivation a graded significance value. The grade depends on:

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

### 0.1 Sets and values

The atom set is A. A value is a subset of A.

| Symbol      | Meaning                       | Implementation |
| ----------- | ----------------------------- | -------------- |
| `a`, `abc`  | `{a}`, `{a,b,c}`              | bitmasks       |
| `v ∨ w`     | union                         | `v \| w`       |
| `v ∧ w`     | intersection                  | `v & w`        |
| `¬v`        | complement within A           | `~v & MASK`    |
| `∅`         | empty value                   | `0`            |
| `σ(ν)`      | evaluation of node sequence ν | `fold_or(nu)`  |
| `x Δ y`     | symmetric difference          | `x ^ y`        |
| `v ∖ w`     | set difference                | `v & ~w`       |
| `[n₁ … nₖ]` | node sequence                 | list           |
| `x ∈ y`     | membership                    | `x in y`       |
| `\|x\|`     | number of atoms in x          | `popcount(x)`  |

The formalisation also uses V\* for sequences of values and 2^A for the set of all values.

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

is the graded significance of working from A toward B.

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

# 1. Atoms and Values

## Definition 1 — Atoms

The atom set is a finite set:

```text
A = {a₀ … a₃₀}.
```

Only finiteness is required by the algebra.

In the implementation, atoms correspond to the available bit positions. The encoding used by the engine is an implementation detail and is not itself part of the algebra.

## Definition 2 — Values

A value is a subset of the atom set:

```text
V = 2^A.
```

For an atom a, the corresponding singleton value is also written a. The empty value is ∅, and the full value is A.

The basic operations are:

```text
v ∨ w = v ∪ w
v ∧ w = v ∩ w
¬v = A ∖ v.
```

These operations satisfy the usual Boolean laws because they are ordinary set operations.

For example:

```text
ab ∨ ac = abc
ab ∧ ac = a
¬a = bc.
```

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

The resulting reference graph is unrestricted and may contain cycles, including canon self-reference and countersign pairs. Cycles carry no decomposition content.

Tier membership and memory-management policy are outside this algebra.

---

# 4. Coverage and Fit

## Definition 8 — Coverage

A node n is **covered** by a value s when:

```text
n ∧ s ≠ ∅.
```

Coverage means that the node shares at least one atom with the value. It does not require the node to be a subset of the value.

## Definition 9 — Underfit and overfit

For a pair (s, ν), let:

```text
u = s ∧ ¬σ(ν)
o = σ(ν) ∧ ¬s.
```

The **underfit** u contains atoms claimed by the head but not supplied by the nodes.

The **overfit** o contains atoms supplied by the nodes but not claimed by the head.

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

Unknown is also the structural representation of an ask: the system has no current content on the relevant side.

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

The goal is read, never rewritten: it scopes targeting (Definition 14), determines the ending (Definition 16), and its nodes may seed slot walks (Definition 15).

## Definition 12 — Derivation

For A = s:ν_A and goal B = t:ν_B, a derivation step has the form:

```text
A ⊢_{M,B} A′.
```

The subscript names the derivation's parameters: the memory it reads — its scope, fixed as trawled at entry (Definition 23) — and the goal it works toward, fixed for the derivation's duration. Writes go to memory (Definition 7), not to the scope: a derivation never consumes what it writes.

The head s of A does not change during the derivation. Only its node sequence changes. Nothing in §§6–9 reads s: licences, endings, bounds, and grades read only the relationship, whose head σ(ν_A) is exact against ν_A at every state by construction. The queued head rides along inert, mattering only after the hop — in absorption, re-entry (§11), and the claim's grounding (§14).

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

Unknown has no witness and therefore cannot license a replacement. Identity is inert because replacing its head with its witness leaves the sequence unchanged.

A replacement never empties the node sequence: both sides of a correspondence carry content.

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

| Relationship band | Licensed targeting                                 |
| ----------------- | -------------------------------------------------- |
| S1                | none; the derivation is done                       |
| S2                | replacements restricted to the misfit region       |
| S3                | replacement permitted; all nodes are in the misfit |
| S4                | none; the derivation is stuck                      |

The restriction reads on both ends of the move: forward, the departed node carries underfit content or the arriving witness adopts overfit content; reverse, the consumed nodes carry the underfit or the arriving head lands in the overfit. An empty underfit therefore bars nothing — an overfit relationship is worked by adoption, on the arrival clause alone.

A targeting replacement is licensed only when it strictly decreases:

```text
|σ(ν_A) Δ σ(ν_B)|.
```

Thus a replacement that affects only shared content, or increases the mismatch, is not a targeting step.

Canon expansion and contraction are different: they preserve the current content and are licensed by the witness alone.

## Definition 15 — Slot derivation

A targeting relationship can be decomposed into slots.

The misfit is carried on both parties: the underfit by nodes of ν_A, the overfit by nodes of ν_B. A node is a **slot** when it carries a misfit atom — an underfit slot of ν_A, an overfit slot of ν_B. The notion is one, read on the two parties: an overfit slot of C(A,B) is an underfit slot of C(B,A).

For an underfit slot, strategy first looks for a licensed replacement at the slot. For the overfit, an adoptive replacement is sought by selection, at any node of ν_A. If none exists, either slot may be explored by a goal-less derivation beginning with:

```text
n:[n].
```

A slot walk is licensed by occurrence of a correspondence — either side of a held kline occurring in the walk's nodes — not by the targeting-scoping rule.

A slot walk ends when:

- from an underfit slot, it reaches content overlapping the overfit;
- from an overfit slot, it reaches content overlapping σ(ν_A), the **anchor**; or
- no unvisited correspondence is available.

The goal is never rewritten. A walk from an overfit slot departs from a node of ν_B and writes only a correspondence; ν_B itself never changes.

Arrival is not absorption. The walk continues under held Canons alone, refining its discovered end to the arrival party's resolution — the overfit content to the goal's witness, the anchor to ν_A's nodes. The departed end already sits at its own party's. The refined state is the terminal.

The terminal is written into memory as a composed correspondence spanning the walk's two ends, with acquisition depth equal to the edges crossed, which the main derivation may then consume. The head is the A-side end: the departed slot or the discovered anchor. The witness holds the head's atoms shared with the goal, together with the B-side end's nodes covering the overfit. A compound node is opaque to licensing — occurrence reads nodes, never the atoms within them — so refinement goes exactly to the consuming resolution and no further: expansion by Definition 13's well-foundedness, contraction by its exact witnessing.

A walk from ν_B that arrives at an underfit slot composes the same correspondence a walk from that slot would; only the direction of discovery differs.

The slot walk is therefore an evidence-construction mechanism: seeded at either misfit location, it discovers a route through memory and turns that route into a new reusable correspondence.

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

A derivation has one of three outcomes:

### Done

```text
fit(C(A,B)) ∈ S1.
```

Equivalently:

```text
σ(ν_A) = σ(ν_B).
```

Node sequences need not be equal.

### Stuck

The derivation is not done and no licensed targeting replacement exists.

This includes:

1. no goal is present; or
2. a goal is present but the current misfit cannot be reduced by any available correspondence, directly or through a slot walk from either party (Definition 15): nothing in M connects either misfit location to the other party's content.

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

A replacement may remove or introduce several atoms in one step. The bound is expressed in atoms, while the run itself proceeds in evidence-sized steps.

#### T2 — Witnessed and traversal cycles

Witnessed replacements may cycle without changing content. For example:

```text
bc → [b,c] → bc.
```

Traversal may also revisit correspondence states without changing the mismatch.

Therefore termination of a mixed derivation is a strategy property rather than a property of the rewrite relation alone.

A strategy must bound witnessed runs and traversal. Suitable policies forbid expand-after-contract of the same witness, and adopt no-revisit for traversal: each consumed correspondence — its signature together with its witness, not the signature alone — is used at most once during a slot walk.

### Confluence

Confluence is intentionally not required.

Different derivation paths may ground different correspondences first and may therefore reach different witnesses. Path dependence is part of the system's intended behaviour.

### Decidability

Because the atom set is finite and well-founded Canon expansion terminates, the set of reachable derivation states is finite in principle. Reachability and non-reachability are therefore decidable in principle, although an exhaustive search may be impractical.

---

# 9. What a Derivation Proves

A completed derivation proves:

```text
σ(ν_A) = σ(ν_B).
```

The final node sequence is a constructive witness for that equality. Every rewrite in the path was licensed by a correspondence held in memory: the path itself is carried as evidence.

The derivation does **not** prove:

- that the queued kline's original head was correct;
- that every correspondence used was factually correct;
- that an S3 correspondence was grounded.

An S3 kline may represent a promise rather than established knowledge. Ratification and evaluation of such promises are outside the rewrite rules.

The relationship:

```text
fit(C(Aᵢ, B))
```

may nevertheless be measured after every step, and its rate of change tracked over steps. This measurement is used by strategy to decide whether continued effort is worthwhile.

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

No directly held kline maps w to the object content. A slot derivation is therefore used to construct that correspondence.

The slot is the node w. The walk starts at the slot identity:

```text
w:[w]
```

and crosses three correspondences:

| Edge                | Correspondence     | Direction      | Occurring side                        |
| ------------------- | ------------------ | -------------- | ------------------------------------- |
| w:[w] → w:[o]       | w:[o] Denotation   | forward        | head w occurs as a node               |
| w:[o] → w:[all]     | all:[o] Denotation | reverse        | witness [o] occurs as a node multiset |
| w:[all] → w:[a,l,l] | all:[a,l,l] Canon  | forward expand | head all occurs as a node             |

The second edge is the crossover. Nothing held maps w to the object content directly: w and all meet only at the shared node o. The walk arrives at o by the forward side of w:[o] and leaves by the reverse side of all:[o] — the same Denotation read from the other side licenses its mirror traversal, [o] ⇉ [all], because its witness occurs in the current nodes. Arrival orients the correspondence; the band does not.

Two details do real work here. Occurrence is read on either side — the mirror clause (Definition 13) and the slot-walk licence (Definition 15): read forward-only, the walk would be stuck at w:[o], since no held kline is headed o. And the no-revisit policy (T2) forbids consuming w:[o] a second time, so the reverse occurrence at [o] cannot bounce the walk back to w — only all:[o] remains, and the walk is forced through the crossover.

The final state overlaps the goal's overfit and is written into memory as the composed correspondence:

```text
w:[a,l,l]
```

with acquisition depth 3, the three edges crossed; the third is the refinement edge (Definition 15).

The main derivation can then apply:

```text
w ⇉ [a,l,l]
```

giving:

```text
[h,m,a,l,l]
```

whose value is:

```text
mhall.
```

The relationship is now Canon, so the derivation is done.

The subject m is never replaced: `m:[m]` is Identity, inert.

Had the Denotations not been held, no replace would reach the object underfit: the derivation would be stuck, and the misfit would ask. Had mhall itself not been held, there would be no goal to check done against.

Read from the answer side, the same correspondences license the mirror derivation; the klines are direction-free, and arrival orients them.

The important point is that the system reaches the answer through held correspondences and the witness transformations they permit; it does not simply replace the question with the answer.

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

and compose atom-weighted: the result is A's coverage fraction, |σ(ν_A) ∧ σ(ν_B)| / |σ(ν_A)|. That fraction reads 1 whenever A's content sits wholly inside B's — an Underfit, still S2. A measure that scores perfect on a misfit is band-inconsistent. Weighing B's overfit as well yields the symmetric form, which is Jaccard.

## Definition 18 — Resolution depth

D̄ is the atom-weighted mean resolution depth of the current content.

Content held at its own resolution has depth 0. Expansion increases depth; contraction decreases it.

Depth is well-defined because licensed Canon expansion terminates (§6).

## Definition 19 — Acquisition depth

Ĥ is the atom-weighted mean acquisition depth of the current content.

Content present at entry has acquisition depth 0.

Content introduced through unratified evidence records the evidence's depth plus one.

Content introduced through ratified evidence inherits the evidence depth without the additional penalty.

Canon-mode rewrites do not change acquisition depth because they do not introduce new content.

Klines written into memory carry their recorded depths; consuming one composes its depth with the current one.

Acquisition depth is stored provenance. It cannot in general be reconstructed from the current node sequence alone, because the sequence does not record the path by which its atoms were acquired: consumption leaves no trace in ν_A.

## Definition 20 — Graded significance

The significance measure is:

```text
γ(A,B) = J(σ(ν_A), σ(ν_B)) · δ^(D̄ + Ĥ)
```

where 0 < δ < 1 is the discount factor. The same factor discounts both depths; both are denominated in edges.

Thus:

```text
significance = content overlap × discount for granularity and acquisition cost.
```

The form is fixed, not free: four requirements force it.

Consequently:

- deeper decomposition lowers significance;
- unratified acquisition lowers significance;
- equal-content states can have different significance if they were reached by different paths.

γ is directional by design: it grades this derivation's effort toward its goal. B's depths are B's own derivation's problem.

### Invariants

Four properties of the measure:

1. **Band-consistency.** γ is 0 exactly at content-disjointness and maximal only at value-equality. Both ends belong to J; the depths only scale down.
2. **Granularity-invariance.** Witnessed moves change γ only through D̄, never through recomposition. Atom-weighted composition is blind to how content is sliced into nodes. An unweighted per-slot mean violates this: expansion alone can raise it at constant content and constant depth.
3. **Granularity-monotonicity.** Expand strictly increases D̄, so strictly decreases γ; contract strictly decreases D̄, so increases γ. This makes gratuitous expansion detectable.
4. **Provenance-monotonicity.** Unratified acquisition strictly increases Ĥ, and nothing in a derivation lowers it — only ratification or re-derivation through ratified licences does. This makes promise-stacking detectable.

The monotonicities are properties of the intended strategy model, not guarantees for arbitrary mixed derivations. Consuming unratified evidence can lower γ even as J rises — the step that wins the answer dips. That dip is the price signal steering strategy toward ratified standing licences.

### Rate of change

The rate of change of γ per step is defined only at this level — a four-band predicate has no useful derivative — and is the signal that cogitation's feedback acts on.

### Exchange

The graded value travels in a KValue (CONTEXT.md) as the sender's assessment, and the acquisition record travels with the kline that earned it: depths are part of what memory holds. Bands need not travel, for they are recomputable from structure.

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

For δ = 1/2, the final significance is:

```text
γ = 2^(−9/5) ≈ 0.29.
```

The answer is therefore complete even though its significance has decreased. The decrease records the cost of reaching the answer through unratified evidence.

Ratification can remove that cost on a subsequent derivation by allowing the same correspondence to be traversed without the unratified penalty.

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

The candidates, ordered by descending γ(A, K), are the goal list. γ is the graded significance (Definition 20): the significance of working from A toward the candidate — content overlap between the two parties, discounted for granularity and acquisition cost. Significance, not band, sets the order.

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

| Token           | Structure                               | Band claim once solved             |
| --------------- | --------------------------------------- | ---------------------------------- |
| `a == b => c d` | `a|ASK:[a's canon nodes]` (the ask) and the goal `b:[c,d]` | S4 ask; goal S1 (Canon) or open S2 |
| `a => b c d`    | `a:[b,c,d]`                             | S1 (Canon) or open S2              |
| `a = a`         | `a:[a]`                                 | S1                                 |
| `a > b`         | `ab:[b]`                                | S2                                 |
| `a = b`         | `a:[b]`                                 | S3                                 |
| `a > a`         | `a:[a]`                                 | S1                                 |
| `a`             | `a|ASK:[]`                              | S4 — the ask                       |
| ask-annotated   | any signature carrying the ASK marker   | S4                                 |

`ASK` is the ASK marker (word-word bit 31): it marks identity, never
content — every atom-space measurement masks it out — and the ask's canon
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

Similarly:

```text
a == b => c d
```

declares goal-targeted training. It creates the queued entry

```text
a:[]
```

— the ask, S4 — and the implied goal

```text
b:[c,d]
```

whose witness is the `=>` block's scaffolding: a Canon when the block
decomposes b exactly, an open covered misfit otherwise. No pair is
created. The goal is a held kline like any other; the engine's own
selection (Definition 22) is unchanged. What the goal adds is a true
answer key: the trainer can compare the trainee's proposals against
b's content rather than against the trainee's own candidate ordering.

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
a:[]
```

which is Unknown and therefore represents an ask. The ask is structural: the Unknown shape is the ask's shape, and no atom, mark, or decree is involved.

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

### Countersigning and ratification

The algebra defines the relevant kline shapes. Ratification is a protocol operation that changes how those klines are treated.

In particular, a trainer's countersign of a proposal creates a standing one-hop correspondence between the proposal's parties. The correspondence then becomes reusable without the acquisition penalty associated with an unratified traversal. The KScript `==` token is not this countersign: it declares the training pair (ask and implied goal) whose proposals the countersign ratifies.

---

# Appendix — Worked example: “Mary had”

The §9 example works an under+over relationship: the question holds content the answer lacks (w, d) and lacks content the answer holds (a, l). This example works the pure overfit relationship — the fragment — where the underfit is empty and the ν_B walk of Definition 15 is the only bridge.

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

No node of ν_A carries an underfit atom: the per-party decomposition yields no A-side slot. No held correspondence adopts the overfit at any node of ν_A — `o:[m]` read reverse swaps m for o and worsens the mismatch, `all:[o]` and `all:[a,l,l]` occur nowhere in ν_A, and no exactly-witnessed group contracts. Targeting alone is stuck, and under a one-party slot definition the misfit would ask — falsely: `o:[m]` connects the parties.

### Step 1 — Walk from the overfit slot

The overfit lives in nodes of ν_B; the overfit slot is the goal's node all. The walk starts at the slot identity:

```text
all:[all]
```

and crosses two correspondences:

| Edge                | Correspondence     | Direction | Occurring side            |
| ------------------- | ------------------ | --------- | ------------------------- |
| all:[all] → all:[o] | all:[o] Denotation | forward   | head all occurs as a node |
| all:[o] → all:[m]   | o:[m] Denotation   | forward   | head o occurs as a node   |

The second edge arrives: content m overlaps σ(ν_A) — the anchor, discovered on arrival. Neither end refines — the anchor is already a node of ν_A, and the departed end is a node of ν_B, at the goal's own witness resolution. The terminal is the arrival state, two edges crossed.

It is written head-ward as the composed correspondence:

```text
m:[m,all]
```

an Overfit kline with acquisition depth 2: the head is the anchor, whose atoms shared with the goal are m itself; the witness holds those shared atoms together with the B-side end's node covering the overfit.

### Step 2 — Adopt

The composed correspondence is in reach of the next hop — its content covers the node m of ν_A, so the trawl finds it — and its forward replacement adopts the overfit:

```text
m ⇉ [m,all]
```

giving:

```text
[m,all,h]
```

whose value is:

```text
mhall.
```

The relationship is Canon, so the derivation is done.

The goal was never rewritten: ν_B holds the same three nodes at done as at entry. The overfit content remains sealed in the compound node all — nothing in this memory consumes bare a or l, so the anchor walk's granularity is already the goal's own, and the measurement reads Ĥ = 2/3, γ = 2^(-2/3) ≈ 0.63.

The important point is the mirror of §9's: the system reaches the answer through held correspondences, and which end of the misfit the derivation departs from is a fact about the relationship, not a restriction on the mechanism. The ask is reserved for the case where no route exists from either party.
