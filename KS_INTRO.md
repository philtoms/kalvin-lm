# KScript — Technical Reference

## 1. Purpose

KScript is a small DSL for compiling scripted kline constructions into a flat list of KLine entries. These entries are loaded into a Kalvin model via the rationalisation pipeline to train the system to recognise and respond to structured input.

KScript targets the Mod32 tokenizer and is designed for small scripts — no more than 10 lines.

## 2. Encoding

The Mod32 tokenizer maps characters to bit positions within a `uint64` node. Two encoding modes are determined automatically from input content:

- **Packed** — all-uppercase-alpha strings are OR'd into a single node with bit 0 clear. Order and multiplicity are destroyed. Used for signatures.
- **Literal** — everything else produces one node per character as `(codepoint << 32) | 0xFFFFFFFF`. Order and identity are preserved.

By convention, KScript capitalises and takes the first character of each word to construct packed signatures:

```
"Mary had a little lamb" ~> MHALL
```

Each character maps to a single bit position (bit 1 through bit 31). The packed token for `MHALL` is the OR of each character's bit:

```
M  = 0b0010 0000 0000 0000 0000
H  = 0b0000 1000 0000 0000 0000
A  = 0b0000 0000 0000 0000 0010
L  = 0b0000 0001 0000 0000 0000
L  = 0b0000 0001 0000 0000 0000
OR = 0b0010 1001 0000 0000 0010
```

Packed encoding does not preserve order and automatically deduplicates. This is not a KScript concern — it is an additional challenge that Kalvin must handle.

Multi-character signatures (MCS) trigger automatic expansion during compilation: one unsigned entry per component character, one canonisation entry mapping the compound to its ordered components, and one unsigned compound entry. This allows the decompiler to recover the original name from packed data.

## 3. Compiled Output

A KScript contains one or more expectations. Each expectation compiles to one or more KLines of the fundamental shape `{signature: [nodes]}`.

The simplest expectation:

```
Q = V
```

Compiles to:

```python
KLine(signature=encode("Q"), nodes=[encode("V")])   # expectation
KLine(signature=encode("Q"), nodes=[])               # unsigned query
KLine(signature=encode("V"), nodes=[])               # unsigned value
```

The first is the relational entry. The second and third are unsigned identities (S4) that register Q and V in the model.

## 4. Operators

Four operators control compilation semantics and map to significance levels:

| Operator | Name          | Direction      | Significance | Output                    |
| -------- | ------------- | -------------- | ------------ | ------------------------- |
| `=`      | UNDERSIGN     | Unidirectional | S1           | `{Q: [V]}`                |
| `==`     | COUNTERSIGN   | Bidirectional  | S1           | `{Q: [V]}, {V: [Q]}`      |
| `>`      | CONNOTATE_FWD | Unidirectional | S3           | `{Q: [V]}`                |
| `=>`     | CANONIZE_FWD  | Per-item fwd   | S2           | `{Q: [V₁]}, {Q: [V₂]}, …` |

Chains (`<=`, `<`) and subscript blocks produce additional entries of the same fundamental shape. The full grammar supports nested expectations but all constructs ultimately reduce to `{Key: [nodes]}` klines.

## 5. The Model

The Kalvin model is a directed knowledge graph of KLines with three-tier memory:

| Tier  | Purpose              | Bound | Written by    | Lifetime    |
| ----- | -------------------- | ----- | ------------- | ----------- |
| STM   | Transitive grounding | 256   | `add`         | Rolling     |
| Frame | Session knowledge    | None  | `promote`     | Per-session |
| Base  | Long-term knowledge  | None  | External only | Persistent  |

Lookups search STM → Frame → Base. The STM is a fast dual-keyed index (by signature and by nodes signature). The frame holds knowledge in temporal insertion order — newer entries are more relevant. For rudimentary training sessions, no base model is used; always start from a blank slate.

KLines are promoted from STM to the frame only when the agent determines they are significant (S1 or S4). This is the sole learning mechanism: insertion into the frame at S1 constitutes learning.

## 6. Significance

Significance is a computed metric describing how well a candidate KLine answers a query KLine. It is the bitwise inverse of a graph-topological distance:

```
significance = (~distance) & MASK64
```

Higher is better. The natural ordering is **S1 > S2 > S3 > S4** by unsigned integer comparison.

| Level | Meaning                                  | Condition                       |
| ----- | ---------------------------------------- | ------------------------------- |
| S1    | Full understanding                       | All nodes match / countersigned |
| S2    | Partial understanding (some nodes match) | Some overlap, some mismatch     |
| S3    | Associative hint (no nodes match)        | Candidates exist, no overlap    |
| S4    | Complete novelty (no candidates)         | Unsigned or no match            |

S1 and S4 are **significants** — the kline is either confirmed or entirely novel. No further processing needed. S2 and S3 are **rationals** — partial relationships queued for background cogitation.

## 7. Rationalisation

Rationalisation is the six-phase pipeline that determines how a new KLine relates to existing knowledge:

1. **Prepare** — assign signature if missing.
2. **Ground check** — does Q already exist in the model? If yes, emit `ground` event, done.
3. **Assess** — fast-path structural checks: unsigned (S4), all-literal (S1), self-grounded (S1), countersigned (S1).
4. **Retrieve candidates** — find klines with signature AND-overlap. No candidates → S4.
5. **Route each candidate** — node-membership test (no model calls). S1 → promote, done. S2/S3 → queue for cogitation.
6. **Integrate** — add Q to model. Act on best result.

Candidate retrieval uses bitwise AND matching: a kline whose signature shares any set bit with the query's signature is a candidate. This is a necessary but not sufficient pre-filter.

## 8. The Training Protocol

The training harness is **event-driven**. It subscribes to `frame` and `done` events and feeds compiled KLines into `agent.rationalise()` one at a time.

### Walkthrough: `Q = V` (blank-slate model)

Compiled entries: `[{Q: [V]}, {Q: None}, {V: None}]`.

**Step 1.** Rationalise `{Q: None}`.

- Zero nodes → S4. Novel identity registered, promoted to frame.
- Event: `frame S4 {Q: None}`.

**Step 2.** Rationalise `{V: None}`.

- `{V: None}` → S4. Novel identity registered.
- Kalvin relates V to existing knowledge: both Q and V are insignificant (S4). Kalvin remembers the temporal order of input (Q was introduced first, then V) and proposes a relationship between them.
- Events: `frame S4 {V: None}`, `frame S3 {Q: [V]}` (proposal).

**Step 3.** The harness observes the S3 proposal. It ratifies by creating and rationalising the countersignature `{V: [Q]}`.

- Rationalise `{V: [Q]}`.
- The model contains `{Q: [V]}` (from the S3 proposal). Its signature is Q, its sole node is V. `make_signature({V: [Q]}.nodes)` = Q. A kline with signature Q and sole node V exists → **countersigned. S1.**
- This S1 on the countersignature bumps `{Q: [V]}` from S3 to S1.
- Events: `frame S1 {V: [Q]}`, `frame S1 {Q: [V]}`.

The relationship is now grounded. Future rationalisations of `{Q: [V]}` will hit the ground check (Phase 2) and emit a `ground` event.

### Proposals and Ratification

Kalvin doesn't wait to be told everything. When it encounters new information, it **proposes** relationships to existing knowledge and publishes them as S3 events. S3 proposals are Kalvin asking: _"are these things related?"_

Ratification is the mechanism by which the harness teaches Kalvin. The harness ratifies a proposal by creating and rationalising the **countersignature** — the mirror entry. This transforms the S3 proposal into S1 grounded understanding.

Without ratification, proposals remain speculative. By omission, the harness communicates: _"no, that guess is wrong."_ No countersignature → no S1 → the proposal decays as STM entries are evicted.

### The Harness Decision Loop

```
for each compiled entry:
    agent.rationalise(entry)
    wait for events

    for each event:
        if S4:
            # novel identity. registered.
        if S1:
            # confirmed understanding. grounded.
        if S3 (proposal):
            # kalvin is guessing at a relationship.
            # ratify by creating and rationalising
            # the countersignature {V: [Q]}.
            # this bumps the proposal to S1.
```

The harness infers what is needed from the event stream alone. No model inspection is required.

## 9. Key Constraints

- **Scale.** KScript/Mod32 is practical for scripts ≤10 lines. Bitmask collisions increase with vocabulary size.
- **Temporal ordering.** The frame records klines in insertion order. Newer = more relevant. This is the basis for learning.
- **No base model.** Rudimentary training sessions start from a blank slate.
- **No invention.** Every kline produced during cogitation must reference signatures already in the model. When the model cannot resolve, no proposal is emitted — the harness infers that scaffolding is needed from the absence of a `frame` event.
- **Ratification is countersignature.** The sole mechanism for turning a proposal into grounded knowledge is the creation and rationalisation of the mirror entry. There is no other path to S1 for composed structures.
