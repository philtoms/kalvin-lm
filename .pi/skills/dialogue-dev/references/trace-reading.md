# Reading a harness trace

The harness prints a per-step trace, then a summary tail. It is a faithful,
non-judging presenter — every line is a fact about what the engine did, not a
verdict. This is the vocabulary for reading it.

## Per-step layout

```
── Step 1  in  S1  MHALL:[SVO] ──
  T01  feed    MHALL:[SVO] S1
        asks     SVO:[]
        asks     MHALL:[]
  T02  feed    SVO:[MHALL] S1 + SVO:[Subject, Verb, Object] S2 + MHALL:[Mary, had, a, little, lamb] S2
```

- **`── Step N  in  <band>  <kline> ──`** — the task for this step. `<band>` is the
  compiled **target** significance (what the script asserts the entry
  should become). `<kline>` is `signature:[nodes]` in scripted labels
  (hex when no label is known).
- **`<turn>  feed  <kline> <band>`** — the harness fed these compiled entries to the
  engine. The `<turn>` is incremented every time the harness feeds another entry. It
  is reset on the next step.
- **`asks <kline>`** — The ask here is the engine's request for klines with this
  signature. The engine always asks about entries it has not previously seen.
- **`propose  <kline> <band> <significance`** — one engine emission this step. `<band>` here
  is the engine's _actual_ output band (S1 ground-and-cascade, S2 propose,
  S3 denote, S4 ask). `(none)` = the engine had nothing to emit.
- **`grounds <kline>`** — an S1 observation: a kline the engine grounded
  internally this step (added to its grounded model). Distinct from `out`:
  groundings are the engine's private S1 state; `out` is what it would say.

## Bands (quick reference)

- **S1** — fully accounted for (identity, canon, or grounded relationship).
- **S2** — relates but diverges; a multi-node misfit proposal.
- **S3** — connects indirectly; a single-node relationship (connote/denote).
- **S4** — the empty ask `{X:[]}`: "I don't know X."

See CONTEXT.md §Significance (Rational) for the precise definitions.

## Annotation headers

```
[Mary had a little lamb]
```

A `[annotation]` line appears when the entry's owning scope's annotation
changes. The annotation is the trainer-facing rationale (the parenthetical
prose in the `.ks`). Use it to form an exploratory expectation. For example:
The annotation for WDMH is (What did mary have) — You know that Mary had a
little lamb, so a useful expectation might be "Mary had a little lamb".
But remember that the engine proposes klines, not prose. Always check the nodes.

Note: a single annotation can appear non-contiguously — source entries and
MTS entries for the same block are split by the encoder's source-before-MTS
partition (see behaviour-notes §Compilation).

## The summary tail

```
── summary ──
  steps: 25
  batch by band: S1=3  S2=0  S3=0  S4=4
  grounded:
      ...
  work_list (pending at end of run):
      ...
```

- **`batch by band`** — counts of engine _emissions_ across the run. A
  factual histogram, not a score.
- **`grounded`** — the final grounded model: everything the engine ended up knowing
  (identities, canons, relationships), in scripted labels.
- **`work_list (pending at end of run)`** — what the engine was still working on when
  turns ran out. **This is the diagnostic.** Distinguish:
  - _Genuine residue_ — signatures the script never makes groundable
    (an unbound `L`; a denotes target like `a:[Det]` where `a` is never an
    identity). Not a bug.
  - _Stalled klines_ — something that should have grounded but the engine
    had no path (the historical fast-route drop; relationships that couldn't
    ground). **This is the work.**

## Worked example 1 — a step that discovers, asks, and is answered

```
[Subject]
── Step 3  in  S3  Mary:[Subject] ──
  offer   Subject:[Subject]
  out   (none)
  ground  Subject:[Subject]
```

Reading: the script feeds `Mary:[Subject]` (target S3, a relationship).
The engine routes it (S3 → slow), unpacks the unseen node `Subject` as an
S4 ask. The harness sees the `{Subject:[]}` ask, finds `Subject:[Subject]`
in the compiled script, and offers it (`offer` line). The engine grounds
it (`ground Subject:[Subject]`). No `out` — the engine had no _proposal_
this step, but its grounded model grew.

The **expectation** here is concrete and checkable: the annotation "Subject"
is a fact about Mary's role, so expect `Subject:[Subject]` to appear in the
`ground` line this step. It does. Met. (Contrast: a step that vanishes —
`out (none)` _and_ nothing in `ground` or `work_list` — is the signal to
suspect the engine dropped the kline at routing. The historical unseen-
signature canon bug was this shape.)

## Worked example 2 — the answering proposal (read this before judging)

A question-shaped entry (`WDMH`, annotation "what did Mary have") is the
class that catches the unwary. There is nothing in the design that locks
this as a question but you know that it is because of the annotation.

**Decide on an exploratory expectation, before reading the trace.**
Decide on a useful expectation and write it down so that you can compare
later. You are looking for a match that you can explore. It might not be
exact. If you wrote down "MHALL:[Mary, had, a, little, lamb]" but the only
match you find is "WDMH:[Mary, had, a, little, lamb]" then report it. It is
useful analysis and in this case it shows that the engine is on the right track.

**Find where that kline actually appears** — it may not be the step you
expect:

```
[Mary had a little lamb]
── Step 16  in  S2  MHALL:[Mary, had, a, little, lamb] ──
  ...
  out   S2  WDMH:[Mary, had, a, little, lamb]
  ground  MHALL:[Mary, had, a, little, lamb]
```

The answer surfaces at step 16 as an `out S2`, the moment `MHALL` grounds
and becomes a graft candidate for the question's misfit. The engine's S2
similar-fit path folded the grounded prime into the question's signature.
That _is_ the engine traversing its grounded model to emit a synthesis.

So the order of judgement is fixed:

1. **Decide what to look for** from the annotation + what has
   grounded, before scanning emissions. Write it down.
2. **Locate** what you wrote down in the trace — the kline itself or a
   near-match (any step, any line kind — `out`, `ground`, or the summary's
   `grounded`).
3. **Report your findings**, so that you can formulate your next step.
   Did the engine meet your expectations? How was it out? What went wrong?
