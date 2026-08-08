# Reading a harness trace

The harness prints a per-step trace, then a summary tail. It is a faithful,
non-judging presenter — every line is a fact about what the engine did, not a
verdict. This is the vocabulary for reading it.

## Per-step layout

```
── Step 1  in  S1  MHALL:[SVO] ──
  out   S4  MHALL:[]
  out   S4  SVO:[]
```

- **`── Step N ──`** — one compiled curriculum entry fed to the engine.
- **`in  <band>  <kline>`** — the entry fed this step. `<band>` is the
  compiled **target** significance (what the curriculum asserts the entry
  should become). `<kline>` is `signature:[nodes]` in scripted labels
  (hex when no label is known).
- **`offer  <kline>`** — the harness fed this S1 identity `X:[X]` to answer
  an S4 ask `{X:[]}` the engine emitted this step (the curriculum defines
  it; the harness answers unsupervised, per-step dedup). No judgement.
- **`out  <band>  <kline>`** — one engine emission this step. `<band>` here
  is the engine's *actual* output band (S1 ground-and-cascade, S2 propose,
  S3 connote, S4 ask). `(none)` = the engine had nothing to emit.
- **`ground  <kline>`** — an S1 observation: a kline the engine grounded
  internally this step (added to its grounded model). Distinct from `out`:
  groundings are K's private S1 state; `out` is what it would say.

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
prose in the `.ks`). Use it to form the trainer expectation: a fact
annotation → expect the fact to ground; a question annotation → expect an
answer. The harness prints it; the trainer (you) judges against it.

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

- **`batch by band`** — counts of engine *emissions* across the run. A
  factual histogram, not a score.
- **`grounded`** — the final grounded model: everything K ended up knowing
  (identities, canons, relationships), in scripted labels.
- **`work_list (pending at end of run)`** — what K was still working on when
  turns ran out. **This is the diagnostic.** Distinguish:
  - *Genuine residue* — signatures the curriculum never makes groundable
    (an unbound `L`; a connotes target like `a:[Det]` where `a` is never an
    identity). Not a bug.
  - *Stalled klines* — something that should have grounded but the engine
    had no path (the historical fast-route drop; relationships that couldn't
    ground). **This is the work.**

## Worked example: a step that discovers, asks, and is answered

```
[Subject]
── Step 3  in  S3  Mary:[Subject] ──
  offer   Subject:[Subject]
  out   (none)
  ground  Subject:[Subject]
```

Reading: the curriculum feeds `Mary:[Subject]` (target S3, a relationship).
The engine routes it (S3 → slow), unpacks the unseen node `Subject` as an
S4 ask. The harness sees the `{Subject:[]}` ask, finds `Subject:[Subject]`
in the compiled curriculum, and offers it (`offer` line). The engine grounds
it (`ground Subject:[Subject]`). No `out` — the engine had no *proposal*
this step, but its grounded model grew. The trainer expectation ("this is a
fact about Mary's role → expect it to ground") is met.

Contrast with a step that vanishes (the historical WDMH bug): an unseen-
signature canon that produced `out (none)` and added nothing to grounded or
work_list — the signature was dropped at routing. That silence is the signal
to suspect the engine.
