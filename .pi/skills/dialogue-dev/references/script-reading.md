# Reading and writing a .ks script

A `.ks` file is a structured document with well-defined syntax: signatures,
relational tokens, and optional semantic annotations. The compiler turns
it into klines with **Target Significance** labels — the answers the
script asserts the trainee should learn to derive. Kalvin's engine never sees the
letters' meanings; those are not encoded into klines. The semantics exist
for the **agent running the session**, so it can tell whether Kalvin is
learning anything useful — and Kalvin's proposals must decode back into
meaningful statements, because Kalvin training is in dialogue form.

Precise definitions live in CONTEXT.md §KScript (Token ID, Target
Significance, Relational Tokens, MTS, Word Binding, Annotation).

## The two layers

- **Structure** — signatures and relational tokens. The actual script
  content fed to Kalvin's engine. A script with structure and no annotations is valid;
  the letters are then just labels to Kalvin and opaque to the agent.
- **Annotations** — parenthetical prose, e.g. `(Mary had a little lamb)`.
  The script's semantic layer. Two jobs: mechanically they resolve Word
  Bindings for the compiler; semantically they instruct the agent, in
  high-level language, what the surrounding structure means. See
  CONTEXT.md §Annotation. They are not for Kalvin — Kalvin knows nothing of them.

## The relational tokens

| Token  | Name          | Emits                   | Structural meaning                                     |
| ------ | ------------- | ----------------------- | ------------------------------------------------------ |
| `==`   | COUNTERSIGNS  | `{A:[B]}` + `{B:[A]}`   | a pairwise denotes; two traversable structures         |
| `=>`   | CANONICALISES | `{A: [B, C, D]}`        | A canonicalises its block operands into a single kline |
| `>`    | CONNOTES      | `{AB: [B]}`             | A connotes B. Becomes IDENTITY when same token.        |
| `<`    | RCONNOTES     | `{BA: [A]}`             | B connotes A, reversed reading (≡ `B > A`).            |
| `=`    | DENOTES       | `{A: [B]}`              | A denotes B.                                           |
| (none) | UNKNOWN       | `{A: []}` or `{A: [A]}` | ask, unless word-bound → identity                      |

`MHALL == SVO` literally means: there is a structure `MHALL:[SVO]` and
its counterpart `SVO:[MHALL]`, so The engine can traverse from one concept to the
other. An agent reading the script can see a route from MHALL via SVO to
other parts of the script. Whether "MHALL is a kind of SVO" is a useful
gloss is up to the agent.

## MTS and annotations

A multi-character signature like `MHALL` or `DMHAL` is an **MTS**: the
compiler expands it into its constituent single-char identities plus one
MTS relationship. A signature that spells its annotation — `DMHAL` under
`(did Mary have a lamb)` — is easy for the agent to decode; `DGI` under
`(did Gulliver live on an island)` compiles fine but throws away most of
the sentence. There is no spelling rule to violate: annotations aid
understanding, they do not enforce MTS. A mismatch is a missed
opportunity for the agent, never an error.

## Reading a script

1. **Read the annotations** — they say what each block means and what it
   is asking. `(did Mary have a lamb)` over a bare `DMHAL` tells you the
   ask; without the annotation you have opaque structure.
2. **Read the first relation in a block** — it defines Kalvin's task:
   - `MHALL == SVO` A countersign. Kalvin is expected to ground this structure at S1.
   - `SVO => S V O` A canon. Kalvin is expected to verify this structure, and propose
     new structures if there are any gaps (eg `WDMH => M D H`).
   - `S = M` a denotation. Kalvin is expected to ground this structure at S1.
   - `O > ALL` a connotation. Kalvin is expected to ground this structure at S1.
   - `DMHAL` an unknown. Kalvin, recognising this structure is incomeplete, is
     expected to complete it. It is a crude but effective question-answer task.
3. **Read the block shape** — a token with indented lines under it is a
   block; the indented items are called scaffolding and they represent the kinds of
   responses that Kalvin might emit as it cogitates the first line in the block.
4. **Read the order** — priming usually precedes questioning. `DMHAL` arrives
   after MHALL is grounded, so Kalvin is already primed with an appropriate answer.
   You know this, but does Kalvin? That's the task.

## Reading a run (the decode loop)

The harness decodes kalvins output into words for you. After a run, read the
decoded trace, grounded, and work_list as sentences, and judge them:

- Does Kalvin's proposal decode back into a meaningful statement? If it's
  semantic gibberish, say so — do not charitably read right-nodes-
  wrong-prose as "on the right track".
- A near-match (right nodes, differently arranged signature) may be a
  finding that Kalvin is on the right track — but check the decoded prose
  before granting it.
- Un-decodable output is itself a clue that Kalvin has gone astray.

The judgement is the agent's job (the harness never judges). Strengthen
or refute the expectation formed from the script and its annotations; report
what you found.

## Authoring a new script

Scripts live in `data/scripts/`. Author a `.ks` to test an engine
theory or shake up a settled engine — never to work around an engine bug
(see the skill's "engine first" discipline).

1. **Write the annotations first.** Each block's parenthetical says what
   it means in high-level language — this is what you (the agent) will
   decode against when reading the run. It can be anything, but keep it
   simple, and keep it consistent. Bad annotations don't show up as compiler
   errors, the just silently mislead you.
2. **Choose tokens that express the teaching.** `==` for mutual
   traversability, `=>` to break down a signtature into more detail, `>` and
   `=` for connotation/denotation edges. Don't be afraid to use multiple connotations
   if you think that steps from `M(ary) > G(irl) > O(bject)` offerts better teaching
   opportunities. But if you do that, make sure you exercise them further down the script.
   Finally, withhold bindings when you want an ask instead of an identity:
   `(Is Mary a girl)IMAG` (see **Reading a script point 2**).
3. **Make signatures easy to decode.** Spell out words annotation where it
   helps `D(et)`, but only if signature is not already annotated, or if you
   need to override it. Inline signatures bind tightly.
4. **Compile-check it.** Always run the harness (or the compiler)
   against the new `.ks` and fix errors before reading anything into
   the trace.

Minimal shape of a script:

```
(Gulliver found an island)
GFAI == SVO =>
   S(ubject) = G
   V(erb) = F
   O(bject) = AI

(an island is territory)
AI > T

(did Gulliver find an island)
DGFAI
```

The first block lays out the first ask: Ground `GFAI == SVO`. The scaffolding
is there to verify that Kalvin is grounding the ask correctly but it has a deliberate
gap. Kalvin won't be able to understand `AI` from this block alone. It needs to wait.
The next block fills the gap. Expect Kalvin to ground this, and then revisit and
complete the first block. The last block is a question. Kalvin needs to answer this
using only the structures it has already grounded - what it has learned so far.
