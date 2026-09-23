# Reading and writing a .ks script

Kalvin is trained through three layers, and a `.ks` script works all three
at once. At the top, **linguistic semantics** — meaningful sentences, and
only meaningful sentences. In the middle, **scaffolding** — the script's
syntax, which binds semantics to structure. At the bottom, **structure** —
the klines the compiler emits, the actual structure used to communicate
with Kalvin. The compiler labels klines with **Target Significance** — the
answers the script asserts the trainee should learn to derive — but the
rationaliser never sees the letters' meanings: those are not encoded into
klines. The semantics exist for the **agent running the session**, so it
can tell whether Kalvin is learning anything useful — and Kalvin's
proposals must decode back into meaningful statements, because Kalvin
training is in dialogue form.

Precise definitions live in CONTEXT.md §KScript (Token ID, Target
Significance, Relational Tokens, MTS, Word Binding, Annotation).

## The three layers

- **Linguistic semantics** — parenthetical prose, e.g. `(Mary had a little
lamb)`. The high layer, and it belongs to the agent alone: Kalvin knows
  nothing of it. Two jobs: mechanically the prose resolves Word Bindings
  for the compiler; semantically it instructs the agent, in high-level
  language, what the surrounding structure means. See CONTEXT.md
  §Annotation. Only meaningful sentences please — this is the layer that
  carries what the script is actually about.
- **Scaffolding** — signatures, relational tokens, block shape: the
  script's syntax, which binds semantics to structure. The trainer's
  layer. A script with scaffolding and no annotations is valid; the
  letters are then just labels to Kalvin and opaque to the agent.
- **Structure** — the klines the compiler emits: the actual content fed to
  Kalvin's rationaliser. The only layer that is ever exchanged with the trainee.

The relational tokens — canonicalises, connotes, denotes — suggest nuanced
relationships through structure, and a script is designed to illustrate
Kalvin's rationalisation structurally. But the terms do not bind Kalvin to
a prescribed structure. They describe an idealised arrangement — the
scaffolding — a trainer can use to help Kalvin build internal pathways to
significant rationalisation. What is being trained on (the scaffolding,
the trainer's ideal) and what is expected of the trainee (pathways it
builds itself) are distinct: the scaffold is an aid, never an obligation.
How a trainer deploys scaffolds across a session or a curriculum is
outside the scope of this document.

## The relational tokens

| Token  | Name          | Emits                            | Structural meaning                                                                       |
| ------ | ------------- | -------------------------------- | ---------------------------------------------------------------------------------------- |
| `==`   | COUNTERSIGNS  | `{A\|ASK:[]}` + goal `{B:[C,D]}` | goal-targeted training: A is the queued ask (S4), B the implied goal from the next block |
| `=>`   | CANONICALISES | `{A: [B, C, D]}`                 | A canonicalises its block operands into a single kline                                   |
| `>`    | CONNOTES      | `{AB: [B]}`                      | A connotes B. Becomes IDENTITY when same token.                                          |
| `<`    | RCONNOTES     | `{BA: [A]}`                      | B connotes A, reversed reading (≡ `B > A`).                                              |
| `=`    | DENOTES       | `{A: [B]}`                       | A denotes B.                                                                             |
| (none) | ASK           | `{A\|ASK: [A]}`                  | ask: a CANON or an IDENTITY flagged as an ASK                                            |

`MHALL == SVO => ...` literally means: queue `MHALL:[]` — the ask at
S4 — with `SVO:[block operands]` as the implied goal. The rationaliser still
selects its own goals (the goal is just another held kline to it); the
goal exists so the agent running the session can grade Kalvin's proposals
against the true answer. Goals don't always have to be Canons:

- `MHALL == N(ursery-rhyme) > S(tory) # goal is CONOTATION`
- `MHALL == WDMH = O < W # goal is CONNOTATED DENOTATION`

## The scaffold grades

A script typically opens with a COUNTERSIGNED kline — a two-part
structure. The lefthand side is a query, the entry kline queued at S4:
unknown, an ask. The righthand side is a scaffolded goal — and it is
optional. The grade of scaffold supplied sets what is expected of the
trainee and how the trainer judges the response.

**No scaffold — the open ask:**

```
(what is the meaning of life)WITMOL
```

The trainee decides the goal. The trainer considers the response from a
narrative perspective, judging its quality rather than its outright
accuracy — and would be very content should the trainee respond with "the
meaning of life is 42".

**An inline answer key — a two-part scaffold without block structure:**

```
(1+2) == 3
(Is the sun hot) == yes
```

The key exists only at the scaffolding layer: the ask is queued, no goal
kline is emitted, and nothing structural reaches Kalvin. The key is for
the agent to read and grade against. One caution: an inline key binds
words too, and the nearest binding wins — check the compiled ask's nodes
(the nodes are what candidate selection reads) before reading anything
into the trace.

**A block goal — the scaffolded goal as structure:**

```
MHALL == SVO =>
   Subject < M
   Verb < H
   Object < Query < ALL =>
     A = Det
     L = Mod
     L = O
```

The goal compiles to a held kline — the harness's answer key (see the
COUNTERSIGNS row above). The rationaliser still selects its own goals; the
scaffold arranges the true answer for grading, it does not prescribe the
route.

## Compounds and annotations

A multi-character signature like `MHALL` or `DMHAL` is a **Compound**: the
compiler expands it into its constituent single-char identities plus one
MTS relationship. A signature that spells its annotation — `DMHAL` under
`(did Mary have a lamb)` — is easy for the agent to decode; `DGI` under
`(did Gulliver live on an island)` compiles fine but throws away most of
the sentence. There is no spelling rule to violate: annotations aid
understanding, they do not enforce MTS. A mismatch is a missed
opportunity for the agent, never an error.

## Reading a script

Read it as a descent through the layers — meaning first, scaffolding
second, structure last.

1. **Read the annotations** — the semantic layer. They say what each block
   means and what it is asking. `(did Mary have a lamb)` over a bare
   `DMHAL` tells you the ask; without the annotation you have opaque
   structure.
2. **Read the first relation in a block** — the scaffolding. It defines
   Kalvin's task. These are the invitations the scaffolding extends — what
   the trainer hopes the structure will bring about — not obligations on
   the rationaliser (see **The three layers**):
   - `MHALL == SVO => ...` goal-targeted training. `MHALL:[]` is queued at
     S4; `SVO:[...]` is the implied goal. Kalvin is expected to propose
     toward the goal — grade its proposals against it.
   - `SVO => S V O` A canon. Kalvin is expected to verify this structure, and propose
     new structures if there are any gaps (eg `WDMH => M D H`).
   - `S = M` a denotation. Kalvin is expected to ground this structure at S1.
   - `O > ALL` a connotation. Kalvin is expected to ground this structure at S1.
   - `DMHAL` an unknown. Kalvin, recognising this structure is incomplete, is
     expected to complete it. It is a crude but effective question-answer task.
3. **Read the block shape** — the scaffolding. A token with indented lines
   under it is a block; the indented items are scaffolds and they
   represent the kinds of responses that Kalvin might emit as it
   cogitates the first line in the block.
4. **Read the order** — priming usually precedes questioning. An ask is
   usually prepared for by training Kalvin on scaffolded facts that lead
   to a rational response:

   ```
   (The hitchhikers guide to the galaxy) == Book =>
     Author = Douglas Adams
     Genre = Humor
     quotes =>
        (the Answer to the Ultimate Question of Life, the Universe, and Everything) = 42
   ```

   A trainee primed on the Guide can trawl `42` out of memory when the
   meaning-of-life ask arrives — the open ask judged narratively, but
   prepared for. Likewise `DMHAL` arrives after MHALL is grounded, so
   Kalvin is already primed with an appropriate answer. You know this,
   but does Kalvin? That's the task.

5. **Read what reaches Kalvin** — the structural layer. Klines, and only
   klines. The annotations, the answer keys, the idealised relationships
   the tokens suggest: none of it is exchanged. Everything above the
   bottom layer is for you.

## Reading a run (the decode loop)

The harness decodes kalvins output into words for you. After a run, read
the decoded trace, grounded, and work_list as sentences, and judge them.
The grade you authored at determines the mode you grade at: a scaffolded
ask (inline key or block goal) is judged against its key — structure and
content; an open ask is judged narratively — there is no true answer, the
question is whether the response would content the trainer. In both
modes:

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

Scripts live in `data/scripts/`. Author a `.ks` to test an rationaliser
theory or shake up a settled rationaliser — never to work around an rationaliser bug
(see the skill's "rationaliser first" discipline).

1. **Write the annotations first.** Each block's parenthetical says what it
   means in high-level language — this is what you (the agent) will
   decode against when reading the run. It can be anything, but keep it
   simple, and keep it consistent. Bad annotations don't show up as compiler
   errors, the just silently mislead you.
2. **Choose tokens that express the teaching.** `==` to pair an ask with
   its true goal, `=>` to break down a signature into more detail, `>` and
   `=` for connotation/denotation edges. Don't be afraid to use multiple connotations
   if you think that steps from `M(ary) > G(irl) > O(bject)` offers better teaching
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

The first block lays out the first ask: `GFAI:[]` is queued at S4 with
`SVO:[...]` its implied goal. The scaffolds are there to verify that
Kalvin is grounding the ask correctly but they have a deliberate gap.
Kalvin won't be able to understand `AI` from this block alone. It needs
to wait. The next block fills the gap. Expect Kalvin to ground this, and
then revisit and complete the first block. The last block is a question.
Kalvin needs to answer this using only the structures it has already
grounded - what it has learned so far.
