# Reading and writing a .ks script

A `.ks` file is a structured document with well-defined syntax: signatures,
relational tokens, and optional semantic annotations. The compiler turns
it into klines with **Target Significance** labels — the answers the
script asserts the trainee should learn to derive. K never sees the
letters' meanings; those are not encoded into klines. The semantics exist
for the **agent running the session**, so it can tell whether K is
learning anything useful — and K's proposals must decode back into
meaningful statements, because Kalvin training is in dialogue form.

Precise definitions live in CONTEXT.md §KScript (Token ID, Target
Significance, Relational Tokens, MTS, Word Binding, Annotation).

## The two layers

- **Structure** — signatures and relational tokens. The actual script
  content fed to K. A script with structure and no annotations is valid;
  the letters are then just labels to K and opaque to the agent.
- **Annotations** — parenthetical prose, e.g. `(Mary had a little lamb)`.
  The script's semantic layer. Two jobs: mechanically they resolve Word
  Bindings for the compiler; semantically they instruct the agent, in
  high-level language, what the surrounding structure means. See
  CONTEXT.md §Annotation. They are not for K — K knows nothing of them.

## The relational tokens

| Token  | Name         | Emits                   | Structural meaning                                     |
| ------ | ------------ | ----------------------- | ------------------------------------------------------ |
| `==`   | COUNTERSIGNS | `{A:[B]}` + `{B:[A]}`   | connotes+denotes shorthand; two traversable structures |
| `=>`   | CANONIZES    | `{A: [B, C, D]}`        | A canonizes its block operands into a single kline     |
| `>`    | CONNOTES     | `{A: [B]}`              | A connotes B                                           |
| `=`    | DENOTES      | `{B: [A]}`              | A denotes B (note the reversed emission)               |
| (none) | UNKNOWN      | `{A: []}` or `{A: [A]}` | ask, unless word-bound → identity                      |

`MHALL == SVO` literally means: there is a structure `MHALL:[SVO]` and
its counterpart `SVO:[MHALL]`, so K can traverse from one concept to the
other. An agent reading the script can see a route from MHALL via SVO to
other parts of the script. Whether "MHALL is a kind of SVO" is a useful
gloss is up to the agent.

A bare signature with no token and no binding compiles to the empty
Unknown `{A: []}` — the structural form of an ask. The same signature
with a binding compiles to an Identity `{A: [A]}`: the binding gives it
a decodable value. Binding chooses the structure; the structure
determines the Target Significance.

## MTS and annotations

A multi-character signature like `MHALL` or `DMHAL` is an **MTS**: the
compiler expands it into its constituent single-char identities plus one
MTS relationship. A signature that spells its annotation — `DMHAL` under
`(did Mary have a lamb)` — is easy for the agent to decode; `DGLI` under
`(did Gulliver live on an island)` compiles fine but throws away most of
the sentence. There is no spelling rule to violate: annotations aid
understanding, they do not enforce MTS. A mismatch is a missed
opportunity for the agent, never an error.

## Reading a script

1. **Read the annotations** — they say what each block means and what it
   is asking. `(did Mary have a lamb)` over a bare `DMHAL` tells you the
   ask; without the annotation you have opaque structure.
2. **Read the block shape** — a token with indented lines under it is a
   block; the indented items are its operands (`=>` block) or its
   counterpart with shared operands (`==` block).
3. **Read the order** — priming precedes questioning. `DMHAL` arrives
   after MHALL is grounded, so K is primed to accept proposals that
   suggest it understands whether Mary had a lamb — speaking in pure
   structural terms.
4. **Read the routes** — `==` pairs and denotations are traversal edges.
   Follow them: MHALL via SVO to the S/V/O operands and onward.

## Reading a run (the decode loop)

The harness decodes labels into words for you. After a run, read the
decoded trace, grounded, and work_list as sentences, and judge them:

- Does K's proposal decode back into a meaningful statement? If it's
  semantic gibberish, say so — do not charitably read right-nodes-
  wrong-prose as "on the right track".
- A near-match (right nodes, differently arranged signature) may be a
  finding that K is on the right track — but check the decoded prose
  before granting it.
- Un-decodable output is itself a clue that K has gone astray.

The judgement is the agent's job (the harness never judges). Strengthen
or refute the expectation formed from the annotations; report what you
found.

## Authoring a new script

Curricula live in `data/scripts/`. Author a `.ks` to test an engine
theory or shake up a settled engine — never to work around an engine bug
(see the skill's "engine first" discipline).

1. **Start from the intent.** Name the one engine path or ambiguity the
   script exercises: a single countersign, a lone unseen canon, a denotes
   with no reciprocal, a withheld identity, a deliberately inverted
   ordering.
2. **Write the annotations first.** Each block's parenthetical says what
   it means in high-level language — this is what you (the agent) will
   decode against when reading the run.
3. **Choose tokens that express the teaching.** `==` for mutual
   traversability, `=>` to aggregate operands, `>` and `=` for
   connotation/denotation edges. Withhold bindings when you want an ask
   instead of an identity.
4. **Make signatures easy to decode.** Spell the annotation where it
   helps; at minimum check each signature against its annotation and
   ask whether an agent reading the run could reconstruct the sentence.
   Mismatches compile fine but waste the annotation.
5. **Use inline annotations only to override a scoped annotation.** A
   scoped annotation binds the characters it can; inline `I(sland)`
   decoration on top of an existing binding is noise. Write `I > T`
   under `(an island is territory)`, and reach for inline only when you
   need to override or supply a binding the scope doesn't give you
   (`I(gloo) > T` binds I to Igloo while T stays bound to territory).
   Inline for genuinely new bindings (e.g. `S(ubject)` in a grammar
   block) is the other legitimate use.
6. **Compile-check it.** Always run the harness (or the compiler)
   against the new `.ks` and fix errors before reading anything into
   the trace.

Minimal shape of a script:

```
(Gulliver found an island)
GFAI == SVO =>
   S(ubject) = G
   V(erb) = F
   O(bject) = I

(an island is territory)
I > T

(did Gulliver find an island)
DGFAI
```

The first block countersigns a true-SVO sentence with its grammar —
G, F, I come from the scoped annotation, so only the grammar labels S,
V, O need inline bindings; the second adds a connotation edge with the
scoped annotation's bindings alone; the last is a bare, word-bound MTS
— the question, answerable only from what the earlier blocks grounded.
