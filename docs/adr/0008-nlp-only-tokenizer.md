# ADR 0008: NLP is the sole tokenizer (Mod removed)

**Date:** 2026-06-16  
**Status:** Accepted. Supersedes the Mod32-fallback and "mixed NLP/Mod32 klines"
clauses of [ADR-0004 — NLP Bindings from Comments](./0004-nlp-bindings-from-comments.md).

> **Scope of this record.** This ADR records a decision and the reasoning behind
> it. It does **not** delete code. The removals it describes are executed by
> follow-on tasks (KB-272 through KB-279); each clause names its owner. A future
> reader of a tree where `src/kalvin/mod_tokenizer.py` is already gone should
> find here *why* it is gone and *why* the NLP data dependency is
> non-negotiable.

## Context

A grilling session (questions Q1–Q8) surfaced four connected defects in the
tokenizer layer. Each was survivable in isolation; together they made the layer
incoherent.

1. **Two coexisting tokenizers with no real seam.** `KTokenizer`
   (`src/kalvin/abstract.py`) nominally has three adapters — `Tokenizer` (BPE),
   `ModTokenizer`, `NLPTokenizer` — but production collapses to two regimes:
   the NLP-BPE path and the Mod bit-packing path. `_default_tokenizer()`
   (`src/kalvin/agent.py`) tries `NLPTokenizer.from_files()` and silently falls
   back to `Mod32Tokenizer` on any failure. The "fallback" is a *different
   encoding scheme*, so a data-less environment quietly produces Mod32 klines
   that are bitwise incompatible with NLP klines. There is no graceful
   degradation here — only a silent regime switch.

2. **A fictional literal mechanism.** `docs/tokenizer-significance.md` documents
   a literal/non-literal distinction that supposedly lives in the significance
   engine: literal nodes contribute "bit 0 only" to signatures, a standalone
   `is_literal()` function tests a `0xFFFFFFFF` literal mask, and "the signature
   builder calls `is_literal()` internally." None of this is true.
   `make_signature` (`src/kalvin/signature.py`) is a plain OR-reduce of raw node
   values — its own docstring states "no masking, no branching, no special
   cases." There is no `is_literal()` function anywhere under `src/`. The
   `0xFFFFFFFF` literal mask exists only *inside the Mod tokenizer's own
   encode/decode* (`src/kalvin/mod_tokenizer.py`) and the NLP decode
   legacy-compat branch (`src/kalvin/nlp_tokenizer.py:117`); it never
   participated in signature construction or significance routing. The
   significance-engine-level literal mechanism was never implemented — it was
   documentation describing code that does not exist.

3. **A `supports_mcs` property that lied about behavior.** `KTokenizer` defines
   `supports_mcs` (`src/kalvin/abstract.py:42`) and `ModTokenizer` overrides it
   to `True` (`src/kalvin/mod_tokenizer.py:61`), with a docstring claiming it
   gates multi-character signature expansion. It is **never read**. A grep for
   `.supports_mcs` across `src/` returns nothing; the only non-definition
   reference is a *negative* test (`tests/test_ks_compiler.py:552`) asserting
   the compiler does **not** reference it. The property exists, is overridden,
   and describes intended behavior that no code path consumes.

4. **An optional data dependency masking a hard one.** Because the Mod32
   fallback exists, the system appears to run with zero external data. In
   reality NLP is the only tokenizer that produces the grammatically meaningful
   nodes the rest of the system is built around; the "optionality" is an
   accident of the fallback, not a designed property. The fallback lets the
   system start in a state that is technically running but substantively wrong.

## Decision

The grilling session resolved all eight questions in one direction: **NLP
becomes the sole tokenizer, and everything that exists only to keep a second
tokenizer alive is removed.**

1. **NLP is the sole tokenizer.** `Mod32Tokenizer`, `Mod64Tokenizer`, and their
   module `src/kalvin/mod_tokenizer.py` are deleted (KB-277/KB-278). `Tokenizer`
   (raw BPE) remains as the subword foundation NLP is built on; it is not a
   second encoding regime.

2. **The "unbound character" Mod32 fallback is removed.** The spec sections that
   define it — `specs/kscript.md` §10.4 (Unbound Characters) and §11.3 (Mod32
   Fallback for Unbound Characters) — are removed (KB-272/KB-273). A character
   the `BindingScope` cannot resolve is raw-BPE-encoded as itself. There is no
   named "unbound" state and no "mixed NLP/Mod32" regime; a single unresolved
   character is simply a one-character BPE token.

3. **`supports_mcs` is deleted.** It is never consumed by the encoder or any
   other path. It documented behavior that was never wired up. Removing it
   (KB-277/KB-278) changes nothing executable.

4. **"MCS" renames to "MTS" (Multi-Token Signature).** "Multi-Character
   Signature" described the Mod world, where one character maps to one bit. Under
   NLP, a bound word BPE-encodes to multiple *tokens* (e.g. "Mary" → `[mar, y]`),
   and the signature is the OR-reduction of those token nodes. The
   `CONTEXT.md` glossary entry is already renamed to **MTS Entry** (CONTEXT.md,
   ~line 147); the rename is propagated across the remaining codebase and specs
   by KB-272/KB-276.

5. **The `is_literal()` literal mechanism is purged as fictional.** As
   established in the Context: `make_signature` is plain OR-reduce with no
   bit-0 special-casing, no masking, and no `is_literal()` call; no
   `is_literal()` function exists under `src/`. The literal mask `0xFFFFFFFF`
   and the bit-0 literal-content flag are not significance-engine concepts —
   they only ever lived inside the Mod encoder (and the NLP decode
   legacy-compat branch), and they die with Mod. The literal narrative in
   `docs/tokenizer-significance.md` is the documentation of a mechanism that was
   never implemented; that document is deleted by KB-275.

6. **NLP data becomes a mandatory system prerequisite.** `_default_tokenizer()`
   no longer falls back; when `NLPTokenizer.from_files()` fails it raises a
   clear, actionable error directing the user to
   `scripts/rebuild-tokenizer-data.sh` (KB-278). Fresh clones must regenerate the
   `data/tokenizer/` assets (~35 MB, gitignored) before the system will start.
   KScript tests that previously ran on the zero-dependency Mod32 default become
   gated behind the existing `requires_nlp_data` skipif (`tests/conftest.py`),
   which already exists for exactly this purpose.

7. **Old persisted model data is abandoned.** Model files (LTM/Frame) created
   under the Mod or mixed regime contain Mod32-encoded nodes that are
   bitwise-incompatible with NLP nodes. They are not migrated; old data is
   expendable. The NLP decode literal-compat branch
   (`src/kalvin/nlp_tokenizer.py:117`, the `(node & 0xFFFFFFFF) == 0xFFFFFFFF`
   case) is deleted (KB-277/KB-278) — it existed only to decode legacy Mod literal
   nodes, and there is no obligation to keep reading dead encodings.

## Alternatives Considered

### Keep a fallback (degenerate) tokenizer for graceful degradation without data
Introduce a minimal tokenizer whose only job is to let the system boot when NLP
data is absent.

**Rejected.** A fallback tokenizer *is* a second tokenizer. The entire point of
this decision is that there is exactly one encoding regime; any fallback that
emits nodes reintroduces the mixed-encoding hazard this decision removes. The
honest version of "graceful degradation without data" is a clear startup error,
not a silently-different encoding (Decision clause 6).

### Keep "unbound" as a resolution-outcome term
Retain "unbound" as a named category even after the Mod32 fallback is gone —
an unresolved character is still *something*, after all.

**Rejected.** "Unbound" was the *name* of the Mod32-fallback case; it did not
describe a behavioral joint that survives the fallback's removal. Once an
unresolved character is simply a one-character BPE token, naming that state
"unbound" adds a term with no distinguishing behavior behind it. The word goes
away with the regime it described.

### Retire the bit-packing OR-reduction significance engine along with Mod
Since Mod was the original home of bit-packed signatures, treat the
OR-reduction significance engine as Mod-specific and replace it wholesale.

**Rejected.** The significance engine is tokenizer-agnostic by construction:
`make_signature` OR-reduces raw `uint64` node values, and significance routing
operates on node membership and bit overlap, not on what the bits mean (see the
@signature and @significance specs). NLP already conforms to it — its node
format `(nlp_type32 << 32) | bpe_id` participates in the same bitwise OR/AND
algebra, and ADR-0007's normalization work treats the engine as
tokenizer-independent. Only the Mod *adapter* and its dead surface area die; the
engine survives intact.

## Consequences

- **Fresh clones lose zero-dependency startup.** A data-less environment can no
  longer run the system; it must regenerate NLP data first. This is the real
  trade-off being accepted: correctness of the single encoding regime in
  exchange for a mandatory build step. The `requires_nlp_data` skipif already
  exists so the test suite degrades cleanly rather than erroring.

- **ADR-0004 is left untouched as immutable history.** ADR-0004 states that
  "Unbound signatures fall back to Mod32 encoding, producing mixed NLP/Mod32
  klines" and that "Mixed NLP/Mod32 klines within the same graph will require
  rationalisation to handle." Those two clauses are **superseded** by this ADR:
  there is no Mod32 fallback and no mixed regime. ADR-0004 is not edited — ADRs
  are immutable historical records — and the supersession is recorded here.

- **`docs/tokenizer-significance.md` will be deleted entirely (KB-275).** It is
  the document that narrated the fictional `is_literal()` mechanism and framed
  Mod and BPE as peers under one significance contract. With Mod gone and the
  literal mechanism exposed as fictional, it has no remaining true content. Its
  deletion is a follow-on consequence recorded here, not an action of this task.

- **The "tokenizers encode dimensionality, not knowledge" thesis loses its
  home.** That framing lived in `docs/tokenizer-significance.md`. It is still
  true and still load-bearing, so it will be backfilled into `specs/tokenizer.md`
  by KB-272 rather than lost with the document it currently inhabits.

- **The significance engine survives intact.** Distance computation,
  band-anchored normalization (ADR-0007), routing, and classification are all
  tokenizer-agnostic. None of them change. Only the Mod adapter, the
  `supports_mcs` property, the `is_literal()` narrative, the Mod32 fallback, and
  the NLP decode legacy-compat branch are removed.
