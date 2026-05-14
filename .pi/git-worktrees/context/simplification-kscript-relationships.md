## Worktree Context: simplification-kscript-relationships

**Task:** Simplification of KScript relationships: remove all BWD operators, reducing from 6 to 4 operators.

## What's Changing

The KScript operator table in `docs/kscript-intro.md` is being simplified from 6 relationships to 4. The BWD/REV operators are being REMOVED:

**REMOVED:**
- `V < Q` (CONNOTATE_BWD / CONNOTATE_REV) — output was `{Q: [V]}`
- `V₁…Vₙ <= Q` (CANONIZE_BWD / CANONIZE_REV) — output was `{Q: [V₁,…Vₙ]}`

**KEPT (new table):**
| Syntax     │ Name          │ Output               │
│ Q == V     │ COUNTERSIGN   │ {Q: [V]}, {V: [Q]}  │
│ Q = V      │ UNDERSIGN     │ {V: [Q]}             │
│ Q > V      │ CONNOTATE     │ {Q: [V]}             │
│ Q => V₁…Vₙ │ CANONIZE      │ {Q: [V₁,…Vₙ]}        │

## Key Rule

Examples that previously used CONNOTATE_BWD (`<`) should be updated to use UNDERSIGN (`=`) instead. For example:
- `M < S` becomes `M = S` (or `S = M` depending on direction semantics — `=` means `{V: [Q]}` where V is right side, Q is left)

Wait — actually looking at the new table more carefully:
- UNDERSIGN `Q = V` produces `{V: [Q]}` — note: the VALUE gets the query as a node, not the other way around!
- CONNOTATE `Q > V` produces `{Q: [V]}`

So where we had `M < S(ubject)` meaning "M is connotated by S" (producing `{S: [M]}`), we need to express this differently. The closest equivalent with UNDERSIGN would be `S = M` which produces `{M: [S]}`. That's a different relationship.

Actually, re-reading the user's instruction: "Updates to this origin document should flow into remaining documents." The user wants me to update the docs to reflect the new 4-operator model. The source code changes (removing BWD from lexer, parser, compiler, tests) should also be done.

## Files to Update

### Source Code (remove BWD operators):
1. `src/kscript/token.py` — Remove `CANONIZE_BWD` and `CONNOTATE_BWD` from TokenType enum (13 → 11 values)
2. `src/kscript/lexer.py` — Remove `<` and `<=` lexing rules  
3. `src/kscript/parser.py` — Remove BWD chain operators from grammar and parsing
4. `src/kscript/compiler.py` — Remove BWD compilation logic, update sig_levels
5. `src/kscript/ast.py` — Update grammar comments to remove BWD
6. `src/kscript/decompiler.py` — Minor: no BWD-specific logic to remove, but update comments
7. `tests/test_kscript.py` — Remove all BWD-specific tests, update examples

### Documentation (4-operator model):
1. `docs/kscript-intro.md` — ORIGIN: Update table from 6 to 4 operators, rename _FWD suffixes, update examples to use `=` instead of `<`
2. `KS_INTRO.md` — Update operator table (remove `<` and `<=` from chains section)
3. `specs/kscript.md` — Major spec update: remove BWD tokens, grammar, compilation rules
4. `data/kscript.md` — Update grammar, remove BWD operators
5. `data/The KScript language SHALL use the follo.md` — Remove BWD from grammar
6. `plans/implement-kscript.md` — Remove BWD from tasks and test cases
7. `.claude/CLAUDE.md` — Update operator table
8. `docs/learning-and-training.md` — Update any BWD references (check if any)
9. `docs/tokenizer-significance.md` — Check for BWD references (probably none)

## Implementation Notes

- `>` should still be classified as both inline and chain in the grammar, but the parser should only consume it as inline (this note stays but becomes simpler without BWD alternatives)
- The `<` character becomes unused in the language (could raise LexerError, or just be removed from the lexer)
- `<=` also becomes unused
- Rename operators: CONNOTATE_FWD → CONNOTATE, CANONIZE_FWD → CANONIZE (drop _FWD suffix)

Run tests after changes to make sure everything passes.
**Branch:** refactor/remove-bwd-operators
**Status:** running
