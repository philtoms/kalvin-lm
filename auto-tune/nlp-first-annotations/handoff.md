# Handoff: NLP-first Curriculum Annotations

## What was done
1. **Fixed CLISupervisor deadlock** (committed to main) — restored pre-loop command poll removed by commit 794b49e
2. **Verified all 7 curricula compile/train with NLPTokenizer** — no changes needed for bare sigs
3. **Wrote spec** `specs/nlp-curriculum-compat.md` and **21 tests** in `tests/test_nlp_curriculum_compat.py`
4. **Auto-tune session torn down** — work is on main directly

## What's next (the new goal)
The user wants all curricula and scaffolding scripts rewritten to be **NLP-first**: every signature gets a parenthetical comment. Patterns:
- **Block comment before multi-char sigs**: `(Alpha Beta Gamma) ABG => ...`
- **Inline on single-char sigs**: `A(lpha) = B(eta)`
- Already-examples in `data/example.ks`, `data/scripts/mw.ks`, `data/chats/example.ks`

## Files to modify

### Curricula (7 files, all in `curricula/`)
Each curriculum needs NLP annotations added to every KScript block:

| File | Content summary | Annotation strategy |
|---|---|---|
| `first-steps.md` | M, H, M == H | `M(ark)`, `H(alo)`, `M(ark) == H(alo)` |
| `first-steps-s2.md` | M, H, M == H, A, MH => H A | inline on all sigs |
| `mhall-svo-single.md` | MHALL == SVO => subscript | block `(Mary Had A Little Lamb)` + inline `S(ubject)` etc. |
| `mhall-svo-equivalence.md` | 5 lessons building MHALL==SVO | block + inline throughout |
| `cascade-pressure.md` | A-J identities, compound canonizes | inline or block for each lesson |
| `conflict-drill.md` | A-D identities, compound canonizes | inline throughout |
| `s3-auto-countersign.md` | M, H, A, P, X, connotation, auto-countersign | inline throughout |

### Standalone scripts (already partially annotated)
| File | Status |
|---|---|
| `data/example.ks` | Already NLP-annotated ✅ |
| `data/scripts/mw.ks` | Already NLP-annotated ✅ |
| `data/chats/example.ks` | Already NLP-annotated ✅ |

## Key reference: binding resolver semantics
- **Block comment** `(W1 W2 W3)` before a multi-char sig: positional zip binding, e.g. `(Mary Had A Little Lamb)` before `MHALL` → M=Mary, H=Had, A=A, L=Little, L=Lamb
- **Inline comment** `S(ubject)`: binds that specific sig char to the word
- **Scope**: block comments are scoped; inline comments on the right side of `=` also bind node sigs
- The existing `data/example.ks` shows the canonical pattern

## Key files in the codebase
- `src/kscript/binding_resolver.py` — walks AST, builds NLPSymbolTable from comments
- `src/kscript/ast_emitter.py` — resolves sig chars to words using symbol table
- `src/kscript/compiler.py` — orchestrates: binding resolver → emitter → encoder
- `src/kalvin/nlp_tokenizer.py` — NLPTokenizer producing 64-bit NLP-BPE nodes

## Commits on main so far
```
cffe645 test: add NLP tokenizer curriculum compatibility tests (21 tests)
47b50ce docs: add NLP tokenizer curriculum compatibility spec
386c3d7 fix(auto-tune): restore pre-loop command poll in CLISupervisor
```
