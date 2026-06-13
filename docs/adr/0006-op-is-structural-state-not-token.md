# `op` records the structural state, not the written token

**Status:** accepted. Supersedes [ADR-0005 — Undersign is S3](./0005-undersign-is-s3.md).

A compiled entry's `op` field (carried in `KDbg`) records the entry's **structural state** — the resulting relationship between signature and nodes — not the relational token that was written in source. The five states are a closed set: **COUNTERSIGNED, UNDERSIGNED, CONNOTED, CANONIZED, IDENTITY**. The first four take past-participle forms because a written token produced them (`==`, `=`, `>`, `=>`); IDENTITY has no suffix because no token acts on it. The proof that `op` is a state and not a token is self-identity: `A = A` writes the UNDERSIGN token but compiles to an IDENTITY kline (`{A: []}`). If `op` were provenance it would record UNDERSIGN; it records IDENTITY, the resulting structure.

**Why this supersedes ADR-0005:** ADR-0005 reclassified undersign as S3 and framed the decision as a single-operator fix. The real decision is broader and deeper: every `op` value is a structural state, not a token, and the four token-derived states are merely *spelled* as token-names in the current code because the token→state mapping is 1:1 for them. ADR-0005's stated consequence — "eliminates the need for `sig_level` as a first-class KLine slot" — was never fully realised: `sig_level()` survived as a display/test helper (`src/kalvin/kline.py`) keyed on the structural state. This ADR absorbs ADR-0005's intent (undersign declares S3) into the general model.

**Consequences:**

1. **Code rename.** The `op` string values change from token-names to state-names: `"COUNTERSIGN"` → `"COUNTERSIGNED"`, `"UNDERSIGN"` → `"UNDERSIGNED"`, `"CONNOTATE"` → `"CONNOTED"`, `"CANONIZE"` → `"CANONIZED"`. `IDENTITY` is unchanged. Touches `_SIG_LEVELS`, `_op_to_str`, `_OP_SYMBOLS`, every `_emit_entry(..., op)` call, `kline_display`'s symbol map, and all test assertions reading `dbg.op`. `TokenType` and the lexer/parser are unaffected — they hold tokens, not states.

2. **Declared vs actual significance stays separate.** The declared significance (compile-time, `_SIG_LEVELS[state]`) is derived from the structural state. The actual significance (runtime, `agent.py`) is derived from structure and model state — it never reads `op`. The two can differ: an UNDERSIGNED entry declares S3 but resolves to S1 at runtime when both sides are grounded.

3. **`sig_level()` retained as a display/test helper.** Despite ADR-0005's claim, the helper survived because display and tests need to read declared significance from `dbg.op`. Keeping it as a helper (not a KLine slot) is the correct resolution.
