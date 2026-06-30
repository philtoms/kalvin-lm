# Implement Dialogue-Driven Training — Plan

## Spec References

- `@specs/dialogue-driven-training.md` — WHAT this plan implements (all DDT-* criteria).
- `@specs/stub-kagent.md` — the deterministic contract double the loop is validated against.
- `@specs/supervisor-decision.md` — escalation path on unresolvable proposals (DDT-20).
- `@specs/harness-server.md` — Trainer participant, bus actions, adapter (unchanged surface).
- `@specs/kscript.md`, `@specs/kline.md`, `@specs/signifier.md` — compile, `is_canon`, `make_signature`.

This plan is an **alternative** to `@plans/implement-rationalise-trainer-significance.md`.
Where they conflict, this plan governs. It does not delete the prior plan; both
coexist until the dialogue-driven model is chosen as the path forward.

## Relationship to Existing Code

The current Trainer (`src/training/trainer/trainer.py`) and Reactor
(`src/training/trainer/reactor.py`) implement the batch-submit + auto-countersign
loop from `@specs/trainer-satisfaction.md`. This plan adds the dialogue-driven
path alongside it; the switchover (replacing the batch loop) is the final phase.
The bus protocol, the KAgent adapter, and the supervisor message surface are
unchanged — only the Trainer's internal loop and its configuration source change.

The dialogue artifact `scripts/dialogue-mhall.json` already exists and is the
canonical example; subword node names in it must be reconciled to real decoded
subwords (see Phase 1, task 1.0).

## Implementation Tasks

### Phase 1 — Decoder (pre-loop configuration)

**1.0 Reconcile the dialogue artifact.** Update `scripts/dialogue-mhall.json`
subword node names to the tokenizer's actual decoded subwords (e.g. `Ma,ry` →
the real `decoded` pair from compiler output). Verify every node list matches a
compiled canon. → supports DDT-1, DDT-4.

**1.1 Dialogue-table loader.** Parse `script` + `turns[]` into a typed
`DialogueTable`/`Turn` structure. Strip `notes` at load. → DDT-1.

**1.2 Symbol resolution against compiled script.** Compile `script` once
(reuse `ks.compiler.compile_source`). Build:
- canon index: node-decoded-label tuple → compiled canon KValue (for DDT-4 retrieval);
- compound/atom label → canonical signature (for relation construction).

→ DDT-3, DDT-4.

**1.3 Single-stage decode.** `decode(table) -> list[DecodedTurn]`. Per turn:
retrieve canon by node-match / resolve atom by label / construct relation from
compound labels; attach significance by band lookup; pass through `actor`, `op`.
→ DDT-2, DDT-3.

**1.4 Pre-decode the whole table** into a flat ordered list at configuration
time. The loop receives only the list. → DDT-2.

### Phase 2 — Stateless Trainer supply rule

**2.1 Held index.** On lesson compile, build `signature → [held klines]` ordered
canon → relation → identity. Withheld = Identities + Canons (incl. subword
canons, unfiltered). → DDT-8, DDT-19.

**2.2 Supply function** (pure): `supply(request_signature, held_index, script) ->
DecodedTurn`. Canon-first with full shadowing (DDT-9); relation only when no
canon shadows it (DDT-11); identity fallback. One kline per call. → DDT-8.

**2.3 Node-terminality significance.** Derive the significance the Trainer
attaches to a supplied entry: terminal nodes → S1, non-terminal → S2. → DDT-12.

**2.4 Stateless response dispatch.** A pure function of `(incoming turn,
held_index, script)`:
- K S4 request → `supply(...)` (DDT-8);
- K S3 proposal → ratify, reciprocal at S1 (DDT-13);
- K S1 terminal → no-op / advance (DDT-14).

No provenance, no open-proposal ledger. → DDT-5.

### Phase 3 — Training loop (dispatch + cursors)

**3.1 Loop driver (dispatch-driven).** The trainer-side loop holds a cursor over
the pre-decoded T-rows; the stub holds its own cursor over the K-rows (see
`@specs/stub-kagent.md`). The loop computes each T via the §2.4 supply function
and validates it against the table; it does not read T turns from the table for
submission. → DDT-22, DDT-23.

**3.2 Greedy cursor dispatch.** Consume a run of same-actor rows whole. The
1:1 canonical table is handled identically to a future multi-row table.
→ DDT-24.

**3.3 Two-sided validation (Model A).** Per turn: validate received K == table K
(stub/table divergence); compute T; validate computed T == table T (supply bug);
submit T. Fail fast with side attribution. → DDT-25.

**3.4 Dual-exhaustion termination.** End the run only when both cursors are empty.
A non-empty stub cursor at trainer exhaustion fails loudly, naming the un-emitted
K-run — this verifies the final K (the closing S1) was consumed and matched, not
trusted. → DDT-26, DDT-27.

**3.5 Opening.** The supply function computes the primary `==` half at S2 via an
opening entry point; the table validates turn 0 like any other T turn. → DDT-7.

**3.6 Synchronous execution** under the stub: submit → drain → validate. No event
loop. → §Training Loop.

### Phase 4 — Satisfaction / learning

**4.1 Declared-band satisfaction.** Mark an entry learned on a Kalvin ground
event whose proposal equals the entry (KLine equality) and whose significance
equals the entry's declared band. `learned` spans all four bands. → DDT-15, DDT-16.

**4.2 Lesson completion** when `learned ⊇ submitted`. Emit `progress:
lesson_complete`. → DDT-17.

**4.3 Run end** on Kalvin's closing S1 countersign of the primary. → DDT-18.

**4.4 Divergences.** Under-reach → not learned, continue. Over-reach → learned +
`progress: over_reach`. (Exercised only by real Kalvin; stub never diverges.)
→ DDT-21.

**4.5 Stall escalation.** Unresolvable request → `@specs/supervisor-decision.md`
path. → DDT-20.

### Phase 5 — Switchover (final)

**5.1** Route the Trainer participant through the dialogue-driven loop for
dialogue-table lessons; retain the legacy batch loop for legacy curricula until
deprecated. Decide retirement of `@specs/trainer-satisfaction.md` and its plan
as a separate step.

## Test Mapping

| Spec ID | Test | Status |
|---------|------|--------|
| DDT-1 | loader parses table/turn fields; notes stripped | todo |
| DDT-2 | decode() returns flat ordered list of DecodedTurn | todo |
| DDT-3 | single-stage: kline from script + sig lookup + actor/op pass-through | todo |
| DDT-4 | canon retrieved by node-list match | todo |
| DDT-5 | supply is pure: same inputs → same output; no state mutated across calls | todo |
| DDT-6 | significance dispatch: S2/S3→respond, S4→supply, S1→advance | todo |
| DDT-7 | trainer opens with primary half at S2; opening computed, table-validated | todo |
| DDT-8 | request for X supplies held klines in canon→relation→identity order | todo |
| DDT-9 | relation behind a canon is never supplied (shadowed) | todo |
| DDT-10 | shadowed relation is K-discovered (S3) then T-ratified (S1) | todo |
| DDT-11 | relation with no canon is supplied on request | todo |
| DDT-12 | terminal nodes→S1, non-terminal→S2 on supplied entries | todo |
| DDT-13 | K S3 proposal → T reciprocal at S1 | todo |
| DDT-14 | K S1 terminal → trainer no-op, advance | todo |
| DDT-15 | entry learned on ground at declared band (kline+sig equal) | todo |
| DDT-16 | S4 satisfies identity; S2 canon; S3 relation; S1 primary | todo |
| DDT-17 | lesson completes when learned ⊇ submitted (all bands) | todo |
| DDT-18 | run ends on dual cursor exhaustion; closing S1 verified not trusted | todo |
| DDT-19 | subword canons withheld, ratifiable, learned at S2 | todo |
| DDT-20 | unresolvable request escalates per supervisor-decision | todo |
| DDT-21 | over-reach learned + over_reach signal; under-reach not learned | todo |
| DDT-22 | loop is dispatch-driven (T computed, table validates), not replay | todo |
| DDT-23 | self-cursored actors (trainer over T-rows, stub over K-rows); no kline matching | todo |
| DDT-24 | greedy per-actor dispatch; multi-row runs handled identically to 1:1 | todo |
| DDT-25 | two-sided validation per turn (K==table, T==table) with side attribution | todo |
| DDT-26 | truncated table fails (non-empty stub cursor at trainer exhaustion) | todo |
| DDT-27 | closing S1 verified (final K-run consumed + matched), not semantically detected | todo |
| DDT-28 | annotation-only turns dropped at decode | todo |

Canonical end-to-end test: the full "Mary had a little lamb" dialogue
(`scripts/dialogue-mhall.json`) runs against `StubKAgent` and every compiled
entry is learned at its declared band, ending on the primary's S1 countersign.

## Design Decisions

**D1 — Trainer is stateless.** Resolved in grill: the Trainer is a pure function
of `(incoming turn, compiled script)`. Open-proposal tracking is Kalvin's state,
not the Trainer's. *Rationale:* concentrates the new intelligence in a
deterministic, testable function; matches the principle that only participants
(supervisors, Kalvin) carry temporal state and cogitate. Rejected alternative: a
provenance ledger of open proposals (unnecessary — Kalvin owns that state).

**D2 — Canon-first with full shadowing.** Resolved in grill: a relation whose
signature carries a canon is never T-supplied; it is K-discovered then
T-ratified. *Rationale:* this is how "make Kalvin work harder" emerges without a
special withholding rule — anything behind a canon is invisible to supply.
Rejected alternative: filter subword canons or supply relations after their canon
grounds (both require state, violating D1).

**D3 — Significance from node-terminality (supplied entries).** Resolved in
grill: terminal nodes → S1, non-terminal → S2. *Rationale:* generative and
self-validating — it caught the `{a:[Det]}` error in the table (Det non-terminal
⇒ S2, not S1). The rule produces the dialogue rather than merely describing it.

**D4 — Pre-decode, single-stage decoder.** Resolved in grill: decode all turns
to a flat list at configuration time; one function per turn. *Rationale:* symbol
resolution is deterministic and total (independent of dialogue history), so JIT
buys nothing; pre-decode validates the table against the script up front and
keeps the loop a pure iterator. Rejected: JIT decode (entangles loop with
script); split significance/structure stages (no useful intermediate).

**D5 — Grounding, not S1, is the satisfaction signal; learned = declared-band
agreement.** Resolved in grill: an identity is learned at S4, a canon at S2, a
relation at S3, only the countersigned primary at S1. *Rationale:* separates
"understood and in LTM" from "countersigned"; lets the loop terminate at the
structural band the author declared. Forward-compatible with the deferred
event-kind change (ground→S1, frame→S2–S4).

**D6 — Symmetric significance contract.** Resolved in grill: either party may
send any band; S2/S3 = proposal, S4 = request, S1 = terminal. *Rationale:*
symmetry is a design feature — there is no structural reason Kalvin cannot
assume a supervisor role. K-originated S2 is deferred to a later grill.

**D7 — Subword canons are not filtered.** Resolved in grill: withheld and
learned at S2 like any canon. *Rationale:* subword structure is future-proofing
(reconstructing words for novel queries); all subword entries become grounded.

**D8 — Dispatch-driven loop; table is the shared source of truth.** Resolved in
grill: both the trainer-side loop and the stub read the same pre-decoded table,
each consuming only its own rows via a self-held cursor. The trainer *computes*
its turns (supply function) and the table *validates* them; the stub *scripts*
its turns from the table. *Rationale:* dispatch is the only loop that survives
the move to real Kalvin (a replayer can't react to divergence); self-cursored
actors are bus-faithful (the stub owns its turn like real Kalvin). Rejected:
replay (orphaned `scripts/dialogue_runner.py` — replaced, not patched);
trigger-keyed stub (the cursor dissolves the matching problem the orphaned
`ResponseRow` solved).

**D9 — Greedy cursor dispatch; 1:1 is not a constraint.** Resolved in grill: a
run of consecutive same-actor rows is consumed whole by that actor. The
canonical table happens to be 1:1 T/K alternating; future tables may have
longer runs and are handled identically. *Rationale:* keeps the loop
format-agnostic and makes the table the sole determinant of run shape.

**D10 — Two-sided validation (Model A) + dual-exhaustion termination.** Resolved
in grill: each turn validates K-against-table then T-against-table (precise
failure attribution: stub/table vs supply bug); the run ends only when both
cursors are empty, so the final K (the closing S1) is verified by construction,
not trusted. *Rationale:* bootstrap exists to validate the trainer's brain
(explicit T-validation is the point); dual-exhaustion guarantees the closing S1
is reached and matched. Rejected: implicit T-validation via K divergence (loses
the precise attribution); trainer-side semantic detection of the closing S1
(violates statelessness, D1).

**Deferred (later grills).** Multi-primary scripts; Trainer intentionality
(submitting S1–S4 to signal intent); the event-kind → significance change;
"Kalvin never derives a withheld kline"; how real Kalvin produces requests and
grounds; multi-row K-run timing (moot for the 1:1 table).

## Status

- Spec: `@specs/dialogue-driven-training.md` — written; Training Loop section
  and dispatch/cursor/dual-exhaustion criteria added (DDT-22..28); DDT-18
  reframed to cursor exhaustion.
- Glossary: `@CONTEXT.md` updated (Canon added, Ratify loosened, MTS Entry
  demoted, Learned added).
- Artifact: `scripts/dialogue-mhall.json` exists in the grilled simple format
  (`{script, turns[]}`, actors T/K); subword node names need reconciliation
  (task 1.0).
- **To replace (previous design generation):** `scripts/dialogue_runner.py`
  (legend format + replay driver — KeyErrors on the current artifact) and the
  trigger-keyed `ResponseRow` model in `@specs/stub-kagent.md`. The stub becomes
  a self-cursored table reader; the runner becomes the dispatch loop (Phase 3).
  These are replaced, not patched.
- Implementation: not started.
- Open: switchover strategy and retirement of the prior spec/plan (Phase 5).
