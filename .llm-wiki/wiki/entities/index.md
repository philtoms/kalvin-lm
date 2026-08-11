# entities

## Concepts

- [Auto-Tune](auto-tune.md) — The project's experimental loop for tuning Kalvin's rationalisation behaviour — an LLM coding agent runs repeated sessions, observes the reactor/cogitator/rationaliser, edits the significance-model code, and re-runs.
- [cogitate()](cogitate.md) — The engine function implementing cogitation — one full LIFO pass over the work-list, emitting semantic predicates (ask S4 / countersign S3 / propose S2 / ground S1).
- [CONTEXT.md](contextmd.md) — The project's domain glossary — the source of truth for terminology. Source is the truth document; CONTEXT.md's glossary maps the terms the source uses.
- [expand](expand.md) — A cogitation strategy for the S2 arm — grades grounded candidates via kalvin.expand.expand. The lean harness default. Emits few proposals; exhibits silent synthesis on mhall.
- [Harness (implementation)](harness.md) — The training harness runtime — the multi-agent WebSocket server and message bus, plus the lean synchronous dialogue harness. Both drive the K engine.
- [_is_groundable()](isgroundable.md) — The engine predicate that decides whether a kline can be grounded, branching in order: identity → signature grounded; canon → all nodes grounded; relationship → reciprocal grounded; misfit → both.
- [K (engine)](k-engine.md) — The cognitive/reasoning engine — the stateless core that derives one dialogue turn from (state, incoming) and returns (batch, observations). The implementation of Kalvin's rationalisation.
- [Kalvin](kalvin.md) — The rationalising system — an agent whose every response carries significance, a measurement of how well-grounded the response is in what it already knows.
- [KDbg](kdbg.md) — Debug annotation carried on each compiled kline — holds the owning scope's annotation (parens stripped) and a scope level (0 for source, 1 for MTS).
- [KScript](kscript.md) — The DSL that authors training material by declaring klines and labelling them with Target Significance. A compiler/provenance concern — it produces structures and target labels, never a participant's lived significance.
- [LTM (Long-Term Memory)](ltm-long-term-memory.md) — Persistent knowledge that survives across sessions. Structurally identical to Frame; the distinction is semantic. A kline residing in LTM is grounded.
- [Mary's World](marys-world.md) — A reference teaching example used throughout the vision — building knowledge from a blank slate, one kline at a time, scaffolding where understanding is incomplete.
- [mhall](mhall.md) — Test case ('Mary had a little lamb') on which both cogitation strategies reach the same grounded model but diverge on S2 emissions.
- [_promote()](promote.md) — The engine cascade that grounds groundable entries to fixed point at S1. Does not emit S2 proposals — promotion is the fast S1 path.
- [rationalise()](rationalise.md) — The engine function driving one full rationalisation turn — resets observations and _incoming, runs route → cogitate → _promote, and emits the turn's batch.
- [route()](route.md) — The engine function that dispatches an incoming kline on its structural significance (sig_level), not on the producer's compiled stamp. The entry to fast/slow routing.
- [similar_fit](similarfit.md) — A cogitation strategy for the S2 arm — the graft heuristic. Emits many proposals (12 on mhall), including the canonical synthesis at step 16.
- [STM (Short-Term Memory)](stm-short-term-memory.md) — The lowest tier in the write cascade and Kalvin's event register — every write reaches it. Empty at session start.
- [TokenEncoder](tokenencoder.md) — The KScript compiler stage that resolves word bindings and produces encoded KLines from symbolic entries. Where identities receive their node labels.
- [WDMH](wdmh.md) — Signature for 'we don't know Mary, had a little lamb' — the other half of the WDMH↔MHALL pair at the centre of the silent-synthesis open question.
