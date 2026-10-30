---
type: source
title: "Cogitator trace tables aligned: closers-first, row-owned frames, per-reentry memo"
status: insight
category: architecture
created: 2026-10-30
updated: 2026-10-30
slug: cogitator-trace-table-alignment
---

# Cogitator trace tables aligned: closers-first, row-owned frames, per-reentry memo

Rewrote Cogitator.rationalise (worktree table-driven, src/trs/model.py + dev/dialogue/trs_harness.py) so both mhall.ks and wdmh.ks generate traces matching the user's tables, per rulings: flood-then-distill (parallel-friendly), orthogonal paths, diagnostic harness. The rules that made it work, each earned by a failed intermediate:

- **Candidate order: closers first, then μ-least.** `(0 if F==side(E) else 1, μ(side))`. μ-least alone diverges from the table at the ALL level (a:Det leads, fragments); closers-first alone diverges at MHALL (no closer exists → μ picks MS:Subject). Both keys together reproduce every level. Child batches push reversed so the first candidate's subtree runs to completion before the next candidate explores — the reductive determinism; batches are independent (parallel).
- **Entry-path rule** added to _in_path: a candidate whose sig equals any ancestor ENTRY's sig is illegal (MHALL→HALL→MHALL). The old content-flow rule (o-side candidate flow, first-hop origin exemption) kept alongside.
- **Self-pairing exclusion** (side(c)==side(k)): without it the ALL-canon/SVO-canon identity pairings short-circuit the tabled chains (the ALL entry resolves instantly against its own canon).
- **o-side witness settlement**: an o-side whose region has a held canonical witness (Domain.witness) is settled at pairing — this is how mhall's root closes its SVO side (the goal grounds itself); wdmh's HALL region has no canon so it reenters (the tabled descent). u-sides never witness-settle (would kill the ALLQ chain).
- **Frame ownership by producing ROW, not reentry**: _chain collects ancestor rows; sibling pairings' frames never cover each other's children (no self-satisfaction through siblings — the old reentry-keyed frames let the ALLQ row's ALL frame settle the lamb row's AL child, and the lamb row resolved the ALL reentry first, stealing the slot).
- **Per-reentry seen memo** (id(re_), entry-fp, cand-fp): the global memo starved twin reentries (identical klines from different parents all skipped, died unresolved, grounded nothing).
- **Slot rule: earliest-created RESOLVING row** (not first-to-resolve): with closers-first creation order this picks the reductive chain over instant-closing mirrors; sticky exchange feeds (realx-gated: only exchanges that admitted non-self-witness content) displace resolved-first rows — "the S3 path is the path" (wdmh's HALL slots H:did-have over HALL×MHALL).
- **Resolution frames its reentry's region** (record-only rframes, never settle) and the goal row's grounding mints the claim's o-region — producing the final frame columns MHALL, SVO / WDH, HALL.
- **Vacuous exchanges** (union adjacency self-witness only, Mod:[Mod]) skip the admit and the sticky claim — this was why wdmh's ALL→Q→O chain vanished from the distilled path.
- **Trace events**: Row carries birth/waited/closed clocks + bp rows; the harness re-emits rows per event (pair/reenter, stalled, resolved-BP/grounded) — reproducing the tables' repeated-row temporal shape. Instant rows (resolved at pairing) and exchange-arriving stalls emit once.
- Row/_Reentry are dataclass(eq=False) — recursive __eq__ RecursionError hazard; an identity rindex map replaces rows.index()/min() comparisons.

Result: mhall stores MHALL:[SVO]; wdmh stores DH:[did,have] + WDMH:[MHALL] with the DH stall→exchange row "frame DH / resolved, BP WDH, HALL" and the root stalled→grounded sequence. Tests: 171 passed, 16 xfailed (mid-migration debt: old module-level rationalise semantics, acknowledged stale since e93c192). Related: [[rationalisation-rules]] needs resync.

*Category: architecture*

---
*Captured: 2026-10-30*

## Related

_Add links to related pages._
