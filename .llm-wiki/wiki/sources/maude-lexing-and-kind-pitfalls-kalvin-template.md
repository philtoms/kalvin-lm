---
type: source
title: Maude lexing and kind pitfalls in the Kalvin template
status: insight
category: formal-spec
created: 2026-09-23
updated: 2026-09-23
slug: maude-lexing-and-kind-pitfalls-kalvin-template
---

# Maude lexing and kind pitfalls in the Kalvin template

While updating [[sources/obs-2026-09-15-kalvin-algebra-translated-to-verified-maude-template]]'s file (docs/kalvin-algebra.maude), three Maude 3.5.1 pitfalls surfaced:

1. **A `---` line comment whose content starts with `(` opens an *enclosed* comment** (`---( ... )---`, like `***( ... )***`). The comment then swallows code until a matching closer (or errors), and the reported symptoms are misleading: comment text leaks into "no parse for statement" errors many lines away. Rule: never begin a `---` comment with `(` (even `---   (` with spaces triggers it). Hit three times in one session; fixed by rewording the comment.
2. **Rewrite rules cannot jump kinds.** A rule `< ... > => bridged(...)` between a `MeetState` LHS and `MeetOutcome` RHS fails to parse ("didn't expect token") because the two sorts live in different connected components. Fix: declare a common supersort (`subsorts MeetState MeetOutcome < Meeting .`) so the rule stays within one kind.
3. **Missing variable declarations cascade.** An undeclared variable (`var U : Value`, `vars AP BP : Posn`) produces "bad token" errors on the *variables themselves*, which reads like a lexer problem rather than a declaration problem. Check the declaration block first when Maude says "bad token <identifier>".

The two-party meeting walk (Def 15) encodes cleanly as one-position-per-party states `< s | aDep | aPos | bDep | bPos | M >` with positions `value via kline` (`root` for departures): four step rules (a/b-descent, a/b-ascent) plus one [meet] rule emitting `bridged(slot_a : slot_b)`. Keep rules linear (shared variables expressed as `T == V` conditions) — non-linear LHS patterns over AC memory terms are best avoided.

*Category: formal-spec*

---
*Captured: 2026-09-23*

## Related

_Add links to related pages._
