# Cascade Control — Specification

## Overview

When a new kline enters `rationalise()` in a densely populated model, the candidate retrieval via `model.where(signature)` can return a very large number of candidates (every kline whose signature has ANY bitwise overlap with the query). Each S2/S3 candidate generates expansion proposals, which are published as events, which trigger reactive scaffolding, which adds new entries to the model, which increases the candidate pool — creating an exponential feedback loop (cascade).

This spec defines two controls that together prevent intra-lesson cascade explosion while preserving productive reactive scaffolding.

## Evidence

Auto-tune session: `auto-tune/explore-agent-capability/`
- Run 6 (no cap): 170K log lines, 56,712 budget exhaustion events, 54,059 events for a single entry
- Run 7 (cap=8): 3.9K log lines, 2 budget exhaustion events — **43x log reduction**
- Run 8 (regression test): no regression on conflict-drill curriculum; lesson 4 improved from 11/11 to 12/11

## Root Cause

`model.where(sig)` uses `signifies()` (bitwise AND ≠ 0) for candidate retrieval. In a model with N identity atoms and their cross-products, a single query can match O(N²) candidates. Each candidate generates expansion proposals; each proposal triggers a reactive round; each reactive round can submit new entries that grow the model further.

## Fixes Applied

### CC-1: Slow-path candidate cap in KAgent.rationalise()

`KAgent` has a configurable `max_candidates` parameter (default 8). After Phase 5 retrieves all S2/S3 candidates, if the list exceeds `max_candidates`:

1. Sort by priority: S2 before S3, then by node overlap count (descending)
2. Truncate to `max_candidates`
3. Submit only the top-K to the cogitator

This limits the fan-out from O(N²) to a constant bound, regardless of model density.

### CC-2: Reactor silent-drop after budget exhaustion

`Reactor._handle_reactive()` previously escalated on EVERY event after budget exhaustion, producing thousands of escalation messages. Now:

1. The first event that reaches `max_reactive_rounds` escalates normally
2. All subsequent events (`reactive_rounds > max_reactive_rounds`) are silently dropped — no escalation, no logging, no bus messages

This prevents the reactor from spinning on the event firehose while the cogitator drains.

### CC-3: Satisfaction guard in Trainer

When cogitation generates multiple events for a single entry (initial S1
plus expansion proposals from S3 candidates), the Trainer checks whether
the entry's key is already in the satisfied set before processing S2/S3
events. Already-satisfied entries are skipped — no reactor processing,
no ratify request, no LLM call. This prevents spurious LLM invocations
when S1 arrives via cogitation before pending S3 expansions are processed.

### CC-4: Satisfaction-based lesson completion

Lesson completion uses `len(state.satisfied) >= len(state.submitted)` instead
of reactor event counting (`received_count == expected_count`). With direct
cogitation push, each entry generates multiple events (initial rationalise +
expansion proposals), so event counting fires before all entries are satisfied.

A re-fire guard prevents duplicate lesson completion: the check returns
`False` if the current lesson is already in `lesson_satisfied` or if there
is no current lesson (curriculum position past the end).

## Behavioural Rules

1. The number of S2/S3 candidates submitted to the cogitator per entry must not exceed `max_candidates`.
2. When truncating, S2 candidates must be prioritised over S3 candidates.
3. Among same-level candidates, those with higher node overlap with the query must be prioritised.
4. After the first budget exhaustion escalation, the reactor must silently drop all subsequent S2/S3 events for the remainder of the lesson.
5. The candidate cap must not prevent S1 fast-path matches (S1 candidates are processed before any truncation).
6. The candidate cap must not affect S4 (novel) routing (no candidates for S4).
7. S2/S3 events for already-satisfied entries must be skipped (no reactor/ratify).
8. Lesson completion must be determined by satisfaction count (`len(satisfied) >= len(submitted)`), not event count.
9. Lesson completion must not re-fire when post-completion cogitation events arrive.

## Test Matrix

| ID  | Criterion | Status |
|-----|-----------|--------|
| CC-1 | rationalise caps slow candidates at max_candidates | ✅ |
| CC-2 | S2 candidates prioritised over S3 in truncation | ✅ |
| CC-3 | Higher overlap candidates prioritised within same level | ✅ |
| CC-4 | First budget exhaustion event escalates | ✅ |
| CC-5 | Subsequent events after budget exhaustion are silently dropped | ✅ |
| CC-6 | S1 fast-path is unaffected by cap | ✅ |
| CC-7 | S4 routing is unaffected by cap | ✅ |
| CC-8 | Default max_candidates is 8 | ✅ |
| CC-9 | S2/S3 events for already-satisfied entries are skipped | ✅ auto-tune/direct-cogitation-push |
| CC-10 | Lesson completion uses satisfaction count | ✅ auto-tune/direct-cogitation-push |
| CC-11 | Lesson completion does not re-fire | ✅ auto-tune/direct-cogitation-push |
