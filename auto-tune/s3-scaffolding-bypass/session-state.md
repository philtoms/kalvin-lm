# Auto-Tune Session State

## Goal
Enable S3 training runs with zero LLM calls and zero supervisor interaction. Two changes required:
1. Make `process_s2_s3` return whether auto-countersign succeeded, so the trainer can suppress `ratify_request` when auto-countersign handles it.
2. Design a curriculum that triggers S3 classification and whose entries predictably match the resulting expansion proposals, so auto-countersign always wins.

## Done Criteria
A full training run completes with:
- S3 classification exercised (harness.log confirms S3 events)
- Zero LLM calls
- Zero supervisor interactions (no ratify_request events emitted when auto-countersign succeeds)
- All lessons satisfied
- harness.log clean (no errors, warnings, or escalations)

## Session
- **Name:** s3-scaffolding-bypass
- **Curriculum:** curricula/first-steps-s2.md (baseline), then custom S3 curriculum
- **Branch:** auto-tune/s3-scaffolding-bypass
- **Worktree:** .worktrees/auto-tune/s3-scaffolding-bypass
- **Started:** 2026-06-08

## Current Phase
editing

## Next Action
Code changes are DONE (process_s2_s3 returns bool, trainer suppresses ratify_request). Now need to design S3 curriculum. Key challenge: figuring out which proposal klines will be generated for a given query, and including matching entries in the same lesson. Was investigating expand() proposal mechanics when context ceiling hit. Resume by:
1. Run a test with the code changes + first-steps-s2 to verify ratify suppression works for auto-countersign cases
2. Design a minimal S3 curriculum — read the "Curriculum Design Challenge" section below for the key insight needed
3. Run 2 to verify zero-LLM + zero-ratify

## Run Log

### Run 1 (latest) — baseline with first-steps-s2
- **Code changes:** none (baseline)
- **Observations:** Lesson 5 (`MH => H A`) triggers S3. Proposal is `H => H A` (sig=0x100, nodes=[256, 2]). Auto-countersign fails because no matching entry. LLM cogitation runs (3+ calls). Also confirmed that `ratify_request` is always sent regardless of auto-countersign outcome.
- **Verdict:** baseline — no changes yet, needed to understand proposal shapes

## Patterns & Notes

### Code Changes Made (not yet committed)
1. **`src/trainer/reactor.py`** — `process_s2_s3()` now returns `bool`. Returns `True` when auto-countersign succeeds, `False` when reactive handling is invoked.
2. **`src/trainer/trainer.py`** — `_handle_rationalise()` now captures the return value from `reactor.process_s2_s3()`. Only sends `ratify_request` when auto-countersign did NOT succeed (`if not auto_matched`).

### Architecture Understanding
- **Auto-countersign:** `Reactor._auto_countersign()` compares proposal kline against `_current_entries` using `KLine.__eq__` (signature + nodes match). Only matches within the SAME lesson's compiled entries.
- **Proposal generation:** `propose_expansions()` → `generate_expansions()` produces misfit expansions (underfit, overfit, dual). Proposals are generated from the CANDIDATE's perspective, not the query.
- **Ratify suppression:** DONE — trainer conditionally suppresses `ratify_request` when `auto_matched=True`.

### Curriculum Design Challenge
The hard part is designing a curriculum where auto-countersign always wins. Key findings:

1. **Proposal source:** Proposals come from `propose_expansions(model, candidate, sig)` where `candidate` is the model kline that the query matched against. The proposal tries to "fix" the candidate to match the query.

2. **Identity klines are canonical:** When the candidate is an identity (sig=X, nodes=[]), `candidate_sig == nodes_sig` (both represent X), so `propose_expansions()` returns immediately with NO proposals. The identity is "canonical" — nothing to expand.

3. **S3 against identity still generates events:** Even though propose_expansions yields nothing for identity candidates, the cogitator still publishes the frame event with the query. The trainer's `_handle_rationalise` still gets called. But `event.proposal` might be `None` or the query itself — need to verify this in a run.

4. **`MH => H A` analysis:** Compiles to 5 entries. Entry 3 (sig=0x2100, nodes=[256, 2]) triggers S3 against H identity (sig=0x100, nodes=[]). From harness log, the proposal IS generated: `H => H A` (sig=0x100, nodes=[256, 2]). But this comes from expand() finding connotation paths, not directly from propose_expansions on the identity.

5. **expand() produces connotation candidates:** The expand() function doesn't just look at direct candidates — it follows connotation paths (S3 hops). This means proposals can come from candidates found via multi-hop traversal, not just direct model lookups.

6. **Promotion creates new candidates:** `promote_participating` runs when S1 is found, creating new klines in the model that become candidates for subsequent entries in the same lesson. This changes the proposal landscape mid-lesson.

### Strategy for Curriculum Design
Rather than predicting exact proposals, consider:
- **Approach A:** Include all possible proposal shapes as entries in the lesson (brute-force matching). But this may cause those entries themselves to trigger more S3 events.
- **Approach B:** Use a curriculum where S3 triggers against identity klines only. Since identities are canonical and produce no proposals, the S3 event has no proposal to countersign. Need to verify: does auto-countersign fire when proposal=None? Does the trainer still send ratify_request?
- **Approach C:** Use countersign entries (M == H) which create reciprocal klines. Submit a canonize that triggers S3 against the reciprocal kline (not the identity). The reciprocal kline IS a misfit (its sig != nodes_sig), so propose_expansions WILL generate proposals.

**Next agent should investigate Approach B first** — it's the simplest. Check what happens when the S3 candidate is an identity (canonical, no proposals). Does the system handle this gracefully? If so, a curriculum that only triggers S3 against identities would need zero matching entries.

## Files Modified
- `src/trainer/reactor.py` — `process_s2_s3()` returns `bool`
- `src/trainer/trainer.py` — conditional `ratify_request` suppression
