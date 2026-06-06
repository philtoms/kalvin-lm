# Session State File Format

The session state file lives at `auto-tune/<name>/session-state.md`.

It is the **single source of truth** for resuming an auto-tune session. The agent reads it at the start of each loop iteration and updates it after every observation phase (step 3d).

## Template

```markdown
# Auto-Tune Session State

## Goal
<what specific improvement you're targeting>

## Done Criteria
<observable outcome that means the goal is met, or "open session">

## Session
- **Name:** <name>
- **Curriculum:** <curriculum-path>
- **Branch:** auto-tune/<name>
- **Started:** <date>

## Current Phase
<one of: running, observing, editing, resetting, documenting, complete>

## Next Action
<what the agent should do next — e.g., "start run 5", "fix crash in src/foo.py", "document results">

## Run Log

### Run N (latest)
- **Code changes:** <what you changed before this run, or "baseline">
- **Observation:** <1-3 sentence summary of what happened>
- **Verdict:** <met / improved / regressed / crashed / no change>

### Run N-1
- **Code changes:** ...
- **Observation:** ...
- **Verdict:** ...

(older runs summary — one line each)
- Run N-2: <one-line summary>
- Run N-3: <one-line summary>

## Patterns & Notes
<recurring observations, hypotheses, things to try next — bullet points>

## Files Modified
<list of source files changed during this session, with brief notes>
```

## Rules

1. **Update after every observation.** Never leave the state file stale. The next reader must be able to continue seamlessly.
2. **Keep run summaries compact.** 1-3 sentences per run. Detailed logs live in `harness.log` and `events.jsonl` — the state file points to them, it doesn't duplicate them.
3. **Next Action must be specific.** Not "continue tuning" but "start run 6, test whether increasing budget fixes the escalation at lesson 3."
4. **Always move the latest run to the top.** The latest run gets the full template; older runs collapse to one-liners.
5. **Patterns section accumulates insight.** This is where cross-run learning lives. Update it when you notice something.
