# Troubleshooting

| Problem                                | Solution                                                                                           |
| -------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `step` times out                       | Supervisor may be stuck. Check `status.json`. Kill stale processes: `lsof -ti :8765 \| xargs kill` |
| Port 8765 in use                       | Kill stale harness: `lsof -ti :8765 \| xargs kill`                                                 |
| `lessons_completed` already 3 on start | Stale curriculum state file. Delete `curricula/<slug>.json` and reset                              |
| Supervisor won't connect               | Harness not ready. Check `harness.pid`, wait, retry                                                |
| No rationalise events in log           | Curriculum is all fast-path S1. That's correct but boring — consider a more complex curriculum     |
| Events.jsonl shows `lesson: null`      | Fixed — Trainer now captures label before advancing position. Run auto-review to catch regressions |
