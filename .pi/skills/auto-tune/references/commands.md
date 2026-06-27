# Auto-Tune Commands

All commands require `PYTHONPATH=src` and are invoked as:

```bash
PYTHONPATH=src $AT_PYTHON -m training.participants.auto_tune <command> --session <name> [options]
```

Where `$AT_PYTHON` is the absolute path to the main repo's `.venv/bin/python`, captured before entering the worktree:

```bash
AT_PYTHON="$(pwd)/.venv/bin/python"
```

## Worktree Convention

- `init` creates a git worktree at `.worktrees/auto-tune/<name>/` with branch `auto-tune/<name>`
- All subsequent commands run from **inside the worktree**: `cd .worktrees/auto-tune/<name>/`
- The main repo stays on its current branch — full isolation

## CLI Commands

| Command                                                               | Purpose                                                                        |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `init --session <name> --curriculum <path> [--host <h>] [--port <p>]` | Create worktree, session directory, config.json, git branch `auto-tune/<name>` |
| `teardown --session <name>`                                           | Remove worktree and delete branch (run from main repo)                         |
| `start-harness --session <name>`                                      | Start harness server (background), wait for ready                              |
| `stop-harness --session <name>`                                       | Graceful shutdown (SIGTERM → SIGKILL on timeout)                               |
| `start-supervisor --session <name>`                                   | Start CLI supervisor (background), wait for connected                          |
| `stop-supervisor --session <name>`                                    | Send shutdown command, wait for exit                                           |
| `send --session <name> --command '<json>'`                            | Write command, return immediately                                              |
| `events --session <name> [--after <seq>]`                             | Print events after given sequence number                                       |
| `step --session <name> --command '<json>'`                            | Write command, block until next event, print it                                |
| `status --session <name>`                                             | Print status.json                                                              |
| `snapshot --session <name>`                                           | Capture state, events, model, git metadata to `runs/<n>/`                      |
| `restore --session <name> --run <n>`                                  | Restore state and model from a snapshot                                        |
| `reset --session <name> [--fresh-model]`                              | Delete curriculum state, truncate events, optionally delete model              |

## Supervisor Commands (sent via `step` or `send`)

| Command JSON                            | Effect                                   |
| --------------------------------------- | ---------------------------------------- |
| `{"action": "start"}`                   | Begin training session                   |
| `{"action": "continue"}`                | No-op — acknowledge event, wait for next |
| `{"action": "ratify"}`                  | Countersign latest pending proposal      |
| `{"action": "restart"}`                 | Reset training state and restart         |
| `{"action": "pause"}`                   | Pause training                           |
| `{"action": "resume"}`                  | Resume training                          |
| `{"action": "stop"}`                    | End training session                     |
| `{"action": "shutdown"}`                | Graceful supervisor exit                 |
| `{"action": "guidance", "text": "..."}` | Send guidance to trainer                 |
| `{"action": "goal", "text": "..."}`     | Set a training goal                      |
| `{"action": "save"}`                    | Persist Kalvin model                     |
| `{"action": "load"}`                    | Load Kalvin model                        |

## Event Types (received from supervisor)

| Type             | When                                                                  |
| ---------------- | --------------------------------------------------------------------- |
| `connected`      | Supervisor connected to harness                                       |
| `progress`       | Training progress (started, lesson_complete, complete, amended, etc.) |
| `rationalise`    | Kalvin rationalisation event (ground or frame)                        |
| `ratify_request` | S2/S3 proposal needing ratification                                   |
| `escalation`     | Trainer stuck (budget_exhaustion or low_confidence)                   |
| `disconnected`   | Supervisor disconnected                                               |
