# Session Lifecycle

All commands use `$AT_PYTHON` (absolute path to main repo's `.venv/bin/python`) and run from inside the worktree.

## Setup

```bash
# From the main project repo:
AT_PYTHON="$(pwd)/.venv/bin/python"

# Init creates worktree + session
PYTHONPATH=src $AT_PYTHON -m training.auto_tune init \
  --session <name> --curriculum curricula/first-steps.md

# Cd into the worktree
cd .worktrees/auto-tune/<name>
```

## First Run

```bash
# 1. Start processes
PYTHONPATH=src $AT_PYTHON -m training.auto_tune start-harness --session <name>
PYTHONPATH=src $AT_PYTHON -m training.auto_tune start-supervisor --session <name>

# 2. Run training (one event at a time)
PYTHONPATH=src $AT_PYTHON -m training.auto_tune step \
  --session <name> --command '{"action": "start"}'
PYTHONPATH=src $AT_PYTHON -m training.auto_tune step \
  --session <name> --command '{"action": "continue"}'
# ... repeat through all events ...

# 3. Snapshot results
PYTHONPATH=src $AT_PYTHON -m training.auto_tune snapshot --session <name>

# 4. Observe
cat auto-tune/<name>/harness.log

# 5. Stop processes
PYTHONPATH=src $AT_PYTHON -m training.auto_tune stop-supervisor --session <name>
PYTHONPATH=src $AT_PYTHON -m training.auto_tune stop-harness --session <name>
```

## Between Runs (the tuning loop)

```bash
# Reset state for a fresh run
PYTHONPATH=src $AT_PYTHON -m training.auto_tune reset --session <name>

# Restart processes
PYTHONPATH=src $AT_PYTHON -m training.auto_tune start-harness --session <name>
PYTHONPATH=src $AT_PYTHON -m training.auto_tune start-supervisor --session <name>

# Run again...
```

## Teardown

When the session is complete and the worktree is no longer needed:

```bash
# Cd back to the main repo first
cd <main-repo-root>
PYTHONPATH=src .venv/bin/python -m training.auto_tune teardown --session <name>
```

This removes the worktree directory and deletes the `auto-tune/<name>` branch.
