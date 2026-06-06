# Session Lifecycle

## First Run

```bash
# 1. Init
PYTHONPATH=src .venv/bin/python -m participants.auto_tune init \
  --session <name> --curriculum curricula/first-steps.md

# 2. Start processes
PYTHONPATH=src .venv/bin/python -m participants.auto_tune start-harness --session <name>
PYTHONPATH=src .venv/bin/python -m participants.auto_tune start-supervisor --session <name>

# 3. Run training (one event at a time)
PYTHONPATH=src .venv/bin/python -m participants.auto_tune step \
  --session <name> --command '{"action": "start"}'
PYTHONPATH=src .venv/bin/python -m participants.auto_tune step \
  --session <name> --command '{"action": "continue"}'
# ... repeat through all events ...

# 4. Snapshot results
PYTHONPATH=src .venv/bin/python -m participants.auto_tune snapshot --session <name>

# 5. Observe
cat auto-tune/<name>/harness.log

# 6. Stop processes
PYTHONPATH=src .venv/bin/python -m participants.auto_tune stop-supervisor --session <name>
PYTHONPATH=src .venv/bin/python -m participants.auto_tune stop-harness --session <name>
```

## Between Runs (the tuning loop)

```bash
# Reset state for a fresh run
PYTHONPATH=src .venv/bin/python -m participants.auto_tune reset --session <name>

# Restart processes
PYTHONPATH=src .venv/bin/python -m participants.auto_tune start-harness --session <name>
PYTHONPATH=src .venv/bin/python -m participants.auto_tune start-supervisor --session <name>

# Run again...
```
