"""Canonical role constants for the harness bus routing."""

# The agent being trained — Kalvin itself.
TRAINEE_ROLE = "trainee"

# The training agent providing instruction and evaluation.
TRAINER_ROLE = "trainer"

# Supervisor role for TUI, Slack, and future AI agents that monitor and
# intervene in training sessions.
SUPERVISOR_ROLE = "supervisor"

__all__ = ["SUPERVISOR_ROLE", "TRAINEE_ROLE", "TRAINER_ROLE"]
