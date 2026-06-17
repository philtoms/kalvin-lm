"""Harness multi-agent runtime infrastructure."""

from training.harness.adapter import KAgentAdapter
from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE, TRAINER_ROLE
from training.harness.message import Message
from training.harness.protocols import Participant

__all__ = [
    "KAgentAdapter",
    "Message",
    "MessageBus",
    "Participant",
    "SUPERVISOR_ROLE",
    "TRAINEE_ROLE",
    "TRAINER_ROLE",
]
