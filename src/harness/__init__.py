"""Harness multi-agent runtime infrastructure."""

from harness.adapter import KAgentAdapter
from harness.bus import MessageBus
from harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE, TRAINER_ROLE
from harness.message import Message
from harness.protocols import Participant

__all__ = [
    "KAgentAdapter",
    "Message",
    "MessageBus",
    "Participant",
    "SUPERVISOR_ROLE",
    "TRAINEE_ROLE",
    "TRAINER_ROLE",
]
