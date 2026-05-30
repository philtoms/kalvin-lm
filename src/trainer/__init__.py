"""Trainer package — embedded harness participant that drives the training loop."""

from trainer.cogitation import (
    ESCALATION_THRESHOLD,
    CogitationRequest,
    CogitationResult,
    Cogitator,
    ConversationTurn,
    MisfitInfo,
)
from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.trainer import Trainer

__all__ = [
    "Trainer",
    "Curriculum",
    "CurriculumState",
    "EntryKey",
    "Cogitator",
    "CogitationRequest",
    "CogitationResult",
    "MisfitInfo",
    "ConversationTurn",
    "ESCALATION_THRESHOLD",
]
