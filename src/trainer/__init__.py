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
from trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
    Lesson,
)
from trainer.curriculum_generator import (
    CurriculumGenerationError,
    CurriculumGenerator,
)
from trainer.reactor import Action, Reactor
from trainer.trainer import Trainer

__all__ = [
    "Action",
    "Trainer",
    "Reactor",
    "Curriculum",
    "CurriculumState",
    "CurriculumDocument",
    "CurriculumDocument",  # re-exported from curriculum_document
    "CurriculumGenerationError",
    "CurriculumGenerator",
    "CurriculumParseError",
    "EntryKey",
    "Lesson",
    "Cogitator",
    "CogitationRequest",
    "CogitationResult",
    "MisfitInfo",
    "ConversationTurn",
    "ESCALATION_THRESHOLD",
]
