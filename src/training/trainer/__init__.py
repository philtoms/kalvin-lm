"""Trainer package — embedded harness participant that drives the training loop."""

from training.trainer.curriculum import Curriculum, CurriculumState, EntryKey
from training.trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
    Lesson,
)
from training.trainer.curriculum_generator import (
    CurriculumGenerationError,
    CurriculumGenerator,
)
from training.trainer.reactor import Reactor
from training.trainer.trainer import Trainer

__all__ = [
    "Trainer",
    "Reactor",
    "Curriculum",
    "CurriculumState",
    "CurriculumDocument",
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
