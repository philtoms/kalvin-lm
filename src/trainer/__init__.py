"""Trainer package — embedded harness participant that drives the training loop."""

from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.trainer import Trainer

__all__ = ["Trainer", "Curriculum", "CurriculumState", "EntryKey"]
