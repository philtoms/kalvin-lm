"""Tests for the Participant runtime-checkable protocol."""

from __future__ import annotations

from harness.protocols import Participant


class _ValidParticipant:
    """Minimal class satisfying the Participant protocol."""

    role: str = "trainee"

    def on_message(self, message: object) -> None:
        pass


class _MissingRole:
    """Class with on_message but no role attribute."""

    def on_message(self, message: object) -> None:
        pass


class _MissingOnMessage:
    """Class with role but no on_message method."""

    role: str = "trainer"


def test_participant_has_role_attribute() -> None:
    obj = _ValidParticipant()
    assert isinstance(obj, Participant)


def test_participant_missing_role_not_instance() -> None:
    obj = _MissingRole()
    assert not isinstance(obj, Participant)


def test_participant_missing_on_message_not_instance() -> None:
    obj = _MissingOnMessage()
    assert not isinstance(obj, Participant)
