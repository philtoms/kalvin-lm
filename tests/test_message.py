"""Tests for the Message dataclass."""

import pytest

from training.harness.message import Message


def test_message_role_field():
    msg = Message(role="trainee", action="submit", message="data")
    assert msg.role == "trainee"


def test_message_repr_shows_role():
    msg = Message(role="trainee", action="submit", message="data")
    r = repr(msg)
    assert "role=" in r
    assert "address=" not in r


def test_message_frozen():
    msg = Message(role="trainee", action="submit", message="data")
    with pytest.raises(AttributeError):
        msg.role = "mentor"


def test_message_sender_default():
    msg = Message(role="trainee", action="submit", message="data")
    assert msg.sender is None
