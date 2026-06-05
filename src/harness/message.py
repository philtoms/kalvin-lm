"""Role-based message for the multi-agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """A unit of inter-participant communication routed by role.

    The harness does not interpret ``action`` or ``message`` — it routes
    by ``role`` only.

    Attributes:
        role: Recipient role string.
        action: Verb interpreted by the recipient.
        message: Arbitrary payload.
        sender: Optional sender role, set by the bus or sender context.
    """

    role: str
    action: str
    message: Any
    sender: str | None = field(default=None)

    def __repr__(self) -> str:
        return (
            f"Message(role={self.role!r}, action={self.action!r}, "
            f"sender={self.sender!r})"
        )
