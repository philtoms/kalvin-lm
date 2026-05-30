"""Addressed message for the multi-agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """A unit of inter-participant communication routed by address.

    The harness does not interpret ``action`` or ``message`` — it routes
    by ``address`` only.

    Attributes:
        address: Recipient address string.
        action: Verb interpreted by the recipient.
        message: Arbitrary payload.
        sender: Optional sender address, set by the bus or sender context.
    """

    address: str
    action: str
    message: Any
    sender: str | None = field(default=None)

    def __repr__(self) -> str:
        return (
            f"Message(address={self.address!r}, action={self.action!r}, "
            f"sender={self.sender!r})"
        )
