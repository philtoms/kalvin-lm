"""Shared protocols for the multi-agent harness."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.message import Message


@runtime_checkable
class Participant(Protocol):
    """Protocol for any participant in the harness.

    Every participant (embedded or client) must have a unique address
    and handle incoming messages via ``on_message``.

    The ``on_message`` signature is compatible with the
    ``Callable[[Message], None]`` expected by :meth:`MessageBus.subscribe`.
    """

    address: str

    def on_message(self, message: Message) -> None:
        """Receive and handle a message from the bus."""
        ...
