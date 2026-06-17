"""Thread-safe role-based message bus for the multi-agent harness."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from harness.message import Message

# Sentinel value to signal the event loop to stop.
_STOP_SENTINEL: Message | None = None

# Role used for wildcard (diagnostic) subscribers.
WILDCARD_ROLE = "*"


class MessageBus:
    """Role-based message router with single-dispatch event loop.

    Thread-safe: ``send()`` may be called from any thread.  All handler
    dispatch occurs on the thread running ``run()``.

    Usage::

        bus = MessageBus()
        bus.subscribe("trainee", my_handler)
        bus.send(Message(role="trainee", action="submit", message="..."))
        # In a dedicated thread:
        bus.run()
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Message | None] = queue.Queue()
        self._handlers: dict[str, list[Callable[[Message], None]]] = {}
        self._wildcards: list[Callable[[Message], None]] = []
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def subscribe(self, role: str, handler: Callable[[Message], None]) -> None:
        """Register *handler* for *role*.

        Multiple handlers per role are allowed.  Use the wildcard role
        ``"*"`` to receive every dispatched message (for diagnostic listeners).
        """
        with self._lock:
            if role == WILDCARD_ROLE:
                self._wildcards.append(handler)
            else:
                self._handlers.setdefault(role, []).append(handler)

    def send(self, msg: Message) -> None:
        """Enqueue *msg* for dispatch.  Safe to call from any thread."""
        self._queue.put(msg)

    def run(self) -> None:
        """Event loop: dequeue messages and dispatch to handlers.

        Blocks until :meth:`stop` is called (which enqueues a sentinel).
        """
        while True:
            msg = self._queue.get()
            if msg is _STOP_SENTINEL:
                break
            self._dispatch(msg)

    def stop(self) -> None:
        """Signal the event loop to exit gracefully."""
        self._queue.put(_STOP_SENTINEL)

    # -- internal ------------------------------------------------------------

    def _dispatch(self, msg: Message) -> None:
        """Dispatch *msg* to role-specific and wildcard handlers."""
        with self._lock:
            wildcards = list(self._wildcards)
            handlers = list(self._handlers.get(msg.role, []))

        for handler in handlers:
            handler(msg)

        for handler in wildcards:
            handler(msg)

        # Unrouted messages with a sender get an error reply.
        if not handlers and msg.sender is not None:
            error_msg = Message(
                role=msg.sender,
                action="error",
                message=f"unknown role: {msg.role}",
            )
            self._queue.put(error_msg)
