"""Thread-safe addressed message bus for the multi-agent harness."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from harness.message import Message

# Sentinel value to signal the event loop to stop.
_STOP_SENTINEL: Message | None = None

# Address used for wildcard (diagnostic) subscribers.
WILDCARD_ADDRESS = "*"


class MessageBus:
    """Addressed message router with single-dispatch event loop.

    Thread-safe: ``send()`` may be called from any thread.  All handler
    dispatch occurs on the thread running ``run()``.

    Usage::

        bus = MessageBus()
        bus.subscribe("kalvin", my_handler)
        bus.send(Message(address="kalvin", action="submit", message="..."))
        # In a dedicated thread:
        bus.run()
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Message | None] = queue.Queue()
        self._handlers: dict[str, list[Callable[[Message], None]]] = {}
        self._wildcards: list[Callable[[Message], None]] = []
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def subscribe(self, address: str, handler: Callable[[Message], None]) -> None:
        """Register *handler* for *address*.

        Multiple handlers per address are allowed.  Use the wildcard address
        ``"*"`` to receive every dispatched message (for diagnostic listeners).
        """
        with self._lock:
            if address == WILDCARD_ADDRESS:
                self._wildcards.append(handler)
            else:
                self._handlers.setdefault(address, []).append(handler)

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
        """Dispatch *msg* to address-specific and wildcard handlers."""
        # Collect wildcard handlers first (always invoked).
        with self._lock:
            wildcards = list(self._wildcards)
            handlers = list(self._handlers.get(msg.address, []))

        # Deliver to address-specific handlers.
        for handler in handlers:
            handler(msg)

        # Deliver to wildcard (diagnostic) handlers.
        for handler in wildcards:
            handler(msg)

        # If no address-specific handler was found and the message has a
        # sender, send an error back to the sender.
        if not handlers and msg.sender is not None:
            error_msg = Message(
                address=msg.sender,
                action="error",
                message=f"unknown address: {msg.address}",
            )
            # Re-enqueue the error so it goes through the normal dispatch
            # path on the same event-loop thread.
            self._queue.put(error_msg)
