"""KAgent adapter — bridge between the KAgent pipeline and the message bus.

Receives harness messages sent to role ``trainee``, compiles KScript source
into entries, submits them one at a time to :meth:`KAgent.rationalise`, and
routes KAgent callbacks back to the original sender via the bus.

The adapter implements the :class:`Participant` protocol (to receive bus
messages) and provides ``on_event()`` so KAgent can call it directly as
its adapter callback (replacing the internal EventBus).

Thread model
------------
``on_message`` executes on the bus dispatch thread.
``on_event`` is called from the Cogitator background thread via
``KAgent._publish`` → ``adapter.on_event``.  The sender map (a plain dict)
is written on the bus thread and read on the Cogitator thread.  Under CPython's
GIL, individual dict reads/writes are atomic, so no explicit locking is needed.
If the adapter is ever used outside CPython, a ``threading.Lock`` should be
added around sender-map access.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from harness.bus import MessageBus
from harness.constants import TRAINEE_ROLE, TRAINER_ROLE, SUPERVISOR_ROLE
from harness.message import Message
from harness.protocols import Participant
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kscript.compiler import compile_source
from kscript.decompiler import Decompiler

if TYPE_CHECKING:
    pass


# Protocol for the kagent parameter — avoids importing KAgent (circular dep)
# while giving mypy the methods we actually call.
class _KAgentLike(Protocol):
    def rationalise(self, kline: KLine) -> bool: ...
    def countersign(self, kline: KLine) -> bool: ...
    def save(self, path, format=None) -> None: ...
    def codec(self) -> object: ...

logger = logging.getLogger(__name__)

# Type alias for the sender-map key: (signature, frozen_nodes).
EntryKey = tuple[int, tuple[int, ...]]


class KAgentAdapter:
    """Thin integration layer between KAgent and the role-based message bus.

    Parameters
    ----------
    bus:
        The message bus to subscribe to and send responses on.
    role:
        The role to subscribe to on the bus (default ``"trainee"``).
    kagent:
        Optional KAgent instance.  Can also be set later via :meth:`bind`.
        This two-phase wiring avoids a circular construction dependency:
        ``adapter = KAgentAdapter(bus)`` → ``agent = KAgent(adapter=adapter)``
        → ``adapter.bind(agent)``.
    """

    def __init__(
        self,
        bus: MessageBus,
        role: str = TRAINEE_ROLE,
        kagent: Optional[_KAgentLike] = None,
    ):
        self._bus = bus
        self._role = role
        self._kagent: _KAgentLike | None = kagent

        # Sender map: (entry.signature, tuple(entry.nodes)) → sender role.
        # Populated in on_message (bus thread), read in on_event (cogitator thread).
        self._sender_map: dict[EntryKey, str] = {}

        # Subscribe to the bus at construction time.
        bus.subscribe(self._role, self.on_message)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def role(self) -> str:
        """The bus role this adapter is subscribed to."""
        return self._role

    @property
    def kagent(self) -> _KAgentLike | None:
        """The KAgent instance, or ``None`` if not yet bound."""
        return self._kagent

    # ── Late binding ───────────────────────────────────────────────────

    def bind(self, kagent: _KAgentLike) -> None:
        """Bind a KAgent instance after construction.

        Use this when the KAgent needs the adapter as its callback during
        its own construction, creating a circular dependency.
        """
        self._kagent = kagent

    # ── Participant protocol ───────────────────────────────────────────

    def on_message(self, msg: Message) -> None:
        """Handle an incoming bus message sent to this adapter's role.

        Actions
        -------
        submit:
            Compile KScript source from ``msg.message``, record sender per
            entry in the sender map, and call ``kagent.rationalise(entry)``
            for each compiled entry.
        countersign:
            Call ``kagent.countersign(kline)`` with the KLine in
            ``msg.message``.
        save:
            Persist Kalvin's model to disk via agent_codec.
        load:
            Load Kalvin's model from disk via agent_codec.
        """
        if msg.action == "submit":
            self._handle_submit(msg)
        elif msg.action == "countersign":
            self._handle_countersign(msg)
        elif msg.action == "save":
            self._handle_save(msg)
        elif msg.action == "load":
            self._handle_load(msg)
        else:
            logger.warning("Unknown action %r from %s", msg.action, msg.sender)

    # ── Adapter callback (KAgent → adapter) ────────────────────────────

    def on_event(self, event: RationaliseEvent) -> None:
        """Receive a rationalisation event from the KAgent.

        The KAgent calls this directly (via ``_publish``) instead of using
        an internal EventBus.  The event is wrapped into a :class:`Message`
        routed to the original sender (looked up in the sender map) and
        sent via the bus.

        Orphan events (no sender in the map, e.g. "done" idle events from
        the Cogitator) are silently dropped.
        """
        key: EntryKey = (event.query.signature, tuple(event.query.nodes))
        sender = self._sender_map.get(key)

        if sender is None:
            logger.debug("Orphan event (no sender): %s", event)
            return

        response = Message(
            role=sender,
            action=event.kind,
            message=event,
        )
        self._bus.send(response)

    # ── Internal handlers ──────────────────────────────────────────────

    def _handle_submit(self, msg: Message) -> None:
        """Compile KScript source and submit each entry to KAgent."""
        if self._kagent is None:
            logger.error("No KAgent bound; cannot submit")
            return

        try:
            entries = compile_source(msg.message)
        except Exception as exc:
            # Compilation error (LexerError, ParseError, etc.) — report back.
            logger.error("Compilation error: %s", exc)
            error_msg = Message(
                role=msg.sender or "",
                action="error",
                message=str(exc),
            )
            self._bus.send(error_msg)
            return

        logger.info("Submitting %d compiled entries to KAgent", len(entries))
        # Pre-register all entries in STM so that countersign pairs
        # (e.g. from M == H compiling to {M: H} and {H: M}) can find
        # each other during the countersign check in rationalise().
        if hasattr(self._kagent, 'model'):
            for entry in entries:
                self._kagent.model.add_stm(entry)
        for entry in entries:
            # Record sender for this entry so callbacks can be routed.
            key: EntryKey = (entry.signature, tuple(entry.nodes))
            sender = msg.sender or ""
            self._sender_map[key] = sender
            # Fire-and-forget: rationalise returns bool but we don't check.
            # Events come back via on_event callback.
            self._kagent.rationalise(entry)

    def _handle_countersign(self, msg: Message) -> None:
        """Forward a countersign request to the KAgent."""
        if self._kagent is None:
            logger.error("No KAgent bound; cannot countersign")
            return

        kline = msg.message
        logger.info("Countersign: %s", kline)
        self._kagent.countersign(kline)

    def _handle_save(self, msg: Message) -> None:
        """Persist Kalvin's model to disk via agent_codec.

        ``msg.message`` is the file path (or None for default).
        Sends a confirmation or error back to the sender.
        """
        if self._kagent is None:
            logger.error("No KAgent bound; cannot save")
            return

        path = msg.message or "data/agent.bin"
        try:
            self._kagent.save(path)
            logger.info("Kalvin model saved to %s", path)
            self._bus.send(Message(
                role=msg.sender or SUPERVISOR_ROLE,
                action="saved",
                message={"path": str(path)},
            ))
        except Exception as exc:
            logger.error("Failed to save Kalvin model: %s", exc)
            self._bus.send(Message(
                role=msg.sender or SUPERVISOR_ROLE,
                action="error",
                message=f"Save failed: {exc}",
            ))

    def _handle_load(self, msg: Message) -> None:
        """Load Kalvin's model from disk via agent_codec.

        ``msg.message`` is the file path (or None for default).
        Reconstructs the KAgent with the loaded model, replacing the
        current one. Sends a confirmation or error back to the sender.
        """
        if self._kagent is None:
            logger.error("No KAgent bound; cannot load")
            return

        path = msg.message or "data/agent.bin"
        try:
            from kalvin.agent_codec import AgentCodec
            model, activity = AgentCodec.load(path)

            # Replace the KAgent's model and activity in-place
            self._kagent._model = model
            self._kagent._activity = activity
            # Rebind the cogitator's model reference
            self._kagent._cogitator._model = model

            logger.info("Kalvin model loaded from %s", path)
            self._bus.send(Message(
                role=msg.sender or SUPERVISOR_ROLE,
                action="loaded",
                message={"path": str(path), "frame_size": len(model)},
            ))
        except Exception as exc:
            logger.error("Failed to load Kalvin model: %s", exc)
            self._bus.send(Message(
                role=msg.sender or SUPERVISOR_ROLE,
                action="error",
                message=f"Load failed: {exc}",
            ))
