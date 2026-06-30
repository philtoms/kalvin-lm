"""KAgent adapter — bridge between the KAgent pipeline and the message bus.

Receives harness messages sent to role ``trainee``, compiles KScript source
into entries, submits them one at a time to :meth:`KAgent.rationalise`, and
routes KAgent callbacks back to the original sender via the bus.

The adapter implements the :class:`Participant` protocol (to receive bus
messages) and provides ``on_event()`` so KAgent can call it directly as
its adapter callback (replacing the internal EventBus).

KValue exchange
---------------
The bus exchanges **KValues** — a KLine paired with a sender's significance
assessment (@kvalue spec §Definition) — not bare KLines:

- ``submit`` compiles KScript source via :func:`compile_source` (which now
  returns ``list[KValue]``) and forwards each KValue to
  :meth:`KAgent.rationalise`. The Model API stays KLine-based (plan D2):
  STM pre-registration and the sender-map key read ``entry.kline``.
- ``countersign`` materialises the inbound bus payload to a :class:`KValue`
  (see :func:`_materialise_kvalue`) and forwards it to
  :meth:`KAgent.countersign`.
- ``on_event`` reads the sender-map key off ``event.query.kline``. The
  event's ``query``/``proposal`` are KValues that carry their own
  significance (@kvalue spec KE-3); there is no top-level significance field.

Countersign bus payload contract (shared with the Reactor, KB-356)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``countersign`` action's payload may arrive in three forms (see
:func:`_materialise_kvalue`): a live :class:`KValue` (the in-process
auto-countersign path), a wire dict ``{signature, nodes, significance}``,
or a legacy bare :class:`KLine` (wrapped at :data:`SIG_S1`, per KP-2:
countersign is an S1 ratification). Significance rides on the KValue;
recovering a significance lost at a future remote (WebSocket) boundary is
explicitly **out of scope** (@kvalue spec §What a KValue is Not).

Thread model
------------
``on_message`` executes on the bus dispatch thread.
``on_event`` is called from the Cogitator background thread via
``KAgent._publish`` → ``adapter.on_event``.

Model access: :class:`~kalvin.model.Model` and :class:`~kalvin.stm.STM` are
internally guarded by re-entrant locks, so any model read performed
here in ``on_event`` — or by any other subscriber — observes a consistent,
atomic snapshot and needs no adapter-level locking. (The rationalisation
*event count* still depends on the async Cogitator's processing timing but
every individual model operation is atomic and deadlock-free.)

Sender map: the sender map (a plain dict) is written on the bus thread and read
on the Cogitator thread.  Under CPython's GIL, individual dict reads/writes are
atomic, so no explicit locking is needed.  If the adapter is ever used outside
CPython, a ``threading.Lock`` should be added around sender-map access.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.paths import agent_bin
from ks.compiler import compile_source
from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from training.harness.message import Message

if TYPE_CHECKING:
    pass


# Protocol for the kagent parameter — avoids importing KAgent (circular dep)
# while giving mypy the methods we actually call.
class _KAgentLike(Protocol):
    def rationalise(self, value: KValue) -> bool: ...
    def countersign(self, value: KValue) -> bool: ...
    def save(self, path, format=None) -> None: ...
    def codec(self) -> object: ...


logger = logging.getLogger(__name__)

# Type alias for the sender-map key: (signature, frozen_nodes).
EntryKey = tuple[int, tuple[int, ...]]


def _materialise_kvalue(obj: object) -> KValue:
    """Materialise an inbound countersign payload to a :class:`KValue`.

    This is the inbound mirror of the harness's outbound encoder: a
    countersign frame that traversed the WebSocket arrives as a plain
    ``dict`` — the canonical KValue wire shape
    ``{"signature": int, "nodes": list[int], "significance": int}``.

    Three input forms are accepted (see the §Countersign Bus Payload Contract
    in the module docstring):

    - **Live :class:`KValue`** — returned unchanged. This is the in-process
      production path: the reactor's auto-countersign posts the proposal
      KValue directly on the bus with no WebSocket round-trip.
    - **Wire dict** ``{"signature", "nodes", "significance"}`` — built into
      ``KValue(KLine(obj["signature"], obj["nodes"]), obj["significance"])``.
      A dict missing ``significance`` raises ``TypeError`` — fail-loud,
      mirroring the existing philosophy.
    - **Legacy :class:`KLine`** — wrapped at :data:`SIG_S1`
      (``KValue(kline, SIG_S1)``). Per @kvalue spec KP-2 the act of
      countersigning is an S1 ratification. Preserves backwards compatibility
      with any caller still sending a bare KLine.

    Any other type raises ``TypeError`` so malformed payloads fail loudly
    rather than being coerced into something meaningless.

    .. note::
       A future *remote* supervisor that loses significance at the WebSocket
       boundary is explicitly **out of scope** — no significance recovery is
       attempted here. Significance rides on the KValue (@kvalue spec KE-3);
       today every participant that exchanges KValues is embedded in-process.
    """
    if isinstance(obj, KValue):
        return obj
    if isinstance(obj, KLine):
        return KValue(obj, SIG_S1)
    if isinstance(obj, dict):
        if "significance" not in obj:
            raise TypeError(
                "countersign wire dict missing required 'significance' key; "
                f"got keys {sorted(obj.keys())!r}"
            )
        return KValue(KLine(obj["signature"], obj["nodes"]), obj["significance"])
    raise TypeError(
        f"countersign payload must be a KValue, KLine, or wire dict, got {type(obj).__name__}"
    )


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
        kagent: _KAgentLike | None = None,
        tokenizer: KTokenizer | None = None,
        signifier: KSignifier | None = None,
    ):
        self._bus = bus
        self._role = role
        self._kagent: _KAgentLike | None = kagent
        self._tokenizer: KTokenizer | None = tokenizer
        self._signifier: KSignifier | None = signifier

        # Sender map: (signature, frozen_nodes) → sender role. Written in
        # on_message (bus thread), read in on_event (cogitator thread).
        self._sender_map: dict[EntryKey, str] = {}

        bus.subscribe(self._role, self.on_message)

    # Properties

    @property
    def role(self) -> str:
        """The bus role this adapter is subscribed to."""
        return self._role

    @property
    def kagent(self) -> _KAgentLike | None:
        """The KAgent instance, or ``None`` if not yet bound."""
        return self._kagent

    @property
    def tokenizer(self) -> KTokenizer | None:
        """The tokenizer used for KScript compilation, or ``None`` if unset."""
        return self._tokenizer

    # Late binding

    def bind(self, kagent: _KAgentLike) -> None:
        """Bind a KAgent instance after construction.

        Use this when the KAgent needs the adapter as its callback during
        its own construction, creating a circular dependency.
        """
        self._kagent = kagent

    # Participant protocol

    def on_message(self, msg: Message) -> None:
        """Handle an incoming bus message sent to this adapter's role.

        Actions
        -------
        submit:
            Compile KScript source from ``msg.message``, record sender per
            entry in the sender map, and call ``kagent.rationalise(entry)``
            for each compiled entry.
        countersign:
            Call ``kagent.countersign(kvalue)``. The payload in ``msg.message``
            is materialised to a :class:`KValue` at this inbound boundary
            (see :func:`_materialise_kvalue`) — it may arrive as a live
            KValue, a wire dict, or a legacy bare KLine.
        rationalise:
            Call ``kagent.rationalise(kvalue)``. The payload is materialised
            to a :class:`KValue` and delivered straight to rationalisation —
            no recompile, no reciprocal, no forced significance. This is the
            path a participant uses to hand Kalvin a KValue with its own
            declared significance (the two-way significance dialog).
        save:
            Persist Kalvin's model to disk via agent_codec.
        load:
            Load Kalvin's model from disk via agent_codec.
        """
        if msg.action == "submit":
            self._handle_submit(msg)
        elif msg.action == "countersign":
            self._handle_countersign(msg)
        elif msg.action == "rationalise":
            self._handle_rationalise(msg)
        elif msg.action == "save":
            self._handle_save(msg)
        elif msg.action == "load":
            self._handle_load(msg)
        elif msg.action == "drain":
            self._handle_drain(msg)
        else:
            logger.warning("Unknown action %r from %s", msg.action, msg.sender)

    # Adapter callback (KAgent → adapter)

    def on_event(self, event: RationaliseEvent) -> None:
        """Receive a rationalisation event from the KAgent.

        The KAgent calls this directly (via ``_publish``) instead of using
        an internal EventBus.  The event is wrapped into a :class:`Message`
        routed to the original sender (looked up in the sender map) and
        sent via the bus.

        The event's ``query`` is a :class:`KValue` (KB-354); the sender-map
        key is read off ``event.query.kline``. There is no top-level
        ``significance`` field on the event (@kvalue spec KE-3) —
        significance rides on the KValue.

        Orphan events (no sender in the map, e.g. "done" idle events from
        the Cogitator) are silently dropped.
        """
        key: EntryKey = (event.query.kline.signature, tuple(event.query.kline.nodes))
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

    # Internal handlers

    def drain(self, timeout: float | None = None) -> bool:
        """Drain pending cogitation work items from the KAgent.

        Returns True if drained within *timeout*, False if timed out.
        No-op if no KAgent is bound.
        """
        if self._kagent is None:
            return True
        return self._kagent.cogitate_drain(timeout)

    def _handle_submit(self, msg: Message) -> None:
        """Compile KScript source and submit each entry to KAgent."""
        if self._kagent is None:
            logger.error("No KAgent bound; cannot submit")
            return

        try:
            entries = compile_source(
                msg.message, tokenizer=self._tokenizer, signifier=self._signifier
            )
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
        # Pre-register all entries in STM so countersign pairs (e.g. from
        # `M == H` compiling to {M: H} and {H: M}) can find each other
        # during rationalise(). The Model API stays KLine-based (D2): pass
        # ``entry.kline`` at the boundary, never the KValue.
        if hasattr(self._kagent, "model"):
            for entry in entries:
                self._kagent.model.add_to_stm(entry.kline)
        for entry in entries:
            key: EntryKey = (entry.kline.signature, tuple(entry.kline.nodes))
            self._sender_map[key] = msg.sender or ""
            # rationalise takes a KValue (KB-354); the agent reads value.kline.
            self._kagent.rationalise(entry)  # fire-and-forget; events come via on_event

    def _handle_countersign(self, msg: Message) -> None:
        """Forward a countersign request to the KAgent.

        The payload in ``msg.message`` is materialised to a :class:`KValue`
        via :func:`_materialise_kvalue` and handed to
        :meth:`KAgent.countersign`, which consumes a KValue. Three payload
        forms are accepted (see :func:`_materialise_kvalue`): a live
        :class:`KValue` (the in-process auto-countersign path), a canonical
        wire dict ``{"signature", "nodes", "significance"}``, or a legacy
        bare :class:`KLine` (wrapped at :data:`SIG_S1`).
        """
        if self._kagent is None:
            logger.error("No KAgent bound; cannot countersign")
            return

        kvalue = _materialise_kvalue(msg.message)
        logger.info("Countersign: %s", kvalue)
        self._kagent.countersign(kvalue)

    def _handle_rationalise(self, msg: Message) -> None:
        """Deliver a participant-constructed KValue straight to rationalisation.

        The payload in ``msg.message`` is materialised to a :class:`KValue`
        via :func:`_materialise_kvalue` and handed to
        :meth:`KAgent.rationalise`. Unlike ``submit`` (which recompiles
        KScript source, re-deriving significance from structure) and
        ``countersign`` (which builds the reciprocal kline at SIG_S1), this
        action delivers the KValue as-is — the significance on the KValue is
        the sender's declared assessment, carried straight into the
        significance-comparison gate (@agent spec §Rationalisation).

        Three payload forms are accepted, same as ``countersign`` (see
        :func:`_materialise_kvalue`): a live :class:`KValue`, a wire dict
        ``{"signature", "nodes", "significance"}``, or a legacy bare
        :class:`KLine` (wrapped at SIG_S1).
        """
        if self._kagent is None:
            logger.error("No KAgent bound; cannot rationalise")
            return

        kvalue = _materialise_kvalue(msg.message)
        # Record the sender so events Kalvin emits about this kline route
        # back to the participant that handed it in. Mirrors ``_handle_submit``;
        # without this, a paced KValue submission's events would be orphaned
        # (on_event drops anything absent from the sender map). The recurrence
        # drop-signal path is unaffected: the KAgent drops those (no events),
        # so the extra entry is simply unused.
        key: EntryKey = (kvalue.kline.signature, tuple(kvalue.kline.nodes))
        self._sender_map[key] = msg.sender or ""
        logger.info("Rationalise (direct): %s", kvalue)
        self._kagent.rationalise(kvalue)  # fire-and-forget; events via on_event

    def _handle_save(self, msg: Message) -> None:
        """Persist Kalvin's model to disk via agent_codec.

        ``msg.message`` is the file path (or None for default).
        Sends a confirmation or error back to the sender.
        """
        if self._kagent is None:
            logger.error("No KAgent bound; cannot save")
            return

        path = msg.message or str(agent_bin())
        try:
            self._kagent.save(path)
            logger.info("Kalvin model saved to %s", path)
            self._bus.send(
                Message(
                    role=msg.sender or SUPERVISOR_ROLE,
                    action="saved",
                    message={"path": str(path)},
                )
            )
        except Exception as exc:
            logger.error("Failed to save Kalvin model: %s", exc)
            self._bus.send(
                Message(
                    role=msg.sender or SUPERVISOR_ROLE,
                    action="error",
                    message=f"Save failed: {exc}",
                )
            )

    def _handle_load(self, msg: Message) -> None:
        """Load Kalvin's model from disk via agent_codec.

        ``msg.message`` is the file path (or None for default).
        Reconstructs the KAgent with the loaded model, replacing the
        current one. Sends a confirmation or error back to the sender.
        """
        if self._kagent is None:
            logger.error("No KAgent bound; cannot load")
            return

        path = msg.message or str(agent_bin())
        try:
            from kalvin.agent_codec import AgentCodec

            model, activity = AgentCodec.load(path)

            self._kagent._model = model
            self._kagent._activity = activity
            self._kagent._cogitator._model = model  # rebind cogitator's model ref

            logger.info("Kalvin model loaded from %s", path)
            self._bus.send(
                Message(
                    role=msg.sender or SUPERVISOR_ROLE,
                    action="loaded",
                    message={"path": str(path), "frame_size": len(model)},
                )
            )
        except Exception as exc:
            logger.error("Failed to load Kalvin model: %s", exc)
            self._bus.send(
                Message(
                    role=msg.sender or SUPERVISOR_ROLE,
                    action="error",
                    message=f"Load failed: {exc}",
                )
            )

    def _handle_drain(self, msg: Message) -> None:
        """Drain pending cogitation work items.

        Waits for the Cogitator to finish processing all queued work items,
        then responds with a confirmation. This ensures that events from
        previous lessons are fully processed before a new lesson starts.
        """
        timeout = msg.message if isinstance(msg.message, (int, float)) else None
        if self._kagent is None:
            logger.debug("Drain: no KAgent bound — responding immediately")
        else:
            logger.info("Draining cogitator...")
            drained = self._kagent.cogitate_drain(timeout=timeout or 30.0)
            if drained:
                logger.info("Cogitator drained")
            else:
                logger.warning("Cogitator drain timed out")

        self._bus.send(
            Message(
                role=msg.sender or SUPERVISOR_ROLE,
                action="drained",
                message={"status": "ok"},
            )
        )
