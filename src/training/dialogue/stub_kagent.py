"""Self-cursored table-driven StubKAgent (spec ``@specs/stub-kagent.md`` ST-1..13).

A deterministic contract double for :class:`kalvin.agent.KAgent`. Instead of
rationalising, the stub **emits its prescribed K-turns from a shared dialogue
table**, advancing its own cursor over the table's K-rows on each
:meth:`rationalise` call. It is **one of the two self-cursored actors** in the
training loop (``@specs/dialogue-driven-training.md`` §Training Loop); the
trainer-side loop is the other.

The stub is a contract double, not a Kalvin: no model, no cogitation, no
expansion, no misfit. The dialogue table scripts it deterministically. It never
matches the submitted kline — it emits its next K-run from its cursor (ST-5,
ST-12). When real Kalvin is reconciled to the same table, the stub is retired.

Wiring mirrors real ``KAgent`` (spec §Definition)::

    adapter = KAgentAdapter(bus)
    stub = StubKAgent(adapter, kturns)
    adapter.bind(stub)

``kturns`` is the ``actor == "K"`` subsequence of the pre-decoded dialogue table
(:func:`training.dialogue.decoder.decode`). The stub holds its own cursor and
emits its next K-run on each ``rationalise`` call.
"""

from __future__ import annotations

from collections.abc import Sequence

from kalvin.events import RationaliseEvent
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn

# Every stub emission is a ``frame`` event whose significance is read off
# ``proposal.significance`` (ST-11). The legacy ``ground`` kind is reserved for
# the future S1-only reading; until the global event-kind change lands, the
# trainer's satisfaction logic keys on significance, not on ``kind``.
_EVENT_KIND = "frame"


class StubKAgent:
    """Self-cursored reader of the shared dialogue table's K-rows (ST-1..13).

    Parameters
    ----------
    adapter:
        Anything with an ``on_event(event)`` method — typically a
        :class:`training.harness.adapter.KAgentAdapter`. The stub calls
        ``adapter.on_event`` for each emission.
    kturns:
        The stub's ordered K-rows: the ``actor == "K"`` subsequence of the
        pre-decoded dialogue table. Construction asserts the cursor starts at 0.
    """

    def __init__(
        self,
        adapter,  # _AdapterCallback: has on_event(event). Imported lazily to keep
        # the dependency direction kalvin <- training (the stub never imports the
        # concrete harness adapter).
        kturns: Sequence[DecodedTurn],
    ) -> None:
        self._adapter = adapter
        self._kturns: tuple[DecodedTurn, ...] = tuple(kturns)
        self._cursor = 0  # index of the next K-row to emit (ST-5: starts at 0)
        self._grounded: set[int] = set()  # observational only (ST-10)

    # ── _KAgentLike ────────────────────────────────────────────────────────

    def rationalise(self, value: KValue) -> bool:
        """Emit the next K-run from the cursor (ST-2, ST-3, ST-4, ST-5, ST-13).

        1. ``value`` is recorded as the current submission — it supplies the
           ``query`` voice of every event this call emits (ST-13).
        2. If the cursor is at end, return ``True`` with no events: the normal
           end-of-run signal. Whether the trainer-side was also exhausted is the
           loop's dual-exhaustion gate, not the stub's (ST-6).
        3. Otherwise advance the cursor through the current K-run (consecutive
           K-rows up to a run boundary or end) and emit each as a ``frame`` event
           with ``query = value``, ``proposal = <the K-row's KValue>``, in
           authored order. Add each grounded/countersigned kline's signature to
           ``grounded``.

        The stub never inspects ``value`` to decide what to emit (ST-5);
        ``value`` only supplies the ``query`` voice.
        """
        if self._cursor >= len(self._kturns):
            return True  # ST-6: normal end-of-run signal

        # Consume the current K-run: consecutive K-rows. For the canonical 1:1
        # table a K-run is a single row; the greedy model handles longer runs
        # identically (the table's structure delimits runs by actor change).
        while self._cursor < len(self._kturns):
            kturn = self._kturns[self._cursor]
            self._cursor += 1
            self._emit(kturn, value)
            # A run boundary is the end of consecutive K-rows. For the canonical
            # 1:1 table every K-row is its own run; we emit one and stop, so the
            # trainer-side gets to validate and compute its next T before the
            # next K-run. (Multi-row K-run timing is deferred per spec §Out of
            # Scope; the 1:1 table makes this loop emit exactly one row.)
            break

        return True

    def countersign(self, value: KValue) -> bool:
        """No-op returning ``True`` (ST-8).

        The stub does not self-ratify in response to trainer countersigns in the
        bootstrap dialogue (ratification is by *submission*, not countersign).
        Exists only to satisfy ``_KAgentLike``.
        """
        return True

    def save(self, path, format=None) -> None:
        """No-op. The stub has no persistent model."""
        return None

    def codec(self) -> None:
        """Placeholder returning ``None`` (spec §save / codec)."""
        return None

    # ── adapter-compat ─────────────────────────────────────────────────────
    #
    # Not part of _KAgentLike, but the harness adapter's drain() / drain action
    # call ``kagent.cogitate_drain(...)`` unconditionally. The stub resolves
    # everything synchronously inside rationalise(), so there is never pending
    # work — a no-op ``True`` makes it a drop-in for any draining code path.

    def cogitate_drain(self, timeout: float | None = None) -> bool:
        """No-op returning ``True`` — the stub has no async cogitation to drain."""
        return True

    # ── inspection ─────────────────────────────────────────────────────────

    @property
    def cursor(self) -> int:
        """Index of the next K-row to emit (ST-5)."""
        return self._cursor

    @property
    def exhausted(self) -> bool:
        """True when every K-row has been emitted (the dual-exhaustion gate)."""
        return self._cursor >= len(self._kturns)

    @property
    def grounded(self) -> frozenset[int]:
        """Signatures of klines the stub has grounded or ratified.

        Observational only (ST-10): the stub never consults this set to decide
        what to emit — atom reuse is table-prescribed.
        """
        return frozenset(self._grounded)

    # ── internals ──────────────────────────────────────────────────────────

    def _emit(self, kturn: DecodedTurn, query: KValue) -> None:
        """Emit one K-row as a ``frame`` event (ST-2, ST-3, ST-4, ST-11, ST-13).

        ``query`` is always the submitted KValue (ST-13) so the adapter's
        sender-map lookup (keyed on ``event.query.kline``) routes the response
        back to the original sender. Significance is carried on
        ``proposal.significance``, never on the event ``kind`` (ST-11).
        """
        event = RationaliseEvent(
            kind=_EVENT_KIND,
            query=query,
            proposal=kturn.value,
        )
        self._adapter.on_event(event)
        self._grounded.add(kturn.value.kline.signature)
