"""Table-driven stub KAgent — a deterministic contract double for :class:`KAgent`.

Spec: ``@specs/stub-kagent.md``. The stub exists to develop the trainer's paced-loop
and satisfaction logic independently of real-Kalvin cogitation. It implements the
same ``_KAgentLike`` surface as :class:`~kalvin.agent.KAgent` but, instead of
rationalising, it emits ``RationaliseEvent``s from an authored **Response Table**:
on each submission it looks up the matching row and emits that row's ``requests``
(``frame`` at ``SIG_S4``), ``grounds`` (``frame`` at their structural band), and
``countersigns`` (``frame`` at ``SIG_S1``), in order.

It is a contract double, not a Kalvin: it has **no model, no cogitation, no
expansion, no misfit** (spec §Out of Scope). Wiring mirrors real ``KAgent`` —
``adapter = KAgentAdapter(bus)`` → ``stub = StubKAgent(adapter, rows)`` →
``adapter.bind(stub)`` — but the stub only needs an ``on_event`` callback, so it
defines a local :class:`_AdapterCallback` Protocol and never imports
``training.harness.adapter`` (the dependency direction is ``kalvin ← training``).

Design decisions (see ``plans/impl/stub-kagent.md`` §Design decisions):

- **No ``model`` attribute.** The harness adapter guards STM pre-registration with
  ``hasattr(kagent, "model")``; a model-less stub is correctly skipped.
- **``"initial"`` trigger fires on the first ``rationalise()`` call**, before kline
  matching, and consumes the row. Do not also give the first submission a
  concrete-trigger row — one row per submission.
- **``grounded`` is observational, not behavioural.** Atom reuse is table-prescribed
  (spec §Atom Reuse, ST-10), so the stub never consults ``grounded`` to decide what
  to emit; it only records signatures for inspection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from kalvin.events import RationaliseEvent
from kalvin.kvalue import KValue

if TYPE_CHECKING:
    from os import PathLike


# The ``"initial"`` sentinel: a row whose trigger is this string fires on the first
# ``rationalise()`` call, regardless of the submitted kline (spec §Definitions).
INITIAL = "initial"


class _AdapterCallback(Protocol):
    """The adapter surface the stub calls back on.

    Mirrors :class:`~kalvin.agent.KAgentAdapter` (the real KAgent defines its own
    equivalent Protocol). The stub only needs ``on_event``; it does not import the
    concrete harness adapter.
    """

    def on_event(self, event: RationaliseEvent) -> None: ...


@dataclass(frozen=True)
class ResponseRow:
    """One row of the Response Table — the stub's prescribed response to a trigger.

    The row fires at most once (spec §Behavioural Rules 3). On fire, the stub emits
    ``requests``, then ``grounds``, then ``countersigns`` (spec §rationalise, ST-7),
    each as a ``frame`` event whose ``query`` is the submitted KValue and whose
    ``proposal`` is the table-authored KValue.

    Attributes
    ----------
    trigger:
        The KValue whose kline this row responds to, matched by KLine equality
        (KV-2 — significance ignored), **or** the sentinel :data:`INITIAL`
        (``"initial"``) to fire on the first ``rationalise()`` call.
    requests:
        KValues the stub proposes back at ``SIG_S4`` ("I'm missing this operand").
    grounds:
        KValues the stub grounds at their structural band (``SIG_S2`` canon,
        ``SIG_S3`` relation, ``SIG_S4`` atom). The band is read off each KValue's
        own significance — the table authors it.
    countersigns:
        KValues the stub self-ratifies at ``SIG_S1`` (a primary's completion).
    """

    trigger: KValue | str
    requests: tuple[KValue, ...] = ()
    grounds: tuple[KValue, ...] = ()
    countersigns: tuple[KValue, ...] = ()


class StubKAgent:
    """Table-driven deterministic stand-in for :class:`~kalvin.agent.KAgent`.

    Construction mirrors real ``KAgent``::

        adapter = KAgentAdapter(bus)
        stub = StubKAgent(adapter, rows)
        adapter.bind(stub)

    Parameters
    ----------
    adapter:
        Anything with an ``on_event(event)`` method — typically a
        :class:`~training.harness.adapter.KAgentAdapter`. The stub calls
        ``adapter.on_event`` for each emission.
    rows:
        The Response Table. At most one row may carry the :data:`INITIAL` trigger;
        duplicates among concrete triggers collapse by KLine equality (KV-2).
    """

    def __init__(
        self,
        adapter: _AdapterCallback,
        rows: Sequence[ResponseRow],
    ) -> None:
        self._adapter = adapter

        initial_rows: list[ResponseRow] = []
        pending: dict[KValue, ResponseRow] = {}
        for r in rows:
            if isinstance(r.trigger, str):
                if r.trigger != INITIAL:
                    raise ValueError(
                        f"invalid string trigger {r.trigger!r}; "
                        f"only {INITIAL!r} is allowed"
                    )
                initial_rows.append(r)
            else:
                # Keyed by the trigger KValue. KValue hashes/compares by kline
                # only (KV-2), so a submitted KValue matches a row whose trigger
                # is the same kline regardless of either side's significance.
                pending[r.trigger] = r

        if len(initial_rows) > 1:
            raise ValueError(
                f"at most one {INITIAL!r} row is allowed; got {len(initial_rows)}"
            )

        self._initial_row: ResponseRow | None = initial_rows[0] if initial_rows else None
        self._pending_rows: dict[KValue, ResponseRow] = pending

        self._fired: list[KValue] = []          # triggers fired, in fire order
        self._grounded: set[int] = set()         # grounded kline signatures
        self._first_call = True

    # ── _KAgentLike ────────────────────────────────────────────────────────

    def rationalise(self, value: KValue) -> bool:
        """Emit the row prescribed for ``value`` (if any), then return ``True``.

        On the first call, if an :data:`INITIAL` row exists it fires and consumes
        that row (spec §Definitions, §Single Cascade). Otherwise the submission is
        matched against pending rows by KLine equality (KV-2). A match fires the
        row's ``requests`` → ``grounds`` → ``countersigns`` in order and consumes
        it; no match returns ``True`` silently (spec §Divergences, ST-6).
        """
        if self._first_call and self._initial_row is not None:
            self._emit(self._initial_row, value)
            self._first_call = False
            return True

        self._first_call = False

        row = self._pending_rows.pop(value, None)
        if row is None:
            # Table/trainer divergence (spec §Divergences): nothing prescribed.
            # Recovery is the trainer's concern, not the stub's.
            return True

        self._emit(row, value)
        self._fired.append(value)
        return True

    def countersign(self, value: KValue) -> bool:
        """No-op returning ``True`` (spec §countersign, ST-8).

        The trainer ratifies by *submission*, not countersign (the
        absent-until-ratify decision). ``countersign`` exists only to satisfy
        ``_KAgentLike``.
        """
        return True

    def save(self, path: str | bytes | PathLike[str] | PathLike[bytes], format=None) -> None:
        """No-op. The stub has no persistent model (spec §save / codec)."""
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
    def fired(self) -> tuple[KValue, ...]:
        """Triggers fired so far, in fire order (each row fires at most once)."""
        return tuple(self._fired)

    @property
    def grounded(self) -> frozenset[int]:
        """Signatures of klines the stub has grounded or ratified.

        Observational only (spec §Atom Reuse, ST-10): the stub never consults
        this set to decide what to emit — atom reuse is table-prescribed.
        """
        return frozenset(self._grounded)

    # ── internals ──────────────────────────────────────────────────────────

    def _emit(self, row: ResponseRow, query: KValue) -> None:
        """Emit a row's requests, grounds, then countersigns (spec §rationalise).

        Every emission is a ``frame`` event (spec §Event Kinds the Stub Emits);
        significance is carried on ``proposal.significance``, never on the event
        (ST-11). ``query`` is always the submitted KValue so the adapter's
        sender-map lookup (keyed on ``event.query.kline``) routes the response
        back to the original sender.
        """
        kind = "frame"
        for req in row.requests:
            self._adapter.on_event(RationaliseEvent(kind, query, req))
        for ground in row.grounds:
            self._adapter.on_event(RationaliseEvent(kind, query, ground))
            self._grounded.add(ground.kline.signature)
        for cs in row.countersigns:
            self._adapter.on_event(RationaliseEvent(kind, query, cs))
            self._grounded.add(cs.kline.signature)  # ratified → grounded
