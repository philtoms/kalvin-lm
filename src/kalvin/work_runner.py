"""WorkRunner — background processor for rational work items (S2/S3).

The WorkRunner is the slow-path of the rationalisation pipeline. It is a thin
threading dispatcher: it dequeues ``WorkItem`` instances, invokes functions
from :mod:`kalvin.expand` (expand) and :mod:`kalvin.proposals`
(propose_expansions), and routes results to a ``WorkHandler``. All
significance computation lives in :mod:`kalvin.significance`; graph expansion
in :mod:`kalvin.expand`; and expansion-proposal logic in
:mod:`kalvin.proposals`.

Split out of the Engine module so the fast-path (Engine routing)
and slow-path (work-item processing) live in their own modules while sharing the seam
defined here: the Engine submits work items and is the primary
``WorkHandler``.
"""

from __future__ import annotations

import threading
import time as _time
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.expand import expand
from kalvin.kline import KDbg, KLine
from kalvin.kvalue import KValue
from kalvin.model import Model
from kalvin.proposals import propose_expansions
from kalvin.significance import SIG_S4, BandLayout

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier
    from kalvin.engine import EngineAdapter


# Cogitation Handler Protocol


@runtime_checkable
class WorkHandler(Protocol):
    """Protocol for handling work-item results.

    The WorkRunner calls these methods when it discovers significant
    results during background graph expansion.
    """

    def on_s1(self, query: KValue, candidate: KLine) -> None:
        """Called when the runner discovers an S1 (exact) result.

        ``query`` is the original inbound KValue; ``candidate`` is the
        KLine (from the model) that reached S1.
        """
        ...

    def on_expansion(
        self,
        query: KValue,
        proposal: KLine,
        significance: int,
        original_candidate: KLine | None = None,
    ) -> None:
        """Called when an expansion proposal is generated (S2/S3).

        ``query`` is the original inbound KValue; ``proposal`` is the
        expansion-proposal KLine carrying the ``expand()``-computed significance.
        """
        ...


# Work Item


class WorkItem(NamedTuple):
    """A single query|candidate pair queued for background processing.

    ``query`` is a KValue (carries the declared significance into the slow
    path); ``candidate`` is the KLine from the model; ``level`` is the
    routing classification ("S2" or "S3").
    """

    query: KValue
    candidate: KLine
    level: str  # "S2" or "S3"


# WorkRunner


class WorkRunner:
    """Background processor for rational work items (S2/S3).

    Receives individual query|candidate|level work items,
    computes deep significance (expand()), and processes results.
    Parameters
    ----------
    model:
        Model instance for distance computation and countersignature checks.
    adapter:
        Adapter for receiving events. Must implement ``on_event(event)``.
        The EventBus class satisfies this protocol via its ``on_event`` method.
    handler:
        WorkHandler implementation. Called when the runner discovers
        significant results (S1 matches and S2/S3 expansion proposals).
        The Engine is the primary implementation.
    timeout:
        Idle seconds before emitting "done" so subscribers can realign.
        Does not halt the thread. Default 2.0.
    """

    def __init__(
        self,
        model: Model,
        adapter: EngineAdapter,
        handler: WorkHandler,
        signifier: KSignifier,
        timeout: float = 2.0,
    ):
        self._model = model
        self._adapter = adapter
        self._handler = handler
        self._signifier = signifier
        self._timeout = timeout

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._backlog: list[WorkItem] = []
        self._stop = threading.Event()
        self._processing = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, item: WorkItem) -> None:
        """Queue a work item for background processing."""
        with self._condition:
            self._backlog.append(item)
            self._condition.notify()

    def join(self, timeout: float | None = None) -> None:
        """Stop the runner thread and wait for it to finish."""
        self._stop.set()
        with self._condition:
            self._condition.notify()
        self._thread.join(timeout=timeout)

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until the backlog is empty and the current work item finishes.

        Does NOT stop the thread — the WorkRunner remains alive and will
        accept new work items after draining.

        Returns True if drained within *timeout*, False if timed out.
        """
        deadline = None
        if timeout is not None:
            deadline = _time.monotonic() + timeout

        while True:
            with self._condition:
                if not self._backlog and not self._processing:
                    return True
                self._condition.wait(timeout=0.5)

            if deadline is not None and _time.monotonic() >= deadline:
                return False

    def _run(self) -> None:
        """Background thread: process work items."""
        idle_time = 0.0
        while not self._stop.is_set():
            with self._condition:
                while not self._backlog and not self._stop.is_set():
                    self._condition.wait(timeout=0.5)
                    idle_time += 0.5
                    if idle_time >= self._timeout:
                        done_k = KLine(0, [], dbg=KDbg(label="done"))
                        done_value = KValue(done_k, SIG_S4)
                        self._adapter.on_event(RationaliseEvent("done", done_value, done_value))
                        idle_time = 0.0
                idle_time = 0.0
                if self._stop.is_set() and not self._backlog:
                    return
                self._processing = True
                item = self._backlog.pop(0)

            self._run_work_item(item)
            with self._condition:
                self._processing = False
                self._condition.notify_all()

    def _run_work_item(self, item: WorkItem) -> None:
        """Expand a work item, classifying each yield against boundaries.

        Work items arrive routed as S2 or S3 only (see Engine._route). The
        pair is expanded and each yield classified; a terminal S1 (distance
        1) discovered during expansion is a genuine structural exact match
        and triggers ``on_s1``.

        ``item.query`` is a KValue (the original inbound); the model API
        (``expand``) stays KLine-based, so ``query_kline`` is extracted here.
        """
        query_value, candidate, level = item
        query_kline = query_value.kline

        layout = BandLayout()

        for kv in expand(self._model, query_kline, candidate, self._signifier):
            band = layout.classify(kv.significance)

            if band == "S4":
                continue

            if band == "S1":
                self._handler.on_s1(query_value, candidate)
                break
            else:
                # kv.kline is the expanded (possibly misfit) candidate.
                # The query voice on the published event is the WorkItem's
                # original inbound KValue.
                for proposal, sig in propose_expansions(
                    self._model, kv.kline, kv.significance, self._signifier
                ):
                    self._handler.on_expansion(
                        query_value,
                        proposal,
                        sig,
                        original_candidate=kv.kline,
                    )
