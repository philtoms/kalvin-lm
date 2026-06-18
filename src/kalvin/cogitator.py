"""Cogitator — background processor for rational work items (S2/S3).

The Cogitator is the slow-path of the rationalisation pipeline (see
@agent spec, §Cogitation). It is a thin threading dispatcher: it dequeues
``WorkItem`` instances, invokes functions from :mod:`kalvin.expand`
(boundaries, classify, expand, propose_expansions), and routes results to a
``CogitationHandler``. All significance computation and expansion-proposal
logic lives in :mod:`kalvin.expand`.

Split out of ``agent.py`` so the fast-path (agent routing) and slow-path
(cogitation) live in their own modules while sharing the seam defined here.

See specs/cogitator.md for the cogitator specification.
See specs/agent.md for the agent specification (the seam: the agent submits
work items and is the primary CogitationHandler).
See specs/cogitator-drain.md for inter-lesson drain semantics.
"""

from __future__ import annotations

import threading
import time as _time
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.expand import boundaries, classify, expand, propose_expansions
from kalvin.kline import KDbg, KLine
from kalvin.model import Model

if TYPE_CHECKING:
    from kalvin.agent import KAgentAdapter


# Cogitation Handler Protocol


@runtime_checkable
class CogitationHandler(Protocol):
    """Protocol for handling cogitation results.

    The Cogitator calls these methods when it discovers significant
    results during background graph expansion.
    """

    def on_s1(self, query: KLine, candidate: KLine) -> None:
        """Called when cogitation discovers an S1 (exact) result."""
        ...

    def on_expansion(
        self,
        query: KLine,
        proposal: KLine,
        significance: int,
        original_candidate: KLine | None = None,
    ) -> None:
        """Called when an expansion proposal is generated (S2/S3)."""
        ...


# Work Item


class WorkItem(NamedTuple):
    """A single query|candidate pair queued for cogitation."""

    query: KLine
    candidate: KLine
    level: str  # "S2" or "S3"


# Cogitator


class Cogitator:
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
        CogitationHandler implementation. Called when cogitation discovers
        significant results (S1 matches and S2/S3 expansion proposals).
        The KAgent is the primary implementation.
    timeout:
        Idle seconds before emitting "done" so subscribers can realign.
        Does not halt the thread. Default 2.0.
    """

    def __init__(
        self,
        model: Model,
        adapter: KAgentAdapter,
        handler: CogitationHandler,
        timeout: float = 2.0,
    ):
        self._model = model
        self._adapter = adapter
        self._handler = handler
        self._timeout = timeout

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._backlog: list[WorkItem] = []
        self._stop = threading.Event()
        self._processing = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, item: WorkItem) -> None:
        """Queue a work item for background cogitation."""
        with self._condition:
            self._backlog.append(item)
            self._condition.notify()

    def join(self, timeout: float | None = None) -> None:
        """Stop the cogitation thread and wait for it to finish."""
        self._stop.set()
        with self._condition:
            self._condition.notify()
        self._thread.join(timeout=timeout)

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until the backlog is empty and the current work item finishes.

        Does NOT stop the thread — the Cogitator remains alive and will
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
                        self._adapter.on_event(RationaliseEvent("done", done_k, done_k, 0))
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

        Work items arrive routed as S2 or S3 only (see KAgent._route). The
        pair is expanded and each yield classified; a terminal S1 (distance
        1) discovered during expansion is a genuine structural exact match
        and triggers ``on_s1``.
        """
        query, candidate, level = item

        s12, s23, s34 = boundaries()

        for qc in expand(self._model, query, candidate):
            band = classify(qc.significance, s12, s23, s34)

            if band == "S4":
                continue

            if band == "S1":
                self._handler.on_s1(query, candidate)
                break
            else:
                # qc.candidate is the expanded (possibly misfit) candidate;
                # qc.query is the correct query context for connotation yields.
                for proposal, sig in propose_expansions(self._model, qc.candidate, qc.significance):
                    self._handler.on_expansion(
                        qc.query,
                        proposal,
                        sig,
                        original_candidate=qc.candidate,
                    )
