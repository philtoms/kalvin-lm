"""WorkRunner — background runner for work-list items.

The Engine feeds memory via the rationaliser: the fast path resolves
directly, the rest queue on the work list. Those queued work-list items are
submitted here. Each item runs one pass of cogitation
(:func:`kalvin.cogitator.cogitate`) over the shared :class:`Memory`; the
pass's emissions are routed to a ``WorkHandler``.
"""

from __future__ import annotations

import threading
import time as _time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kalvin.cogitator import cogitate
from kalvin.events import RationaliseEvent
from kalvin.kline import KDbg, KLine
from kalvin.kvalue import KValue
from kalvin.memory import Memory
from kalvin.significance import SIG_S4

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.engine import EngineAdapter


# Work Handler Protocol


@runtime_checkable
class WorkHandler(Protocol):
    """Protocol for handling cogitation emissions.

    The WorkRunner calls these methods for the emissions a cogitation
    pass produces while working a submitted item.
    """

    def on_emission(self, query: KLine, emission: KValue) -> None:
        """Called once per emission of the cogitation pass working ``query``.

        ``query`` is the submitted work-list kline; ``emission`` is a
        cogitation emission (a proposal KValue carrying its own
        significance).
        """
        ...


# WorkRunner


class WorkRunner:
    """Background runner for work-list items.

    Receives queued work-list klines, runs one cogitation pass over the
    shared memory per item, and routes each emission to the handler.

    Parameters
    ----------
    state:
        The shared Memory the cogitation passes operate on.
    adapter:
        Adapter for receiving events ("done" idle events). Must implement
        ``on_event(event)``.
    handler:
        WorkHandler implementation — called with each cogitation emission.
        The Engine is the primary implementation.
    timeout:
        Idle seconds before emitting "done" so subscribers can realign.
        Does not halt the thread. Default 2.0.
    """

    def __init__(
        self,
        state: Memory,
        adapter: EngineAdapter,
        handler: WorkHandler,
        timeout: float = 2.0,
    ):
        self._state = state
        self._adapter = adapter
        self._handler = handler
        self._timeout = timeout

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._backlog: list[KLine] = []
        self._stop = threading.Event()
        self._processing = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, kline: KLine) -> None:
        """Queue a work-list kline for background cogitation."""
        with self._condition:
            self._backlog.append(kline)
            self._condition.notify()

    def join(self, timeout: float | None = None) -> None:
        """Stop the runner's thread and wait for it to finish."""
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
        """Background thread: run one cogitation pass per submitted item."""
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

    def _run_work_item(self, kline: KLine) -> None:
        """One cogitation pass over the work list, per submitted item.

        The pass operates on the shared memory's work list (the submitted
        kline among its entries); each emission goes to the handler, with
        the submitted kline as its query voice.
        """
        for emission in cogitate(self._state):
            self._handler.on_emission(kline, emission)
