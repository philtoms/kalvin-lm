"""Engine — orchestrator of the rationalisation pipeline.

The Engine rationalises KValues against the Memory via the fast/slow split:
  - Fast path: the rationaliser (:mod:`kalvin.rationaliser`) feeds memory —
    S1-stamped queries ground on receipt, S4 is refused, no model calls.
  - Slow path: the queued work-list items are submitted to the WorkRunner
    (:mod:`kalvin.work_runner`), which cogitates each item
    (:mod:`kalvin.cogitator`) in a background thread.

Events (fast-path grounding is silent): each cogitation emission is
published as a ``frame`` event via the adapter; the runner publishes
``done`` idle events. Serialization is the Memory state snapshot (JSON).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.events import EventBus, RationaliseEvent  # EventBus: test/dev fallback
from kalvin.kline import KLine, sig_level
from kalvin.kvalue import KValue
from kalvin.memory import Memory
from kalvin.paths import agent_bin
from kalvin.rationaliser import Rationaliser
from kalvin.significance import SIG_S1, structural_sig
from kalvin.signifier import NLPSignifier
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.tokenizer import TiktokenNotInstalledError
from kalvin.work_runner import (
    WorkHandler,
    WorkRunner,
)

__all__ = [
    # WorkHandler, WorkRunner are re-exported from
    # kalvin.work_runner (their canonical import location).
    "WorkHandler",
    "WorkRunner",
    "Engine",
    "EngineAdapter",
    "Agent",
]

_log = logging.getLogger(__name__)

# Default tokenizer factory


def _default_tokenizer() -> KTokenizer:
    """Create the default kalvin tokenizer (the sole production tokenizer).

    The kalvin tokenizer is mandatory — there is no fallback.  If the data
    files are missing, the BPE backend (tiktoken/rustbpe) cannot be loaded,
    or the data files are unreadable, this raises :class:`RuntimeError`
    instructing the user to regenerate the data via
    ``scripts/rebuild-tokenizer-data.sh``.
    """
    try:
        return BPETokenizer()
    except (FileNotFoundError, ImportError, OSError, TiktokenNotInstalledError) as exc:
        raise RuntimeError(
            "Tokenizer data is required but unavailable. "
            "Run `bash scripts/rebuild-tokenizer-data.sh` to generate data/tokenizer/."
        ) from exc


# Default signifier factory


def _default_signifier() -> KSignifier:
    """Create the default kalvin signifier (the sole production signifier)."""
    return NLPSignifier()


# EngineAdapter Protocol


@runtime_checkable
class EngineAdapter(Protocol):
    """Protocol for receiving rationalisation events from Engine.

    Any object with an ``on_event(RationaliseEvent)`` method satisfies this
    protocol.  The concrete ``EngineAdapter`` in ``harness/adapter.py`` is
    the canonical production implementation; ``EventBus`` (in ``events.py``)
    is the standard test/dev adapter.

    Note: the name ``EngineAdapter`` intentionally mirrors the concrete class
    in ``harness/adapter.py`` — that class satisfies this protocol implicitly.
    """

    def on_event(self, event: RationaliseEvent) -> None: ...


# Engine


class Engine:
    """Orchestrator of the rationalisation pipeline.

    Parameters
    ----------
    tokenizer:
        Tokenizer instance. Defaults to the kalvin BPETokenizer (the sole
        production tokenizer). Used for encoding text to nodes.
    state:
        Memory instance serving as base memory. Defaults to an empty Memory
        built on the signifier.
    signifier:
        Signifier for the default state. Defaults to the kalvin NLPSignifier.
    adapter:
        Adapter for receiving events. Must implement ``on_event(event)``.
        Required — pass an ``EventBus`` for test/dev use, or a
        ``EngineAdapter`` (from ``harness.adapter``) for production.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        state: Memory | None = None,
        signifier: KSignifier | None = None,
        *,
        adapter: EngineAdapter,
    ):
        self._tokenizer = tokenizer if tokenizer else _default_tokenizer()
        self._signifier = signifier if signifier is not None else _default_signifier()
        self._state = state if state is not None else Memory(self._signifier)

        self._adapter: EngineAdapter = adapter

        self._rationaliser = Rationaliser(self._state)
        self._runner = WorkRunner(
            state=self._state,
            adapter=self._adapter,
            handler=self,
            timeout=2.0,
        )

    # Properties

    @property
    def state(self) -> Memory:
        return self._state

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def signifier(self) -> KSignifier:
        return self._signifier

    @property
    def events(self) -> EngineAdapter:
        """The adapter, exposed for event inspection (e.g. ``.subscribe()`` on EventBus)."""
        return self._adapter

    @property
    def runner(self) -> WorkRunner:
        return self._runner

    # Rationalisation

    def rationalise(self, value: KValue) -> bool:
        """Feed memory via the rationaliser; submit the queued work-list items to the runner.

        Fast path: the rationaliser grounds S1-stamped queries on receipt and
        refuses S4. Slow path: everything else queues on the work list, and
        each queued item is submitted to the work runner, which cogitates it.

        Returns True if the fast path resolved the query, False if it was
        queued (rational).
        """
        before = len(self._state.work_list)
        self._rationaliser.rationalise([value])
        queued = self._state.work_list[before:]
        for kline in queued:
            self._runner.submit(kline)
        return not queued

    # WorkHandler protocol

    def on_emission(self, query: KLine, emission: KValue) -> None:
        """WorkHandler.on_emission: publish a frame event for a cogitation emission.

        ``query`` is the submitted work-list kline; its voice on the event
        carries the structurally derived significance (the declared
        significance does not ride the work list).
        """
        query_value = KValue(query, structural_sig(sig_level(query, self._signifier)))
        self._publish("frame", query_value, emission)

    def runner_join(self, timeout: float | None = None) -> None:
        """Stop the work runner and wait for it to finish."""
        self._runner.join(timeout)

    def runner_drain(self, timeout: float | None = None) -> bool:
        """Drain pending work items without stopping the thread.

        Returns True if drained within *timeout*, False if timed out.
        """
        return self._runner.drain(timeout)

    # Events

    def _publish(self, kind: str, query_value: KValue, proposal_value: KValue) -> None:
        """Publish a rationalisation event via the adapter.

        ``query_value`` is the query voice (the submitted or queued kline);
        ``proposal_value`` is Kalvin's assessment.
        """
        self._adapter.on_event(RationaliseEvent(kind, query_value, proposal_value))

    def countersign(self, value: KValue) -> bool:
        """Generate the reciprocal kline ({Q:[V]} → {V:[Q]}) and rationalise it.

        The reciprocal kline is wrapped in a KValue at ``SIG_S1`` — the act of
        countersigning is an S1 ratification. Requires non-empty nodes;
        returns the result of ``rationalise``.
        """
        kline = value.kline
        reciprocal_sig = self._signifier.signature_of(kline.nodes)
        reciprocal = KLine(reciprocal_sig, [kline.signature])
        reciprocal_value = KValue(reciprocal, SIG_S1)
        return self.rationalise(reciprocal_value)

    # Frame info

    def frame_size(self) -> int:
        return sum(len(bucket) for bucket in self._state.frame.values())

    # Memory rebinding

    def rebind(self, state: Memory) -> None:
        """Swap the memory this engine (and its rationaliser and runner) operate on."""
        self._state = state
        self._rationaliser = Rationaliser(state)
        self._runner._state = state  # rebind the runner's state ref

    # Serialization — the Memory state snapshot (JSON).

    def to_dict(self) -> dict:
        return self._state.to_dict()

    @classmethod
    def from_dict(cls, data: dict, adapter: EngineAdapter | None = None) -> Engine:
        state = Memory.from_dict(_default_signifier(), data)
        return cls(state=state, adapter=adapter or EventBus())

    def save(self, path: str | Path) -> None:
        self._state.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        adapter: EngineAdapter | None = None,
    ) -> Engine:
        state = Memory.load(_default_signifier(), path or str(agent_bin()))
        return cls(state=state, adapter=adapter or EventBus())


# Backward-compatible alias
Agent = Engine
