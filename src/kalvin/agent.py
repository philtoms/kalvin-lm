"""KAgent — orchestrator of the rationalisation pipeline.

The KAgent rationalises KLines against the Model using a fast/slow split:
  - Fast path: routing (node membership) — no model calls. S1/S4 resolve instantly.
  - Slow path: cogitation — expand() per work item in a background thread.

The Cogitator (slow path) lives in :mod:`kalvin.cogitator`; this module
imports and wires it. All significance computation and expansion-proposal
logic lives in :mod:`kalvin.expand`.

Serialization is delegated to the AgentCodec module (see agent_codec.py).

See specs/agent.md for the agent specification.
See specs/cogitator.md for the cogitator specification.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.agent_codec import AgentCodec
from kalvin.cogitator import (
    CogitationHandler,
    Cogitator,
    WorkItem,
)
from kalvin.events import EventBus, RationaliseEvent  # EventBus: test/dev fallback
from kalvin.expand import (
    SIG_S1,
    SIG_S4,
    is_countersigned,
    is_s1,
    promote_participating,
)
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.model import Model
from kalvin.signifier import NLPSignifier
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.tokenizer import TiktokenNotInstalledError

__all__ = [
    # CogitationHandler, Cogitator, WorkItem are re-exported from
    # kalvin.cogitator (their canonical import location).
    "CogitationHandler",
    "Cogitator",
    "WorkItem",
    "KAgent",
    "KAgentAdapter",
    "Agent",
]

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
        return NLPTokenizer()
    except (FileNotFoundError, ImportError, OSError, TiktokenNotInstalledError) as exc:
        raise RuntimeError(
            "Tokenizer data is required but unavailable. "
            "Run `bash scripts/rebuild-tokenizer-data.sh` to generate data/tokenizer/."
        ) from exc


# Default signifier factory


def _default_signifier() -> KSignifier:
    """Create the default kalvin signifier (the sole production signifier)."""
    return NLPSignifier()


# KAgentAdapter Protocol


@runtime_checkable
class KAgentAdapter(Protocol):
    """Protocol for receiving rationalisation events from KAgent.

    Any object with an ``on_event(RationaliseEvent)`` method satisfies this
    protocol.  The concrete ``KAgentAdapter`` in ``harness/adapter.py`` is
    the canonical production implementation; ``EventBus`` (in ``events.py``)
    is the standard test/dev adapter.

    Note: the name ``KAgentAdapter`` intentionally mirrors the concrete class
    in ``harness/adapter.py`` — that class satisfies this protocol implicitly.
    """

    def on_event(self, event: RationaliseEvent) -> None: ...


# KAgent


class KAgent:
    """Orchestrator of the rationalisation pipeline.

    Parameters
    ----------
    tokenizer:
        Tokenizer instance. Defaults to the kalvin NLPTokenizer (the sole
        production tokenizer). Used for encoding text to nodes.
    model:
        Model instance serving as base memory. Defaults to empty Model.
    adapter:
        Adapter for receiving events. Must implement ``on_event(event)``.
        Required — pass an ``EventBus`` for test/dev use, or a
        ``KAgentAdapter`` (from ``harness.adapter``) for production.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        model: Model | None = None,
        signifier: KSignifier | None = None,
        *,
        adapter: KAgentAdapter,
    ):
        self._tokenizer = tokenizer if tokenizer else _default_tokenizer()
        self._signifier = signifier if signifier is not None else _default_signifier()
        self._model = model if model is not None else Model(signifier=self._signifier)
        self._activity: Counter = Counter()

        self._adapter: KAgentAdapter = adapter

        self._cogitator = Cogitator(
            model=self._model,
            adapter=self._adapter,
            handler=self,
            signifier=self._signifier,
            timeout=2.0,
        )

    # Properties

    @property
    def model(self) -> Model:
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def signifier(self) -> KSignifier:
        return self._signifier

    @property
    def events(self) -> KAgentAdapter:
        """The adapter, exposed for event inspection (e.g. ``.subscribe()`` on EventBus)."""
        return self._adapter

    @property
    def cogitator(self) -> Cogitator:
        return self._cogitator

    # Routing

    @staticmethod
    def _route(query: KLine, candidate: KLine) -> str:
        """Fast classification — node-membership test only. No model call.

        Routes cogitated candidates between S2 and S3 only:
          - S2: at least one query node is a candidate node (partial or
            full overlap).
          - S3: no node overlap.

        S1 (full overlap) is intentionally NOT routed here — true S1 is a
        structural property established by ``expand()`` / ``is_s1()``, not
        by node membership. S4 (empty query) never reaches routing because
        identity klines are resolved on the fast path in ``rationalise``
        before any candidate is submitted to the cogitator.
        """
        candidate_nodes = set(candidate.nodes)
        match_count = sum(1 for n in query.nodes if n in candidate_nodes)

        if match_count > 0:
            return "S2"
        return "S3"

    # Rationalisation

    def rationalise(self, value: KValue) -> bool:
        """Rationalise a KValue into the model.

        Operates on ``value.kline`` (the objective structure) for every model
        call and routing decision — the Model API stays KLine-based (plan D2).
        ``value.significance`` (the sender's declared assessment) is carried
        through as the query voice on published events but is **not consumed**
        for behaviour (deferred — see @agent spec §Rationalisation).

        Fast path: routing (no model calls). S1/S4 resolve instantly.
        Slow path: S2/S3 queued as individual work items for cogitation.

        Returns True if significant (S1, S4), False if rational (S2, S3).
        """
        kline = value.kline
        # Prepare — callers must provide a set signature (see @specs/agent.md
        # §Phase 1). This is a presence check, not a value-test: 0 is an
        # ordinary signature value (the empty node set's signature).
        assert kline.signature is not None, (
            "KLine.signature must be set before rationalise; callers compute "
            "it via signifier.make_signature(nodes)."
        )

        # Ground check (Frame/LTM/Base only — not STM)
        if self._model.grounded(kline):
            self._model.add_to_stm(kline)
            self._publish("ground", value, KValue(kline, SIG_S1))
            return True

        if not kline.nodes:
            self._model.add_to_ltm(kline)
            self._publish("frame", value, KValue(kline, SIG_S4))  # S4
            return True

        expected_sig = self._signifier.make_signature(kline.nodes)
        if kline.signature == expected_sig:
            all_resolved = all(
                (node_kl := self._model.find(n)) is not None and self._model.grounded(node_kl)
                for n in kline.nodes
            )
            if all_resolved:
                self._model.add_to_ltm(kline)
                self._publish("frame", value, KValue(kline, SIG_S1))  # S1
                return True

        # Register in STM before the ratification check so sequential
        # countersign pairs (e.g. from `M == H` compiling to {M: H} and
        # {H: M}) can find each other via is_countersigned.
        self._model.add_to_stm(kline)

        # Ratification — countersigned in the model → S1. Only countersign
        # produces reciprocal klines; undersign/connotate share a structure
        # (a single node entry in opposite directions) and are handled below.
        if is_countersigned(self._model, kline, self._signifier):
            self._model.add_to_ltm(kline)
            self._publish("frame", value, KValue(kline, SIG_S1))  # S1
            return True

        # Retrieve candidates (exclude self to prevent trivial match)
        candidates = [
            kl
            for kl in self._model.where(kline.signature)
            if kl is not kline and (kl.signature != kline.signature or kl.nodes != kline.nodes)
        ]

        if not candidates:
            self._model.add_to_ltm(kline)
            self._publish("frame", value, KValue(kline, SIG_S4))  # S4 — novel
            return True

        # DEVELOPMENT-ONLY — candidate fan-out cap.
        # rationalise→expand is exponential by design; the internal logic
        # that bounds expansion is still being refined. 
        # Remove it entirely once expansion is bounded internally.
        _DEV_MAX_CANDIDATES = 8
        if len(candidates) > _DEV_MAX_CANDIDATES:
            candidates = candidates[:_DEV_MAX_CANDIDATES]

        for candidate in candidates:
            level = self._route(kline, candidate)
            # The query KValue flows into the cogitator so the declared
            # significance rides the slow path's published events (KE-2).
            self._cogitator.submit(WorkItem(value, candidate, level))

        return False

    # Graph Expansion Resolution

    # CogitationHandler protocol

    def on_s1(self, query_value: KValue, candidate: KLine) -> None:
        """CogitationHandler.on_s1: structural check, promote, publish frame event.

        ``query_value`` is the original inbound KValue (KE-2); its kline is the
        query voice for promotion. The candidate kline becomes the proposal,
        wrapped at ``SIG_S1`` (S1 ratification).
        """
        query = query_value.kline
        if is_s1(self._model, candidate, self._signifier):
            promote_participating(self._model, query, candidate, self._signifier)
        self._publish("frame", query_value, KValue(candidate, SIG_S1))

    def on_expansion(
        self,
        query_value: KValue,
        proposal: KLine,
        significance: int,
        original_candidate: KLine | None = None,
    ) -> None:
        """CogitationHandler.on_expansion: write proposal to Frame, publish frame event.

        The proposal kline carries the ``expand()``-computed significance (KP-3),
        not a band-representative value. ``query_value`` is the original inbound
        KValue (KE-2).

        ``original_candidate`` is retained on the signature for the cogitator's
        dispatch but is no longer carried onto the event (the ``candidate``
        field is gone). It is intentionally unused here.
        """
        del original_candidate  # retained for dispatch compatibility; not on the event
        self._model.add_to_frame(proposal)
        self._publish("frame", query_value, KValue(proposal, significance))

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate thread and wait for it to finish."""
        self._cogitator.join(timeout)

    def cogitate_drain(self, timeout: float | None = None) -> bool:
        """Drain pending cogitation work items without stopping the thread.

        Returns True if drained within *timeout*, False if timed out.
        """
        return self._cogitator.drain(timeout)

    # Events

    def _publish(self, kind: str, query_value: KValue, proposal_value: KValue) -> None:
        """Publish a rationalisation event via the adapter.

        ``query_value`` is the inbound KValue (the sender's declared
        assessment); ``proposal_value`` is Kalvin's assessment of it. On the
        fast path both wrap the same immutable KLine (KE-1).
        """
        self._adapter.on_event(RationaliseEvent(kind, query_value, proposal_value))

    def countersign(self, value: KValue) -> bool:
        """Generate the reciprocal kline ({Q:[V]} → {V:[Q]}) and rationalise it.

        The reciprocal kline is wrapped in a KValue at ``SIG_S1`` — the act of
        countersigning is an S1 ratification (KP-2). Requires non-empty nodes;
        returns the result of ``rationalise``.
        """
        kline = value.kline
        reciprocal_sig = self._signifier.make_signature(kline.nodes)
        reciprocal = KLine(reciprocal_sig, [kline.signature])
        reciprocal_value = KValue(reciprocal, SIG_S1)
        return self.rationalise(reciprocal_value)

    # Frame info

    def frame_size(self) -> int:
        return len(self._model)

    def codec(self) -> AgentCodec:
        return AgentCodec(self._model, self._activity)

    # Serialization — all delegate to AgentCodec.

    def to_bytes(self) -> bytes:
        return self.codec().to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes, adapter: KAgentAdapter | None = None) -> KAgent:
        model, activity = AgentCodec.from_bytes(data)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent

    def to_dict(self) -> dict:
        return self.codec().to_dict()

    @classmethod
    def from_dict(cls, data: dict, adapter: KAgentAdapter | None = None) -> KAgent:
        model, activity = AgentCodec.from_dict(data)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent

    def save(self, path: str | Path, format: Literal["bin", "json"] | None = None) -> None:
        self.codec().save(path, format)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        format: Literal["bin", "json"] | None = None,
        adapter: KAgentAdapter | None = None,
    ) -> KAgent:
        model, activity = AgentCodec.load(path, format)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent


# Backward-compatible alias
Agent = KAgent
