"""Rationaliser — orchestrator of the rationalisation pipeline.

The Rationaliser rationalises KLines against the Model using a fast/slow split:
  - Fast path: routing (node membership) — no model calls. S1/S4 resolve instantly.
  - Slow path: cogitation — expand() per work item in a background thread.

The Cogitator (slow path) lives in :mod:`kalvin.cogitator`; this module
imports and wires it. All significance computation lives in
:mod:`kalvin.significance`; graph expansion in :mod:`kalvin.expand`; and
expansion-proposal logic in :mod:`kalvin.proposals`.

Serialization is delegated to the AgentCodec module (see agent_codec.py).
"""

from __future__ import annotations

import logging
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
from kalvin.significance import (
    SIG_S1,
    SIG_S2,
    SIG_S4,
    structural_sig,
)
from kalvin.kline import KLine, is_canon, is_identity, sig_level
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
    "Rationaliser",
    "RationaliserAdapter",
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


# RationaliserAdapter Protocol


@runtime_checkable
class RationaliserAdapter(Protocol):
    """Protocol for receiving rationalisation events from Rationaliser.

    Any object with an ``on_event(RationaliseEvent)`` method satisfies this
    protocol.  The concrete ``RationaliserAdapter`` in ``harness/adapter.py`` is
    the canonical production implementation; ``EventBus`` (in ``events.py``)
    is the standard test/dev adapter.

    Note: the name ``RationaliserAdapter`` intentionally mirrors the concrete class
    in ``harness/adapter.py`` — that class satisfies this protocol implicitly.
    """

    def on_event(self, event: RationaliseEvent) -> None: ...


# Rationaliser


class Rationaliser:
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
        ``RationaliserAdapter`` (from ``harness.adapter``) for production.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        model: Model | None = None,
        signifier: KSignifier | None = None,
        *,
        adapter: RationaliserAdapter,
    ):
        self._tokenizer = tokenizer if tokenizer else _default_tokenizer()
        self._signifier = signifier if signifier is not None else _default_signifier()
        self._model = model if model is not None else Model(signifier=self._signifier)
        self._activity: Counter = Counter()

        self._adapter: RationaliserAdapter = adapter

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
    def events(self) -> RationaliserAdapter:
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
        structural property established by ``expand()`` / ``model.grounded()``, not
        by node membership. S4 (empty query) never reaches routing because
        Unknown klines are resolved on the fast path in ``rationalise``
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
        ``value.significance`` (the sender's declared assessment) is the
        counterpart to Kalvin's own assessment in a two-way significance
        dialog. It is consumed by the significance-comparison gate below
        (MVP: an S4 disagreement drops the query). The query voice on
        published events also carries it (KE-2).

        Fast path: routing (no model calls). S1/S4 resolve instantly.
        Slow path: S2/S3 queued as individual work items for cogitation.

        Returns True if significant (S1, S4), False if rational (S2, S3).
        """
        kline = value.kline
        # Prepare — callers must provide a set signature. This is a presence
        # check, not a value-test: 0 is an ordinary signature value (the
        # empty node set's signature).
        assert kline.signature is not None, (
            "KLine.signature must be set before rationalise; callers compute "
            "it via signifier.signature_of(nodes)."
        )

        # Significance-comparison gate — Kalvin compares its own derived
        # significance to the sender's declared significance. MVP: when the
        # sender declares S4 and Kalvin derives otherwise, Kalvin drops the
        # query (returns True, no STM write, no event). This sits before the
        # ground check because a recurring proposal is already in Frame, so a
        # post-ground gate would be inert against its target.
        #
        # S4 is the sentinel SIG_S4 (= 0), detected by value: classify()
        # collapses the S3|S4 boundary (0 classifies as S3), so the band
        # function cannot be used to detect S4. The derived band is the
        # structural band (sig_level → structural_sig) with the one model-state
        # fork: a structurally-S2 misfit whose reciprocal countersigner is
        # present upgrades to S1. Only an Unknown ask (empty-nodes Unknown)
        # derives SIG_S4, so an Unknown kline declared S4 agrees here and is
        # never dropped.
        derived_sig = structural_sig(sig_level(kline, self._signifier))
        if derived_sig == SIG_S2 and self._model.is_countersigned(kline):
            derived_sig = SIG_S1
        if value.significance == SIG_S4 and derived_sig != SIG_S4:
            return True  # drop — sender declares S4; Kalvin derives otherwise

        # Ground check (Frame/LTM/Base only — not STM)
        if self._model.grounded(kline):
            self._model.add_to_stm(kline)
            self._publish("ground", value, KValue(kline, SIG_S1))
            return True

        if not kline.nodes:
            self._model.add_to_ltm(kline)
            self._publish("frame", value, KValue(kline, SIG_S4))  # S4
            return True

        if is_identity(kline):
            self._model.add_to_ltm(kline)
            self._publish("frame", value, KValue(kline, SIG_S1))  # S1
            return True

        expected_sig = self._signifier.signature_of(kline.nodes)
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
        # {H: M}) can find each other via model.is_countersigned.
        self._model.add_to_stm(kline)

        # Ratification — countersigned in the model → S1. Only countersign
        # produces reciprocal klines; denote/connote share a structure
        # (a single node entry in opposite directions) and are handled below.
        if self._model.is_countersigned(kline):
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

    # Promotion

    def _promote_participating(self, query: KLine, candidate: KLine) -> None:
        """Promote klines that structurally participated in a ratification event.

        After S1 ratification between query and candidate, promote:
        1. The query and candidate themselves (always)
        2. Any STM kline whose signature is a node value in the query or
           candidate AND whose nodes are empty (Unknown frame), a single
           non-literal node (countersign/denote pair), or a canonical
           composition (canonization entry).

        Does NOT promote cogitator expansion proposals (multi-node non-
        canonical klines) that merely share signature bits.
        """
        model = self._model
        signifier = self._signifier

        # Signatures of node values participating in query/candidate.
        node_sigs: set[int] = set()
        for n in query.nodes:
            node_sigs.add(n)
        for n in candidate.nodes:
            node_sigs.add(n)
        node_sigs.add(query.signature)
        node_sigs.add(candidate.signature)

        to_promote: list[KLine] = []
        for kl in model.iter_stm():
            if kl.signature not in node_sigs:
                continue
            # Promote structural klines: Unknown frames, single-node entries,
            # or canonical compositions.
            if not kl.nodes:
                to_promote.append(kl)
            elif isinstance(kl.nodes, int):
                to_promote.append(kl)
            elif isinstance(kl.nodes, list) and len(kl.nodes) == 1:
                to_promote.append(kl)
            elif is_canon(kl, signifier):
                to_promote.append(kl)

        _log.info(
            "_promote_participating: query=%#x candidate=%#x promoting %d structural + 2",
            query.signature,
            candidate.signature,
            len(to_promote),
        )

        to_promote.extend([query, candidate])

        for kl in to_promote:
            model.add_to_ltm(kl)

    # Graph Expansion Resolution

    # CogitationHandler protocol

    def on_s1(self, query_value: KValue, candidate: KLine) -> None:
        """CogitationHandler.on_s1: promote, publish frame event.

        ``query_value`` is the original inbound KValue (KE-2); its kline is the
        query voice for promotion. The candidate kline becomes the proposal,
        wrapped at ``SIG_S1`` (S1 ratification).
        """
        query = query_value.kline
        self._promote_participating(query, candidate)
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
        reciprocal_sig = self._signifier.signature_of(kline.nodes)
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
    def from_bytes(cls, data: bytes, adapter: RationaliserAdapter | None = None) -> Rationaliser:
        model, activity = AgentCodec.from_bytes(data)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent

    def to_dict(self) -> dict:
        return self.codec().to_dict()

    @classmethod
    def from_dict(cls, data: dict, adapter: RationaliserAdapter | None = None) -> Rationaliser:
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
        adapter: RationaliserAdapter | None = None,
    ) -> Rationaliser:
        model, activity = AgentCodec.load(path, format)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent


# Backward-compatible alias
Agent = Rationaliser
