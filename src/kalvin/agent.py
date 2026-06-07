"""KAgent — orchestrator of the rationalisation pipeline.

The KAgent rationalises KLines against the Model using a fast/slow split:
  - Fast path: routing (node membership) — no model calls. S1/S4 resolve instantly.
  - Slow path: cogitation — expand() per work item in a background thread.

The Cogitator is a thin threading dispatcher: it dequeues work items, calls
functions from expand.py (boundaries, classify, expand, propose_expansions),
and routes results to a CogitationHandler. All significance computation and
expansion proposal logic lives in expand.py.

Serialization is delegated to the AgentCodec module (see agent_codec.py).

See specs/agent.md for the full specification.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

from kalvin.agent_codec import AgentCodec
from kalvin.events import EventBus, RationaliseEvent  # EventBus: test/dev fallback
from kalvin.expand import (
    D_MAX,
    boundaries,
    classify,
    expand,
    is_countersigned,
    is_s1,
    promote_participating,
    propose_expansions,
)
from kalvin.kline import KLine
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.model import Model
from kalvin.signature import is_literal_node, make_signature

# ── KAgentAdapter Protocol ─────────────────────────────────────────────

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

    def on_event(self, event: RationaliseEvent) -> None:
        ...


# ── Cogitation Handler Protocol ────────────────────────────────────────

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


# ── Work Item ─────────────────────────────────────────────────────────

class WorkItem(NamedTuple):
    """A single query|candidate pair queued for cogitation."""
    query: KLine
    candidate: KLine
    level: str  # "S2" or "S3"


# ── Cogitator ─────────────────────────────────────────────────────────

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
            import time as _time
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
                        done_k = KLine(0, [], dbg_text="done")
                        self._adapter.on_event(
                            RationaliseEvent("done", done_k, done_k, 0)
                        )
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
        """Expand a work item, classifying each yield against boundaries."""
        query, candidate, _level = item
        s12, s23, s34 = boundaries()

        for qc in expand(self._model, query, candidate):
            band = classify(qc.significance, s12, s23, s34)

            if band == "S4":
                continue  # demote: below S3|S4 boundary

            if band == "S1":
                self._handler.on_s1(query, candidate)
                break  # query fully resolved — skip remaining expansions
            else:
                # S2 or S3: propose expansions, route to handler
                # Note: qc.candidate (not WorkItem.candidate) is the expanded
                # candidate that may be a misfit. qc.query (not WorkItem.query)
                # is the correct query context for connotation yields.
                for proposal, sig in propose_expansions(
                    self._model, qc.candidate, qc.significance
                ):
                    self._handler.on_expansion(
                        qc.query, proposal, sig, original_candidate=qc.candidate,
                    )


# ── KAgent ────────────────────────────────────────────────────────────

class KAgent:
    """Orchestrator of the rationalisation pipeline.

    Parameters
    ----------
    tokenizer:
        Tokenizer instance. Defaults to Mod32Tokenizer.
        Used for encoding text to nodes.
    model:
        Model instance serving as base knowledge graph. Defaults to empty Model.
    adapter:
        Adapter for receiving events. Must implement ``on_event(event)``.
        Required — pass an ``EventBus`` for test/dev use, or a
        ``KAgentAdapter`` (from ``harness.adapter``) for production.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        model: Model | None = None,
        *,
        adapter: KAgentAdapter,
    ):
        self._tokenizer = tokenizer if tokenizer else Mod32Tokenizer()
        self._model = model if model is not None else Model()
        self._activity: Counter = Counter()

        # Adapter — receives events via on_event()
        self._adapter: KAgentAdapter = adapter

        # Cogitator — KAgent is the CogitationHandler
        self._cogitator = Cogitator(
            model=self._model,
            adapter=self._adapter,
            handler=self,
            timeout=2.0,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def model(self) -> Model:
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def events(self) -> KAgentAdapter:
        """The adapter, exposed for event inspection (e.g. ``.subscribe()`` on EventBus)."""
        return self._adapter

    @property
    def cogitator(self) -> Cogitator:
        return self._cogitator

    # ── Routing ───────────────────────────────────────────────────────

    @staticmethod
    def _route(query: KLine, candidate: KLine) -> str:
        """Fast classification — node-membership test only. No model call.

        Returns "S1", "S2", "S3", or "S4" (for empty query).
        """
        nodes = query.nodes
        total = len(nodes)

        if total == 0:
            return "S4"

        candidate_nodes = set(candidate.nodes)
        match_count = sum(1 for n in nodes if n in candidate_nodes)

        if match_count == total:
            return "S1"
        elif match_count > 0:
            return "S2"
        else:
            return "S3"

    # ── Rationalisation ───────────────────────────────────────────────

    def rationalise(self, kline: KLine) -> bool:
        """Rationalise a KLine into the knowledge graph.

        Fast path: routing (no model calls). S1/S4 resolve instantly.
        Slow path: S2/S3 queued as individual work items for cogitation.

        Returns True if significant (S1, S4), False if rational (S2, S3).
        """
        # Phase 1: Prepare
        if kline.signature == 0 and kline.nodes:
            kline.signature = make_signature(kline.nodes)

        # Phase 2: Ground check (Frame/LTM/Base only — not STM)
        if kline.signature != 0 and self._model.grounded(kline):
            self._model.add_stm(kline)
            self._publish("ground", kline, kline, D_MAX - 1)
            return True

        # Phase 3: Assess
        if not kline.nodes:
            self._model.add_ltm(kline)
            self._publish("frame", kline, kline, 0)  # S4
            return True

        if all(is_literal_node(n) for n in kline.nodes):
            self._model.add_ltm(kline)
            self._publish("frame", kline, kline, D_MAX - 1)  # S1
            return True

        expected_sig = make_signature(kline.nodes)
        if kline.signature == expected_sig:
            all_resolved = all(
                is_literal_node(n) or self._model.find(n) is not None
                for n in kline.nodes
            )
            if all_resolved:
                self._model.add_ltm(kline)
                self._publish("frame", kline, kline, D_MAX - 1)  # S1
                return True

        # Phase 3 (continued): Register in STM before ratification check.
        # This ensures that sequential countersign pairs (e.g. from M == H
        # compiling to {M: H} and {H: M}) can find each other via
        # is_countersigned: the first entry in STM becomes visible when
        # the second entry's countersign check runs.
        self._model.add_stm(kline)

        # Phase 3 (continued): Ratification — countersigned in the model → S1.
        # Only countersign (==) entries can pass this check, because only
        # countersign produces reciprocal klines. Undersign (=) and connotate
        # (>) compile to the same kline structure — a single node entry in
        # opposite directions — so Kalvin treats them identically.
        if is_countersigned(self._model, kline):
            self._model.add_ltm(kline)
            self._publish("frame", kline, kline, D_MAX - 1)  # S1
            return True

        # Phase 3 (continued): Undersign ratification — single-node entries
        # where both signature and node are grounded identities. Only applies
        # to entries compiled from S1 operators (countersign, undersign).
        # S3 connotate entries with the same structure must go through slow path.
        if (getattr(kline, 'sig_level', None) == 'S1'
            and len(kline.nodes) == 1
            and not is_literal_node(kline.nodes[0])
            and self._model.find(kline.signature) is not None
            and self._model.find(kline.nodes[0]) is not None):
            self._model.add_ltm(kline)
            self._publish("frame", kline, kline, D_MAX - 1)  # S1
            return True

        # Phase 3b: Graph expansion for single-node entries with unknown
        # nodes. When a connotate or undersign references a node that
        # doesn't exist in the model, attempt to resolve it by expanding
        # the graph from the signature's context.
        if (len(kline.nodes) == 1
            and not is_literal_node(kline.nodes[0])
            and self._model.find(kline.nodes[0]) is None
            and self._model.find(kline.signature) is not None):
            if self._resolve_unknown_via_graph(kline.signature, kline.nodes[0]):
                # Unknown node grounded via graph expansion.
                # Now both sides are grounded — accept as S1.
                self._model.add_ltm(kline)
                self._publish("frame", kline, kline, D_MAX - 1)
                return True

        # Phase 4: Retrieve candidates (exclude self to prevent trivial match)
        candidates = [
            kl for kl in self._model.where(kline.signature)
            if kl is not kline
            and (kl.signature != kline.signature or kl.nodes != kline.nodes)
        ]

        if not candidates:
            # S4 — novel, no candidates
            self._model.add_ltm(kline)
            self._publish("frame", kline, kline, 0)
            return True

        # Phase 5: Route candidates — scan for S1 first, defer cogitation.
        # Previous code submitted S2/S3 work items to the cogitator inline
        # during iteration. If a later candidate matched S1 and broke the
        # loop, the earlier work items were already queued and processed
        # needlessly. We now collect slow candidates and only submit them
        # if no S1 is found in the full candidate list.

        found_s1 = False
        slow_candidates: list[tuple[KLine, str]] = []
        for candidate in candidates:
            level = self._route(kline, candidate)

            if level == "S1":
                # Fast response — confirmed, done
                promote_participating(self._model, kline, candidate)
                self._publish("frame", kline, candidate, D_MAX - 1)
                found_s1 = True
                break
            else:
                # Defer — only submit if no S1 found
                slow_candidates.append((candidate, level))

        if found_s1:
            return True

        # No S1 found — submit deferred slow candidates to cogitator
        for candidate, level in slow_candidates:
            self._cogitator.submit(WorkItem(kline, candidate, level))

        # All candidates routed as S2/S3
        return False

    # ── Graph Expansion Resolution ───────────────────────────────────

    def _resolve_unknown_via_graph(
        self, signature: int, unknown_node: int
    ) -> bool:
        """Try to ground an unknown node through graph expansion.

        Given a single-node entry {sig: [unknown]} where unknown is not in
        the model, attempts to ground the unknown node by verifying that the
        signature participates in a known structure (canonization or mapping).

        Strategy:
        1. Find klines whose nodes contain the signature — i.e., structures
           the signature participates in.
        2. Verify at least one such structure is grounded (S1).
        3. If confirmed, create a frame for the unknown node, making it
           a grounded identity in the model.

        Returns True if the unknown node was grounded, False otherwise.
        """
        # Check if signature participates in any grounded structure
        for containing_kline in self._model.klines():
            if signature not in containing_kline.nodes:
                continue
            if is_s1(self._model, containing_kline):
                # Signature participates in a grounded structure.
                # Create a frame for the unknown node.
                unknown_frame = KLine(unknown_node, [])
                self._model.add_ltm(unknown_frame)
                self._publish("frame", unknown_frame, unknown_frame, 0)
                return True

        return None

    # ── CogitationHandler protocol ────────────────────────────────────

    def on_s1(self, query: KLine, candidate: KLine) -> None:
        """CogitationHandler.on_s1: structural check, promote, publish frame event."""
        if is_s1(self._model, candidate):
            promote_participating(self._model, query, candidate)
        self._publish("frame", query, candidate, D_MAX - 1)

    def on_expansion(
        self,
        query: KLine,
        proposal: KLine,
        significance: int,
        original_candidate: KLine | None = None,
    ) -> None:
        """CogitationHandler.on_expansion: write proposal to Frame, publish frame event."""
        self._model.add_frame(proposal)
        event = RationaliseEvent(
            "frame", query, proposal, significance,
            candidate=original_candidate,
        )
        self._adapter.on_event(event)

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate thread and wait for it to finish."""
        self._cogitator.join(timeout)

    def cogitate_drain(self, timeout: float | None = None) -> bool:
        """Drain pending cogitation work items without stopping the thread.

        Returns True if drained within *timeout*, False if timed out.
        """
        return self._cogitator.drain(timeout)

    # ── Events ────────────────────────────────────────────────────────

    def _publish(self, kind: str, query: KLine, proposal: KLine, significance: int) -> None:
        """Publish a rationalisation event via the adapter."""
        self._adapter.on_event(RationaliseEvent(kind, query, proposal, significance))

    def countersign(self, kline: KLine) -> bool:
        """Generate the reciprocal kline and rationalise it.

        For {Q: [V]}, the reciprocal is {V: [Q]}.

        Precondition: kline.nodes is not empty.
        Postcondition: A reciprocal kline is rationalised. Returns the result.

        Args:
            kline: The kline to countersign.

        Returns:
            Result of rationalise(reciprocal).
        """
        reciprocal_sig = make_signature(kline.nodes)
        reciprocal = KLine(reciprocal_sig, [kline.signature])
        return self.rationalise(reciprocal)

    # ── Frame info ────────────────────────────────────────────────────

    def frame_size(self) -> int:
        return len(self._model)

    # ── Codec ──────────────────────────────────────────────────────────

    def codec(self) -> AgentCodec:
        """Return an AgentCodec for this agent's model and activity."""
        return AgentCodec(self._model, self._activity)

    # ── Serialization (delegates to AgentCodec) ───────────────────────

    def to_bytes(self) -> bytes:
        """Serialize to binary."""
        return self.codec().to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes, adapter: KAgentAdapter | None = None) -> KAgent:
        """Deserialize from binary."""
        model, activity = AgentCodec.from_bytes(data)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return self.codec().to_dict()

    @classmethod
    def from_dict(cls, data: dict, adapter: KAgentAdapter | None = None) -> KAgent:
        """Deserialize from dict."""
        model, activity = AgentCodec.from_dict(data)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent

    def save(self, path: str | Path, format: Literal["bin", "json"] | None = None) -> None:
        """Persist to file."""
        self.codec().save(path, format)

    @classmethod
    def load(
        cls, path: str | Path = "data/agent.bin", format: Literal["bin", "json"] | None = None,
        adapter: KAgentAdapter | None = None,
    ) -> KAgent:
        """Load from file."""
        model, activity = AgentCodec.load(path, format)
        agent = cls(model=model, adapter=adapter or EventBus())
        agent._activity = activity
        return agent


# Backward-compatible alias
Agent = KAgent
