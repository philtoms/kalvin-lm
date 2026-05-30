"""KAgent — orchestrator of the Kalvin rationalisation pipeline.

The KAgent rationalises KLines against the Model using a fast/slow split:
  - Fast path: routing (node membership) — no model calls. S1/S4 resolve instantly.
  - Slow path: cogitation — model.expand() per work item in a background thread.

Serialization is delegated to the AgentCodec module (see agent_codec.py).

See specs/agent.md for the full specification.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

from kalvin.agent_codec import AgentCodec
from kalvin.events import EventBus, RationaliseEvent
from kalvin.kline import KLine
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.model import D_MAX, MASK64, Model, QueryCandidate
from kalvin.signature import is_literal_node, make_signature

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

    def on_expansion(self, query: KLine, proposal: KLine, significance: int) -> None:
        """Called when an expansion proposal is generated (S2/S3)."""
        ...


# ── Distance Constants ────────────────────────────────────────────────

# S2|S3 boundary — packed distance threshold separating S2 from S3.
# Sits between max single-node S2 distance (MAX_HOP=100) and min S3 packed
# distance (_pack(2+_S3_BIAS)=121), ensuring clean separation.
_S2_S3_DISTANCE = 100


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
    computes deep significance (model.expand), and processes results.
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
        adapter: Any,
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
                item = self._backlog.pop(0)

            self._run_work_item(item)

    # ── Boundary Computation ───────────────────────────────────────────

    @staticmethod
    def _boundaries() -> tuple[int, int, int]:
        """Return the three significance boundaries (S1|S2, S2|S3, S3|S4).

        S1|S2 = D_MAX - 1              (only exact S1 qualifies)
        S2|S3 = ~_S2_S3_DISTANCE       (packed distance threshold)
        S3|S4 = 0                       (only complete unresolvable is S4)
        """
        s12 = D_MAX - 1
        s23 = (~_S2_S3_DISTANCE) & MASK64
        s34 = 0
        return s12, s23, s34

    @staticmethod
    def _classify(sig: int, s12: int, s23: int, s34: int) -> str:
        """Classify a significance value against three boundaries.

        Returns "S1", "S2", "S3", or "S4".
        """
        if sig >= s12:
            return "S1"
        elif sig >= s23:
            return "S2"
        elif sig >= s34:
            return "S3"
        else:
            return "S4"

    def _run_work_item(self, item: WorkItem) -> None:
        """Expand a work item, classifying each yield against boundaries.

        Each yielded QC is classified against fixed boundaries.
        S4 results are demoted. S1 results trigger promotion (if structurally
        grounded) and re-rationalisation. S2/S3 results are processed for
        expansion.
        """
        query, candidate, _level = item
        s12, s23, s34 = self._boundaries()

        for qc in self._model.expand(query, candidate):
            band = self._classify(qc.significance, s12, s23, s34)

            if band == "S4":
                continue  # demote: below S3|S4 boundary

            if band == "S1":
                if self._model.is_s1(candidate):
                    self._model.promote_participating(query, candidate)
                self._handler.on_s1(query, candidate)
            else:
                self._process(qc)  # S2 or S3: expansion

    def _process(self, item: QueryCandidate) -> None:
        """Process a single expanded result: S2/S3 expansion only.
        """
        query, candidate, significance = item

        candidate_sig = candidate.signature
        nodes_sig = make_signature(candidate.nodes)

        if candidate_sig == nodes_sig:
            return  # canonical — nothing to expand

        underfit, overfit = self._model.classify_misfit(candidate)

        if not underfit and not overfit:
            return  # neither — nothing to expand

        underfit_gap = candidate_sig & ~nodes_sig
        overfit_mask = nodes_sig & ~candidate_sig

        for proposal, companions in self._model.generate_expansions(
            candidate, underfit_gap, overfit_mask
        ):
            self._handler.on_expansion(query, proposal, significance)
            for companion in companions:
                self._handler.on_expansion(query, companion, significance)


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
        Defaults to a new ``EventBus`` instance for backward compatibility.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        model: Model | None = None,
        adapter: Any = None,
    ):
        self._tokenizer = tokenizer if tokenizer else Mod32Tokenizer()
        self._model = model if model is not None else Model()
        self._activity: Counter = Counter()

        # Adapter — receives events via on_event()
        self._adapter: Any = adapter if adapter is not None else EventBus()

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
    def events(self) -> Any:
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

        # Phase 2: Ground check
        if kline.signature != 0 and self._model.exists(kline):
            self._publish("ground", kline, kline, D_MAX - 1)
            return True

        # Phase 3: Assess
        if not kline.nodes:
            self._model.add(kline)
            self._model.promote(kline)
            self._publish("frame", kline, kline, 0)  # S4
            return True

        if all(is_literal_node(n) for n in kline.nodes):
            self._model.add(kline)
            self._model.promote(kline)
            self._publish("frame", kline, kline, D_MAX - 1)  # S1
            return True

        expected_sig = make_signature(kline.nodes)
        if kline.signature == expected_sig:
            all_resolved = all(
                is_literal_node(n) or self._model.find(n) is not None
                for n in kline.nodes
            )
            if all_resolved:
                self._model.add(kline)
                self._model.promote(kline)
                self._publish("frame", kline, kline, D_MAX - 1)  # S1
                return True

        # Phase 3 (continued): Ratification — countersigned in the model → S1
        if self._model.is_countersigned(kline):
            self._model.add(kline)
            self._model.promote(kline)
            self._publish("frame", kline, kline, D_MAX - 1)  # S1
            return True

        # Phase 4: Retrieve candidates
        candidates = self._model.where(kline.signature)

        if not candidates:
            # S4 — novel, no candidates
            self._model.add(kline)
            self._model.promote(kline)
            self._publish("frame", kline, kline, 0)
            return True

        # Phase 5: Route each candidate — fast path on S1, submit S2/S3
        self._model.add(kline)

        found_s1 = False
        for candidate in candidates:
            level = self._route(kline, candidate)

            if level == "S1":
                # Fast response — confirmed, done
                self._model.promote_participating(kline, candidate)
                self._publish("frame", kline, candidate, D_MAX - 1)
                found_s1 = True
                break
            else:
                # S2 or S3 — submit as work item for cogitation
                self._cogitator.submit(WorkItem(kline, candidate, level))

        if found_s1:
            return True

        # All candidates routed as S2/S3
        return False

    def countersign(self, kline: KLine) -> bool:
        """Generate the reciprocal kline and rationalise it.

        The reciprocal swaps signature and nodes:
          - reciprocal_sig = make_signature(kline.nodes)
          - reciprocal_nodes = [kline.signature]

        This is the ratification primitive: by countersigning a proposal,
        the Agent adds the reciprocal relationship to its model, enabling
        future fast-path resolution via the countersigned check.

        See specs/harness.md §Agent.countersign, HRN-9, HRN-16.

        Args:
            kline: The kline to countersign.

        Returns:
            The result of rationalise(reciprocal).
        """
        reciprocal_sig = make_signature(kline.nodes)
        reciprocal = KLine(reciprocal_sig, [kline.signature])
        return self.rationalise(reciprocal)

    # ── CogitationHandler protocol ────────────────────────────────────

    def on_s1(self, query: KLine, candidate: KLine) -> None:
        """CogitationHandler.on_s1: re-rationalise when S1 is discovered."""
        self.rationalise(query)

    def on_expansion(self, query: KLine, proposal: KLine, significance: int) -> None:
        """CogitationHandler.on_expansion: publish expansion as frame event."""
        self._publish("frame", query, proposal, significance)

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate thread and wait for it to finish."""
        self._cogitator.join(timeout)

    # ── Events ────────────────────────────────────────────────────────

    def _publish(self, kind: str, query: KLine, proposal: KLine, significance: int) -> None:
        """Publish a rationalisation event via the adapter."""
        self._adapter.on_event(RationaliseEvent(kind, query, proposal, significance))

    # ── Countersign ───────────────────────────────────────────────────

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
        from kalvin.signature import make_signature
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
    def from_bytes(cls, data: bytes) -> KAgent:
        """Deserialize from binary."""
        model, activity = AgentCodec.from_bytes(data)
        agent = cls(model=model)
        agent._activity = activity
        return agent

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return self.codec().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> KAgent:
        """Deserialize from dict."""
        model, activity = AgentCodec.from_dict(data)
        agent = cls(model=model)
        agent._activity = activity
        return agent

    def save(self, path: str | Path, format: Literal["bin", "json"] | None = None) -> None:
        """Persist to file."""
        self.codec().save(path, format)

    @classmethod
    def load(
        cls, path: str | Path = "data/agent.bin", format: Literal["bin", "json"] | None = None
    ) -> KAgent:
        """Load from file."""
        model, activity = AgentCodec.load(path, format)
        agent = cls(model=model)
        agent._activity = activity
        return agent


# Backward-compatible alias
Agent = KAgent
