"""Agent — orchestrator of the Kalvin rationalisation pipeline.

The Agent rationalises KLines against the Model using a fast/slow split:
  - Fast path: routing (node membership) — no model calls. S1/S4 resolve instantly.
  - Slow path: cogitation — model.expand() per work item in a background thread.

See specs/agent.md for the full specification.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal, Any, NamedTuple
from collections import Counter

import json

from kalvin.kline import KLine
from kalvin.events import EventBus, RationaliseEvent
from kalvin.model import Model, QueryCandidate, D_MAX, MASK64
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.signature import make_signature


# ── Sampling ─────────────────────────────────────────────────────────

class Sampling:
    """Response sampling parameters for cogitation.

    Values follow LLM convention so user intuition transfers directly.
    Applied per-yield inside the cogitation stream — no batch collection.

    Attributes
    ----------
    temperature:
        Significance scaling. (0, ∞). Default 1.0 (identity).
        < 1 conservative (only high-significance connotations pass),
        > 1 exploratory (low-significance connotations pass more freely).
    top_k:
        Maximum connotations processed per work item. Default 40.
        0 = unlimited.
    top_p:
        Cumulative significance threshold for early stopping. Default 0.95.
        When accumulated significance reaches this fraction of D_MAX,
        further connotations are demoted (never processed).
        1.0 = no early stopping.
    """

    __slots__ = ("temperature", "top_k", "top_p")

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1.0], got {top_p}")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def __repr__(self) -> str:
        return (
            f"Sampling(temperature={self.temperature}, "
            f"top_k={self.top_k}, top_p={self.top_p})"
        )


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
    Sampling parameters control the streaming consumption of expand()
    yields — temperature gates per-yield quality, top_k caps exploration
    budget, and top_p triggers early stopping on sufficient evidence.

    Parameters
    ----------
    model:
        Model instance for distance computation and countersignature checks.
    event_bus:
        EventBus for publishing events.
    on_s1:
        Callback invoked when cogitation discovers an S1 result.
        Receives (query_kline, candidate_kline).
    timeout:
        Idle seconds before emitting "done" so subscribers can realign.
        Does not halt the thread. Default 2.0.
    sampling:
        Sampling parameters. Defaults to temperature=1.0, top_k=40,
        top_p=0.95.
    """

    def __init__(
        self,
        model: Model,
        event_bus: EventBus,
        on_s1: Any,  # Callable[[KLine, KLine], None]
        timeout: float = 2.0,
        sampling: Sampling | None = None,
    ):
        self._model = model
        self._event_bus = event_bus
        self._on_s1 = on_s1
        self._timeout = timeout
        self.sampling = sampling if sampling is not None else Sampling()
        self.publish_threshold: int = D_MAX - 1

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
                        self._event_bus.publish(
                            RationaliseEvent("done", done_k, done_k, 0)
                        )
                        idle_time = 0.0
                idle_time = 0.0
                if self._stop.is_set() and not self._backlog:
                    return
                item = self._backlog.pop(0)

            self._run_work_item(item)

    def _adjust(self, significance: int) -> int:
        """Apply temperature to a significance value.

        Works in distance space where division by τ is natural:
          τ < 1 → distance increases → significance drops → conservative
          τ = 1 → identity
          τ > 1 → distance decreases → significance rises → exploratory
        """
        if self.sampling.temperature == 1.0:
            return significance
        distance = (~significance) & MASK64
        adjusted = min(int(distance / self.sampling.temperature), D_MAX)
        return (~adjusted) & MASK64

    def _run_work_item(self, item: WorkItem) -> None:
        """Expand a work item with streaming sampling.

        Consumes expand() yields one at a time. Temperature gates
        per-yield quality, top_k caps the exploration budget, and top_p
        triggers early stopping when sufficient evidence has accumulated.
        """
        query, candidate, level = item
        s = self.sampling

        evidence_target = int(s.top_p * D_MAX)
        count = 0
        cumulative = 0

        for qc in self._model.expand(query, candidate, level):
            # Temperature gate — per-yield quality filter
            adjusted = self._adjust(qc.significance)
            if adjusted < self.publish_threshold:
                continue  # demote: insufficient significance after temperature

            count += 1
            cumulative += adjusted

            self._process(qc)

            # Top-p: sufficient evidence accumulated?
            if s.top_p < 1.0 and cumulative >= evidence_target:
                break  # demote remainder — diminishing returns

            # Top-k: exploration budget exhausted?
            if 0 < s.top_k <= count:
                break  # demote remainder — budget spent

    def _process(self, item: QueryCandidate) -> None:
        """Process a single expanded result: check countersignature."""
        query, candidate, significance = item

        # Check countersignature — may upgrade to S1
        if self._model.is_countersigned(query, candidate):
            self._model.add(candidate)
            self._on_s1(query, candidate)


# ── Agent ─────────────────────────────────────────────────────────────

class Agent:
    """Orchestrator of the rationalisation pipeline.

    Parameters
    ----------
    tokenizer:
        Tokenizer instance. Defaults to Mod32Tokenizer.
        Used for is_literal tests during signature construction and
        the Assess phase.
    model:
        Model instance serving as base knowledge graph. Defaults to empty Model.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        model: Model | None = None,
    ):
        self._tokenizer = tokenizer if tokenizer else Mod32Tokenizer()
        self._model = model if model is not None else Model(
            is_literal_fn=self._tokenizer.is_literal,
        )
        self._activity: Counter = Counter()

        # Pub/sub
        self._event_bus = EventBus()

        # Cogitator
        self._cogitator = Cogitator(
            model=self._model,
            event_bus=self._event_bus,
            on_s1=self._on_cogitate_s1,
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
    def events(self) -> EventBus:
        return self._event_bus

    @property
    def cogitator(self) -> Cogitator:
        return self._cogitator

    @property
    def sampling(self) -> Sampling:
        """Cogitation sampling parameters (temperature, top_k, top_p)."""
        return self._cogitator.sampling

    @sampling.setter
    def sampling(self, value: Sampling) -> None:
        self._cogitator.sampling = value

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
            kline.signature = make_signature(kline.nodes, self._tokenizer.is_literal)

        # Phase 2: Ground check
        if kline.signature != 0 and self._model.exists(kline):
            self._publish("ground", kline, kline, D_MAX - 1)
            return True

        # Phase 3: Assess
        if not kline.nodes:
            self._model.add(kline)
            self._publish("frame", kline, kline, 0)  # S4
            return True

        if all(self._tokenizer.is_literal(n) for n in kline.nodes):
            self._model.add(kline)
            self._publish("frame", kline, kline, D_MAX - 1)  # S1
            return True

        expected_sig = make_signature(kline.nodes, self._tokenizer.is_literal)
        if kline.signature == expected_sig:
            all_resolved = all(
                self._tokenizer.is_literal(n) or self._model.find(n) is not None
                for n in kline.nodes
            )
            if all_resolved:
                self._model.add(kline)
                self._publish("frame", kline, kline, D_MAX - 1)  # S1
                return True

        # Phase 4: Retrieve candidates
        candidates = self._model.where(kline.signature)

        if not candidates:
            # S4 — novel, no candidates
            self._model.add(kline)
            self._publish("frame", kline, kline, 0)
            return True

        # Phase 5: Route each candidate — fast path on S1, submit S2/S3
        self._model.add(kline)

        found_s1 = False
        for candidate in candidates:
            level = self._route(kline, candidate)

            if level == "S1":
                # Fast response — confirmed, done
                self._model.promote(kline)
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

    # ── Cogitation callback ───────────────────────────────────────────

    def _on_cogitate_s1(self, query: KLine, candidate: KLine) -> None:
        """Called by Cogitator when an S1 result is discovered."""
        self.rationalise(query)

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate thread and wait for it to finish."""
        self._cogitator.join(timeout)

    # ── Events ────────────────────────────────────────────────────────

    def _publish(self, kind: str, query: KLine, value: KLine, significance: int) -> None:
        """Publish a rationalisation event."""
        if kind == "frame" and significance in (D_MAX - 1, 0):
            self._model.promote(value)
        self._event_bus.publish(RationaliseEvent(kind, query, value, significance))

    # ── Frame info ────────────────────────────────────────────────────

    def frame_size(self) -> int:
        return len(self._model)

    # ── Serialization ─────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialize to binary."""
        from struct import pack, unpack

        klines = [kl for kl in self._model if kl is not None]
        parts: list[bytes] = []
        parts.append(pack("<I", len(klines)))
        for kl in klines:
            parts.append(pack("<Q", kl.signature))
            parts.append(pack("<I", len(kl.nodes)))
            for n in kl.nodes:
                parts.append(pack("<Q", n))

        parts.append(pack("<I", len(self._activity)))
        for key, count in self._activity.items():
            parts.append(pack("<Q", key))
            parts.append(pack("<I", count))

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> Agent:
        """Deserialize from binary."""
        from struct import unpack

        offset = 0
        kline_count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4

        klines: list[KLine] = []
        for _ in range(kline_count):
            sig = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            n_count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            nodes = []
            for _ in range(n_count):
                nodes.append(unpack("<Q", data[offset:offset + 8])[0])
                offset += 8
            klines.append(KLine(sig, nodes))

        activity_count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        activity: Counter = Counter()
        for _ in range(activity_count):
            key = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            activity[key] = count

        model = Model(is_literal_fn=Mod32Tokenizer().is_literal)
        for kl in klines:
            model.add(kl)
            model.promote(kl)
        agent = cls(model=model)
        agent._activity = activity
        return agent

    def to_dict(self) -> dict:
        return {
            "klines": [
                {"signature": kl.signature, "nodes": kl.nodes}
                for kl in self._model
            ],
            "activity": {str(k): v for k, v in self._activity.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Agent:
        klines = [
            KLine(item["signature"], item["nodes"])
            for item in data.get("klines", [])
        ]
        model = Model(is_literal_fn=Mod32Tokenizer().is_literal)
        for kl in klines:
            model.add(kl)
            model.promote(kl)
        agent = cls(model=model)
        if "activity" in data:
            agent._activity = Counter({int(k): v for k, v in data["activity"].items()})
        return agent

    def save(self, path: str | Path, format: Literal["bin", "json"] | None = None) -> None:
        path = Path(path)
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "bin"
        if format == "bin":
            path.write_bytes(self.to_bytes())
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path = "data/agent.bin", format: Literal["bin", "json"] | None = None) -> Agent:
        path = Path(path)
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "bin"
        if format == "bin":
            return cls.from_bytes(path.read_bytes())
        else:
            return cls.from_dict(json.loads(path.read_text()))
