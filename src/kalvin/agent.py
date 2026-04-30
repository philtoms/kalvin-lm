"""Agent — orchestrator of the Kalvin rationalisation pipeline.

The Agent rationalises KLines against the Model, delegates significance
computation, and integrates results back into the knowledge graph.
Encoding is the caller's responsibility — the Agent consumes KLines
produced externally via the Tokenizer.

See specs/agent.md for the full specification.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal, Any
from collections import Counter

import json

from kalvin.kline import KLine
from kalvin.events import EventBus, RationaliseEvent
from kalvin.model import Model
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.signature import make_signature
from kalvin.significance import (
    significance_pipeline,
    compute_significance,
    SignificanceResult,
    D_MAX,
)


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
        self._backlog_lock = threading.Lock()
        self._backlog_condition = threading.Condition(self._backlog_lock)
        self._backlog: list[KLine] = []
        self._cogitate_stop = threading.Event()
        self._cogitate_thread = threading.Thread(target=self._cogitate, daemon=True)
        self._cogitate_thread.start()

        # Cogitation parameters
        self._d_cogitate = 2
        self._max_cogitate_passes = 3
        self._cogitate_timeout = 2.0
        self._cogitate_passes: dict[int, int] = {}  # id(kline) -> pass count

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

    # ── Rationalisation ───────────────────────────────────────────────

    def rationalise(self, kline: KLine) -> bool:
        """Rationalise a KLine into the knowledge graph.

        6-phase pipeline:
          1. PREPARE — assign signature if missing.
          2. GROUND CHECK — already exists?
          3. ASSESS — structural fast-paths.
          4. RETRIEVE CANDIDATES — signature overlap.
          5. COMPUTE SIGNIFICANCE — per-candidate pipeline.
          6. INTEGRATE — add to model, act on result.

        Returns True if significant (S1, S4), False if rational (S2, S3).
        """
        # Phase 1: Prepare
        if kline.signature == 0 and kline.nodes:
            kline.signature = make_signature(kline.nodes, self._tokenizer.is_literal)

        # Phase 2: Ground check
        if kline.signature != 0 and self._model.exists(kline):
            self._publish("ground", kline, kline, D_MAX - 1)  # S1 = all bits set
            return True

        # Phase 3: Assess
        # Unsigned (no nodes)
        if not kline.nodes:
            self._model.add(kline)
            self._publish("frame", kline, kline, 0)  # S4 = 0
            return True

        # All-literal fast path
        if all(self._tokenizer.is_literal(n) for n in kline.nodes):
            self._model.add(kline)
            self._publish("frame", kline, kline, D_MAX - 1)  # S1
            return True

        # Self-grounded canonical test
        expected_sig = make_signature(kline.nodes, self._tokenizer.is_literal)
        if kline.signature == expected_sig:
            # Check if every node resolves in the model (or is literal)
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

        # Phase 5: Compute significance
        results = significance_pipeline(kline, candidates, self._model)

        # Phase 6: Integrate
        self._model.add(kline)

        # Find best result
        best_candidate, best_result = max(results, key=lambda r: r[1].significance)

        if best_result.level == "S1":
            self._model.promote(kline)
            self._publish("frame", kline, best_candidate, best_result.significance)
            return True
        elif best_result.level == "S4":
            # Novel — no candidates matched (pipeline returned S4)
            self._publish("frame", kline, kline, 0)
            return True
        else:
            # S2 or S3 — queue for cogitation
            self._queue_cogitation(kline)
            return False

    # ── Cogitation ────────────────────────────────────────────────────

    def _queue_cogitation(self, kline: KLine) -> None:
        """Queue a KLine for background cogitation."""
        with self._backlog_condition:
            self._backlog.append(kline)
            self._backlog_condition.notify()

    def _cogitate(self) -> None:
        """Background thread: process rational KLines via deeper graph traversal."""
        idle_time = 0.0
        while not self._cogitate_stop.is_set():
            with self._backlog_condition:
                while not self._backlog and not self._cogitate_stop.is_set():
                    self._backlog_condition.wait(timeout=0.5)
                    idle_time += 0.5
                    if idle_time >= self._cogitate_timeout:
                        done_k = KLine(0, [], dbg_text="done")
                        self._publish("done", done_k, done_k, 0)
                        return
                idle_time = 0.0
                if self._cogitate_stop.is_set() and not self._backlog:
                    return
                kline = self._backlog.pop(0)

            # Check pass limit
            kid = id(kline)
            passes = self._cogitate_passes.get(kid, 0)
            if passes >= self._max_cogitate_passes:
                self._cogitate_passes.pop(kid, None)
                continue
            self._cogitate_passes[kid] = passes + 1

            # Expand graph context
            candidates = self._model.query(kline.signature, depth=self._d_cogitate)

            found_s1 = False
            for candidate in candidates:
                # Run significance pipeline
                result = compute_significance(kline, candidate, self._model)

                # Test for countersignature
                if self._model.is_countersigned(kline, candidate):
                    result = SignificanceResult(
                        significance=D_MAX - 1,
                        distance=0,
                        level="S1",
                        match_count=len(kline.nodes),
                        total_nodes=len(kline.nodes),
                    )

                if result.level == "S1":
                    self._model.add(candidate)
                    found_s1 = True
                    break

            if found_s1:
                self._cogitate_passes.pop(kid, None)
                self.rationalise(kline)
            # Otherwise it will remain at S2/S3

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate thread and wait for it to finish."""
        self._cogitate_stop.set()
        with self._backlog_condition:
            self._backlog_condition.notify()
        self._cogitate_thread.join(timeout=timeout)

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
