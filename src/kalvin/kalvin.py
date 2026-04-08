"""Kalvin - Knowledge graph with tokenization support."""

from __future__ import annotations

import threading
from pathlib import Path
from struct import pack, unpack
from typing import Literal, Any
from collections import Counter
from typing import Iterator, Tuple

import json

from kalvin.abstract import KAgent, KLine, KModel, KNodes, KNone, KSig, KTokenizer, KSignificance
from kalvin.events import EventBus, RationaliseEvent
from kalvin.model import Model
from kalvin.significance import Int32Significance
from kalvin.tokenizer import Tokenizer


class Kalvin(KAgent):
    """Kalvin agent with NLP features for knowledge graph operations."""

    __dictionary: dict[str, Any] = {}
    __nlp_type: dict[str, Any] = {}
    __frames: list[KModel] = []

    def __init__(
        self,
        tokenizer: KTokenizer | None = None,
        model: Model | None = None,
        activity: Counter | None = None,
        significance: KSignificance | None = None,
        dictionary: str | None = None,
        nlp_detail: str = "nlp_type32"
    ):
        """Initialize Kalvin with optional model, tokenizer, and significance.

        Args:
            model: Optional Model instance
            activity: Optional Counter for tracking token activity
            tokenizer: Optional KTokenizer instance
            significance: Optional KSignificance instance (defaults to Int32Significance)
            dictionary: Path to grammar dictionary JSON file
            nlp_detail: NLP detail level for type encoding (e.g., "nlp_type32")
        """
        self._model = model if model else Model()
        self._tokenizer = tokenizer if tokenizer else Tokenizer.from_directory()
        self._significance = significance if significance else Int32Significance()
        self._ws_token = self._tokenizer.encode(" ")[0]
        self._activity = activity if activity else Counter()
        self._unrecognised_tokens = set()
        self._dictionary_path = dictionary
        self._nlp_detail = nlp_detail
        self._dictionary: dict[str, Any] = {}
        self._nlp_type: dict[str, Any] = {}
        self.__frames.append(self._model)

        # Pub/sub
        self._event_bus = EventBus()
        self._backlog_lock = threading.Lock()
        self._backlog_condition = threading.Condition(self._backlog_lock)
        self._backlog: list[tuple[KLine, Iterator[KLine]]] = []
        self._cogitate_stop = threading.Event()
        self._cogitate_thread = threading.Thread(target=self._cogitate, daemon=True)
        self._cogitate_thread.start()

        # === Tokenization ===
        if not self._dictionary_path:
            self._dictionary = Kalvin.__dictionary
            self._nlp_type = Kalvin.__nlp_type
            self._dictionary_path = "data/tokenizer/simplestories-1_grammar.json"

        if not self._dictionary:
            with open(self._dictionary_path, "r") as f:
                self._dictionary = json.load(f)
                Kalvin.__dictionary = self._dictionary

            with open(self._dictionary_path.replace("grammar", self._nlp_detail), "r") as f:
                self._nlp_type = json.load(f)
                Kalvin.__nlp_type = self._nlp_type

    # === KAgent interface ===

    @property
    def model(self) -> Model:
        """Get the knowledge graph model."""
        return self._model

    @property
    def tokenizer(self) -> KTokenizer:
        """Get the tokenizer for encoding/decoding text."""
        return self._tokenizer

    @property
    def significance(self) -> KSignificance:
        """Get the significance instance for S1-S4 operations."""
        return self._significance

    @property
    def events(self) -> EventBus:
        """Get the event bus for subscribing to rationalisation events."""
        return self._event_bus

    def _emit(self, kind: str, kline: KLine, query: KLine) -> None:
        """Emit a rationalisation event."""
        self._event_bus.publish(RationaliseEvent(kind, kline, query))

    def encode(self, text: str, nlp_detail: str = "nlp_type32") -> KLine | None:
        """Encode a string to a KLine.

        Args:
            text: Input string to encode

        Returns:
            KLine representing the encoded text, or None if duplicate
        """
        ks_key = 0
        ks_nodes = []
        tokens = self._tokenizer.encode(text)
        self._activity.update(tokens)

        for token in tokens:
            decode = ""
            if token in self._dictionary:
                entry = self._dictionary[token]
                signature = entry[nlp_detail]
                decode = entry["text"]
            elif token == self._ws_token:
                signature = token
            else:
                signature = self._nlp_type["POS_X"]
                self._unrecognised_tokens.add(token)

            signature |= token
            self.rationalise(KLine(signature, nodes=[token], dbg_text=decode))
            # Add token kline to model
            self._model.add(KLine(signature, nodes=[token], dbg_text=decode))
            ks_nodes.append(signature)
            ks_key |= signature

        kline = KLine(signature=ks_key, nodes=ks_nodes, dbg_text=text)
        self.rationalise(kline)
        self._model.add(kline)
        return kline

    def decode(self, token_sig: KSig) -> str:
        """Decode a list of KNodes (token IDs) back to a string.

        Args:
            nodes: List of KNode integers (token IDs)

        Returns:
            Decoded string
        """
        kline = self.model.find_kline(token_sig)
        if kline is None:
            return ""
        knodes = []
        for node in kline.nodes:
            knode = self.model.find_kline(node)
            if knode:
                # Get first node from knode's node list
                node_list = knode.as_node_list()
                if node_list:
                    knodes.append(node_list[0])
        return self._tokenizer.decode(knodes)

    def prune(self, level: int = 1) -> "Kalvin":
        """Prune model to activity level + untracked activity.

        Args:
            level: activity level to keep
        """
        pruned_model: list[KLine] = []
        pruned_activity = Counter()

        for kline in self._model:  # retain untracked activity
            if kline.signature not in self._activity:
                pruned_model.append(kline)

        for key, count in self._activity.items():
            if count >= level:
                kline = self._model.find_kline(key)
                if kline.signature:
                    pruned_model.append(kline)
                    pruned_activity[key] = count

        model = Model(pruned_model)
        return Kalvin(model=model, activity=pruned_activity)

    def get_frame(self) -> KModel:
        """Return the current frame context if it is in bounds, otherwise create a new one.

        Returns:
            New KModel frame instance
        """
        if len(self.__frames) > 0 and len(self.__frames[-1].klines) < 100:
            frame = self.__frames.pop()
        else:
            frame = Model([], self._model)

        self.__frames.append(frame)
        return frame

    def rationalise(self, kline: KLine, frame: KModel | None = None) -> None:
        """Rationalise a KLine query in frame context.

        Emits events via the event bus as significance is established.
        Fast kline results are emitted inline; slow kline results are 
        queued for the cogitate background thread.

        Args:
            kline: KLine to rationalise
            frame: Optional KModel frame context
        """
        is_top_level = frame is None
        if is_top_level:
            frame = self.get_frame()

        # Test early to prevent infinite recursion
        if frame.exists(kline):
            if is_top_level:
                self._emit("complete", kline, kline)
            return

        #bring nodes into frame
        for n in kline.nodes:
            if self.tokenizer.is_literal(n):
                continue

            nk = self._model.find_kline(n)
            if nk == KNone:
                nk = KLine(signature=n, nodes=None) # new token node (also at S4)
            self.rationalise(nk, frame=frame)

        fast, slow = frame.query(kline)

        # Queue slow stream for cogitate background thread
        with self._backlog_condition:
            self._backlog.append((kline, slow))
            self._backlog_condition.notify()

        for fk in fast:
            sig = self.signify(kline, fk)
            self._emit("fast", fk, kline)
            if self._significance.has_s1(sig):
                # Create a KLine with the significance for upgrading
                cs = KLine(signature=sig, nodes=fk.nodes)
                sv = self._significance.calculate(frame, kline, cs)
                frame.upgrade(cs, sv)
                if is_top_level:
                    self._emit("complete", cs, kline)
                return

        frame.add(kline)
        if is_top_level:
            self._emit("complete", kline, kline)

    def _cogitate(self) -> None:
        """Background thread that processes slow stream klines.
        """
        while not self._cogitate_stop.is_set():
            with self._backlog_condition:
                while not self._backlog and not self._cogitate_stop.is_set():
                    self._backlog_condition.wait(timeout=0.5)
                if self._cogitate_stop.is_set() and not self._backlog:
                    return
                qk, slow = self._backlog.pop(0)

            for sk in slow:
                sig = self.signify(qk, sk)
                if self._significance.has_s1(sig):
                    cs = KLine(signature=sig, nodes=sk.nodes)
                    sv = self._significance.calculate(self._model, qk, cs)
                    self._model.upgrade(cs, sv)
                self._emit("slow", sk, qk)

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate background thread and wait for it to finish."""
        self._cogitate_stop.set()
        with self._backlog_condition:
            self._backlog_condition.notify()
        self._cogitate_thread.join(timeout=timeout)
            

    def signify(self, k1: KLine, k2: KLine, s: KSig | None = None) -> KSig:
        """Establish significance relationship between two KLines.

        Calculates internal significance of k1:k2 relationship.
        If s is provided and less significant than internal, returns internal.
        Otherwise, creates links at the requested significance level.

        Args:
            k1: First KLine
            k2: Second KLine
            s: Optional requested significance level. If None, returns internal.

        Returns:
            Significance value (higher = more significant: S1 > S2 > S3 > S4)
        """
        # Calculate internal significance
        internal = self._significance.calculate(self._model, k1, k2)

        # If no specific level requested, or internal is more significant, return internal
        if s is None or internal >= s:
            return internal

        # Requested level is more significant than internal - create links at level s

        # S1: Create countersigned links
        if self._significance.has_s1(s):
            # Add bidirectional links - k1 gets k2's signature as node, k2 gets k1's
            link1 = KLine(k1.signature, [k2.signature])
            link2 = KLine(k2.signature, [k1.signature])
            self._model.add(link1)
            self._model.add(link2)
            return s

        # S2: Check compound match (k2.nodes OR'd together equals k1.signature)
        if self._significance.has_s2(s):
            compound = 0
            for node in k2.nodes:
                compound |= node
            if compound == k1.signature:
                # S2 verified - create link
                link = KLine(k1.signature, k2.nodes)
                self._model.add(link)
                return s

        # S3: Create connotated link
        if self._significance.has_s3(s):
            link = KLine(k1.signature, [k2.signature])
            self._model.add(link)
            return s

        # S4: No significance
        return self._significance.S4


    def model_size(self) -> int:
        """Return the number of KLines in the model.

        Returns:
            Number of KLines stored in the model
        """
        return len(self._model)

    # === Serialization ===

    def to_bytes(self) -> bytes:
        """Serialize model to binary format.

        Binary layout:
        - 4 bytes: metadata length (uint32)
        - N bytes: JSON-encoded metadata (UTF-8)
        - 4 bytes: number of klines (uint32)
        - For each kline:
          - 8 bytes: signature (uint64)
          - 4 bytes: node count (uint32)
          - N * 8 bytes: nodes (uint64 each)
        - 4 bytes: number of activity entries (uint32)
        - For each activity entry:
          - 8 bytes: key (uint64)
          - 4 bytes: count (uint32)
        """
        # Serialize metadata
        metadata = {
            "dictionary": self._dictionary_path,
            "nlp_detail": self._nlp_detail,
        }
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        parts = [pack("<I", len(metadata_bytes)), metadata_bytes]

        # Serialize klines
        parts.append(pack("<I", len(self._model)))
        for kline in self._model:
            node_list = kline.as_node_list()
            parts.append(pack("<Q", kline.signature))
            parts.append(pack("<I", len(node_list)))
            for node in node_list:
                parts.append(pack("<Q", node))

        # Serialize activity Counter
        activity = self._activity
        parts.append(pack("<I", len(activity)))
        for key, count in activity.items():
            parts.append(pack("<Q", key))
            parts.append(pack("<I", count))

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Kalvin":
        """Deserialize model from binary format."""
        from collections import Counter

        offset = 0

        # Deserialize metadata
        metadata_len = unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        metadata = json.loads(data[offset : offset + metadata_len].decode("utf-8"))
        offset += metadata_len
        dictionary = metadata.get("dictionary")
        nlp_detail = metadata.get("nlp_detail", "nlp_type32")

        # Deserialize klines
        kline_count = unpack("<I", data[offset : offset + 4])[0]
        offset += 4

        klines: list[KLine] = []
        for _ in range(kline_count):
            signature = unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            node_count = unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            nodes: KNodes = []
            for _ in range(node_count):
                node = unpack("<Q", data[offset : offset + 8])[0]
                offset += 8
                nodes.append(node)
            klines.append(KLine(signature=signature, nodes=nodes))

        # Deserialize activity Counter
        activity_count = unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        activity: Counter = Counter()
        for _ in range(activity_count):
            key = unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            count = unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            activity[key] = count

        model = Model(klines=klines)
        return cls(model=model, activity=activity, dictionary=dictionary, nlp_detail=nlp_detail)

    def to_dict(self) -> dict:
        """Serialize model to dict (for JSON)."""
        return {
            "metadata": {
                "dictionary": self._dictionary_path,
                "nlp_detail": self._nlp_detail,
            },
            "klines": [{"signature": kline.signature, "nodes": kline.nodes} for kline in self._model],
            "activity": {str(k): v for k, v in self._activity.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Kalvin":
        """Deserialize model from dict."""
        from collections import Counter

        # Extract metadata if present
        metadata = data.get("metadata", {})
        dictionary = metadata.get("dictionary")
        nlp_detail = metadata.get("nlp_detail", "nlp_type32")

        klines = [KLine(signature=item["signature"], nodes=item["nodes"]) for item in data["klines"]]
        model = Model(klines=klines)
        activity = Counter()
        if "activity" in data:
            activity.update({KSig(k): v for k, v in data["activity"].items()})
        return cls(model=model, activity=activity, dictionary=dictionary, nlp_detail=nlp_detail)

    def save(
        self,
        path: str | Path,
        format: Literal["bin", "json"] | None = None,
    ) -> None:
        """Save model to file.

        Args:
            path: File path to save to
            format: 'bin' (default, compact) or 'json' (human-readable)
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "bin"

        if format == "bin":
            path.write_bytes(self.to_bytes())
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path = "data/kalvin.bin",
        format: Literal["bin", "json"] | None = None,
    ) -> "Kalvin":
        """Load model from file.

        Args:
            path: File path to load from
            format: 'bin', 'json', or None (auto-detect from extension)

        Returns:
            Loaded Model instance
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "bin"

        if format == "bin":
            return cls.from_bytes(path.read_bytes())
        else:
            return cls.from_dict(json.loads(path.read_text()))
