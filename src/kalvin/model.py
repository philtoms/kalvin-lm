"""Model - Knowledge graph with tokenization support."""

from __future__ import annotations

import threading
from pathlib import Path
from struct import pack, unpack
from typing import Literal, Any
from collections import Counter

import json

from kalvin.abstract import KModel, KLine, KFrame, KNodes, KNone, KSig, KTokenizer, KSignificance
from kalvin.events import EventBus, RationaliseEvent
from kalvin.frame import Frame
from kalvin.significance import Int32Significance
from kalvin.tokenizer import Tokenizer


class Model(KModel):
    """KLine model."""

    __dictionary: dict[str, Any] = {}
    __nlp_type: dict[str, Any] = {}
    __frames: list[KFrame] = []

    def __init__(
        self,
        tokenizer: KTokenizer | None = None,
        frame: Frame | None = None,
        activity: Counter | None = None,
        significance: KSignificance | None = None,
        dictionary: str | None = None,
        nlp_detail: str = "nlp_type32",
        dev: bool = False,
    ):
        """Initialize Model with optional frame, tokenizer, and significance.

        Args:
            frame: Optional Frame instance
            activity: Optional Counter for tracking token activity
            tokenizer: Optional KTokenizer instance
            significance: Optional KSignificance instance (defaults to Int32Significance)
            dictionary: Path to grammar dictionary JSON file
            nlp_detail: NLP detail level for type encoding (e.g., "nlp_type32")
        """
        self._dev = dev
        self._frame = frame if frame else Frame()
        self._tokenizer = tokenizer if tokenizer else Tokenizer.from_directory()
        self._significance = significance if significance else Int32Significance()
        self._ws_token = self._tokenizer.encode(" ")[0]
        self._activity = activity if activity else Counter()
        self._unrecognised_tokens = set()
        self._dictionary_path = dictionary
        self._nlp_detail = nlp_detail
        self._dictionary: dict[str, Any] = {}
        self._nlp_type: dict[str, Any] = {}
        self.__frames.append(self._frame)

        # Pub/sub
        self._event_bus = EventBus()
        self._backlog_lock = threading.Lock()
        self._backlog_condition = threading.Condition(self._backlog_lock)
        self._backlog: list[tuple[KFrame, KLine, KLine, int]] = []
        self._cogitate_stop = threading.Event()
        self._cogitate_thread = threading.Thread(target=self._cogitate, daemon=True)
        self._cogitate_thread.start()

        # === Tokenization ===
        if not self._dictionary_path:
            self._dictionary = Model.__dictionary
            self._nlp_type = Model.__nlp_type
            self._dictionary_path = "data/tokenizer/simplestories-1_grammar.json"

        if not self._dictionary:
            with open(self._dictionary_path, "r") as f:
                self._dictionary = json.load(f)
                Model.__dictionary = self._dictionary

            with open(self._dictionary_path.replace("grammar", self._nlp_detail), "r") as f:
                self._nlp_type = json.load(f)
                Model.__nlp_type = self._nlp_type

    def _get_frame(self) -> KFrame:
        """Return the current frame context if it is in bounds, otherwise create a new one.

        Returns:
            New KFrame frame instance
        """
        if len(self.__frames) > 1 and len(self.__frames[-1].klines) < 100:
            return self.__frames[-1]
        
        frame = Frame([], self._frame)
        self.__frames.append(frame)
        return frame
    
    def _signify(self, frame: KFrame, query: KLine, target: KLine) -> KSig:
        """Calculate significance between query and target KLines.

        Significance is comparable - higher = more significant.
        S1 > S2 > S3 > S4.

        Args:
            frame: The Frame containing the KLines (for descendant lookup)
            query: The query KLine
            target: The target KLine to compare against

        Returns:
            Significance value
        """
        # Get nodes as lists for comparison
        query_nodes = query.as_node_list()
        target_nodes = target.as_node_list()

        # Handle empty node lists
        if not query_nodes or not target_nodes:
            return self._significance.S4

        min_len = min(len(query_nodes), len(target_nodes))

        # Count S1 matches: positional equality (up to min length)
        s1_match_positions = set(
            i for i in range(min_len) if self._significance.equal(query_nodes[i], target_nodes[i])
        )
        s1_matches = len(s1_match_positions)

        # S2 -> S1: All nodes match
        if s1_matches == min_len and len(query_nodes) == len(target_nodes):
                return self._significance.S1

        # S2: Partial match (some positional matches exist)
        if s1_matches > 0:
            s1_pct = (s1_matches * 100) // min_len

            # S2 matches: nodes at different positions
            target_set = set(target_nodes)
            s2_matches = 0
            for i, node in enumerate(query_nodes):
                if i in s1_match_positions:
                    continue  # Already counted as S1
                if node in target_set:
                    s2_matches += 1

            s2_pct = (s2_matches * 100) // len(query_nodes) if query_nodes else 0
            return self._significance.build_s2(s1_pct, s2_pct)

        # S3: No positional matches, check unordered and generational
        target_set = set(target_nodes)
        query_set = set(query_nodes)

        # S3-Unordered S1: query nodes that exist in target (any position)
        unordered_s1_matches = query_set & target_set
        s3_s1_pct = (
            (len(unordered_s1_matches) * 100) // len(query_set) if query_set else 0
        )

        # S3-Unordered S2: query nodes whose children match target nodes
        s3_s2_matches = 0
        for node in query_nodes:
            if node in target_set:
                continue  # Already S1 match
            # Check if node's children intersect with target
            node_kline = frame.find_kline(node)
            if node_kline is not KNone:
                node_children = set(node_kline.as_node_list())
                if node_children & target_set:
                    s3_s2_matches += 1

        s3_s2_pct = (
            (s3_s2_matches * 100) // len(query_nodes) if query_nodes else 0
        )

        # S3-Generational: query nodes whose descendants (at any depth) match target nodes
        gen_matches = 0
        for node in query_nodes:
            if node in target_set:
                continue  # Already S1 match
            # Collect all descendants of this node
            descendants = frame.get_all_descendants(node)
            if descendants & target_set:
                gen_matches += 1

        gen_pct = (gen_matches * 100) // len(query_nodes) if query_nodes else 0

        if s3_s1_pct > 0 or s3_s2_pct > 0 or gen_pct > 0:
            return self._significance.build_s3(s3_s1_pct, s3_s2_pct, gen_pct)

        # S4: No match
        return self._significance.S4
    
    # === KModel interface ===

    @property
    def frame(self) -> Frame:
        """Get the knowledge graph frame."""
        return self._frame

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
            # Add token kline to frame
            self._frame.add(KLine(signature, nodes=[token], dbg_text=decode))
            ks_nodes.append(signature)
            ks_key |= signature

        kline = KLine(signature=ks_key, nodes=ks_nodes, dbg_text=text)
        self.rationalise(kline)
        self._frame.add(kline)
        return kline

    def decode(self, token_sig: KSig) -> str:
        """Decode a list of KNodes (token IDs) back to a string.

        Args:
            nodes: List of KNode integers (token IDs)

        Returns:
            Decoded string
        """
        kline = self.frame.find_kline(token_sig)
        if kline is None:
            return ""
        knodes = []
        for node in kline.as_node_list():
            knode = self.frame.find_kline(node)
            if knode:
                # Get first node from knode's node list
                node_list = knode.as_node_list()
                if node_list:
                    knodes.append(node_list[0])
        return self._tokenizer.decode(knodes)

    def prune(self, level: int = 1) -> "Model":
        """Prune frame to activity level + untracked activity.

        Args:
            level: activity level to keep
        """
        pruned_frame: list[KLine] = []
        pruned_activity = Counter()

        for kline in self._frame:  # retain untracked activity
            if kline.signature not in self._activity:
                pruned_frame.append(kline)

        for key, count in self._activity.items():
            if count >= level:
                kline = self._frame.find_kline(key)
                if kline.signature:
                    pruned_frame.append(kline)
                    pruned_activity[key] = count

        frame = Frame(pruned_frame)
        return Model(frame=frame, activity=pruned_activity)

    def rationalise(self, qk: KLine, frame: KFrame | None = None) -> bool:
        """Rationalise a KLine query in frame context.

        Emits events via the event bus as significance is established.
        - S1 and S4 results (significants) are fast tracked
        - S2 and S3 results (rationals) are queued for cogitation.

        Args:
            kline: KLine to rationalise
            frame: Optional KFrame frame context
        
        Returns:
            True if significant (S1, S4), False if rational (S2, S3)
        """

        is_top_level = frame is None
        if is_top_level:
            frame = self._get_frame()

        # Test early to prevent infinite recursion
        if frame.exists(qk):
            if is_top_level:
                self._emit("complete", qk, qk)
            return True

        # Identity (S1)
        if self._significance.equal(qk.signature, qk.nodes):
            frame.add(qk)
            self._emit("fast", qk, qk)
            return True

        # Unsigned (S4)
        if self._significance.is_unsigned(qk.nodes):
            frame.add(qk)
            self._emit("fast", qk, qk)
            return True

        # Expand query into frame and rationalise
        fk_list = list(frame.query(qk, 100)) if self._dev else frame.query(qk, 100)
        for fk in fk_list:
            sig = self._signify(frame, qk, fk)
            if sig == self._significance.S1 or sig == self._significance.S4:
                self._frame.add(qk)
                self._emit("fast", qk, fk)
            else:
                # rational (S2, S3): queue for cogitate background thread
                with self._backlog_condition:
                    self._backlog.append((frame, qk, fk, sig))
                    self._backlog_condition.notify()

        return False

    def _cogitate(self) -> None:
        """Background thread that processes rational klines (S2, S3).
        """
        while not self._cogitate_stop.is_set():
            with self._backlog_condition:
                while not self._backlog and not self._cogitate_stop.is_set():
                    self._backlog_condition.wait(timeout=0.5)
                if self._cogitate_stop.is_set() and not self._backlog:
                    return
                frame, qk, sk, sig = self._backlog.pop()

            self.rationalise(qk, frame)

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate background thread and wait for it to finish."""
        self._cogitate_stop.set()
        with self._backlog_condition:
            self._backlog_condition.notify()
        self._cogitate_thread.join(timeout=timeout)


     # === Significance calculation ===

    def frame_size(self) -> int:
        """Return the number of KLines in the frame.

        Returns:
            Number of KLines stored in the frame
        """
        return len(self._frame)

    # === Serialization ===

    def to_bytes(self) -> bytes:
        """Serialize frame to binary format.

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
        parts.append(pack("<I", len(self._frame)))
        for kline in self._frame:
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
    def from_bytes(cls, data: bytes) -> "Model":
        """Deserialize frame from binary format."""
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

        frame = Frame(klines=klines)
        return cls(frame=frame, activity=activity, dictionary=dictionary, nlp_detail=nlp_detail)

    def to_dict(self) -> dict:
        """Serialize frame to dict (for JSON)."""
        return {
            "metadata": {
                "dictionary": self._dictionary_path,
                "nlp_detail": self._nlp_detail,
            },
            "klines": [{"signature": kline.signature, "nodes": kline.as_node_list()} for kline in self._frame],
            "activity": {str(k): v for k, v in self._activity.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        """Deserialize frame from dict."""
        from collections import Counter

        # Extract metadata if present
        metadata = data.get("metadata", {})
        dictionary = metadata.get("dictionary")
        nlp_detail = metadata.get("nlp_detail", "nlp_type32")

        klines = [KLine(signature=item["signature"], nodes=item["nodes"]) for item in data["klines"]]
        frame = Frame(klines=klines)
        activity = Counter()
        if "activity" in data:
            activity.update({KSig(k): v for k, v in data["activity"].items()})
        return cls(frame=frame, activity=activity, dictionary=dictionary, nlp_detail=nlp_detail)

    def save(
        self,
        path: str | Path,
        format: Literal["bin", "json"] | None = None,
    ) -> None:
        """Save frame to file.

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
        path: str | Path = "data/model.bin",
        format: Literal["bin", "json"] | None = None,
    ) -> "Model":
        """Load frame from file.

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
