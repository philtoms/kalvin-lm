"""Agent - Knowledge graph with tokenization support."""

from __future__ import annotations

import threading
from pathlib import Path
from struct import pack, unpack
from typing import Literal, Any, cast
from collections import Counter

import json

from kalvin.abstract import KAgent, KLine, KModel, KNodes, KSig, KTokenizer, KSignificance, KGraph
from kalvin.events import EventBus, RationaliseEvent
from kalvin.model import Model
from kalvin.significance import Int32Significance
from kalvin.tokenizer import Tokenizer


class Agent(KAgent):
    """KLine agent."""

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
        nlp_detail: str = "nlp_type32",
        dev: bool = False,
    ):
        """Initialize Agent with optional frame, tokenizer, and significance.

        Args:
            frame: Optional Model instance (base frame for the agent)
            activity: Optional Counter for tracking token activity
            tokenizer: Optional KTokenizer instance
            significance: Optional KSignificance instance (defaults to Int32Significance)
            dictionary: Path to grammar dictionary JSON file
            nlp_detail: NLP detail level for type encoding (e.g., "nlp_type32")
        """
        self._dev = dev
        self._model = model if model else Model()
        self._tokenizer = tokenizer if tokenizer else Tokenizer.from_directory()
        self._sig = significance if significance else Int32Significance()
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
        self._backlog: list[tuple[KLine, KModel]] = []
        self._cogitate_stop = threading.Event()
        self._cogitate_thread = threading.Thread(target=self._cogitate, daemon=True)
        self._cogitate_thread.start()

        # === Tokenization ===
        if not self._dictionary_path:
            self._dictionary = Agent.__dictionary
            self._nlp_type = Agent.__nlp_type
            self._dictionary_path = "data/tokenizer/simplestories-1_grammar.json"

        if not self._dictionary:
            with open(self._dictionary_path, "r") as f:
                self._dictionary = json.load(f)
                Agent.__dictionary = self._dictionary

            with open(self._dictionary_path.replace("grammar", self._nlp_detail), "r") as f:
                self._nlp_type = json.load(f)
                Agent.__nlp_type = self._nlp_type

    def _emit(self, kind: str, query: KLine, value: KLine, significance: int) -> None:
        """Emit a rationalisation event."""
        self._event_bus.publish(RationaliseEvent(kind, query, value,significance))

    def _get_frame(self) -> KModel:
        """Return the current frame context if it is in bounds, otherwise create a new one.

        Returns:
            New KModel frame instance
        """
        if len(self.__frames) > 1 and len(self.__frames[-1].klines) < 100:
            return self.__frames[-1]
        
        frame = Model([], self._model)
        self.__frames.append(frame)
        return frame
    
    def _signify(self, kline: KLine, frame: KModel) -> KSig:
        """Calculate the significance of a kline signature if it exists.
        
        Args:
            kline: the kline to be signified
            frame: the current model frame used to ground S1
        """
        sig = kline.signature
        nodes = kline.nodes

        # Identity: S1
        if self._sig.is_identity(sig, nodes):
            return self._sig.S1

        # Unsigned: S4
        if self._sig.is_unsigned(nodes):
            return self._sig.S4

        # Signed: S1 | S3
        if self._sig.is_signed(nodes):
            if frame.exists(kline):
                nodes_sig = cast(KSig, nodes)
                if self._tokenizer.is_literal(nodes_sig):
                    return self._sig.S1
                if frame.exists(KLine(nodes_sig, sig)):
                    return self._sig.S1
            return self._sig.S3

        # Canonisation: S1 | S2 
                  
        nodes_sig = self._tokenizer.make_signature(cast(list,nodes))
        
        if sig == nodes_sig:
            for n in kline.as_node_list():
                if not frame.find_kline(nodes_sig) and not self._tokenizer.is_literal(n):
                    # Partially canonised: S2
                    return self._sig.S2
                
            # Fully canonised: S1
            return self._sig.S1

        # Underfit|Overfit: S2
        if sig & nodes_sig:
            return self._sig.S2 #if sig.bit_count() > n_sig.bit_count() else self._sig.S3 

        # Connotation: S3
        return self._sig.S3

    def _rational_signify(self, qk: KLine, level: KSig, frame: KModel, graph: KGraph) -> bool:
        # Expand query into frame and rationalise
        for fk in graph:
            sig = self._calculate(qk, fk, frame)
            if sig == self._sig.S1:
                self._model.add(fk)
                return True
        return False
        
    def _calculate(self, query: KLine, target: KLine, frame: KModel) -> KSig:
        """Calculate rational significance between query and target KLines.

        Significance is comparable - higher = more significant.
        S1 > S2 > S3 > S4.

        Args:
            frame: The Model containing the KLines (for descendant lookup)
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
            return self._sig.S4

        min_len = min(len(query_nodes), len(target_nodes))

        # Count S1 matches: positional equality (up to min length)
        s1_match_positions = set(
            i for i in range(min_len) if self._sig.is_identity(query_nodes[i], target_nodes[i])
        )
        s1_matches = len(s1_match_positions)

        # S1: signed nodes match
        if query.is_signed() and target.is_signed():
            if self._sig.is_identity(query.nodes, target.nodes):
                return self._sig.S1
            
        # S2 -> S1: All nodes match
        if query.is_canonized() and target.is_canonized():                
            if s1_matches == min_len and len(query_nodes) == len(target_nodes):
                return self._sig.S1

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
            return self._sig.build_s2(s1_pct, s2_pct)

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
            if node_kline:
                node_children = set(node_kline.as_node_list())
                if node_children & target_set:
                    s3_s2_matches += 1

        s3_s2_pct = (
            (s3_s2_matches * 100) // len(query_nodes) if query_nodes else 0
        )

        # S3-Generational: descendants of unmatched query nodes match descendants of unmatched target nodes
        unmatched_q = [n for n in query_nodes if n not in target_set]
        unmatched_t = [n for n in target_nodes if n not in query_set]
        if unmatched_q and unmatched_t:
            q_desc: set[KNodes] = set()
            for node in unmatched_q:
                q_desc.update(frame.get_all_descendants(node))
            t_desc: set[KNodes] = set()
            for node in unmatched_t:
                t_desc.update(frame.get_all_descendants(node))
            overlap = q_desc & t_desc
            gen_pct = (len(overlap) * 100) // len(t_desc) if t_desc else 0
        else:
            gen_pct = 0

        if s3_s1_pct > 0 or s3_s2_pct > 0 or gen_pct > 0:
            return self._sig.build_s3(s3_s1_pct, s3_s2_pct, gen_pct)

        # S4: No match
        return self._sig.S4
    
    # === KAgent interface ===

    @property
    def model(self) -> Model:
        """Get the knowledge graph model (base frame)."""
        return self._model

    @property
    def tokenizer(self) -> KTokenizer:
        """Get the tokenizer for encoding/decoding text."""
        return self._tokenizer

    @property
    def significance(self) -> KSignificance:
        """Get the significance instance for S1-S4 operations."""
        return self._sig

    @property
    def events(self) -> EventBus:
        """Get the event bus for subscribing to rationalisation events."""
        return self._event_bus

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
            # self.rationalise(KLine(signature, nodes=[token], dbg_text=decode))
            # Add token kline to frame
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
        for node in kline.as_node_list():
            knode = self.model.find_kline(node)
            if knode:
                # Get first node from knode's node list
                node_list = knode.as_node_list()
                if node_list:
                    knodes.append(node_list[0])
        return self._tokenizer.decode(knodes)

    def prune(self, level: int = 1) -> KAgent:
        """Prune frame to activity level + untracked activity.

        Args:
            level: activity level to keep
        """
        pruned_frame: list[KLine] = []
        pruned_activity = Counter()

        for kline in self._model:  # retain untracked activity
            if kline.signature not in self._activity:
                pruned_frame.append(kline)

        for key, count in self._activity.items():
            if count >= level:
                kline = self._model.find_kline(key)
                if kline:
                    pruned_frame.append(kline)
                    pruned_activity[key] = count

        frame = Model(pruned_frame)
        return Agent(model=frame, activity=pruned_activity)

    def rationalise(self, qk: KLine, frame: KModel | None = None, graph: KGraph | None = None) -> bool:
        """Rationalise a KLine for significance in frame context.

        Emits events via the event bus as significance is established.
        - S1 and S4 results (significants) are fast tracked
        - S2 and S3 results (rationals) are queued for cogitation.

        Args:
            kline: KLine to rationalise
            frame: Optional KModel frame context
        
        Returns:
            True if significant (S1, S4), False if rational (S2, S3)
        """

        is_top_level = frame is None
        if is_top_level:
            frame = self._get_frame()
        
        if graph is None:
            graph = frame.query_graph(qk.signature, 100)

        # Test early to prevent infinite recursion
        if frame.exists(qk):
            if is_top_level:
                self._emit("ground", qk, qk, self._sig.S1)
            return True

        sig_level = self._signify(qk, frame)

        # Significant (S1, S4):
        if sig_level == self._sig.S1 or sig_level == self._sig.S4:
            # bring query kline to attention (dedup=FALSE)
            self._model.add(qk, dedup=False)
            self._emit("frame", qk, qk, sig_level)
            return True

        # Rational (S2, S3): 
        if (sig_level == self._sig.S2 or sig_level == self._sig.S3):
            # try rational significance before committing to long term cogitation
            if self._rational_signify(qk, sig_level, frame, graph):
                if self.rationalise(qk, frame, graph):
                    return True

        # Queue for cogitation
        with self._backlog_condition:
            frame.add(qk, dedup=True)
            self._backlog.append((qk, frame))
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
                qk, frame = self._backlog.pop()

            self.rationalise(qk, frame)

    def cogitate_join(self, timeout: float | None = None) -> None:
        """Stop the cogitate background thread and wait for it to finish."""
        self._cogitate_stop.set()
        with self._backlog_condition:
            self._backlog_condition.notify()
        self._cogitate_thread.join(timeout=timeout)

    def frame_size(self) -> int:
        """Return the number of KLines in the frame.

        Returns:
            Number of KLines stored in the frame
        """
        return len(self._model)

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
    def from_bytes(cls, data: bytes) -> "Agent":
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

        model = Model(klines=klines)
        return cls(model=model, activity=activity, dictionary=dictionary, nlp_detail=nlp_detail)

    def to_dict(self) -> dict:
        """Serialize model to dict (for JSON)."""
        return {
            "metadata": {
                "dictionary": self._dictionary_path,
                "nlp_detail": self._nlp_detail,
            },
            "klines": [{"signature": kline.signature, "nodes": kline.as_node_list()} for kline in self._model],
            "activity": {str(k): v for k, v in self._activity.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
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
        path: str | Path = "data/agent.bin",
        format: Literal["bin", "json"] | None = None,
    ) -> "Agent":
        """Load model from file.

        Args:
            path: File path to load from
            format: 'bin', 'json', or None (auto-detect from extension)

        Returns:
            Loaded Agent instance
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "bin"

        if format == "bin":
            return cls.from_bytes(path.read_bytes())
        else:
            return cls.from_dict(json.loads(path.read_text()))
