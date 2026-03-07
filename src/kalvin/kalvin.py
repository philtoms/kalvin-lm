"""Kalvin - Knowledge graph with tokenization support."""

from __future__ import annotations

from pathlib import Path
from struct import pack, unpack
from typing import Literal, Any
from collections import Counter
import json

from kalvin.agent import KAgent
from kalvin.model import KLine, KNode, KNone, Model
from kalvin.significance import (
    calculate_significance,
    has_s1,
    get_s2,
    get_s3,
    S1, S2, S3, S4
)
from kalvin.tokenizer import Tokenizer


class Kalvin(KAgent):
    """Kalvin agent with NLP features for knowledge graph operations."""

    __dictionary: dict[int, Any] = {}
    __nlp_type: dict[str, Any] = {}

    def __init__(
        self,
        model: Model | None = None,
        activity: Counter | None = None,
        tokenizer: Tokenizer | None = None,
        dictionary: str | None = None,
        nlp_detail: str = "nlp_type32"
    ):
        """Initialize Kalvin with optional model and tokenizer.

        Args:
            model: Optional Model instance
            tokenizer: Optional Tokenizer instance
            dictionary: Path to grammar dictionary JSON file
            nlp_detail: NLP detail level for type encoding (e.g., "nlp_type32")
        """
        self._model = model if model else Model()
        self._tokenizer = tokenizer if tokenizer else Tokenizer.from_directory()
        self._ws_token = self._tokenizer.encode(" ")[0]
        self._activity = activity if activity else Counter()
        self._unrecognised_tokens = set()
        self._dictionary_path = dictionary
        self._nlp_detail = nlp_detail
        self._dictionary: dict[int, Any] = {}
        self._nlp_type: dict[str, Any] = {}

        # === Tokenization ===
        if not self._dictionary_path:
            self._dictionary = Kalvin.__dictionary
            self._nlp_type = Kalvin.__nlp_type
            self._dictionary_path = "data/tokenizer/simplestories-1_grammar.json"

        if not self._dictionary:
            with open(self._dictionary_path, "r") as f:
                str_dict = json.load(f)
                self._dictionary = {}

            for key, value in str_dict.items():
                key = int(key)
                self._dictionary[key] = value
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
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer for encoding/decoding text."""
        return self._tokenizer

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
            self.rationalise(KLine(signature, nodes=[token], dbg_text=decode), True)
            ks_nodes.append(signature)
            ks_key |= signature

        kline = KLine(signature=ks_key, nodes=ks_nodes, dbg_text=text)
        self.rationalise(kline, True) 
        return kline

    def decode(self, token_sig: int | None) -> str:
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
                knodes.append(knode.nodes[0])
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
        return Kalvin(model, pruned_activity)

    def rationalise(self, kline: KLine, train: bool = False):
        """Rationalise a KLine.

        Args:
            kline: KLine to rationalise
            train: If True, enforce training mode (dedup by signature only)
        """
        fast, slow = self._model.query(kline.signature)
        for kf in fast:
            if kf.signature == self._ws_token:
                continue

            if kf.signature == kline.signature:
                return

            # CS = calculate_significance(self.model,kline,kf)
            # if has_s1(CS):
            #     return

        if not self._model.add(kline, train):
            return

    def signify(self, k1: KLine, k2: KLine, S: int | None = None) -> int:
        """Establish significance relationship between two KLines.

        Calculates internal significance of k1:k2 relationship.
        If requested s is higher (more significant) than internal:
        - S1: Adds bidirectional links, returns S1
        - S2: Verifies compound signature of k2.nodes == k1.signature
        - S3: Adds bidirectional links, returns S3

        Args:
            k1: First KLine
            k2: Second KLine
            s: Optional requested significance level (S1/S2/S3 bit flags)

        Returns:
            The resulting significance value
        """
        # Calculate internal significance
        CS = calculate_significance(self._model, k1, k2)

        # No requested level or internal already sufficient
        if S is None or CS >= S:
            return CS

        # S1 requested and higher than internal
        if has_s1(S):
            self._model.add(KLine(signature=k1.signature, nodes=[k2.signature]))
            self._model.add(KLine(signature=k2.signature, nodes=[k1.signature]))
            return S1
        # S2 requested - verify compound signature
        if get_s2(S) > 0:
            compound = 0
            for node in k2.nodes:
                compound |= node
            verify_sig = compound & k1.signature
            # Signatures must be fully overlapping
            if compound == verify_sig or k1.signature == verify_sig:
                return S2
            # Verification failed, continue to S3 check

        # S3 requested and higher than internal
        if get_s3(S) > 0:
            self._model.add(KLine(signature=k1.signature, nodes=[k2.signature]))
            return S3

        return CS

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
            parts.append(pack("<Q", kline.signature))
            parts.append(pack("<I", len(kline.nodes)))
            for node in kline.nodes:
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
            nodes: list[KNode] = []
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
        return cls(model, activity, dictionary=dictionary, nlp_detail=nlp_detail)

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
            activity.update({int(k): v for k, v in data["activity"].items()})
        return cls(model, activity, dictionary=dictionary, nlp_detail=nlp_detail)

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
