"""Kalvin - Knowledge graph with tokenization support."""

from dataclasses import dataclass
from pathlib import Path
from struct import pack, unpack
from typing import Literal
from collections import Counter
import json

from kalvin.model import KLine, KNode, Model
from kalvin.tokenizer import Tokenizer


@dataclass
class Kalvin:
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
            nlp_detail: NLP detail type (e.g., "nlp_type32")
        """
        self.model = model if model else Model()
        self.tokenizer = tokenizer if tokenizer else Tokenizer.from_directory()
        self.ws_token = self.tokenizer.encode(" ")[0]
        self.activity = activity if activity else Counter()
        self.unrecognised_tokens = set()
        self.nlp_detail = nlp_detail

        # === Tokenization ===
        if not dictionary:
            dictionary = "/Volumes/USB-Backup/ai/data/tidy-ts/simplestories-1_grammar.json"
        self.dictionary_path = dictionary
        with open(dictionary, "r") as f:
            str_dict = json.load(f)
            self.dictionary = {}

        for key, value in str_dict.items():
            key = int(key)
            self.dictionary[key] = value

        with open(dictionary.replace("grammar", self.nlp_detail), "r") as f:
            self.nlp_type = json.load(f)
    # === Tokenization ===

    def encode(self, text: str, nlp_detail: str = "nlp_type32") -> KLine | None:
        """Encode a string to a KLine.

        Args:
            text: Input string to encode

        Returns:
            KLine representing the encoded text, or None if duplicate
        """
        ks_key = 0
        ks_nodes = []
        tokens = self.tokenizer.encode(text)
        self.activity.update(tokens)
        for token in tokens:
            if token in self.dictionary:
                entry = self.dictionary[token]
                s_key = entry[nlp_detail]
            elif token == self.ws_token:
                s_key = token
            else:
                s_key = self.nlp_type["POS_X"]
                self.unrecognised_tokens.add(token)

            s_key |= token
            self.model.add(KLine(s_key, [token]), True)
            ks_nodes.append(s_key)
            ks_key |= s_key

        kline = KLine(s_key=ks_key, nodes=ks_nodes)
        return kline if self.model.add(kline, True) is not None else None

    def decode(self, token_sig: int | None) -> str:
        """Decode a list of KNodes (token IDs) back to a string.

        Args:
            nodes: List of KNode integers (token IDs)

        Returns:
            Decoded string
        """
        kline = self.model.find_by_key(token_sig)
        if kline is None:
            return ""
        knodes = []
        for node in kline.nodes:
            knode = self.model.find_by_key(node)
            if knode:
                knodes.append(knode.nodes[0])
        return self.tokenizer.decode(knodes)

    def prune(self, level: int = 1) -> "Kalvin":
        """Prune model to activity level + untracked activity.

        Args:
            level: activity level to keep
        """
        pruned_model: list[KLine] = []
        pruned_activity = Counter()

        for kline in self.model:  # retain untracked activity
            if kline.s_key not in self.activity:
                pruned_model.append(kline)

        for key, count in self.activity.items():
            if count >= level:
                kline = self.model.find_by_key(key)
                if kline:
                    pruned_model.append(kline)
                    pruned_activity[key] = count

        model = Model(pruned_model)
        return Kalvin(model, pruned_activity)

    def model_size(self) -> int:
        """Return the number of KLines in the model.

        Returns:
            Number of KLines stored in the model
        """
        return len(self.model)

    # === Serialization ===

    def to_bytes(self) -> bytes:
        """Serialize model to binary format.

        Binary layout:
        - 4 bytes: metadata length (uint32)
        - N bytes: JSON-encoded metadata (UTF-8)
        - 4 bytes: number of klines (uint32)
        - For each kline:
          - 8 bytes: s_key (uint64)
          - 4 bytes: node count (uint32)
          - N * 8 bytes: nodes (uint64 each)
        - 4 bytes: number of activity entries (uint32)
        - For each activity entry:
          - 8 bytes: key (uint64)
          - 4 bytes: count (uint32)
        """
        # Serialize metadata
        metadata = {
            "dictionary": self.dictionary_path,
            "nlp_detail": self.nlp_detail,
        }
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        parts = [pack("<I", len(metadata_bytes)), metadata_bytes]

        # Serialize klines
        parts.append(pack("<I", len(self.model)))
        for kline in self.model:
            parts.append(pack("<Q", kline.s_key))
            parts.append(pack("<I", len(kline.nodes)))
            for node in kline.nodes:
                parts.append(pack("<Q", node))

        # Serialize activity Counter
        activity = self.activity
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
            s_key = unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            node_count = unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            nodes: list[KNode] = []
            for _ in range(node_count):
                node = unpack("<Q", data[offset : offset + 8])[0]
                offset += 8
                nodes.append(node)
            klines.append(KLine(s_key=s_key, nodes=nodes))

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
                "dictionary": self.dictionary_path,
                "nlp_detail": self.nlp_detail,
            },
            "klines": [{"s_key": kline.s_key, "nodes": kline.nodes} for kline in self.model],
            "activity": {str(k): v for k, v in self.activity.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Kalvin":
        """Deserialize model from dict."""
        from collections import Counter

        # Extract metadata if present
        metadata = data.get("metadata", {})
        dictionary = metadata.get("dictionary")
        nlp_detail = metadata.get("nlp_detail", "nlp_type32")

        klines = [KLine(s_key=item["s_key"], nodes=item["nodes"]) for item in data["klines"]]
        model = Model(klines=klines)
        activity = Counter()
        if "activity" in data:
            activity.update({int(k): v for k, v in data["activity"].items()})
        return cls(model, activity, dictionary=dictionary, nlp_detail=nlp_detail)

    def save(
        self,
        path: str | Path,
        format: Literal["binary", "json"] = "binary",
    ) -> None:
        """Save model to file.

        Args:
            path: File path to save to
            format: 'binary' (default, compact) or 'json' (human-readable)
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "binary"

        if format == "binary":
            path.write_bytes(self.to_bytes())
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path = "data/kalvin.bin",
        format: Literal["binary", "json"] | None = None,
    ) -> "Kalvin":
        """Load model from file.

        Args:
            path: File path to load from
            format: 'binary', 'json', or None (auto-detect from extension)

        Returns:
            Loaded Model instance
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "binary"

        if format == "binary":
            return cls.from_bytes(path.read_bytes())
        else:
            return cls.from_dict(json.loads(path.read_text()))
