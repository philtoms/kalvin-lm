#!/usr/bin/env python3
"""Tag a BPE vocabulary with NLP types from a grammar dictionary.

Takes a trained BPE tokenizer (from ``Tokenizer.train()``) and an NLP grammar
dictionary (from ``nlp_analyzer.py``), and produces a new grammar dict where
every BPE token — including subwords — is annotated with NLP type information.

The matching algorithm works **from vocab to grammar**:

1. **Exact match**: If the decoded BPE token string appears as ``text`` in the
   grammar dict, inherit that entry directly.
2. **BPE decomposition**: For each grammar word, BPE-decompose it and build a
   reverse map (subword token → parent words). Subword tokens inherit the
   highest-count parent's NLP types.
3. **Substring fallback**: For remaining alphabetic tokens, find the grammar
   word whose text contains the token as a substring (highest-count parent).
4. **Special-token rules**: Deterministic annotation for whitespace, punctuation,
   digits, and control characters.
5. **Unknown fallback**: Tokens that match none of the above get ``POS_X``
   (nlp_type32 = 65536).

Usage:
    # Tag a trained tokenizer's vocab
    python dev/nlp/tag_vocab.py \\
        --grammar data/tokenizer/simplestories-1_grammar.json \\
        --tokenizer-dir data/tokenizer \\
        --tokenizer-name tokenizer-32768 \\
        --output data/tokenizer/tagged_grammar.json

    # Dry run (print stats only)
    python dev/nlp/tag_vocab.py \\
        --grammar data/tokenizer/simplestories-1_grammar.json \\
        --tokenizer-dir data/tokenizer \\
        --tokenizer-name tokenizer-32768 \\
        --dry-run
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from kalvin.tokenizer import Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────

# POS_X fallback for unknown tokens (bit 16 set).
UNKNOWN_NLP_TYPE = 65536


# ── Vocab loading ─────────────────────────────────────────────────────────


def load_bpe_vocab(tokenizer: Tokenizer) -> dict[int, str]:
    """Load the BPE vocabulary from a trained tokenizer.

    Uses ``get_mergeable_ranks()`` from the underlying rustbpe tokenizer to
    get all (bytes, rank) pairs without going through tiktoken.

    Args:
        tokenizer: A trained ``Tokenizer`` instance.

    Returns:
        Dict mapping token ID (rank) to decoded string.
    """
    # Access the underlying rustbpe tokenizer
    inner = tokenizer._tokenizer
    if hasattr(inner, "get_mergeable_ranks"):
        ranks = inner.get_mergeable_ranks()
        vocab = {}
        for raw_bytes, rank in ranks:
            vocab[rank] = raw_bytes.decode("utf-8", errors="replace")
        return vocab

    # Fallback: decode individual token IDs
    vocab = {}
    for i in range(tokenizer.vocab_size):
        vocab[i] = tokenizer.decode([i])
    return vocab


def load_bpe_vocab_from_bin(bin_path: Path) -> dict[int, str]:
    """Load BPE vocab directly from a .bin file.

    Each entry has 'b64' (base64-encoded bytes) and 'rank' (the token ID).

    Args:
        bin_path: Path to the tokenizer .bin file.

    Returns:
        Dict mapping token ID to decoded string.
    """
    data = json.loads(bin_path.read_text())
    return {
        item["rank"]: base64.b64decode(item["b64"]).decode("utf-8", errors="replace")
        for item in data
    }


# ── Grammar index ─────────────────────────────────────────────────────────


class GrammarIndex:
    """Pre-processed grammar dict for fast matching.

    Builds several lookup structures from a grammar dict:

    - ``by_text``: Maps lowercase text → (text, count, entry) for exact matching.
    - ``bpe_parents``: Maps BPE token ID → list of (count, text, entry) for
      subword inheritance via BPE decomposition.
    - ``sorted_entries``: All entries sorted by count descending for substring
      fallback matching.
    """

    def __init__(self, grammar: dict, bpe_tokenizer: Tokenizer | None = None):
        self.by_text: dict[str, tuple[str, int, dict]] = {}
        self.bpe_parents: dict[int, list[tuple[int, str, dict]]] = {}
        self.sorted_entries: list[tuple[int, str, dict]] = []

        # Build text lookup
        for tid_str, entry in grammar.items():
            text = entry.get("text", "")
            count = entry.get("count", 0)
            if not text:
                continue
            key = text.lower()
            if key not in self.by_text or count > self.by_text[key][1]:
                self.by_text[key] = (text, count, entry)
            self.sorted_entries.append((count, text, entry))

        # Sort by count descending
        self.sorted_entries.sort(key=lambda x: -x[0])

        # Build BPE decomposition map
        if bpe_tokenizer is not None:
            self._build_bpe_map(bpe_tokenizer)

    def _build_bpe_map(self, bpe_tokenizer: Tokenizer) -> None:
        """BPE-decompose each grammar word and build reverse subword map."""
        for _count, text, entry in self.sorted_entries:
            if not text.strip():
                continue
            try:
                bpe_ids = bpe_tokenizer.encode(text)
            except Exception:
                continue
            if len(bpe_ids) <= 1:
                continue
            for bpe_id in bpe_ids:
                if bpe_id not in self.bpe_parents:
                    self.bpe_parents[bpe_id] = []
                self.bpe_parents[bpe_id].append((_count, text, entry))

    def find_exact(self, token_text: str) -> dict | None:
        """Find an exact text match in the grammar (case-insensitive)."""
        hit = self.by_text.get(token_text.lower())
        return hit[2] if hit else None

    def find_bpe_parent(self, bpe_id: int) -> dict | None:
        """Find the best parent for a BPE subword via decomposition map."""
        candidates = self.bpe_parents.get(bpe_id)
        if not candidates:
            return None
        # Highest count, then longest text
        candidates.sort(key=lambda c: (-c[0], -len(c[1])))
        return candidates[0][2]

    def find_substring(self, token_text: str) -> dict | None:
        """Find the best parent by substring match (fallback)."""
        fragment = token_text.lower()
        for _count, text, entry in self.sorted_entries:
            if fragment in text.lower():
                return entry
        return None


# ── Special-token classification ───────────────────────────────────────────


def classify_special(token_str: str) -> dict | None:
    """Classify a token as special (whitespace, punct, digit, control).

    Returns a partial grammar entry with pos/pos_fine/dep/morph, or None
    if the token is not a special token.
    """
    if re.match(r"^[\x00-\x08\x0b\x0c\x0e-\x1f]+$", token_str):
        return {"pos": "X", "pos_fine": "", "dep": "", "morph": ""}
    if re.match(r"^\s+$", token_str):
        return {"pos": "SPACE", "pos_fine": "_SP", "dep": "", "morph": ""}
    if re.match(r"^\d+$", token_str):
        return {"pos": "NUM", "pos_fine": "CD", "dep": "nummod", "morph": ""}
    if re.match(r"^[^\w\s]+$", token_str):
        return {"pos": "PUNCT", "pos_fine": _punct_fine_tag(token_str), "dep": "punct", "morph": ""}
    return None


def _punct_fine_tag(token_str: str) -> str:
    """Map a punctuation string to its fine POS tag."""
    first = token_str[0] if token_str else "."
    if first == ".":
        return "."
    if first == ",":
        return ","
    if first in "!?":
        return "."
    if first == ":":
        return ":"
    if first == ";":
        return "."
    if first == '"':
        return "``"
    if first == "'":
        return "''"
    if first == "`":
        return "``"
    if first == "-":
        return "HYPH"
    if first == "\u2014":
        return "NFP"
    if first in "()[]{}":
        return "-LRB-" if first in "([" else "-RRB-"
    return "."


# ── NLP type computation ──────────────────────────────────────────────────


def compute_nlp_type32(pos: str, dep: str, morph: str) -> int:
    """Compute 32-bit NLP type from POS/DEP/MORPH flags.

    Bit layout:
        Bits 0-16:  Coarse POS (17 universal tags)
        Bits 17-24: Simplified dependency groups (8 groups)
        Bits 25-31: Simplified morph features (7 features)
    """
    result = 0

    # POS bits (0-16)
    pos_map = {
        "ADJ": 1,
        "ADP": 2,
        "ADV": 4,
        "AUX": 8,
        "CCONJ": 16,
        "DET": 32,
        "INTJ": 64,
        "NOUN": 128,
        "NUM": 256,
        "PART": 512,
        "PRON": 1024,
        "PROPN": 2048,
        "PUNCT": 4096,
        "SCONJ": 8192,
        "SYM": 16384,
        "VERB": 32768,
        "X": 65536,
        "SPACE": 131072,
    }
    result |= pos_map.get(pos, 65536)  # Default to X

    # DEP bits (17-24)
    dep_map = {
        "nsubj": 1 << 17,
        "nsubjpass": 1 << 17,
        "csubj": 1 << 17,
        "obj": 1 << 18,
        "iobj": 1 << 18,
        "dobj": 1 << 18,
        "root": 1 << 19,
        "amod": 1 << 20,
        "advmod": 1 << 20,
        "det": 1 << 21,
        "case": 1 << 21,
        "mark": 1 << 21,
        "aux": 1 << 22,
        "auxpass": 1 << 22,
        "cop": 1 << 22,
        "compound": 1 << 23,
        "flat": 1 << 23,
        "punct": 1 << 24,
        "neg": 1 << 24,
    }
    if dep in dep_map:
        result |= dep_map[dep]

    # MORPH bits (25-31)
    if "Number=Sing" in morph:
        result |= 1 << 25
    if "Number=Plur" in morph:
        result |= 1 << 26
    if "Tense=Past" in morph:
        result |= 1 << 27
    if "Tense=Pres" in morph:
        result |= 1 << 28
    if "VerbForm=Inf" in morph:
        result |= 1 << 29
    if "VerbForm=Part" in morph:
        result |= 1 << 30
    if "Polarity=Neg" in morph:
        result |= 1 << 31

    return result


# ── Tagging engine ─────────────────────────────────────────────────────────


def tag_vocab(
    vocab: dict[int, str],
    grammar_index: GrammarIndex,
) -> dict[str, dict]:
    """Tag every BPE token with NLP type information.

    For each token in the vocab, establishes the nearest grammar word and
    inherits its NLP type. Matching priority:

    1. Exact text match in grammar.
    2. BPE decomposition parent (if ``GrammarIndex`` was built with a tokenizer).
    3. Substring match against grammar texts.
    4. Special-token classification (whitespace, punct, digit, control).
    5. POS_X fallback for unresolvable tokens.

    Args:
        vocab: Dict mapping BPE token ID to decoded string.
        grammar_index: Pre-built ``GrammarIndex`` for fast matching.

    Returns:
        New grammar dict keyed by string BPE token IDs, with full NLP annotations.
    """
    tagged: dict[str, dict] = {}

    # Tracking stats
    stats = {
        "exact": 0,
        "bpe_parent": 0,
        "substring": 0,
        "special": 0,
        "unknown": 0,
    }

    for token_id, token_str in vocab.items():
        entry = _resolve_token(token_id, token_str, grammar_index, stats)
        tagged[str(token_id)] = entry

    total = len(vocab)
    print("\nTagging Results")
    print(f"{'=' * 50}")
    print(f"Total vocab tokens:     {total:>8,}")
    print(f"  Exact match:          {stats['exact']:>8,}")
    print(f"  BPE decomposition:    {stats['bpe_parent']:>8,}")
    print(f"  Substring fallback:   {stats['substring']:>8,}")
    print(f"  Special tokens:       {stats['special']:>8,}")
    print(f"  Unknown (POS_X):      {stats['unknown']:>8,}")
    coverage = (total - stats["unknown"]) / total * 100 if total else 0
    print(f"  Coverage:             {coverage:>7.1f}%")
    print(f"{'=' * 50}\n")

    return tagged


def _is_space_prefixed_alpha(s: str) -> bool:
    """Check if string is a space followed by purely alphabetic characters."""
    if not s or not s[0].isspace():
        return False
    rest = s.lstrip()
    return bool(rest) and rest.isalpha()


def _resolve_token(
    token_id: int,
    token_str: str,
    grammar_index: GrammarIndex,
    stats: dict[str, int],
) -> dict:
    """Resolve a single vocab token to a grammar entry."""
    # 1. Exact match
    parent = grammar_index.find_exact(token_str)
    if parent is not None:
        stats["exact"] += 1
        return _make_entry(token_id, token_str, parent)

    # 1b. Exact match with leading space stripped (BPE tokens like " the")
    if _is_space_prefixed_alpha(token_str):
        parent = grammar_index.find_exact(token_str.lstrip())
        if parent is not None:
            stats["exact"] += 1
            return _make_entry(token_id, token_str, parent)

    # 2. BPE decomposition parent
    parent = grammar_index.find_bpe_parent(token_id)
    if parent is not None:
        stats["bpe_parent"] += 1
        return _make_entry(token_id, token_str, parent)

    # 3. Substring match (alphabetic or space-prefixed alphabetic)
    if token_str.isalpha() or _is_space_prefixed_alpha(token_str):
        # Strip leading space for matching against grammar texts
        match_text = token_str.lstrip()
        parent = grammar_index.find_substring(match_text)
        if parent is not None:
            stats["substring"] += 1
            return _make_entry(token_id, token_str, parent)

    # 4. Special-token rules
    special = classify_special(token_str)
    if special is not None:
        stats["special"] += 1
        pos = special["pos"]
        pos_fine = special["pos_fine"]
        dep = special["dep"]
        morph = special["morph"]
        return {
            "text": token_str,
            "pos": pos,
            "pos_fine": pos_fine,
            "dep": dep,
            "morph": morph,
            "count": 0,
            "tokens": [token_id],
            "frequency_pct": 0.0,
            "nlp_type32": compute_nlp_type32(pos, dep, morph),
        }

    # 5. Unknown fallback — POS_X
    stats["unknown"] += 1
    return {
        "text": token_str,
        "pos": "X",
        "pos_fine": "",
        "dep": "",
        "morph": "",
        "count": 0,
        "tokens": [token_id],
        "frequency_pct": 0.0,
        "nlp_type32": UNKNOWN_NLP_TYPE,
    }


def _make_entry(token_id: int, token_str: str, parent: dict) -> dict:
    """Create a grammar entry inheriting from a parent word."""
    return {
        "text": token_str,
        "pos": parent.get("pos", "X"),
        "pos_fine": parent.get("pos_fine", ""),
        "dep": parent.get("dep", ""),
        "morph": parent.get("morph", ""),
        "count": 0,
        "tokens": [token_id],
        "frequency_pct": 0.0,
        "nlp_type32": parent.get("nlp_type32", UNKNOWN_NLP_TYPE),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag a BPE vocabulary with NLP types from a grammar dictionary",
    )
    parser.add_argument(
        "--grammar",
        type=Path,
        required=True,
        help="Path to the NLP grammar dictionary JSON (from nlp_analyzer.py)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Directory containing the BPE tokenizer files (default: data/tokenizer)",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="tokenizer-32768",
        help="Tokenizer variant name (default: tokenizer-32768)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the tagged grammar JSON (default: same as --grammar with _tagged suffix)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing the output file",
    )
    args = parser.parse_args()

    # Load grammar
    print(f"Loading grammar: {args.grammar}")
    grammar = json.loads(args.grammar.read_text())
    print(f"  {len(grammar)} grammar entries")

    # Load BPE tokenizer (for BPE decomposition of grammar words)
    print(f"Loading tokenizer: {args.tokenizer_dir}/{args.tokenizer_name}")
    bpe_tokenizer = Tokenizer.from_directory(str(args.tokenizer_dir), args.tokenizer_name)
    print(f"  vocab_size={bpe_tokenizer.vocab_size}")

    # Load BPE vocab directly from .bin file (authoritative source)
    bin_path = args.tokenizer_dir / f"{args.tokenizer_name}.bin"
    vocab = load_bpe_vocab_from_bin(bin_path)
    print(f"  loaded {len(vocab)} tokens from {bin_path.name}")

    # Sanity: vocab size must match tokenizer
    assert len(vocab) == bpe_tokenizer.vocab_size, (
        f"Vocab mismatch: .bin has {len(vocab)} tokens, "
        f"tokenizer reports {bpe_tokenizer.vocab_size}"
    )

    # Build grammar index (with BPE decomposition)
    print("Building grammar index with BPE decomposition...")
    grammar_index = GrammarIndex(grammar, bpe_tokenizer)
    print(
        f"  {len(grammar_index.by_text)} unique texts, "
        f"{len(grammar_index.bpe_parents)} BPE subword mappings"
    )

    # Tag every vocab token
    print("Tagging vocab tokens...")
    tagged = tag_vocab(vocab, grammar_index)

    # Write output
    if args.dry_run:
        print("Dry run — not writing output file.")
    else:
        out_path = args.output or args.grammar.parent / (
            args.grammar.stem.rsplit("_", 1)[0] + "_tagged_grammar.json"
        )
        print(f"Writing tagged grammar to: {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tagged, f, indent=2, ensure_ascii=False)
        print(f"Done. Wrote {len(tagged)} entries.")


if __name__ == "__main__":
    main()
