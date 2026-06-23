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
5. **Unknown fallback**: Tokens that match none of the above get the
   lowest-selection-probability bit derived from the fine-type legend (the
   residue of the most frequent type), written as the generic ``sig_word``
   key so the kalvin base tokenizer can read the output without any NLP
   coupling.

**sig_word derivation**: each entry's ``sig_word`` is a single 32-bit value
with exactly one bit set. It is derived from a fine-type legend
(``*_nlp_fine_types.json``, passed via ``--fine-types``): the entry's
pos/pos_fine/dep/morph fields are matched against four family maps, and the
rarest matched type (largest nlp_fine_type value) wins. Its position is folded
into 32 bits via ``1 << (position % 32)``. This produces a far sparser type
word than the legacy OR-all ``nlp_type32``, tightening ``signifies`` overlap.

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
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.tokenizer import Tokenizer

# The unknown-token fallback bit is no longer a hardcoded constant; it is
# derived from the fine-type legend in FineTypeMaps.fallback_bit (the residue
# of the lowest-value, most-frequent type). See build_fine_type_maps.


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
    inner = tokenizer._bpe
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
                bpe_ids = bpe_tokenizer.encode_bpe(text)
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


# ── Fine-type sig_word computation ─────────────────────────────────────────
#
# Each output ``sig_word`` is a single 32-bit value with exactly one bit set,
# derived from a fine-type legend (``*_nlp_fine_types.json`` from
# nlp_analyzer.py). The legend maps each NLP type (POS / POS_FINE / DEP /
# MORPH) to an integer ``1 << position``, where ``position`` is its rank in
# the round-robin frequency ordering (frequent type -> low position -> low
# value; rare type -> high position -> high value).
#
# Derivation (see ``compute_sig_word``):
#   1. Match the entry's pos, pos_fine, dep, and each morph feature against
#      four family maps. A field contributes a candidate only when its key
#      exists in the map.
#   2. Select the candidate with the largest nlp_fine_type value — the
#      rarest matched type (most informative).
#   3. Emit ``1 << ((winner.bit_length()-1) % 32)``.
#
# The ``% 32`` folds the unbounded round-robin positions (149 types) into the
# 32-bit sig_word width. Within-token selection is unambiguous (one winner);
# the fold causes between-token collisions (unrelated types share a bit),
# which is the intended cost of 32-bit compression.


def _morph_flag_name(feature: str) -> str:
    """Forward-transform a morph feature to its legend flag name.

    Mirrors ``NLPFineTypeRegistry._flag_name`` in nlp_analyzer.py so a feature
    like "Number=Sing" maps to the exact legend key "MORPH_NUMBER_SING". The
    legend key is not reversed back to a feature string because camelCase keys
    (e.g. ``VerbForm`` -> ``VERBFORM``) are lossy to reverse; lookup always
    forward-transforms.
    """
    if "=" in feature:
        key, value = feature.split("=", 1)
        return f"MORPH_{key.upper()}_{value.upper()}"
    return f"MORPH_{feature.upper()}"


@dataclass
class FineTypeMaps:
    """Four family maps for deriving a single-bit 32-bit ``sig_word``.

    Built from a fine-type legend by ``build_fine_type_maps``. Each family map
    keys a raw grammar field value to its nlp_fine_type integer.

    Attributes:
        pos: ``{POS_TAG: value}`` keyed by uppercase POS (e.g. "DET").
        pos_fine: ``{POS_FINE_TAG: value}`` keyed verbatim (e.g. "DT", ".").
        dep: ``{dep_label: value}`` keyed lowercase (e.g. "det").
        morph: ``{flag_name: value}`` keyed by legend flag name
            (e.g. "MORPH_NUMBER_SING"); lookup forward-transforms each feature.
        fallback_bit: single-bit value used when no family matches — the
            residue of the lowest-value (most frequent) type in the legend.
        by_value: reverse map ``{nlp_fine_type_value: flag_name}`` for naming
            the winning type. Legend values are distinct powers of two, so
            this is injective.
    """

    pos: dict[str, int]
    pos_fine: dict[str, int]
    dep: dict[str, int]
    morph: dict[str, int]
    fallback_bit: int
    by_value: dict[int, str]


def build_fine_type_maps(legend: dict[str, int]) -> FineTypeMaps:
    """Build the four family maps from a fine-type legend.

    Args:
        legend: ``{flag_name: nlp_fine_type_value}`` from
            ``*_nlp_fine_types.json``.

    Returns:
        A populated ``FineTypeMaps``. ``fallback_bit`` is the residue bit of
        the legend's lowest-value type (lowest selection probability).
    """
    pos: dict[str, int] = {}
    pos_fine: dict[str, int] = {}
    dep: dict[str, int] = {}
    morph: dict[str, int] = {}
    for flag_name, value in legend.items():
        if flag_name.startswith("POS_FINE_"):
            pos_fine[flag_name[len("POS_FINE_") :]] = value
        elif flag_name.startswith("POS_"):
            pos[flag_name[len("POS_") :]] = value
        elif flag_name.startswith("DEP_"):
            # Grammar dep fields are lowercase ("det"); legend is uppercase.
            dep[flag_name[len("DEP_") :].lower()] = value
        elif flag_name.startswith("MORPH_"):
            morph[flag_name] = value

    if legend:
        min_value = min(legend.values())
        fallback_bit = 1 << ((min_value.bit_length() - 1) % 32)
    else:
        fallback_bit = 1  # degenerate: empty legend -> bit 0
    # Reverse map value -> flag_name. Legend values are distinct powers of
    # two, so this is injective and unambiguously names a winner.
    by_value = {v: k for k, v in legend.items()}
    return FineTypeMaps(pos, pos_fine, dep, morph, fallback_bit, by_value)


def compute_sig_word(entry: dict, maps: FineTypeMaps) -> tuple[int, str]:
    """Derive a single-bit 32-bit ``sig_word`` and its type name from an entry.

    Matches the entry's ``pos``, ``pos_fine``, ``dep``, and each ``morph``
    feature against ``maps`` and emits ``1 << ((max.bit_length()-1) % 32)``
    where ``max`` is the largest nlp_fine_type value among all matches (the
    rarest matched type). If nothing matches, the bit is ``maps.fallback_bit``
    and the name is ``""`` (no type was selected).

    Args:
        entry: Grammar entry with optional ``pos``, ``pos_fine``, ``dep``,
            ``morph`` fields. Absent or empty fields contribute no candidate.
        maps: The four family maps built from the fine-type legend.

    Returns:
        ``(sig_word, sig_type)`` — a 32-bit integer with exactly one bit set,
        and the winning type's legend flag name (e.g. ``"POS_FINE_NNP"``), or
        ``""`` when the fallback bit was used.
    """
    candidates: list[int] = []

    pos = entry.get("pos", "")
    if pos and pos in maps.pos:
        candidates.append(maps.pos[pos])

    pos_fine = entry.get("pos_fine", "")
    if pos_fine and pos_fine in maps.pos_fine:
        candidates.append(maps.pos_fine[pos_fine])

    dep = entry.get("dep", "")
    if dep and dep.lower() in maps.dep:
        candidates.append(maps.dep[dep.lower()])

    morph = entry.get("morph", "")
    if morph:
        for feature in morph.split("|"):
            feature = feature.strip()
            if feature:
                flag = _morph_flag_name(feature)
                if flag in maps.morph:
                    candidates.append(maps.morph[flag])

    if not candidates:
        return maps.fallback_bit, ""
    winner = max(candidates)
    return 1 << ((winner.bit_length() - 1) % 32), maps.by_value[winner]


# ── Tagging engine ─────────────────────────────────────────────────────────


def tag_vocab(
    vocab: dict[int, str],
    grammar_index: GrammarIndex,
    maps: FineTypeMaps,
) -> dict[str, dict]:
    """Tag every BPE token with NLP type information.

    For each token in the vocab, establishes the nearest grammar word and
    inherits its NLP type. Matching priority:

    1. Exact text match in grammar.
    2. BPE decomposition parent (if ``GrammarIndex`` was built with a tokenizer).
    3. Substring match against grammar texts.
    4. Special-token classification (whitespace, punct, digit, control).
    5. Fallback bit for unresolvable tokens.

    Each resolved entry's ``sig_word`` is a single bit derived via
    ``compute_sig_word`` from the fine-type legend: the rarest matched NLP
    type, folded into 32 bits.

    Args:
        vocab: Dict mapping BPE token ID to decoded string.
        grammar_index: Pre-built ``GrammarIndex`` for fast matching.
        maps: Fine-type family maps for ``sig_word`` derivation.

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
        entry = _resolve_token(token_id, token_str, grammar_index, maps, stats)
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
    maps: FineTypeMaps,
    stats: dict[str, int],
) -> dict:
    """Resolve a single vocab token to a grammar entry."""
    # 1. Exact match
    parent = grammar_index.find_exact(token_str)
    if parent is not None:
        stats["exact"] += 1
        return _make_entry(token_id, token_str, parent, maps)

    # 1b. Exact match with leading space stripped (BPE tokens like " the")
    if _is_space_prefixed_alpha(token_str):
        parent = grammar_index.find_exact(token_str.lstrip())
        if parent is not None:
            stats["exact"] += 1
            return _make_entry(token_id, token_str, parent, maps)

    # 2. BPE decomposition parent
    parent = grammar_index.find_bpe_parent(token_id)
    if parent is not None:
        stats["bpe_parent"] += 1
        return _make_entry(token_id, token_str, parent, maps)

    # 3. Substring match (alphabetic or space-prefixed alphabetic)
    if token_str.isalpha() or _is_space_prefixed_alpha(token_str):
        # Strip leading space for matching against grammar texts
        match_text = token_str.lstrip()
        parent = grammar_index.find_substring(match_text)
        if parent is not None:
            stats["substring"] += 1
            return _make_entry(token_id, token_str, parent, maps)

    # 4. Special-token rules — sig_word derived via the fine-type maps;
    #    values absent from the legend (e.g. POS_SPACE) simply match nothing
    #    and fall through to the fallback bit inside compute_sig_word.
    special = classify_special(token_str)
    if special is not None:
        stats["special"] += 1
        sig_word, sig_type = compute_sig_word(special, maps)
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
            "sig_word": sig_word,
            "sig_type": sig_type,
        }

    # 5. Unknown fallback — no NLP information; use the lowest-selection-
    #    probability bit. ``pos`` is recorded as "X" for traceability only.
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
        "sig_word": maps.fallback_bit,
        "sig_type": "",
    }


def _make_entry(token_id: int, token_str: str, parent: dict, maps: FineTypeMaps) -> dict:
    """Create a grammar entry inheriting from a parent word.

    ``sig_word`` is derived from the parent's NLP fields via the fine-type
    maps (a single bit — the rarest matched type), not copied from the
    parent's multi-bit ``nlp_type32``. ``sig_type`` names the winning type.
    """
    sig_word, sig_type = compute_sig_word(parent, maps)
    return {
        "text": token_str,
        "pos": parent.get("pos", "X"),
        "pos_fine": parent.get("pos_fine", ""),
        "dep": parent.get("dep", ""),
        "morph": parent.get("morph", ""),
        "count": 0,
        "tokens": [token_id],
        "frequency_pct": 0.0,
        "sig_word": sig_word,
        "sig_type": sig_type,
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag a BPE vocabulary with NLP types from a grammar dictionary",
    )
    parser.add_argument(
        "--grammar",
        type=Path,
        default="data/tokenizer/simplestories-1_grammar.json",
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
    parser.add_argument(
        "--fine-types",
        type=Path,
        default=None,
        help=(
            "Path to the fine-type legend JSON (from nlp_analyzer.py's "
            "*_nlp_fine_types.json). Each sig_word is derived as a single "
            "bit: the rarest matched NLP type, folded into 32 bits. Default: "
            "inferred from --grammar ({stem}_nlp_fine_types.json in the same dir)."
        ),
    )
    args = parser.parse_args()

    # Load grammar
    print(f"Loading grammar: {args.grammar}")
    grammar = json.loads(args.grammar.read_text())
    print(f"  {len(grammar)} grammar entries")

    # Load fine-type legend (default: alongside the grammar)
    fine_types_path = args.fine_types
    if fine_types_path is None:
        base = args.grammar.stem.removesuffix("_grammar")
        fine_types_path = args.grammar.parent / f"{base}_nlp_fine_types.json"
    if not fine_types_path.exists():
        raise SystemExit(f"Fine-type legend not found: {fine_types_path}")
    print(f"Loading fine-type legend: {fine_types_path}")
    fine_legend = json.loads(fine_types_path.read_text())
    maps = build_fine_type_maps(fine_legend)
    print(
        f"  {len(fine_legend)} fine types; "
        f"fallback bit = {maps.fallback_bit:#x}"
    )

    # Load BPE tokenizer (for BPE decomposition of grammar words)
    print(f"Loading tokenizer: {args.tokenizer_dir}/{args.tokenizer_name}")
    bpe_tokenizer = NLPTokenizer(
        tokenizer_path=args.tokenizer_dir,
        tokenizer_name=args.tokenizer_name,
    )
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
    tagged = tag_vocab(vocab, grammar_index, maps)

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
