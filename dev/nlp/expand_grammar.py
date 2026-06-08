#!/usr/bin/env python3
"""
Expand NLP grammar dictionary coverage.

Analyzes uncovered BPE tokens in a grammar dictionary and fills gaps using three
mechanisms:

1. **Special-token annotation**: Assigns NLP tags to whitespace, punctuation,
   digits, and control characters using deterministic pattern rules (no spaCy needed).
2. **Subword token inheritance**: Resolves BPE subword fragments by inheriting
   POS/DEP/MORPH from parent words found in the grammar dict.
3. **Multi-source merge**: Combines grammar dicts from multiple sources (e.g.,
   different corpora) without overwriting existing entries.

Usage:
    # Expand grammar with all strategies
    python dev/nlp/expand_grammar.py \\
        --grammar data/tokenizer/simplestories-1_grammar.json \\
        --output data/tokenizer/simplestories-1_grammar.json

    # Dry run (print stats only)
    python dev/nlp/expand_grammar.py \\
        --grammar data/tokenizer/simplestories-1_grammar.json \\
        --dry-run

    # Merge an additional pre-computed grammar
    python dev/nlp/expand_grammar.py \\
        --grammar data/tokenizer/simplestories-1_grammar.json \\
        --extra-grammar data/tokenizer/openwebtext_grammar.json \\
        --output data/tokenizer/simplestories-1_grammar.json
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

from nlp_analyzer import compute_nlp_type32, compute_nlp_type48
from kalvin.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Token vocab loading
# ---------------------------------------------------------------------------

def load_tokenizer_vocab(tokenizer_dir: Path, tokenizer_name: str) -> dict[int, str]:
    """Load the full tokenizer vocabulary: token_id -> decoded string.

    Args:
        tokenizer_dir: Directory containing tokenizer files.
        tokenizer_name: Tokenizer variant name (e.g., 'tokenizer-32768').

    Returns:
        Dict mapping token ID (int) to decoded string.
    """
    t = Tokenizer.from_directory(path=str(tokenizer_dir), name=tokenizer_name)
    vocab = {}
    for i in range(t.vocab_size):
        vocab[i] = t.decode([i])
    return vocab


def load_tokenizer_vocab_from_bin(bin_path: Path) -> dict[int, str]:
    """Load tokenizer vocab directly from the .bin file without tiktoken.

    Each entry in the JSON-wrapped .bin file has 'b64' (base64-encoded bytes)
    and 'rank' (the token ID).

    Args:
        bin_path: Path to tokenizer-32768.bin file.

    Returns:
        Dict mapping token ID (rank) to decoded string.
    """
    data = json.loads(bin_path.read_text())
    vocab = {}
    for item in data:
        token_id = item["rank"]
        decoded = base64.b64decode(item["b64"]).decode("utf-8", errors="replace")
        vocab[token_id] = decoded
    return vocab


# ---------------------------------------------------------------------------
# Step 1: Categorize uncovered tokens
# ---------------------------------------------------------------------------

def categorize_uncovered(
    tokenizer_vocab: dict[int, str],
    grammar: dict,
) -> dict[str, list[int]]:
    """Categorize every uncovered BPE token into descriptive buckets.

    Buckets:
        - subword_fragment: alphabetic but never appears as a standalone word
          in any grammar entry's ``text`` field (e.g., "zing", "yhead")
        - single_letter: single alphabetic character (e.g., "B", "z")
        - digit: matches ``^\\d+$`` (e.g., "0", "19")
        - whitespace: contains only whitespace chars (spaces, ``\\n``, ``\\t``)
        - punctuation: matches ``^[^\\w\\s]+$`` (e.g., "..")
        - control_char: matches ``^[\\x00-\\x1f]+$`` excluding ``\\t`` and ``\\n``
        - rare_word: alphabetic tokens that are full words but not in the
          grammar dict (fallback bucket)

    Args:
        tokenizer_vocab: Full token ID -> decoded string mapping.
        grammar: Existing grammar dict (keys are string token IDs).

    Returns:
        Dict mapping category name -> list of uncovered token IDs.
    """
    covered_ids = {int(k) for k in grammar.keys()}

    # Collect all text values in grammar for subword matching
    grammar_texts = set()
    for entry in grammar.values():
        text = entry.get("text", "")
        if text:
            grammar_texts.add(text)

    categories: dict[str, list[int]] = {
        "subword_fragment": [],
        "single_letter": [],
        "digit": [],
        "whitespace": [],
        "punctuation": [],
        "control_char": [],
        "rare_word": [],
    }

    for token_id, token_str in tokenizer_vocab.items():
        if token_id in covered_ids:
            continue

        # Control characters: \x00-\x1f excluding \t (0x09) and \n (0x0a)
        if re.match(r"^[\x00-\x08\x0b\x0c\x0e-\x1f]+$", token_str):
            categories["control_char"].append(token_id)
        # Whitespace-only
        elif re.match(r"^\s+$", token_str):
            categories["whitespace"].append(token_id)
        # Digit-only
        elif re.match(r"^\d+$", token_str):
            categories["digit"].append(token_id)
        # Punctuation (non-word, non-whitespace)
        elif re.match(r"^[^\w\s]+$", token_str):
            categories["punctuation"].append(token_id)
        # Single alphabetic letter
        elif len(token_str) == 1 and token_str.isalpha():
            categories["single_letter"].append(token_id)
        # Alphabetic but not in grammar texts -> subword fragment
        elif token_str.isalpha() and token_str not in grammar_texts:
            categories["subword_fragment"].append(token_id)
        # Fallback: rare word or mixed content
        else:
            categories["rare_word"].append(token_id)

    return categories


def print_category_summary(
    categories: dict[str, list[int]],
    total_vocab: int,
    covered_count: int,
) -> None:
    """Print a formatted summary of uncovered token categories."""
    uncovered = sum(len(v) for v in categories.values())
    pct = (covered_count / total_vocab * 100) if total_vocab else 0

    print(f"\n{'='*60}")
    print(f"Grammar Coverage Report")
    print(f"{'='*60}")
    print(f"Total vocab tokens:     {total_vocab:>8,}")
    print(f"Covered tokens:         {covered_count:>8,}  ({pct:.1f}%)")
    print(f"Uncovered tokens:       {uncovered:>8,}  ({100-pct:.1f}%)")
    print(f"{'-'*60}")
    print(f"Uncovered by category:")
    for cat, ids in categories.items():
        print(f"  {cat:<22s} {len(ids):>6,}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Step 2: Special-token annotation rules
# ---------------------------------------------------------------------------

def _compute_nlp_fine_type(
    pos: str,
    pos_fine: str,
    dep: str,
    morph: str,
    fine_legend_reverse: dict[str, dict[str, int]],
) -> int:
    """Compute nlp_fine_type from POS/DEP/MORPH using the fine-type legend.

    Args:
        pos: Coarse POS tag.
        pos_fine: Fine POS tag.
        dep: Dependency label.
        morph: Pipe-separated morph features.
        fine_legend_reverse: Pre-built reverse lookup:
            {"pos": {"NOUN": bit_value, ...}, "pos_fine": {...}, "dep": {...}, "morph": {...}}

    Returns:
        Integer fine-type bit pattern.
    """
    result = 0

    if pos and pos in fine_legend_reverse.get("pos", {}):
        result |= fine_legend_reverse["pos"][pos]

    if pos_fine and pos_fine in fine_legend_reverse.get("pos_fine", {}):
        result |= fine_legend_reverse["pos_fine"][pos_fine]

    if dep and dep in fine_legend_reverse.get("dep", {}):
        result |= fine_legend_reverse["dep"][dep]

    if morph:
        for feature in morph.split("|"):
            feature = feature.strip()
            if feature and feature in fine_legend_reverse.get("morph", {}):
                result |= fine_legend_reverse["morph"][feature]

    return result


def build_fine_legend_reverse(fine_legend: dict[str, int]) -> dict[str, dict[str, int]]:
    """Build a reverse lookup from the fine-type legend.

    Converts {"POS_NOUN": 128, "DEP_ROOT": 256, "MORPH_NUMBER_SING": 512}
    into {"pos": {"NOUN": 128}, "dep": {"ROOT": 256}, "morph": {"Number=Sing": 512}}

    Args:
        fine_legend: The fine-type legend dict from ``*_nlp_fine_types.json``.

    Returns:
        Nested dict: category -> value -> bit_value.
    """
    reverse: dict[str, dict[str, int]] = {
        "pos": {},
        "pos_fine": {},
        "dep": {},
        "morph": {},
    }

    for flag_name, bit_value in fine_legend.items():
        if flag_name.startswith("POS_FINE_"):
            # pos_fine category: POS_FINE_NN -> "NN"
            value = flag_name[len("POS_FINE_"):]
            reverse["pos_fine"][value] = bit_value
        elif flag_name.startswith("POS_"):
            # pos category: POS_NOUN -> "NOUN"
            value = flag_name[len("POS_"):]
            reverse["pos"][value] = bit_value
        elif flag_name.startswith("DEP_"):
            # dep category: DEP_ROOT -> "root" (stored lowercase in grammar)
            value = flag_name[len("DEP_"):].lower()
            reverse["dep"][value] = bit_value
        elif flag_name.startswith("MORPH_"):
            # morph category: MORPH_NUMBER_SING -> "Number=Sing"
            raw = flag_name[len("MORPH_"):]
            if "_" in raw:
                parts = raw.split("_", 1)
                value = parts[0].capitalize() + "=" + parts[1].capitalize()
            else:
                value = raw.capitalize()
            reverse["morph"][value] = bit_value

    return reverse


def annotate_special_tokens(
    tokenizer_vocab: dict[int, str],
    grammar: dict,
    fine_legend_reverse: dict[str, dict[str, int]],
) -> dict:
    """Annotate special tokens (whitespace, punctuation, digits, control chars).

    Assigns NLP annotations determinable by pattern alone, without running spaCy.
    Never overwrites existing grammar entries.

    Args:
        tokenizer_vocab: Full token ID -> decoded string mapping.
        grammar: Existing grammar dict (keys are string token IDs).
        fine_legend_reverse: Reverse lookup from fine-type legend.

    Returns:
        Updated grammar dict with new entries added.
    """
    covered_ids = {int(k) for k in grammar.keys()}
    new_entries = 0

    for token_id, token_str in tokenizer_vocab.items():
        if token_id in covered_ids:
            continue

        entry = _classify_special_token(token_id, token_str, fine_legend_reverse)
        if entry is not None:
            grammar[str(token_id)] = entry
            new_entries += 1

    print(f"  Special-token annotation: added {new_entries} new entries")
    return grammar


def _classify_special_token(
    token_id: int,
    token_str: str,
    fine_legend_reverse: dict[str, dict[str, int]],
) -> dict | None:
    """Classify a single token using pattern rules.

    Returns a grammar entry dict, or None if the token doesn't match any
    special-token pattern.
    """
    pos = ""
    pos_fine = ""
    dep = ""
    morph = ""

    # Control characters: \x00-\x1f excluding \t (0x09) and \n (0x0a)
    if re.match(r"^[\x00-\x08\x0b\x0c\x0e-\x1f]+$", token_str):
        pos = "X"
        pos_fine = ""
        dep = ""
        morph = ""
    # Whitespace-only tokens
    elif re.match(r"^\s+$", token_str):
        pos = "SPACE"
        pos_fine = "_SP"
        dep = ""
        morph = ""
    # Digit-only tokens
    elif re.match(r"^\d+$", token_str):
        pos = "NUM"
        pos_fine = "CD"
        dep = "nummod"
        morph = ""
    # Punctuation (non-word, non-whitespace)
    elif re.match(r"^[^\w\s]+$", token_str):
        pos = "PUNCT"
        pos_fine = _punct_fine_tag(token_str)
        dep = "punct"
        morph = ""
    else:
        return None  # Not a special token

    nlp_type32 = compute_nlp_type32(pos, dep, morph)
    nlp_type48 = compute_nlp_type48(pos, dep, morph)
    nlp_fine_type = _compute_nlp_fine_type(pos, pos_fine, dep, morph, fine_legend_reverse)

    return {
        "text": token_str,
        "pos": pos,
        "pos_fine": pos_fine,
        "dep": dep,
        "morph": morph,
        "count": 0,
        "tokens": [token_id],
        "frequency_pct": 0.0,
        "nlp_type32": nlp_type32,
        "nlp_type48": nlp_type48,
        "nlp_fine_type": nlp_fine_type,
    }


def _punct_fine_tag(token_str: str) -> str:
    """Map a punctuation string to its fine POS tag.

    Uses the first character for classification.
    """
    first = token_str[0] if token_str else "."
    # Map common punctuation characters to spaCy fine POS tags
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
    if first == "\u2014":  # em dash
        return "NFP"
    if first in "()":
        return "-LRB-" if first == "(" else "-RRB-"
    if first in "[]":
        return "-LRB-" if first == "[" else "-RRB-"
    if first in "{}":
        return "-LRB-" if first == "{" else "-RRB-"
    return "."


# ---------------------------------------------------------------------------
# Step 3: Subword token inheritance
# ---------------------------------------------------------------------------

def inherit_subword_types(
    tokenizer_vocab: dict[int, str],
    grammar: dict,
    fine_legend_reverse: dict[str, dict[str, int]],
) -> dict:
    """Resolve subword fragment tokens by inheriting from parent words.

    For each uncovered alphabetic subword fragment, search the grammar for a
    parent word whose ``text`` field contains the fragment as a contiguous
    substring. If multiple parents match, prefer the one with the highest
    ``count``.

    Args:
        tokenizer_vocab: Full token ID -> decoded string mapping.
        grammar: Existing grammar dict.
        fine_legend_reverse: Reverse lookup from fine-type legend.

    Returns:
        Updated grammar dict with inherited entries added.
    """
    covered_ids = {int(k) for k in grammar.keys()}

    # Collect all grammar text entries with their counts, sorted by count desc
    grammar_entries = []
    for tid_str, entry in grammar.items():
        text = entry.get("text", "")
        count = entry.get("count", 0)
        if text and text.strip():
            grammar_entries.append((text, count, entry))

    # Sort by count descending (so first match is most frequent)
    grammar_entries.sort(key=lambda x: -x[1])

    # Build a lookup: for each unique lowercase text, keep the best entry
    best_by_text: dict[str, tuple] = {}
    for text, count, entry in grammar_entries:
        key = text.lower()
        if key not in best_by_text:
            best_by_text[key] = (text, count, entry)

    # Collect all parent texts for substring matching
    parent_texts = list(best_by_text.keys())

    resolved = 0
    unresolved = 0

    for token_id, token_str in tokenizer_vocab.items():
        if token_id in covered_ids:
            continue

        # Only process alphabetic tokens that could be subword fragments
        if not token_str.isalpha():
            continue

        # Skip if already covered by special-token annotation
        if str(token_id) in grammar:
            continue

        # Find parent: search for a grammar text containing this fragment
        fragment_lower = token_str.lower()
        parent_entry = None

        for parent_text_lower in parent_texts:
            if fragment_lower in parent_text_lower:
                parent_entry = best_by_text[parent_text_lower]
                break

        if parent_entry is None:
            unresolved += 1
            continue

        _parent_text, _parent_count, parent = parent_entry
        pos = parent.get("pos", "")
        pos_fine = parent.get("pos_fine", "")
        dep = parent.get("dep", "")
        morph = parent.get("morph", "")

        nlp_type32 = compute_nlp_type32(pos, dep, morph)
        nlp_type48 = compute_nlp_type48(pos, dep, morph)
        nlp_fine_type = _compute_nlp_fine_type(pos, pos_fine, dep, morph, fine_legend_reverse)

        grammar[str(token_id)] = {
            "text": token_str,
            "pos": pos,
            "pos_fine": pos_fine,
            "dep": dep,
            "morph": morph,
            "count": 0,
            "tokens": [token_id],
            "frequency_pct": 0.0,
            "nlp_type32": nlp_type32,
            "nlp_type48": nlp_type48,
            "nlp_fine_type": nlp_fine_type,
        }
        resolved += 1

    print(f"  Subword inheritance: resolved {resolved}, unresolved {unresolved}")
    return grammar


# ---------------------------------------------------------------------------
# Step 4: Multi-source merge pipeline
# ---------------------------------------------------------------------------

def merge_grammars(base: dict, *others: dict) -> dict:
    """Merge grammar dicts from different sources.

    - Existing entries in ``base`` are never overwritten by ``others``.
    - New entries from ``others`` are added to ``base``.
    - ``count`` and ``frequency_pct`` from ``others`` are reset to 0.

    Args:
        base: The base grammar dict (modified in place and returned).
        *others: Additional grammar dicts to merge in.

    Returns:
        The merged grammar dict (same object as ``base``).
    """
    for other in others:
        for key, entry in other.items():
            if key not in base:
                # New entry -- add with reset count/frequency
                new_entry = dict(entry)
                new_entry["count"] = 0
                new_entry["frequency_pct"] = 0.0
                base[key] = new_entry
    return base


# ---------------------------------------------------------------------------
# Step 5: Manual annotation for BPE artifacts
# ---------------------------------------------------------------------------

# Hardcoded annotations for BPE tokens that no automated strategy can resolve.
# These are contraction stems (e.g., "didn" from "didn't"), the negation clitic
# "'t", newline-composite punctuation (e.g., '"\n\n'), and the underscore "_".
#
# Contraction stems are BPE fragments that never appear as standalone words in
# any corpus.  The newline-prefixed punctuation tokens are BPE artifacts that
# encode formatting (e.g., closing quote + paragraph break) as single tokens.
_MANUAL_ANNOTATIONS: list[dict] = [
    # -- Contraction stems (auxiliary verb stems without the "'t" negation) --
    # spaCy tags "did" (in "didn't") as AUX/VBD/aux/Tense=Past|VerbForm=Fin
    {"id": 1840, "text": "didn",    "pos": "AUX", "pos_fine": "VBD", "dep": "aux",   "morph": "Tense=Past|VerbForm=Fin"},
    # spaCy tags "could" (in "couldn't") as AUX/MD/aux/VerbForm=Fin
    {"id": 2392, "text": "couldn",  "pos": "AUX", "pos_fine": "MD",  "dep": "aux",   "morph": "VerbForm=Fin"},
    # spaCy tags "was" (in "wasn't") as AUX/VBD/aux/Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
    {"id": 3698, "text": "wasn",    "pos": "AUX", "pos_fine": "VBD", "dep": "aux",   "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin"},
    # spaCy tags "does" (in "doesn't") as AUX/VBZ/aux/Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
    {"id": 4413, "text": "doesn",   "pos": "AUX", "pos_fine": "VBZ", "dep": "aux",   "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"},
    # spaCy tags "is" (in "isn't") as AUX/VBZ/aux/Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
    {"id": 4976, "text": "isn",     "pos": "AUX", "pos_fine": "VBZ", "dep": "aux",   "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"},
    # Capitalized form of "isn"
    {"id": 5470, "text": "Isn",     "pos": "AUX", "pos_fine": "VBZ", "dep": "aux",   "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"},
    # spaCy tags "would" (in "wouldn't") as AUX/MD/aux/VerbForm=Fin
    {"id": 5736, "text": "wouldn",  "pos": "AUX", "pos_fine": "MD",  "dep": "aux",   "morph": "VerbForm=Fin"},
    # spaCy tags "had" (in "hadn't") as AUX/VBD/aux/Tense=Past|VerbForm=Fin
    {"id": 6590, "text": "hadn",    "pos": "AUX", "pos_fine": "VBD", "dep": "aux",   "morph": "Tense=Past|VerbForm=Fin"},
    # spaCy tags "should" (in "shouldn't") as AUX/MD/aux/VerbForm=Fin
    {"id": 6939, "text": "shouldn", "pos": "AUX", "pos_fine": "MD",  "dep": "aux",   "morph": "Polarity=Neg"},
    # spaCy tags "were" (in "weren't") as AUX/VBD/aux/Mood=Ind|Tense=Past|VerbForm=Fin
    {"id": 8868, "text": "weren",   "pos": "AUX", "pos_fine": "VBD", "dep": "aux",   "morph": "Mood=Ind|Tense=Past|VerbForm=Fin"},
    # Capitalized form of "wouldn"
    {"id": 10919, "text": "Wouldn", "pos": "AUX", "pos_fine": "MD",  "dep": "aux",   "morph": "VerbForm=Fin"},
    # Capitalized form of "wasn"
    {"id": 13135, "text": "Wasn",   "pos": "AUX", "pos_fine": "VBD", "dep": "aux",   "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin"},
    # Capitalized form of "shouldn"
    {"id": 15827, "text": "Shouldn","pos": "AUX", "pos_fine": "MD",  "dep": "aux",   "morph": "Polarity=Neg"},
    # Capitalized form of "hadn"
    {"id": 16191, "text": "Hadn",   "pos": "AUX", "pos_fine": "VBD", "dep": "aux",   "morph": "Tense=Past|VerbForm=Fin"},
    # Capitalized form of "doesn"
    {"id": 16534, "text": "Doesn",  "pos": "AUX", "pos_fine": "VBZ", "dep": "aux",   "morph": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"},
    # "cannot" — spaCy tags "can" as AUX/MD/aux/VerbForm=Fin
    {"id": 1969, "text": "cannot",  "pos": "AUX", "pos_fine": "MD",  "dep": "aux",   "morph": "VerbForm=Fin"},

    # -- Special symbols --
    # Underscore — used as a placeholder or structural symbol
    {"id": 95,   "text": "_",       "pos": "SYM", "pos_fine": "NFP", "dep": "",      "morph": ""},

    # -- Negation clitic --
    # "'t" — the contracted negation particle from don't, isn't, etc.
    # spaCy tags "n't" as PART/RB/neg/Polarity=Neg
    {"id": 832,  "text": "'t",      "pos": "PART","pos_fine": "RB",  "dep": "neg",   "morph": "Polarity=Neg"},

    # -- Newline-composite punctuation (BPE formatting artifacts) --
    # Double-quote + paragraph break
    {"id": 1110,  "text": '"\n\n',  "pos": "PUNCT","pos_fine": "``", "dep": "punct", "morph": ""},
    # Hyphen + paragraph break
    {"id": 5163,  "text": "-\n\n",  "pos": "PUNCT","pos_fine": "HYPH","dep": "punct", "morph": ""},
    # Triple-dash separator + paragraph break (section break)
    {"id": 5190,  "text": "---\n\n","pos": "PUNCT","pos_fine": "NFP", "dep": "punct", "morph": ""},
    # Mixed quote + paragraph break
    {"id": 13974, "text": "'\"\n\n","pos": "PUNCT","pos_fine": "''", "dep": "punct", "morph": ""},
    # Colon + paragraph break
    {"id": 13986, "text": ":\n\n",  "pos": "PUNCT","pos_fine": ":",  "dep": "punct", "morph": ""},
]


def annotate_manual_tokens(
    grammar: dict,
    fine_legend_reverse: dict[str, dict[str, int]],
) -> dict:
    """Add hardcoded NLP annotations for BPE tokens that automated strategies miss.

    Covers contraction stems (e.g., "didn", "couldn"), the negation clitic "'t",
    newline-composite punctuation, and the underscore.  Each entry is added only
    if the token ID is not already present in the grammar dict.

    Args:
        grammar: Existing grammar dict (keys are string token IDs).
        fine_legend_reverse: Reverse lookup from fine-type legend.

    Returns:
        Updated grammar dict with new entries added.
    """
    new_entries = 0

    for spec in _MANUAL_ANNOTATIONS:
        token_id = spec["id"]
        if str(token_id) in grammar:
            continue

        pos = spec["pos"]
        pos_fine = spec["pos_fine"]
        dep = spec["dep"]
        morph = spec["morph"]

        nlp_type32 = compute_nlp_type32(pos, dep, morph)
        nlp_type48 = compute_nlp_type48(pos, dep, morph)
        nlp_fine_type = _compute_nlp_fine_type(pos, pos_fine, dep, morph, fine_legend_reverse)

        grammar[str(token_id)] = {
            "text": spec["text"],
            "pos": pos,
            "pos_fine": pos_fine,
            "dep": dep,
            "morph": morph,
            "count": 0,
            "tokens": [token_id],
            "frequency_pct": 0.0,
            "nlp_type32": nlp_type32,
            "nlp_type48": nlp_type48,
            "nlp_fine_type": nlp_fine_type,
        }
        new_entries += 1

    print(f"  Manual annotation: added {new_entries} new entries")
    return grammar


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_fine_legend(grammar_path: Path) -> dict:
    """Load the fine-type legend that accompanies a grammar dict.

    Looks for ``{stem}_nlp_fine_types.json`` in the same directory.
    """
    stem = grammar_path.stem.rsplit("_", 1)[0]  # e.g., "simplestories-1_grammar" -> "simplestories-1"
    fine_path = grammar_path.parent / f"{stem}_nlp_fine_types.json"
    if fine_path.exists():
        return json.loads(fine_path.read_text())
    raise FileNotFoundError(f"Fine-type legend not found: {fine_path}")


def expand_grammar(
    grammar_path: Path,
    tokenizer_dir: Path,
    tokenizer_name: str,
    extra_grammars: list[Path] | None = None,
    output: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the full expansion pipeline.

    1. Load base grammar.
    2. Load tokenizer vocab.
    3. Merge any extra grammar files.
    4. Apply special-token rules.
    5. Apply subword inheritance.
    6. Apply manual annotations for BPE artifacts.
    7. Save or print stats.

    Args:
        grammar_path: Path to base grammar JSON.
        tokenizer_dir: Directory containing tokenizer files.
        tokenizer_name: Tokenizer variant name.
        extra_grammars: Optional list of pre-computed grammar JSONs to merge.
        output: Path to write expanded grammar. If None, uses grammar_path.
        dry_run: If True, print stats without writing.

    Returns:
        The expanded grammar dict.
    """
    # 1. Load base grammar
    print(f"Loading base grammar: {grammar_path}")
    grammar = json.loads(grammar_path.read_text())
    initial_count = len(grammar)

    # 2. Load tokenizer vocab
    print(f"Loading tokenizer vocab from {tokenizer_dir}/{tokenizer_name}")
    tokenizer_vocab = load_tokenizer_vocab(tokenizer_dir, tokenizer_name)
    total_vocab = len(tokenizer_vocab)

    # Load fine-type legend
    fine_legend = load_fine_legend(grammar_path)
    fine_legend_reverse = build_fine_legend_reverse(fine_legend)

    # Categorize uncovered tokens (before expansion)
    categories_before = categorize_uncovered(tokenizer_vocab, grammar)
    print_category_summary(categories_before, total_vocab, initial_count)

    # 3. Merge extra grammars
    if extra_grammars:
        for extra_path in extra_grammars:
            print(f"Merging extra grammar: {extra_path}")
            extra = json.loads(extra_path.read_text())
            merge_grammars(grammar, extra)
            print(f"  After merge: {len(grammar)} entries")

    # 4. Apply special-token annotation
    print("Applying special-token annotation...")
    grammar = annotate_special_tokens(tokenizer_vocab, grammar, fine_legend_reverse)

    # 5. Apply subword inheritance
    print("Applying subword inheritance...")
    grammar = inherit_subword_types(tokenizer_vocab, grammar, fine_legend_reverse)

    # 6. Apply manual annotations for BPE artifacts
    print("Applying manual annotations...")
    grammar = annotate_manual_tokens(grammar, fine_legend_reverse)

    # 7. Report final stats
    final_count = len(grammar)
    categories_after = categorize_uncovered(tokenizer_vocab, grammar)
    print(f"\n{'='*60}")
    print(f"Expansion Results")
    print(f"{'='*60}")
    print(f"Initial coverage: {initial_count}/{total_vocab} ({initial_count/total_vocab*100:.1f}%)")
    print(f"Final coverage:   {final_count}/{total_vocab} ({final_count/total_vocab*100:.1f}%)")
    print(f"New entries:      {final_count - initial_count}")
    remaining = sum(len(v) for v in categories_after.values())
    print(f"Still uncovered:  {remaining}")
    if remaining > 0:
        print(f"\nRemaining uncovered by category:")
        for cat, ids in categories_after.items():
            if ids:
                print(f"  {cat:<22s} {len(ids):>6,}")
    print(f"{'='*60}\n")

    if dry_run:
        print("Dry run -- not writing output file.")
    else:
        out_path = output or grammar_path
        print(f"Writing expanded grammar to: {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(grammar, f, indent=2, ensure_ascii=False)
        print(f"Done. Wrote {final_count} entries.")

    return grammar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand NLP grammar dictionary coverage"
    )
    parser.add_argument(
        "--grammar",
        type=Path,
        required=True,
        help="Path to the base grammar JSON file to expand",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Directory containing tokenizer files (default: data/tokenizer)",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="tokenizer-32768",
        help="Tokenizer variant name (default: tokenizer-32768)",
    )
    parser.add_argument(
        "--extra-grammar",
        type=Path,
        action="append",
        default=None,
        help="Path to a pre-computed grammar JSON to merge in. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the expanded grammar JSON (default: same as --grammar)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print before/after coverage stats without writing the output file",
    )
    args = parser.parse_args()

    expand_grammar(
        grammar_path=args.grammar,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_name=args.tokenizer_name,
        extra_grammars=args.extra_grammar,
        output=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
