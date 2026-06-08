"""Tests for dev/nlp/expand_grammar.py grammar expansion logic."""

import json
import sys
from pathlib import Path

import pytest

# Ensure imports work
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "dev" / "nlp"))

from expand_grammar import (
    annotate_special_tokens,
    build_fine_legend_reverse,
    categorize_uncovered,
    inherit_subword_types,
    merge_grammars,
    _compute_nlp_fine_type,
    _classify_special_token,
    _punct_fine_tag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vocab(entries: dict[int, str]) -> dict[int, str]:
    """Create a tokenizer vocab dict from {token_id: decoded_string}."""
    return dict(entries)


def _make_grammar(entries: dict[int, dict]) -> dict:
    """Create a grammar dict with string keys from int-keyed entries."""
    return {str(k): v for k, v in entries.items()}


def _sample_fine_legend() -> dict[str, int]:
    """Return a minimal fine-type legend for testing."""
    return {
        "POS_SPACE": 1,
        "POS_PUNCT": 2,
        "POS_NUM": 4,
        "POS_X": 8,
        "POS_NOUN": 16,
        "POS_VERB": 32,
        "POS_FINE__SP": 64,
        "POS_FINE_CD": 128,
        "POS_FINE_NN": 256,
        "POS_FINE_VBD": 512,
        "DEP_PUNCT": 1024,
        "DEP_NUMMOD": 2048,
        "DEP_ROOT": 4096,
        "DEP_NSUBJ": 8192,
        "MORPH_NUMBER_SING": 16384,
    }


# ---------------------------------------------------------------------------
# Step 1: categorize_uncovered tests
# ---------------------------------------------------------------------------

class TestCategorizeUncovered:
    """Tests for the categorize_uncovered function."""

    def test_all_covered(self):
        """When every token is in the grammar, all categories are empty."""
        vocab = _make_vocab({0: "hello", 1: "world"})
        grammar = _make_grammar({
            0: {"text": "hello"},
            1: {"text": "world"},
        })
        cats = categorize_uncovered(vocab, grammar)
        assert all(len(v) == 0 for v in cats.values())

    def test_whitespace_categorized(self):
        vocab = _make_vocab({0: " ", 1: "hello", 2: "\n"})
        grammar = _make_grammar({1: {"text": "hello"}})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["whitespace"]
        assert 2 in cats["whitespace"]

    def test_digit_categorized(self):
        vocab = _make_vocab({0: "42", 1: "hello", 2: "0"})
        grammar = _make_grammar({1: {"text": "hello"}})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["digit"]
        assert 2 in cats["digit"]

    def test_punctuation_categorized(self):
        vocab = _make_vocab({0: ".", 1: "hello", 2: ".."})
        grammar = _make_grammar({1: {"text": "hello"}})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["punctuation"]
        assert 2 in cats["punctuation"]

    def test_control_char_categorized(self):
        vocab = _make_vocab({0: "\x00", 1: "hello", 2: "\x1f"})
        grammar = _make_grammar({1: {"text": "hello"}})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["control_char"]
        assert 2 in cats["control_char"]

    def test_tab_and_newline_not_control(self):
        """Tab and newline should be whitespace, not control chars."""
        vocab = _make_vocab({0: "\t", 1: "\n"})
        grammar = _make_grammar({})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["whitespace"]
        assert 1 in cats["whitespace"]
        assert len(cats["control_char"]) == 0

    def test_single_letter_categorized(self):
        vocab = _make_vocab({0: "B", 1: "hello", 2: "z"})
        grammar = _make_grammar({1: {"text": "hello"}})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["single_letter"]
        assert 2 in cats["single_letter"]

    def test_subword_fragment_categorized(self):
        """Alphabetic token not appearing as text in grammar is a subword fragment."""
        vocab = _make_vocab({0: "zing", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["subword_fragment"]

    def test_rare_word_categorized(self):
        """Alphabetic token that IS in grammar texts but not as a key -> not subword_fragment.

        Wait, if it IS in grammar texts, it should be rare_word (since it is a known word
        but the token itself is not in the grammar dict). Actually per the spec,
        subword_fragment means "not in grammar texts". If it IS in grammar texts, it
        falls through to rare_word.
        """
        vocab = _make_vocab({0: "hello", 1: "world"})
        # "hello" appears as a text in grammar entry 1, but token 0 is not in grammar
        grammar = _make_grammar({1: {"text": "hello"}})
        cats = categorize_uncovered(vocab, grammar)
        # "hello" is in grammar_texts, so token 0 -> rare_word
        assert 0 in cats["rare_word"]

    def test_mixed_token_to_rare_word(self):
        """Non-alphabetic, non-digit, non-punct, non-space -> rare_word."""
        vocab = _make_vocab({0: "abc123"})
        grammar = _make_grammar({})
        cats = categorize_uncovered(vocab, grammar)
        assert 0 in cats["rare_word"]

    def test_counts_sum_to_uncovered(self):
        """Total across all categories equals uncovered count."""
        vocab = _make_vocab({
            0: "\x00", 1: " ", 2: "42", 3: ".", 4: "B", 5: "zing",
            6: "hello", 7: "world", 8: "\t",
        })
        grammar = _make_grammar({7: {"text": "world"}})
        cats = categorize_uncovered(vocab, grammar)
        total = sum(len(v) for v in cats.values())
        assert total == 8  # 9 vocab - 1 covered


# ---------------------------------------------------------------------------
# Step 2: Special-token annotation tests
# ---------------------------------------------------------------------------

class TestAnnotateSpecialTokens:
    """Tests for the annotate_special_tokens function."""

    def _reverse(self):
        return build_fine_legend_reverse(_sample_fine_legend())

    def test_whitespace_token_annotated(self):
        vocab = _make_vocab({0: " ", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        entry = result["0"]
        assert entry["pos"] == "SPACE"
        assert entry["pos_fine"] == "_SP"
        assert entry["dep"] == ""
        assert entry["morph"] == ""
        assert entry["count"] == 0
        assert entry["frequency_pct"] == 0.0
        assert entry["tokens"] == [0]

    def test_newline_annotated_as_space(self):
        vocab = _make_vocab({0: "\n", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert result["0"]["pos"] == "SPACE"
        assert result["0"]["pos_fine"] == "_SP"

    def test_digit_token_annotated(self):
        vocab = _make_vocab({0: "42", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        entry = result["0"]
        assert entry["pos"] == "NUM"
        assert entry["pos_fine"] == "CD"
        assert entry["dep"] == "nummod"

    def test_digit_zero_annotated(self):
        vocab = _make_vocab({0: "0", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert result["0"]["pos"] == "NUM"

    def test_punctuation_period(self):
        vocab = _make_vocab({0: ".", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        entry = result["0"]
        assert entry["pos"] == "PUNCT"
        assert entry["dep"] == "punct"

    def test_punctuation_comma(self):
        vocab = _make_vocab({0: ",", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert result["0"]["pos"] == "PUNCT"

    def test_control_char_annotated(self):
        vocab = _make_vocab({0: "\x00", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        entry = result["0"]
        assert entry["pos"] == "X"
        assert entry["dep"] == ""

    def test_control_char_1f(self):
        vocab = _make_vocab({0: "\x1f", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert result["0"]["pos"] == "X"

    def test_existing_entries_not_overwritten(self):
        vocab = _make_vocab({0: ".", 1: "hello"})
        grammar = _make_grammar({
            0: {"text": ".", "pos": "CUSTOM"},
            1: {"text": "hello"},
        })
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        # Entry for token 0 should be preserved
        assert result["0"]["pos"] == "CUSTOM"

    def test_alphabetic_token_not_special(self):
        """Alphabetic tokens should not be classified as special."""
        vocab = _make_vocab({0: "abc", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello"}})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert "0" not in result

    def test_nlp_type32_computed(self):
        """Verify nlp_type32 is computed for special tokens."""
        vocab = _make_vocab({0: "."})
        grammar = _make_grammar({})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert "nlp_type32" in result["0"]
        assert isinstance(result["0"]["nlp_type32"], int)
        assert result["0"]["nlp_type32"] > 0

    def test_nlp_type48_computed(self):
        vocab = _make_vocab({0: "."})
        grammar = _make_grammar({})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert "nlp_type48" in result["0"]
        assert isinstance(result["0"]["nlp_type48"], int)

    def test_nlp_fine_type_computed(self):
        vocab = _make_vocab({0: "."})
        grammar = _make_grammar({})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        assert "nlp_fine_type" in result["0"]
        assert isinstance(result["0"]["nlp_fine_type"], int)

    def test_all_required_fields_present(self):
        """Every entry must have all required fields."""
        required = {"text", "pos", "pos_fine", "dep", "morph", "count", "tokens",
                    "frequency_pct", "nlp_type32", "nlp_type48", "nlp_fine_type"}
        vocab = _make_vocab({0: ".", 1: " ", 2: "42", 3: "\x00"})
        grammar = _make_grammar({})
        result = annotate_special_tokens(vocab, grammar, self._reverse())
        for key in result:
            assert required.issubset(set(result[key].keys())), f"Missing fields in {key}"


# ---------------------------------------------------------------------------
# Step 2: _punct_fine_tag tests
# ---------------------------------------------------------------------------

class TestPunctFineTag:
    def test_period(self):
        assert _punct_fine_tag(".") == "."

    def test_comma(self):
        assert _punct_fine_tag(",") == ","

    def test_exclamation(self):
        assert _punct_fine_tag("!") == "."

    def test_question(self):
        assert _punct_fine_tag("?") == "."

    def test_colon(self):
        assert _punct_fine_tag(":") == ":"

    def test_dash(self):
        assert _punct_fine_tag("-") == "HYPH"

    def test_open_paren(self):
        assert _punct_fine_tag("(") == "-LRB-"

    def test_close_paren(self):
        assert _punct_fine_tag(")") == "-RRB-"

    def test_open_bracket(self):
        assert _punct_fine_tag("[") == "-LRB-"

    def test_close_bracket(self):
        assert _punct_fine_tag("]") == "-RRB-"


# ---------------------------------------------------------------------------
# Step 2: build_fine_legend_reverse tests
# ---------------------------------------------------------------------------

class TestBuildFineLegendReverse:
    def test_pos_mapping(self):
        legend = {"POS_NOUN": 16, "POS_VERB": 32}
        rev = build_fine_legend_reverse(legend)
        assert rev["pos"]["NOUN"] == 16
        assert rev["pos"]["VERB"] == 32

    def test_pos_fine_mapping(self):
        legend = {"POS_FINE_NN": 256}
        rev = build_fine_legend_reverse(legend)
        assert rev["pos_fine"]["NN"] == 256

    def test_dep_mapping(self):
        legend = {"DEP_ROOT": 4096}
        rev = build_fine_legend_reverse(legend)
        assert rev["dep"]["root"] == 4096

    def test_morph_mapping(self):
        legend = {"MORPH_NUMBER_SING": 16384}
        rev = build_fine_legend_reverse(legend)
        assert rev["morph"]["Number=Sing"] == 16384

    def test_morph_multi_part(self):
        legend = {"MORPH_VERBFORM_INF": 999}
        rev = build_fine_legend_reverse(legend)
        assert rev["morph"]["Verbform=Inf"] == 999

    def test_all_categories_populated(self):
        legend = _sample_fine_legend()
        rev = build_fine_legend_reverse(legend)
        assert len(rev["pos"]) > 0
        assert len(rev["pos_fine"]) > 0
        assert len(rev["dep"]) > 0
        assert len(rev["morph"]) > 0


# ---------------------------------------------------------------------------
# Step 2: _compute_nlp_fine_type tests
# ---------------------------------------------------------------------------

class TestComputeNlpFineType:
    def test_pos_contribution(self):
        rev = build_fine_legend_reverse(_sample_fine_legend())
        result = _compute_nlp_fine_type("NOUN", "", "", "", rev)
        assert result == 16  # POS_NOUN

    def test_pos_fine_contribution(self):
        rev = build_fine_legend_reverse(_sample_fine_legend())
        result = _compute_nlp_fine_type("", "NN", "", "", rev)
        assert result == 256  # POS_FINE_NN

    def test_dep_contribution(self):
        rev = build_fine_legend_reverse(_sample_fine_legend())
        result = _compute_nlp_fine_type("", "", "root", "", rev)
        assert result == 4096  # DEP_ROOT

    def test_morph_contribution(self):
        rev = build_fine_legend_reverse(_sample_fine_legend())
        result = _compute_nlp_fine_type("", "", "", "Number=Sing", rev)
        assert result == 16384  # MORPH_NUMBER_SING

    def test_combined(self):
        rev = build_fine_legend_reverse(_sample_fine_legend())
        result = _compute_nlp_fine_type("NOUN", "NN", "nsubj", "Number=Sing", rev)
        assert result == 16 | 256 | 8192 | 16384  # NOUN | NN | NSUBJ | NUMBER_SING

    def test_unknown_value_returns_0_for_that_category(self):
        rev = build_fine_legend_reverse(_sample_fine_legend())
        result = _compute_nlp_fine_type("UNKNOWN_POS", "", "", "", rev)
        assert result == 0


# ---------------------------------------------------------------------------
# Step 3: Subword inheritance tests
# ---------------------------------------------------------------------------

class TestInheritSubwordTypes:
    """Tests for the inherit_subword_types function."""

    def _reverse(self):
        return build_fine_legend_reverse(_sample_fine_legend())

    def test_inherit_from_parent_word(self):
        """Fragment 'ning' should inherit from 'running'."""
        vocab = _make_vocab({0: "ning", 1: "running"})
        grammar = _make_grammar({
            1: {
                "text": "running",
                "pos": "VERB",
                "pos_fine": "VBD",
                "dep": "ROOT",
                "morph": "Tense=Past|VerbForm=Fin",
                "count": 100,
                "tokens": [1],
            },
        })
        result = inherit_subword_types(vocab, grammar, self._reverse())
        entry = result["0"]
        assert entry["text"] == "ning"
        assert entry["pos"] == "VERB"
        assert entry["dep"] == "ROOT"
        assert entry["count"] == 0
        assert entry["frequency_pct"] == 0.0
        assert entry["tokens"] == [0]

    def test_prefer_highest_count_parent(self):
        """When multiple parents match, prefer the one with the highest count."""
        vocab = _make_vocab({0: "ing"})
        grammar = _make_grammar({
            1: {"text": "king", "pos": "NOUN", "pos_fine": "NN", "dep": "nsubj",
                "morph": "", "count": 50, "tokens": [1]},
            2: {"text": "running", "pos": "VERB", "pos_fine": "VBD", "dep": "ROOT",
                "morph": "Tense=Past", "count": 200, "tokens": [2]},
        })
        result = inherit_subword_types(vocab, grammar, self._reverse())
        # "ing" appears in both "king" (count 50) and "running" (count 200)
        # Should prefer "running" (highest count)
        entry = result["0"]
        assert entry["pos"] == "VERB"
        assert entry["dep"] == "ROOT"

    def test_no_parent_found(self):
        """Fragment with no parent should remain uncovered."""
        vocab = _make_vocab({0: "xyzabc", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello", "count": 10}})
        result = inherit_subword_types(vocab, grammar, self._reverse())
        assert "0" not in result

    def test_non_alpha_skipped(self):
        """Non-alphabetic tokens should not be processed."""
        vocab = _make_vocab({0: "123", 1: "hello"})
        grammar = _make_grammar({1: {"text": "hello", "count": 10}})
        result = inherit_subword_types(vocab, grammar, self._reverse())
        assert "0" not in result

    def test_already_covered_skipped(self):
        """Tokens already in grammar should not be overwritten."""
        vocab = _make_vocab({0: "ing", 1: "running"})
        grammar = _make_grammar({
            0: {"text": "ing", "pos": "CUSTOM"},
            1: {"text": "running", "pos": "VERB", "count": 100},
        })
        result = inherit_subword_types(vocab, grammar, self._reverse())
        assert result["0"]["pos"] == "CUSTOM"

    def test_nlp_types_computed(self):
        """Verify nlp_type32 and nlp_type48 are computed from parent."""
        vocab = _make_vocab({0: "ning"})
        grammar = _make_grammar({
            1: {"text": "running", "pos": "VERB", "pos_fine": "VBD", "dep": "ROOT",
                "morph": "Tense=Past", "count": 100, "tokens": [1]},
        })
        result = inherit_subword_types(vocab, grammar, self._reverse())
        entry = result["0"]
        assert "nlp_type32" in entry
        assert "nlp_type48" in entry
        assert "nlp_fine_type" in entry
        assert isinstance(entry["nlp_type32"], int)


# ---------------------------------------------------------------------------
# Step 4: merge_grammars tests
# ---------------------------------------------------------------------------

class TestMergeGrammars:
    """Tests for the merge_grammars function."""

    def test_base_entries_preserved(self):
        base = {"0": {"text": "hello", "pos": "NOUN", "count": 10, "frequency_pct": 0.5}}
        other = {"0": {"text": "hello", "pos": "VERB", "count": 20, "frequency_pct": 1.0}}
        result = merge_grammars(base, other)
        assert result["0"]["pos"] == "NOUN"  # Base preserved
        assert result["0"]["count"] == 10  # Base preserved

    def test_new_entries_added(self):
        base = {"0": {"text": "hello", "count": 10, "frequency_pct": 0.5}}
        other = {"1": {"text": "world", "pos": "NOUN", "count": 5, "frequency_pct": 0.3}}
        result = merge_grammars(base, other)
        assert "1" in result
        assert result["1"]["count"] == 0  # Reset
        assert result["1"]["frequency_pct"] == 0.0  # Reset

    def test_count_frequency_reset(self):
        base = {"0": {"text": "hello", "count": 10, "frequency_pct": 0.5}}
        other = {"1": {"text": "world", "count": 999, "frequency_pct": 99.9}}
        result = merge_grammars(base, other)
        assert result["1"]["count"] == 0
        assert result["1"]["frequency_pct"] == 0.0

    def test_multiple_others(self):
        base = {"0": {"text": "hello", "count": 10, "frequency_pct": 0.5}}
        other1 = {"1": {"text": "world", "count": 5, "frequency_pct": 0.3}}
        other2 = {"2": {"text": "foo", "count": 3, "frequency_pct": 0.2}}
        result = merge_grammars(base, other1, other2)
        assert "0" in result
        assert "1" in result
        assert "2" in result

    def test_other_doesnt_overwrite_base(self):
        base = {"0": {"text": "hello", "count": 10, "frequency_pct": 0.5}}
        other = {"0": {"text": "HELLO", "count": 99, "frequency_pct": 99.0},
                 "1": {"text": "world", "count": 5, "frequency_pct": 0.3}}
        result = merge_grammars(base, other)
        assert result["0"]["text"] == "hello"  # Base not overwritten
        assert "1" in result  # New entry added

    def test_returns_base_object(self):
        base = {"0": {"text": "hello", "count": 10, "frequency_pct": 0.5}}
        other = {"1": {"text": "world", "count": 5, "frequency_pct": 0.3}}
        result = merge_grammars(base, other)
        assert result is base  # Same object

    def test_empty_base(self):
        base = {}
        other = {"0": {"text": "hello", "count": 5, "frequency_pct": 0.3}}
        result = merge_grammars(base, other)
        assert "0" in result
        assert result["0"]["count"] == 0

    def test_no_others(self):
        base = {"0": {"text": "hello", "count": 10, "frequency_pct": 0.5}}
        result = merge_grammars(base)
        assert result == base


# ---------------------------------------------------------------------------
# Integration: categorize + annotate + inherit pipeline
# ---------------------------------------------------------------------------

class TestExpansionPipeline:
    """Integration tests for the full expansion pipeline (minus file I/O)."""

    def _reverse(self):
        return build_fine_legend_reverse(_sample_fine_legend())

    def test_full_pipeline_coverage(self):
        """Verify special tokens and subword inheritance work together."""
        vocab = _make_vocab({
            0: ".",          # punctuation
            1: " ",          # whitespace
            2: "42",         # digit
            3: "\x00",       # control char
            4: "ing",        # subword fragment
            5: "running",    # covered word
            6: "B",          # single letter
        })
        grammar = _make_grammar({
            5: {"text": "running", "pos": "VERB", "pos_fine": "VBD", "dep": "ROOT",
                "morph": "", "count": 100, "tokens": [5]},
        })
        rev = self._reverse()

        # Step: annotate special tokens
        grammar = annotate_special_tokens(vocab, grammar, rev)
        assert grammar["0"]["pos"] == "PUNCT"
        assert grammar["1"]["pos"] == "SPACE"
        assert grammar["2"]["pos"] == "NUM"
        assert grammar["3"]["pos"] == "X"

        # Step: inherit subword types
        grammar = inherit_subword_types(vocab, grammar, rev)
        assert grammar["4"]["pos"] == "VERB"
        assert grammar["4"]["text"] == "ing"

        # Single letter is alphabetic but won't match a grammar text
        # It could be inherited if it appears as substring of some word
        # "B" appears in no grammar text, so it remains uncovered
        # (unless "running" contains "B" -- lowercase so no)

        total_covered = len(grammar)
        assert total_covered >= 5  # At least: running, ., space, 42, \x00, ing

    def test_no_overwrite_across_stages(self):
        """Verify that subword inheritance doesn't overwrite special-token entries."""
        # Token 0 is "ing" which looks alphabetic (subword), but if we already
        # covered it via special tokens, inheritance shouldn't touch it.
        vocab = _make_vocab({0: "ing", 1: "running"})
        grammar = _make_grammar({
            1: {"text": "running", "pos": "VERB", "count": 100, "tokens": [1]},
        })
        rev = self._reverse()

        # Manually add "ing" as a special token (contrived, but tests the principle)
        grammar["0"] = {"text": "ing", "pos": "CUSTOM", "count": 0}
        grammar = inherit_subword_types(vocab, grammar, rev)
        assert grammar["0"]["pos"] == "CUSTOM"  # Not overwritten
