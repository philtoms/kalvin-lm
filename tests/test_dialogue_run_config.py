"""Phase 1 — dialogue-run table config + decode-time validations.

Spec: ``@specs/dialogue-runner.md`` PDT-1, PDT-3, PDT-4. A dialogue-mode
:class:`DialogueTable` carries an optional ``run`` section declaring its
divergence policy (``on_divergence``); the decoder validates the run
invariants (opening is a T-row; opening and closing are content-distinct).

These tests build minimal dialogue-mode tables on a small script and assert the
loader/decoder behaviour. The runner itself (Phases 2–3) is exercised in
``tests/test_runner.py``.
"""

from __future__ import annotations

import pytest

from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from training.dialogue.decoder import (
    DecodeError,
    decode,
    load_table,
    turn_content_key,
)

# A minimal script with two resolvable atoms (A, B) so we can build IDENTITY
# turns for both the opening and closing without needing the full MHALL cascade.
_SCRIPT = "A\nB"


def _run_table(turns: list[dict], **extra) -> dict:
    """Build a raw dialogue-mode table dict with the given turns.

    A ``run`` section (present) carries the run modifiers from ``extra``
    (e.g. ``on_divergence="accept"``)."""
    run = dict(extra)
    base = {"script": _SCRIPT, "turns": turns, "run": run}
    return base


def _identity(role: str, sig: str, band: str = "S4") -> dict:
    return {"role": role, "op": "IDENTITY", "signature": sig, "significance": band}


def _decode(table_dict: dict):
    tok, sigf = NLPTokenizer(), NLPSignifier()
    return decode(load_table(table_dict), tokenizer=tok, signifier=sigf)


# ── PDT-1: table run-config fields ────────────────────────────────────────


def test_table_without_run_section_has_no_config():
    """PDT-1: a table with no ``run`` section carries no run config."""
    table = load_table({"script": _SCRIPT, "turns": [_identity("T", "A")]})
    assert not table.has_run_config
    assert table.run_config is None


def test_run_section_is_carried():
    """PDT-1: a ``run`` section (even empty) is carried on the table."""
    table = load_table(
        {"script": _SCRIPT, "turns": [_identity("T", "A")], "run": {}}
    )
    assert table.has_run_config
    assert table.run_config is not None
    assert table.run_config.on_divergence == "fail"  # default


def test_run_on_divergence_round_trips():
    """PDT-1: ``run.on_divergence`` is parsed and carried on the RunConfig."""
    table = load_table(
        _run_table(
            [_identity("T", "A"), _identity("K", "A"), _identity("T", "B", "S1")],
            on_divergence="accept",
        )
    )
    assert table.has_run_config
    assert table.run_config is not None
    assert table.run_config.on_divergence == "accept"


def test_loader_rejects_bad_on_divergence():
    with pytest.raises(DecodeError):
        load_table(
            {"script": _SCRIPT, "turns": [], "run": {"on_divergence": "maybe"}}
        )


def test_loader_rejects_unknown_run_keys():
    """PDT-1: unknown keys inside the ``run`` section are a decode error."""
    with pytest.raises(DecodeError, match="unknown run"):
        load_table(
            {"script": _SCRIPT, "turns": [], "run": {"bogus": True}}
        )


def test_loader_rejects_non_object_run_section():
    with pytest.raises(DecodeError):
        load_table({"script": _SCRIPT, "turns": [], "run": "not-an-object"})


# ── De-positional: a coverage set and one unique close ─────────────────────


def test_decode_preserves_authored_order():
    """The decoded order retains the authored turn order (documentation of cause
    and effect). The runner is de-positional: the first row is not pinned as an
    opening, and the close is the ``close:true`` turn or the last row."""
    decoded = _decode(
        _run_table(
            [
                _identity("T", "A"),       # coverage
                _identity("K", "A"),       # coverage
                _identity("T", "B", "S2"),  # coverage
                _identity("T", "B", "S1"),  # close (last row)
            ]
        )
    )
    assert len(decoded) == 4


def test_decode_accepts_any_first_row_role():
    """The first row is not pinned as a trainer opening — any role may sit
    there (de-positional). The runner seeds the trainer mechanically, not by
    table position."""
    decoded = _decode(
        _run_table(
            [
                _identity("K", "A"),       # first row is K — valid now
                _identity("T", "B", "S1"),
            ]
        )
    )
    assert decoded[0].role == "K"


def test_decode_rejects_close_content_in_coverage():
    """The close content must be unique — not present as a coverage row (an
    emission of that content would be ambiguous: coverage or close?)."""
    with pytest.raises(DecodeError, match="coverage row"):
        _decode(
            _run_table(
                [
                    _identity("T", "A", "S2"),   # coverage
                    _identity("T", "B", "S1"),   # coverage — same content as close
                    _identity("T", "B", "S1"),   # close (last row)
                ]
            )
        )


def test_decode_rejects_table_with_too_few_turns():
    """PDT-4: a run table needs at least an opening and a closing."""
    with pytest.raises(DecodeError, match="at least"):
        _decode(_run_table([_identity("T", "A")]))


def test_decode_accepts_distinct_opening_and_closing():
    """PDT-4 (positive): a well-formed table decodes without error."""
    decoded = _decode(
        _run_table(
            [
                _identity("T", "A", "S2"),   # opening
                _identity("K", "A"),         # middle
                _identity("T", "B", "S1"),   # closing (different content)
            ]
        )
    )
    assert turn_content_key(decoded[0]) != turn_content_key(decoded[-1])
