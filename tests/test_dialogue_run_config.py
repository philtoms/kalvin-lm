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


# ── PDT-3: three zones ──────────────────────────────────────────────────────


def test_decode_keeps_opening_first_and_closing_last():
    """PDT-3: the decoded order retains opening (turn 0) and closing (turn -1)
    in their authored positions; the runner pins them positionally."""
    decoded = _decode(
        _run_table(
            [
                _identity("T", "A"),       # opening
                _identity("K", "A"),       # middle
                _identity("T", "B", "S2"),  # middle
                _identity("T", "B", "S1"),  # closing
            ]
        )
    )
    assert decoded[0].role == "T" and decoded[0].value.kline.signature != 0
    assert decoded[-1].value.significance != decoded[0].value.significance or (
        decoded[-1].role != decoded[0].role
    )


# ── PDT-4: run invariants ───────────────────────────────────────────────────


def test_decode_rejects_non_trainer_opening():
    """PDT-4: the opening must be a trainer (T) row."""
    with pytest.raises(DecodeError, match="trainer"):
        _decode(
            _run_table(
                [
                    _identity("K", "A"),       # opening is K — malformed
                    _identity("T", "B", "S1"),
                ]
            )
        )


def test_decode_rejects_content_equal_opening_and_closing():
    """PDT-4: opening and closing must be content-distinct (role, kline, sig)."""
    with pytest.raises(DecodeError, match="content-equal"):
        _decode(
            _run_table(
                [
                    _identity("T", "A", "S2"),   # opening
                    _identity("T", "A", "S2"),   # closing — identical content
                ]
            )
        )


def test_decode_rejects_closing_content_in_middle():
    """PDT-4: the closing content must be unique — not present as a middle row."""
    with pytest.raises(DecodeError, match="middle row"):
        _decode(
            _run_table(
                [
                    _identity("T", "A", "S2"),   # opening
                    _identity("T", "B", "S1"),   # middle — same content as closing
                    _identity("T", "B", "S1"),   # closing
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
