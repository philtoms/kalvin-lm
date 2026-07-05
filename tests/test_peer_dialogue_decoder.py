"""Phase 1 — peer-dialogue table regime + decode-time validations.

Spec: ``@specs/peer-dialogue.md`` PDT-1, PDT-3, PDT-4. A peer-mode
:class:`DialogueTable` declares its run regime (``mode``) and divergence
policy (``on_divergence``); the decoder validates the peer invariants
(opening is a T-row; opening and closing are content-distinct).

These tests build minimal peer-mode tables on a small script and assert the
loader/decoder behaviour. The sink runner itself (Phases 2–3) is exercised in
``tests/test_peer_runner.py``.
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
_SCRIPT = "(alpha beta)\nA(=alpha)\nB(=beta)"


def _peer_table(turns: list[dict], **extra) -> dict:
    """Build a raw peer-mode table dict with the given turns.

    A ``peer`` section (present) selects peer mode; its modifiers come from
    ``extra`` (e.g. ``on_divergence="accept"``)."""
    peer = dict(extra)
    base = {"script": _SCRIPT, "turns": turns, "peer": peer}
    return base


def _identity(role: str, sig: str, band: str = "S4") -> dict:
    return {"role": role, "op": "IDENTITY", "signature": sig, "significance": band}


def _decode(table_dict: dict):
    tok, sigf = NLPTokenizer(), NLPSignifier()
    return decode(load_table(table_dict), tokenizer=tok, signifier=sigf)


# ── PDT-1: table regime fields ────────────────────────────────────────────


def test_table_without_peer_section_is_ordered():
    """PDT-1: a table with no ``peer`` section is ordered (synchronous run)."""
    table = load_table({"script": _SCRIPT, "turns": [_identity("T", "A")]})
    assert not table.is_peer
    assert table.peer is None


def test_peer_section_selects_peer_mode():
    """PDT-1: a ``peer`` section (even empty) selects peer mode."""
    table = load_table(
        {"script": _SCRIPT, "turns": [_identity("T", "A")], "peer": {}}
    )
    assert table.is_peer
    assert table.peer is not None
    assert table.peer.on_divergence == "fail"  # default


def test_peer_on_divergence_round_trips():
    """PDT-1: ``peer.on_divergence`` is parsed and carried on the PeerConfig."""
    table = load_table(
        _peer_table(
            [_identity("T", "A"), _identity("K", "A"), _identity("T", "B", "S1")],
            on_divergence="accept",
        )
    )
    assert table.is_peer
    assert table.peer is not None
    assert table.peer.on_divergence == "accept"


def test_loader_rejects_bad_on_divergence():
    with pytest.raises(DecodeError):
        load_table(
            {"script": _SCRIPT, "turns": [], "peer": {"on_divergence": "maybe"}}
        )


def test_loader_rejects_unknown_peer_keys():
    """PDT-1: unknown keys inside the ``peer`` section are a decode error."""
    with pytest.raises(DecodeError, match="unknown peer"):
        load_table(
            {"script": _SCRIPT, "turns": [], "peer": {"bogus": True}}
        )


def test_loader_rejects_non_object_peer_section():
    with pytest.raises(DecodeError):
        load_table({"script": _SCRIPT, "turns": [], "peer": "not-an-object"})


# ── PDT-3: three zones ──────────────────────────────────────────────────────


def test_peer_decode_keeps_opening_first_and_closing_last():
    """PDT-3: the decoded order retains opening (turn 0) and closing (turn -1)
    in their authored positions; the runner pins them positionally."""
    decoded = _decode(
        _peer_table(
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


# ── PDT-4: peer invariants ──────────────────────────────────────────────────


def test_peer_decode_rejects_non_trainer_opening():
    """PDT-4: the opening must be a trainer (T) row."""
    with pytest.raises(DecodeError, match="trainer"):
        _decode(
            _peer_table(
                [
                    _identity("K", "A"),       # opening is K — malformed
                    _identity("T", "B", "S1"),
                ]
            )
        )


def test_peer_decode_rejects_content_equal_opening_and_closing():
    """PDT-4: opening and closing must be content-distinct (role, kline, sig)."""
    with pytest.raises(DecodeError, match="content-equal"):
        _decode(
            _peer_table(
                [
                    _identity("T", "A", "S2"),   # opening
                    _identity("T", "A", "S2"),   # closing — identical content
                ]
            )
        )


def test_peer_decode_rejects_closing_content_in_middle():
    """PDT-4: the closing content must be unique — not present as a middle row."""
    with pytest.raises(DecodeError, match="middle row"):
        _decode(
            _peer_table(
                [
                    _identity("T", "A", "S2"),   # opening
                    _identity("T", "B", "S1"),   # middle — same content as closing
                    _identity("T", "B", "S1"),   # closing
                ]
            )
        )


def test_peer_decode_rejects_table_with_too_few_turns():
    """PDT-4: a peer table needs at least an opening and a closing."""
    with pytest.raises(DecodeError, match="at least"):
        _decode(_peer_table([_identity("T", "A")]))


def test_peer_decode_accepts_distinct_opening_and_closing():
    """PDT-4 (positive): a well-formed peer table decodes without error."""
    decoded = _decode(
        _peer_table(
            [
                _identity("T", "A", "S2"),   # opening
                _identity("K", "A"),         # middle
                _identity("T", "B", "S1"),   # closing (different content)
            ]
        )
    )
    assert turn_content_key(decoded[0]) != turn_content_key(decoded[-1])
