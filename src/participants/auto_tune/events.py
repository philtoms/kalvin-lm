"""Event enrichment for auto-tune.

Transforms raw harness WebSocket frames into the enriched auto-tune event
format defined in specs/auto-tune.md §Event Enrichment (rules 23–27).

Each event gets a monotonic ``seq`` counter supplied by the caller.
Rationalise and ratify events receive decompiled KScript source and a full
significance breakdown (raw, normalised, level).
"""

from __future__ import annotations

from kalvin.expand import MASK64, boundaries, classify
from kalvin.kline import KLine, kline_display
from kalvin.mod_tokenizer import Mod32Tokenizer

# Module-level tokenizer for display (stateless across calls)
_tokenizer = Mod32Tokenizer()


# ── Public API ────────────────────────────────────────────────────────


def enrich_event(raw_frame: dict, seq: int) -> dict:
    """Transform a raw harness WebSocket frame into an enriched auto-tune event.

    Args:
        raw_frame: Deserialized WebSocket JSON frame from the harness bus.
        seq: Monotonic sequence number assigned by the caller.

    Returns:
        Enriched event dict ready for writing to events.jsonl.

    Raises:
        ValueError: If the action type is unknown.
    """
    action = raw_frame["action"]
    message = raw_frame["message"]

    if action == "progress":
        return _enrich_progress(message, seq)
    elif action == "event":
        return _enrich_rationalise(message, seq)
    elif action == "ratify_request":
        return _enrich_ratify_request(message, seq)
    elif action == "notify":
        return _enrich_escalation(message, seq)
    else:
        raise ValueError(f"Unknown action: {action!r}")


# ── Action-specific enrichers ─────────────────────────────────────────


def _enrich_progress(message: dict, seq: int) -> dict:
    """Enrich a progress frame (rule 27)."""
    return {
        "seq": seq,
        "type": "progress",
        "status": message["status"],
        "lesson": message.get("lesson_label"),
        "lessons_total": message["lessons_total"],
        "lessons_completed": message["lessons_completed"],
    }


def _enrich_rationalise(message: object, seq: int) -> dict:
    """Enrich a rationalise event frame (rules 23–26)."""
    # message may be a RationaliseEvent instance or a dict after serialization
    if isinstance(message, dict):
        kind = message["kind"]
        query = _to_kline(message["query"])
        proposal = _to_kline(message["proposal"])
        significance = message["significance"]
    else:
        # RationaliseEvent object
        kind = message.kind
        query = _to_kline(message.query)
        proposal = _to_kline(message.proposal)
        significance = message.significance

    return {
        "seq": seq,
        "type": "rationalise",
        "kind": kind,
        "significance": _build_significance(significance),
        "query": _build_kline_display(query),
        "proposal": _build_kline_display(proposal),
    }


def _enrich_ratify_request(message: dict, seq: int) -> dict:
    """Enrich a ratify request frame (rules 23–26)."""
    query = _to_kline(message["query"])
    proposal = _to_kline(message["proposal"])
    significance = message["significance"]

    return {
        "seq": seq,
        "type": "ratify_request",
        "query": _build_kline_display(query),
        "proposal": _build_kline_display(proposal),
        "significance": _build_significance(significance),
    }


def _enrich_escalation(message: dict, seq: int) -> dict:
    """Enrich a notify/escalation frame."""
    return {
        "seq": seq,
        "type": "escalation",
        "reason": message["reason"],
        "detail": message["detail"],
        "lesson_position": message["lesson_position"],
    }


# ── Helpers ───────────────────────────────────────────────────────────


def _build_significance(raw_sig: int) -> dict:
    """Build a Significance Object (§Significance Object).

    Returns dict with raw, normalised, and level fields.

    Normalisation uses the inverted distance (distance = ~raw) against
    S2_S3_DISTANCE so that S1 → [0.99, 1.0], S2 → [0.0, 0.98],
    S3/S4 → 0.0. This ensures S2 < S1 in the normalised value, unlike
    raw/D_MAX which loses float64 precision for distances < 2^64.
    """
    from kalvin.expand import S2_S3_DISTANCE

    s12, s23, s34 = boundaries()
    level = classify(raw_sig, s12, s23, s34)
    if raw_sig == 0:
        normalised = 0.0
    else:
        distance = (~raw_sig) & MASK64
        normalised = max(0.0, 1.0 - distance / S2_S3_DISTANCE)
    return {
        "raw": raw_sig,
        "normalised": normalised,
        "level": level,
    }


def _build_kline_display(kline: KLine) -> dict:
    """Build a KLine Display Object (§KLine Display Object).

    Returns dict with raw (signature, nodes) and decompiled source.
    """
    source = _display_kline(kline)
    return {
        "raw": {"signature": kline.signature, "nodes": kline.nodes},
        "source": source,
    }


def _to_kline(obj: object) -> KLine:
    """Normalise a KLine instance or plain dict to a KLine instance."""
    if isinstance(obj, KLine):
        return obj
    if isinstance(obj, dict):
        return KLine(
            signature=obj["signature"],
            nodes=obj["nodes"],
        )
    raise TypeError(f"Cannot convert {type(obj).__name__} to KLine")


def _display_kline(kline: KLine) -> str:
    """Display a KLine as KScript source string.

    Returns "<unknown>" if display fails.
    """
    try:
        return kline_display(kline, _tokenizer)
    except Exception:
        pass
    return "<unknown>"
