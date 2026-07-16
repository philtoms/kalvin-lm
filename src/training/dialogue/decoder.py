"""Single-stage dialogue-script decoder.

``decode(script)`` turns a :class:`DialogueTable` into a flat ordered list of
:class:`DecodedTurn`, resolving every symbolic label against ``script``. The
training loop receives the decoded list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from ks.compiler import compile_source

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.nlp_tokenizer import NLPTokenizer

# ── Significance band lookup ──────────────────────────────────────────────
#
# Significance is attached by band lookup ("S1"→SIG_S1, …). Independent of
# ``op``. These are the same uint64 inverted-distance constants the compiler
# attaches to compiled entries.
BAND_TO_SIG: dict[str, int] = {
    "S1": SIG_S1,
    "S2": SIG_S2,
    "S3": SIG_S3,
    "S4": SIG_S4,
}

Role = Literal["T", "K"]  # a turn's role: trainer (T) or trainee (K)

# The runner's divergence policy. Lives on :class:`RunConfig`, carried on
# the script's optional ``run`` section.
OnDivergence = Literal["fail", "accept"]

# The script's closed op vocabulary. An unknown op is a decode error.
DIALOGUE_OPS = frozenset({"COUNTERSIGNS", "CANONIZES", "CONNOTES", "DENOTES", "IDENTITY"})


class DecodeError(Exception):
    """A turn could not be decoded: an unknown op, an unresolved symbol, an
    ambiguous/missing canon node-list match, or a band typo."""

# ── Typed script structures ────────────────────────────────────────────────


@dataclass(frozen=True)
class Turn:
    """One row of the dialogue script.

    Structural turns (those carrying ``op``) decode to a :class:`DecodedTurn`.
    Annotation-only turns (``notes`` but no ``op``) are dropped at decode time.
    ``close`` marks a turn as a script close (a boundary marker); the runner
    reads it to know when a script ends.
    """

    role: Role
    op: str | None  # None on annotation-only turns
    signature: str | None
    nodes: tuple[str, ...]
    significance: str | None  # None on annotation-only turns
    notes: str = ""
    close: bool = False  # True when this turn closes a script (a boundary marker)
    # The raw JSON record this turn was loaded from (``None`` for turns built
    # synthetically, e.g. test fixtures). Carried for diagnostics so a trace
    # can show the script row verbatim alongside its decoded form.
    record: dict | None = None

    @property
    def is_annotation_only(self) -> bool:
        """An annotation-only turn carries ``notes`` but no ``op``."""
        return self.op is None


@dataclass(frozen=True)
class RunConfig:
    """Runner configuration, carried on a ``DialogueTable``'s optional ``run``
    section."""

    on_divergence: OnDivergence = "fail"


@dataclass(frozen=True)
class DialogueScript:
    """The source artifact for a lesson.

    ``source`` is the source of truth for kline structure; ``turns`` is the
    exact T/K exchange. ``events`` (optional, same row shape as ``turns``)
    holds expected K groundings the runner verifies white-box, separately
    from the dialogue coverage check. ``run_config`` carries the runner
    modifiers (``None`` when no ``run`` section is present).
    """

    source: str
    turns: tuple[Turn, ...]
    events: tuple[Turn, ...] = ()
    run_config: RunConfig | None = None

    @property
    def has_run_config(self) -> bool:
        """True when this script carries a ``run`` section."""
        return self.run_config is not None


@dataclass(frozen=True)
class DecodedTurn:
    """A turn resolved to a submittable structure.

    ``role`` and ``op`` are carried alongside the KValue as independent axes.
    ``close`` is carried through from :class:`Turn.close` for the runner.
    """

    role: Role
    op: str
    value: KValue
    close: bool = False  # True when this turn closes a script
    # The raw JSON record this decoded turn was loaded from (``None`` for
    # turns built synthetically). Carried for diagnostics.
    record: dict | None = None



# ── Symbol resolution against compiled source ─────────────────────────────


@dataclass(frozen=True)
class _ResolvedScript:
    """Compiled-source indices built once at decode time: canon-by-label,
    relation-by-label, and a general label→KLine index."""

    canon_by_label: dict[str, KLine] = field(default_factory=dict)
    relation_by_label: dict[str, KLine] = field(default_factory=dict)
    labels: dict[str, KLine] = field(default_factory=dict)


def _resolve_script(
    source: str,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> tuple[list[KValue], _ResolvedScript]:
    """Compile ``source`` once and build the canon + label indices."""
    entries = compile_source(source, tokenizer=tokenizer, signifier=signifier, dev=True)

    resolved = _ResolvedScript()
    for e in entries:
        kl = e.kline
        d = kl.dbg
        # Compiled entries from ``compile_source(..., dev=True)`` always carry
        # ``dbg`` (op/label/decoded provenance). An entry without it is a
        # compiler invariant violation, not a decode error.
        if d is None:
            raise DecodeError(
                f"compiled entry 0x{kl.signature:x} has no debug info "
                "— dev compile invariant broken"
            )
        # Canon-by-label: the canonical signature for this label. Populated for
        # any compiled canon (COUNTERSIGNS canons like MHALL, and CANONIZES
        # canons like Det) so both node and signature resolution can prefer the
        # canon when a label names one.
        if d.op in ("CANONIZES", "COUNTERSIGNS") and kl.nodes and d.label:
            resolved.canon_by_label.setdefault(d.label, kl)
        if d.op in ("COUNTERSIGNS", "CONNOTES", "DENOTES") and d.label:
            # Relation-by-label: carries the relation's structural dbg.op, so a
            # constructed-relation turn reports e.g. COUNTERSIGNS, not the
            # CANONIZES of a canon that shares the label.
            resolved.relation_by_label.setdefault(d.label, kl)
        # Label index: compound/atom dbg.label, and subword atom dbg.decoded.
        if d.label:
            resolved.labels.setdefault(d.label, kl)
        if d.decoded:
            resolved.labels.setdefault(d.decoded, kl)
    return entries, resolved


def primaries_from_source(
    source: str,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[KLine]:
    """The ordered source primaries (one per top-level KScript scope).

    Each primary is the kline a trainer opens (R1) for that source. Used by a
    multi-script trainer to open successive scripts after each close.
    """
    from ks.lexer import Lexer
    from ks.parser import OperatorScope, Parser

    kfile = Parser(Lexer(source).tokenize()).parse()
    scope_labels = [
        c.sig.id for c in kfile.constructs if isinstance(c, OperatorScope)
    ]
    if not scope_labels:
        return []
    entries = compile_source(source, tokenizer=tokenizer, signifier=signifier, dev=True)
    first_by_label: dict[str, KLine] = {}
    for e in entries:
        lbl = e.kline.dbg.label if e.kline.dbg else None
        if lbl and lbl not in first_by_label:
            first_by_label[lbl] = e.kline
    primaries: list[KLine] = []
    for label in scope_labels:
        kl = first_by_label.get(label)
        if kl is None:
            raise DecodeError(
                f"source primary {label!r} has no compiled entry — cannot resolve"
            )
        primaries.append(kl)
    return primaries


# ── The single-stage decode ──────────────────────────────────────────────


def _resolve_node_signatures(
    nodes: tuple[str, ...],
    resolved: _ResolvedScript,
    *,
    op: str,
) -> list[int]:
    """Resolve each node label to its canonical signature (canon-preferred,
    atom fallback). Shared by the CANONIZES and constructed-relation branches."""
    node_sigs: list[int] = []
    for n in nodes:
        ncanon = resolved.canon_by_label.get(n)
        if ncanon is not None:
            node_sigs.append(ncanon.signature)
            continue
        nkl = resolved.labels.get(n)
        if nkl is None:
            raise DecodeError(
                f"{op} node {n!r}: label not found in compiled source"
            )
        node_sigs.append(nkl.signature)
    return node_sigs


def _resolve_kline(
    op: str,
    signature: str,
    nodes: tuple[str, ...],
    resolved: _ResolvedScript,
) -> KLine:
    """Resolve a turn's symbolic ``(op, signature, nodes)`` to a KLine.

    The decoder is a resolver, not a gatekeeper: it builds the kline the turn
    declares (declared signature verbatim, nodes resolved to canonical
    signatures) — an author may declare a deliberate misfit. Three branches:
    CANONIZES, IDENTITY, and constructed relation (CONNOTES/DENOTES/
    COUNTERSIGNS).
    """
    if op == "CANONIZES":
        # Build the kline with the declared signature verbatim; nodes resolved
        # to their canonical signatures. No signature-consistency check.
        node_sigs = _resolve_node_signatures(nodes, resolved, op="CANONIZES")
        sig_kl = resolved.canon_by_label.get(signature)
        if sig_kl is None:
            sig_kl = resolved.labels.get(signature)
        if sig_kl is None:
            raise DecodeError(
                f"CANONIZES signature {signature!r}: label not found in compiled source"
            )
        return KLine(sig_kl.signature, node_sigs, dbg=sig_kl.dbg)

    if op == "IDENTITY":
        # Prefer the canon when the label names one (the identity names the
        # concept, not its atoms).
        kl = resolved.canon_by_label.get(signature)
        if kl is None:
            kl = resolved.labels.get(signature)
        if kl is None:
            raise DecodeError(
                f"IDENTITY {signature!r}: label not found in compiled source"
            )
        return KLine(kl.signature, [], dbg=kl.dbg)

    # Constructed relation: resolve node labels to canonical signatures and
    # rebuild the relation KLine.
    node_sigs = _resolve_node_signatures(nodes, resolved, op="relation")
    sig_kl = resolved.relation_by_label.get(signature)
    if sig_kl is None:
        # Fall back to the canon signature (same value) or the label index.
        sig_canon = resolved.canon_by_label.get(signature)
        sig_kl = sig_canon if sig_canon is not None else resolved.labels.get(signature)
    if sig_kl is None:
        raise DecodeError(
            f"relation signature {signature!r}: label not found in compiled source"
        )
    return KLine(sig_kl.signature, node_sigs, dbg=sig_kl.dbg)


def decode(
    script: DialogueScript,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[DecodedTurn]:
    """Pre-decode every turn into a flat ordered ``list[DecodedTurn]``.

    Per turn: resolve the kline from ``source``, attach significance by band
    lookup, pass through ``actor``/``op``, ignore ``notes``. Annotation-only
    turns are dropped. A configuration-time function: call once, hand the
    result to the training loop.
    """
    resolved = _resolve_script(
        script.source, tokenizer=tokenizer, signifier=signifier
    )[1]

    out = _decode_turns(script.turns, resolved, what="turn")
    # ``close`` markers and dialogue-mode invariants are validated on the
    # decoded list (they require symbol resolution to have run).
    _validate_close(out)
    if script.has_run_config:
        _validate_run(out)
    return out


def decode_events(
    script: DialogueScript,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[DecodedTurn]:
    """Decode the script's ``events`` (expected K groundings) for white-box
    verification.

    Same row shape and resolution as :func:`decode`, but the result carries no
    ``close`` semantics (events are groundings, not terminal dialogue content).
    """
    resolved = _resolve_script(
        script.source, tokenizer=tokenizer, signifier=signifier
    )[1]
    return _decode_turns(script.events, resolved, what="event")


def _decode_turns(
    turns: tuple[Turn, ...], resolved, *, what: str
) -> list[DecodedTurn]:
    """Resolve a sequence of :class:`Turn`\ s into :class:`DecodedTurn`\ s.

    Shared by :func:`decode` (turns) and :func:`decode_events` (events).
    Annotation-only rows are dropped. ``what`` labels the row kind in errors.
    """
    out: list[DecodedTurn] = []
    for idx, turn in enumerate(turns):
        # Annotation-only turns (notes, no op) are dropped.
        if turn.is_annotation_only:
            continue
        assert turn.op is not None and turn.significance is not None  # annotation guard
        if turn.signature is None:
            raise DecodeError(f"{what} {idx}: structural row missing 'signature'")
        if turn.op not in DIALOGUE_OPS:
            raise DecodeError(f"{what} {idx}: unknown op {turn.op!r}")
        if turn.significance not in BAND_TO_SIG:
            raise DecodeError(
                f"{what} {idx}: unknown significance {turn.significance!r}"
            )

        kline = _resolve_kline(turn.op, turn.signature, turn.nodes, resolved)
        significance = BAND_TO_SIG[turn.significance]
        out.append(
            DecodedTurn(
                role=turn.role,
                op=turn.op,
                value=KValue(kline, significance),
                close=turn.close,
                record=turn.record,
            )
        )
    return out


# ── Loader ─────────────────────────────────────────────────────────────────


def _turn_from_dict(raw: dict) -> Turn:
    """Build a :class:`Turn` from a raw JSON turn dict.

    A turn with an ``op`` must also carry ``signature`` and ``significance``;
    annotation-only turns carry only ``notes``.
    """
    role = raw.get("role")
    if role not in ("T", "K"):
        raise DecodeError(f"turn role must be 'T' or 'K', got {role!r}")
    op = raw.get("op")
    if op is not None and op not in DIALOGUE_OPS:
        raise DecodeError(f"unknown op {op!r}")
    nodes = tuple(raw.get("nodes", ()) or ())
    if op is not None:
        if "signature" not in raw:
            raise DecodeError(f"structural turn missing 'signature': {raw!r}")
        if "significance" not in raw:
            raise DecodeError(f"structural turn missing 'significance': {raw!r}")
    close = raw.get("close")
    if close is not None and not (isinstance(close, bool) and close):
        raise DecodeError(
            f"'close' must be the boolean true (a source-boundary marker), got {close!r}"
        )
    return Turn(
        role=role,
        op=op,
        signature=raw.get("signature"),
        nodes=nodes,
        significance=raw.get("significance"),
        notes=raw.get("notes", ""),
        close=close,
        record=raw,
    )


# ── Dialogue-run decode-time validations ──────────────────────────────────


def turn_content_key(turn: DecodedTurn) -> tuple[str, int, tuple[int, ...], int]:
    """The content identity of a decoded turn:
    ``(role, kline_signature, kline_nodes_tuple, significance)``.

    Exists because :class:`KValue.__eq__` ignores significance, so matching
    cannot use ``KValue`` equality directly.
    """
    return (
        turn.role,
        turn.value.kline.signature,
        tuple(turn.value.kline.nodes),
        turn.value.significance,
    )


def _validate_close(decoded: list[DecodedTurn]) -> None:
    """Validate the ``close`` markers. Presence-only: a ``close: true`` is
    well-formed on any row (role-agnostic). Tables with no ``close`` are valid
    (the runner ends at the last row)."""
    return


def _validate_run(decoded: list[DecodedTurn]) -> None:
    """Validate the dialogue-mode invariants.

    The close need not be a unique content: it may recur as a coverage row,
    in which case its coverage copies are consumed first and the run closes
    on the post-exhaustion emission (see runner ``_observe``). The only
    requirement is that a close is present (a script needs at least two
    turns: a coverage set and a close).
    """
    if len(decoded) < 2:
        raise DecodeError(
            "dialogue-mode script needs at least two turns (a coverage set and a close)"
        )


def _run_config_from_dict(raw: dict) -> RunConfig:
    """Build a :class:`RunConfig` from a raw ``run`` section dict. Only the
    known modifier ``on_divergence`` is read; unknown keys are a decode error."""
    on_divergence = raw.get("on_divergence", "fail")
    if on_divergence not in ("fail", "accept"):
        raise DecodeError(
            f"run.on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
        )
    unknown = set(raw) - {"on_divergence"}
    if unknown:
        raise DecodeError(f"unknown run section keys: {sorted(unknown)!r}")
    return RunConfig(on_divergence=on_divergence)


def _load_table_file(path: Path) -> DialogueScript:
    """Load a :class:`DialogueTable` from a JSON file (used for ``priors``).

    The loaded script's own ``priors`` resolve recursively by :func:`load_table`.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DecodeError(
            f"dialogue prior script {str(path)!r} could not be read: {exc}"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecodeError(
            f"dialogue prior script {str(path)!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise DecodeError(
            f"dialogue prior script {str(path)!r} must be a JSON object"
        )
    return load_script(raw)


def _collapse_to_single_close(turns: tuple[Turn, ...]) -> tuple[Turn, ...]:
    """Keep only the last ``close`` marker; clear every earlier one.

    A composed multi-file script has a single close; each prior's own
    ``close:true`` is an intermediate source boundary that becomes an ordinary
    coverage row. A script with zero or one close is a no-op.
    """
    last_close = max(
        (i for i, t in enumerate(turns) if t.close), default=-1
    )
    if last_close < 0:
        return turns
    from dataclasses import replace

    return tuple(
        replace(t, close=False) if (t.close and i != last_close) else t
        for i, t in enumerate(turns)
    )


def load_script(raw: dict) -> DialogueScript:
    """Parse a raw ``{source, turns[], run?, priors?}`` dict into a
    :class:`DialogueTable`. Structural fields are validated for shape here;
    symbol resolution happens later in :func:`decode`.
    """
    if "source" not in raw or not isinstance(raw["source"], str):
        raise DecodeError("dialogue script missing string 'source'")
    # ``source`` may be a path to a .ks file or inline source, disambiguated
    # by path-likeness (path separator or ``.ks`` suffix). A path-like value
    # is resolved against the file system; otherwise used verbatim as inline
    # KScript.
    source = raw["source"]
    looks_like_path = ("/" in source) or ("\\" in source) or source.endswith(".ks")
    if looks_like_path:
        script_path = Path(source)
        try:
            source = script_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DecodeError(
                f"dialogue script 'source' path {source!r} could not be read: {exc}"
            ) from exc
    if "turns" not in raw or not isinstance(raw["turns"], list):
        raise DecodeError("dialogue script missing list 'turns'")
    run_raw = raw.get("run")
    if run_raw is not None:
        if not isinstance(run_raw, dict):
            raise DecodeError("'run' section must be an object")
        run_config = _run_config_from_dict(run_raw)
    else:
        run_config = None
    turns = tuple(_turn_from_dict(t) for t in raw["turns"])
    # ``events`` (optional): expected K groundings the runner verifies
    # white-box, same row shape as ``turns``.
    events_raw = raw.get("events")
    if events_raw is not None:
        if not isinstance(events_raw, list):
            raise DecodeError("'events' must be a list of turn rows")
        events = tuple(_turn_from_dict(t) for t in events_raw)
    else:
        events = ()
    # ``priors`` (optional): other script files whose turns run before this
    # script's own, in list order. Each resolves its own ``source``; only its
    # turns (and events) are carried in.
    priors_raw = raw.get("priors")
    if priors_raw is not None:
        if not isinstance(priors_raw, list) or not all(
            isinstance(p, str) for p in priors_raw
        ):
            raise DecodeError("'priors' must be a list of script-file path strings")
        prior_turns: list[Turn] = []
        prior_events: list[Turn] = []
        for prior_path in priors_raw:
            prior_table = _load_table_file(Path(prior_path))
            prior_turns.extend(prior_table.turns)
            prior_events.extend(prior_table.events)
        turns = tuple(prior_turns) + turns
        events = tuple(prior_events) + events
    # Collapse to a single close: in the merged list only the final
    # ``close:true`` is the run's terminal content; earlier ones become
    # ordinary coverage rows.
    turns = _collapse_to_single_close(turns)
    return DialogueScript(source=source, turns=turns, events=events, run_config=run_config)
