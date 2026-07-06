"""Single-stage dialogue-table decoder (spec ``@specs/dialogue-driven-training.md`` §Decoder).

This is Phase 1 of the plan. The decoder is a **pre-loop configuration stage**:
``decode(table)`` turns a :class:`DialogueTable` into a flat ordered list of
:class:`DecodedTurn`, resolving every symbolic label against ``script`` (the
single source of truth for kline structure). The training loop (Phases 2–3)
receives only the decoded list and never touches ``script`` or the symbol map.

Spec mapping
------------
- DDT-1 — :class:`DialogueTable`/``Turn`` typed structure (``script`` + ordered
  ``turns``; each turn: actor, op, signature, nodes, significance, notes).
- DDT-2 — :func:`decode` returns a flat ordered ``list[DecodedTurn]``.
- DDT-3 — single-stage: resolve the kline from ``script``, attach significance
  by band lookup, pass through ``actor``/``op``, ignore ``notes``.
- DDT-5 — canons retrieved by **node-list match** against compiled canons.
- DDT-28 — annotation-only turns (notes, no structural fields) are dropped at
  decode time (not submittable).
"""

from __future__ import annotations

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
# Per spec §Decoder step 2: significance is attached by band lookup ("S1"→
# SIG_S1, …) from the turn. It is independent of ``op`` (DDT-3: op and
# significance are separate axes; they diverge in most turns and neither
# reconstructs the other). These are the same uint64 inverted-distance
# constants the compiler attaches to compiled entries (see ``kalvin.expand``).
BAND_TO_SIG: dict[str, int] = {
    "S1": SIG_S1,
    "S2": SIG_S2,
    "S3": SIG_S3,
    "S4": SIG_S4,
}

Role = Literal["T", "K"]  # a turn's role: trainer (T) or trainee (K)

# The peer runner's divergence policy (spec ``@specs/peer-dialogue.md``
# §Matching). Lives on :class:`PeerConfig`, which selects peer mode by its
# presence on the table.
OnDivergence = Literal["fail", "accept"]

# The table's closed op vocabulary (spec §Dialogue Table). An unknown op is
# a decode error.
DIALOGUE_OPS = frozenset({"COUNTERSIGNED", "CANONIZED", "CONNOTED", "UNDERSIGNED", "IDENTITY"})


class DecodeError(Exception):
    """A turn could not be decoded: an unknown op, an unresolved symbol, an
    ambiguous/missing canon node-list match, or a band typo."""


# ── Typed table structures (DDT-1) ────────────────────────────────────────


@dataclass(frozen=True)
class Turn:
    """One row of the dialogue table (spec §Dialogue Table).

    Structural turns (those carrying ``op``) decode to a :class:`DecodedTurn`.
    Annotation-only turns (``notes`` but no ``op``) are dropped at decode time
    (DDT-28) — they are not submittable klines.

    ``close`` marks a turn as the close of script ``n`` (1, 2, …). It is the
    table's explicit script-boundary authority: the runner reads it to know
    when a script ends (and, for multi-script, when to open the next). A turn
    without ``close`` is not a close. See :class:`DecodedTurn.close`.
    """

    role: Role
    op: str | None  # None on annotation-only turns (DDT-28)
    signature: str | None
    nodes: tuple[str, ...]
    significance: str | None  # None on annotation-only turns
    notes: str = ""
    close: int | None = None  # None unless this turn closes script `close`

    @property
    def is_annotation_only(self) -> bool:
        """An annotation-only turn carries ``notes`` but no structural fields.

        Such turns are dropped at decode (DDT-28): they are commentary, not
        submittable klines. A turn is annotation-only when it has no ``op``.
        """
        return self.op is None


@dataclass(frozen=True)
class PeerConfig:
    """Peer-mode configuration (spec ``@specs/peer-dialogue.md`` §The Table in
    Peer Mode).

    A table is in **peer mode** when it carries a ``peer`` section; the
    section's presence is the mode selector (there is no separate ``mode``
    field). All peer operations and modifiers live in this one block so future
    peer knobs extend it without touching the rest of the table.

    - ``on_divergence`` — the peer runner's divergence policy (default
      ``"fail"``).
    """

    on_divergence: OnDivergence = "fail"


@dataclass(frozen=True)
class DialogueTable:
    """The source artifact for a lesson (spec §Dialogue Table).

    ``script`` is the single source of truth for kline structure (canonical
    signatures, atom values, subword composition). ``turns`` is the exact T/K
    exchange — prescriptive, not predictive.

    ``peer`` selects the run regime (spec ``@specs/peer-dialogue.md`` §The
    Table in Peer Mode): ``None`` (default, no ``peer`` section) drives the
    synchronous :func:`~training.dialogue.runner.run`; a :class:`PeerConfig`
    drives the sink-shaped :class:`~training.dialogue.peer_runner.PeerRunner`.
    The section's presence *is* the peer-mode selector — all peer operations
    and modifiers live in it. The runners consume :class:`DecodedTurn`s, not
    the raw table.
    """

    script: str
    turns: tuple[Turn, ...]
    peer: PeerConfig | None = None

    @property
    def is_peer(self) -> bool:
        """True when this table declares peer mode (carries a ``peer`` section)."""
        return self.peer is not None


@dataclass(frozen=True)
class DecodedTurn:
    """A turn resolved to a submittable structure (spec §Decoded Turn).

    ``role`` and ``op`` are carried alongside the KValue and **never folded
    into it** (DDT-3): ``op`` is the structural state; ``significance`` (on the
    KValue) is the dialogic stance. They are independent axes.

    ``close`` is carried through from :class:`Turn.close`: when set (1, 2, …),
    this turn is the close of script ``close``. The runner — which owns the
    table cursor — reads it as the script-boundary signal; the trainer does
    not detect closes (it is told, via routing, that a script has ended).
    """

    role: Role
    op: str
    value: KValue
    close: int | None = None  # None unless this turn closes script `close`


# backwards-compat alias for the documented `DECODEDTurn` spelling used in the
# spec text ("a flat ordered list of `DecodedTurn`"). Both names refer to the
# same class.
DECODEDTurn = DecodedTurn


# ── Symbol resolution against compiled script ─────────────────────────────


@dataclass(frozen=True)
class _ResolvedScript:
    """Compiled-script indices built once at decode time.

    - ``by_node_labels`` — the canon-retrieval index (DDT-5): maps the tuple of
      *decoded node labels* of each compiled canon to that canon's KValue.
      Retrieval key is exactly the turn's symbolic node list.
    - ``canon_by_label`` — canonical signature per label: a relation node whose
      label names a canon (e.g. ``Det``) resolves to the canon's signature, not
      the atom identity that shares the label. This keeps relation construction
      faithful to the compiled structure (the compiler binds ``A > Det`` to the
      Det canon signature).
    - ``relation_by_label`` — the compiled relation (COUNTERSIGNED/CONNOTED/
      UNDERSIGNED) per label, so a constructed-relation turn carries the
      relation's structural ``dbg.op`` (e.g. ``COUNTERSIGNED``), not the canon's
      (``CANONIZED``) when the signature label names both.
    - ``labels`` — every resolvable label (a compound/atom ``dbg.label`` or a
      subword atom ``dbg.decoded``), mapped to its compiled KLine. Used to
      resolve atom signatures (IDENTITY) and as the label fallback.
    """

    by_node_labels: dict[tuple[str, ...], KValue] = field(default_factory=dict)
    canon_by_label: dict[str, KLine] = field(default_factory=dict)
    relation_by_label: dict[str, KLine] = field(default_factory=dict)
    labels: dict[str, KLine] = field(default_factory=dict)


def _node_decoded_label(entry_value: KValue, by_sig: dict[int, list[KValue]]) -> tuple[str, ...] | None:
    """The decoded-label tuple of a kline's nodes (the DDT-5 retrieval key).

    A canon's own signature is a packed MTS (OR-reduction), so its ``dbg.decoded``
    is empty (the encoder suppresses decode for packed signatures). Each **node**
    is a single-token signature whose compiled entry carries the real subword text
    in ``dbg.decoded`` (or ``dbg.label`` for authored atoms). We resolve each node
    to its compiled entry and read its display label.

    Returns ``None`` when any node has no resolvable compiled entry: such a canon
    is not retrievable by node-list match (its node labels cannot be keyed) and
    is skipped in the index. This lets a script contain canons whose nodes are
    not all first-class compiled klines (e.g. a question script whose atoms are
    only ever referenced as nodes) without poisoning the whole index; the
    retrieval-time error surfaces only if a turn actually requests that canon.
    """

    def _label_for(sig: int) -> str | None:
        hits = by_sig.get(sig)
        if not hits:
            return None  # unresolvable node — canon not node-list-keyable
        d = hits[0].kline.dbg
        if d is None:
            return None
        return d.decoded or d.label

    labels = [_label_for(n) for n in entry_value.kline.nodes]
    if any(lbl is None for lbl in labels):
        return None
    return tuple(lbl for lbl in labels if lbl is not None)


def _resolve_script(
    script: str,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> tuple[list[KValue], _ResolvedScript]:
    """Compile ``script`` once and build the canon + label indices.

    Compilation reuses :func:`ks.compiler.compile_source` (plan task 1.2). The
    indices are:
      - ``by_node_labels`` — node-decoded-label tuple → canon KValue (DDT-5).
        Multiple canons sharing a node-label tuple would be ambiguous; the first
        compiled canon wins and a debug note is kept (the MHALL table has none).
      - ``labels`` — display label → KLine, for atom/compound resolution.
    """
    entries = compile_source(script, tokenizer=tokenizer, signifier=signifier, dev=True)

    by_sig: dict[int, list[KValue]] = {}
    for e in entries:
        by_sig.setdefault(e.kline.signature, []).append(e)

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
        # Canon index: keyed by node-decoded-label tuple (DDT-5).
        if d.op == "CANONIZED" and kl.nodes:
            key = _node_decoded_label(e, by_sig)
            # A canon whose nodes cannot all be resolved (e.g. a question
            # script referencing atoms that are never first-class compiled) is
            # not node-list-keyable; skip it rather than poisoning the index.
            # Ambiguity would mean two canons decompose identically — a script
            # authoring error. Keep the first and let later lookups be stable.
            if key is not None:
                resolved.by_node_labels.setdefault(key, e)
            # Canon-by-label: the canonical signature for this label.
            if d.label:
                resolved.canon_by_label.setdefault(d.label, kl)
        elif d.op in ("COUNTERSIGNED", "CONNOTED", "UNDERSIGNED") and d.label:
            # Relation-by-label: carries the relation's structural dbg.op, so a
            # constructed-relation turn reports e.g. COUNTERSIGNED, not the
            # CANONIZED of a canon that shares the label.
            resolved.relation_by_label.setdefault(d.label, kl)
        # Label index: compound/atom dbg.label, and subword atom dbg.decoded.
        if d.label:
            resolved.labels.setdefault(d.label, kl)
        if d.decoded:
            resolved.labels.setdefault(d.decoded, kl)
    return entries, resolved


def primaries_from_source(
    script: str,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[KLine]:
    """The ordered script primaries (one per top-level KScript scope).

    A multi-script file (e.g. ``MHALL`` then ``WDMH``) has one primary per
    top-level ``OperatorScope``, in source order. Each primary is the first
    compiled entry whose ``dbg.label`` matches the scope's signature id — the
    kline a trainer opens (R1) for that script. Used by a multi-script trainer
    to open successive scripts after each close.

    The AST is the principled source of script boundaries (a compiled list
    alone cannot distinguish a primary from any other labelled entry); this
    helper keeps that AST knowledge in the decoder, next to the other
    compile-time indexing.
    """
    from ks.lexer import Lexer
    from ks.parser import OperatorScope, Parser

    kfile = Parser(Lexer(script).tokenize()).parse()
    scope_labels = [
        c.sig.id for c in kfile.constructs if isinstance(c, OperatorScope)
    ]
    if not scope_labels:
        return []
    entries = compile_source(script, tokenizer=tokenizer, signifier=signifier, dev=True)
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
                f"script primary {label!r} has no compiled entry — cannot resolve"
            )
        primaries.append(kl)
    return primaries


# ── The single-stage decode (DDT-2, DDT-3, DDT-28) ────────────────────────


def _resolve_kline(
    op: str,
    signature: str,
    nodes: tuple[str, ...],
    resolved: _ResolvedScript,
) -> KLine:
    """Resolve a turn's symbolic ``(op, signature, nodes)`` to a KLine.

    Three branches per spec §Decoder step 1:

    - **CANONIZED** — retrieve the compiled canon whose node-decoded labels match
      the turn's ``nodes`` list (node-list match, DDT-5). The turn's node names
      *are* the retrieval key; ``signature`` is a redundant author hint (checked
      for consistency).
    - **IDENTITY** — resolve the atom by label, preferring the canon when the
      label names one (a label may name both a canon and its atoms; the
      identity names the concept, i.e. the canon).
    - **Constructed relation** (``CONNOTED`` / ``UNDERSIGNED`` / ``COUNTERSIGNED``)
      — resolve each node label to its canonical signature and reconstruct the
      relation KLine with the compiler's signifier (``make_signature``). A node
      label may be a compound (e.g. ``subject``) or an atom; both resolve via
      the label index.
    """
    if op == "CANONIZED":
        canon = resolved.by_node_labels.get(nodes)
        if canon is None:
            raise DecodeError(
                f"CANONIZED {signature!r}: no compiled canon matches node-list {list(nodes)}"
            )
        # ``signature`` is an author hint; confirm it names the canon's label.
        canon_dbg = canon.kline.dbg
        if canon_dbg is None:
            raise DecodeError(
                f"CANONIZED node-list {list(nodes)} resolved to a canon with no debug info"
            )
        canon_label = canon_dbg.label
        if canon_label != signature:
            raise DecodeError(
                f"CANONIZED node-list {list(nodes)} resolved to canon "
                f"{canon_label!r}, but the turn's signature is {signature!r}"
            )
        return canon.kline

    if op == "IDENTITY":
        # Prefer the canon when the label names one. A KScript label may name
        # both a canon and its atoms (e.g. ``Det`` names the Det canon AND the
        # atoms D, et, which all carry dbg.label=='Det'); the atoms are compiled
        # before the canon, so a bare ``labels[label]`` lookup would resolve
        # to the first atom rather than the concept. An IDENTITY turn names the
        # *concept* (e.g. K asking "what is Det?" means Det the canon), so the
        # canon signature is correct; a pure-atom label (no canon) falls back
        # to the label index.
        kl = resolved.canon_by_label.get(signature)
        if kl is None:
            kl = resolved.labels.get(signature)
        if kl is None:
            raise DecodeError(
                f"IDENTITY {signature!r}: label not found in compiled script"
            )
        # An atom label may resolve to a multi-node canon (e.g. a subword word);
        # the identity is the bare signature with empty nodes. The compiled
        # entry's ``dbg`` (op/label/decoded provenance) is carried through so the
        # loop's diagnostics and the supply rule can read structure off it.
        return KLine(kl.signature, [], dbg=kl.dbg)

    # Constructed relation: resolve each node label to its canonical signature
    # (preferring the canon when the label names one — e.g. ``Det`` resolves to
    # the Det canon signature, matching the compiler's binding), then rebuild the
    # relation KLine. Atom labels (no canon) fall back to the label index.
    node_sigs: list[int] = []
    for n in nodes:
        ncanon = resolved.canon_by_label.get(n)
        if ncanon is not None:
            node_sigs.append(ncanon.signature)
            continue
        nkl = resolved.labels.get(n)
        if nkl is None:
            raise DecodeError(
                f"relation node {n!r}: label not found in compiled script"
            )
        node_sigs.append(nkl.signature)
    sig_kl = resolved.relation_by_label.get(signature)
    if sig_kl is None:
        # No compiled relation for this label: fall back to the canon signature
        # (same value) or the label index. The dbg may then be CANONIZED, which
        # is acceptable — a relation whose script declares no standalone relation
        # entry is constructed fresh.
        sig_canon = resolved.canon_by_label.get(signature)
        sig_kl = sig_canon if sig_canon is not None else resolved.labels.get(signature)
    if sig_kl is None:
        raise DecodeError(
            f"relation signature {signature!r}: label not found in compiled script"
        )
    # Carry the signature label's compiled ``dbg`` (its op/label provenance); the
    # turn's ``op`` is the dialogue vocabulary and stays on the DecodedTurn.
    return KLine(sig_kl.signature, node_sigs, dbg=sig_kl.dbg)


def decode(
    table: DialogueTable,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
) -> list[DecodedTurn]:
    """Pre-decode every turn into a flat ordered ``list[DecodedTurn]`` (DDT-2).

    Single-stage per turn (DDT-3): resolve the kline from ``script``, attach
    significance by band lookup, pass through ``actor``/``op``, ignore
    ``notes``. Annotation-only turns are dropped (DDT-28).

    This is a configuration-time function: call once, hand the result to the
    training loop. The loop never touches ``script`` again.
    """
    _entries, resolved = _resolve_script(
        table.script, tokenizer=tokenizer, signifier=signifier
    )

    out: list[DecodedTurn] = []
    for idx, turn in enumerate(table.turns):
        # DDT-28: annotation-only turns (notes, no op) are dropped at decode.
        if turn.is_annotation_only:
            continue
        assert turn.op is not None and turn.significance is not None  # annotation guard
        # A structural turn (op set) must carry ``signature`` — enforced at
        # load time (``_turn_from_dict``), so this is a defensive decode check.
        if turn.signature is None:
            raise DecodeError(f"turn {idx}: structural turn missing 'signature'")
        if turn.op not in DIALOGUE_OPS:
            raise DecodeError(f"turn {idx}: unknown op {turn.op!r}")
        if turn.significance not in BAND_TO_SIG:
            raise DecodeError(
                f"turn {idx}: unknown significance {turn.significance!r}"
            )

        kline = _resolve_kline(turn.op, turn.signature, turn.nodes, resolved)
        significance = BAND_TO_SIG[turn.significance]
        out.append(
            DecodedTurn(
                role=turn.role,
                op=turn.op,
                value=KValue(kline, significance),
                close=turn.close,
            )
        )
    # ``close`` markers are validated on the decoded list (they are a property
    # of the exchange, but the per-turn shape check happens here alongside the
    # other decoded-list invariants).
    _validate_close(out)
    # Peer-mode invariants are checked on the decoded list (spec
    # @specs/peer-dialogue.md §Invariants). They require symbol resolution to
    # have run, so they live here, not in the loader.
    if table.is_peer:
        _validate_peer(out)
    return out


# ── Loader (DDT-1) ────────────────────────────────────────────────────────


def _turn_from_dict(raw: dict) -> Turn:
    """Build a :class:`Turn` from a raw JSON turn dict (DDT-1).

    Annotation-only turns carry ``notes`` but no ``op``; their ``signature``,
    ``nodes`` and ``significance`` default to absent (DDT-28). A turn with an
    ``op`` must also carry ``signature`` and ``significance``; a structural
    turn missing them is a malformed-table decode error.
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
    if close is not None and (not isinstance(close, int) or isinstance(close, bool) or close < 1):
        raise DecodeError(
            f"'close' must be a positive int (the script it closes), got {close!r}"
        )
    return Turn(
        role=role,
        op=op,
        signature=raw.get("signature"),
        nodes=nodes,
        significance=raw.get("significance"),
        notes=raw.get("notes", ""),
        close=close,
    )


# ── Peer-run decode-time validations (spec @specs/peer-dialogue.md §Invariants) ─


def turn_content_key(turn: DecodedTurn) -> tuple[str, int, tuple[int, ...], int]:
    """The content identity of a decoded turn (spec @specs/peer-dialogue.md
    §Matching): ``(role, kline_signature, kline_nodes_tuple, significance)``.

    Peer matching keys on role + KLine + significance. The KLine is carried by
    its signature and nodes tuple (the two fields that define KLine equality,
    ``@specs/kline.md``). This helper exists because :class:`KValue.__eq__` is
    structural-only and ignores significance (``src/kalvin/kvalue.py``), so peer
    matching cannot use ``KValue`` equality directly — significance must be
    compared explicitly alongside the kline.
    """
    return (
        turn.role,
        turn.value.kline.signature,
        tuple(turn.value.kline.nodes),
        turn.value.significance,
    )


def _validate_close(decoded: list[DecodedTurn]) -> None:
    """Validate the ``close`` markers (spec §Dialogue Table).

    A ``close`` turn closes a script; its ``n`` (1, 2, …) names which. Rules:
    - a ``close`` must be on a trainee (K) row (the trainee reaches the
      expected conclusion; a trainer does not close a script);
    - ``close`` values, if present, are contiguous starting at 1
      (``close:1`` then ``close:2`` …) — a skipped or missing ``close:1`` is a
      malformed table;
    - at most one turn closes a given script.

    Tables with no ``close`` markers are valid (single-script, backward
    compatible): the runner ends at the last row and no script boundary fires.
    """
    seen: set[int] = set()
    expected = 1
    for idx, turn in enumerate(decoded):
        if turn.close is None:
            continue
        if turn.role != "K":
            raise DecodeError(
                f"turn {idx}: a 'close' must be on a trainee (K) row, got role {turn.role!r}"
            )
        if turn.close != expected:
            raise DecodeError(
                f"turn {idx}: close markers must be contiguous from 1, "
                f"got close:{turn.close} expected close:{expected}"
            )
        if turn.close in seen:
            raise DecodeError(
                f"turn {idx}: duplicate close marker for script {turn.close}"
            )
        seen.add(turn.close)
        expected += 1


def _validate_peer(decoded: list[DecodedTurn]) -> None:
    """Validate the peer-mode invariants (spec @specs/peer-dialogue.md §Invariants).

    - The opening (``decoded[0]``) is a trainer (``T``) row.
    - The closing (``decoded[-1]``) is content-distinct from the opening **and**
      from every middle row: its ``(role, kline, significance)`` content key
      appears exactly once in the table. A closing that duplicates the opening
      would make coverage degenerate (the opening satisfies it); a closing that
      duplicates a middle row would make the positional "consumed last"
      semantics ambiguous (which occurrence is the closing?).
    """
    if len(decoded) < 2:
        raise DecodeError(
            "peer-mode table needs at least an opening and a closing turn"
        )
    if decoded[0].role != "T":
        raise DecodeError(
            f"peer-mode opening must be a trainer (T) row, got role {decoded[0].role!r}"
        )
    closing_key = turn_content_key(decoded[-1])
    middle_keys = {turn_content_key(t) for t in decoded[1:-1]}
    opening_key = turn_content_key(decoded[0])
    if closing_key == opening_key:
        raise DecodeError(
            "peer-mode opening and closing are content-equal "
            "(same role, kline, significance) — malformed table"
        )
    if closing_key in middle_keys:
        raise DecodeError(
            "peer-mode closing content also appears as a middle row "
            "— the closing must be a unique terminal content"
        )


def _peer_config_from_dict(raw: dict) -> PeerConfig:
    """Build a :class:`PeerConfig` from a raw ``peer`` section dict.

    The section's presence selects peer mode. Only the known modifier
    ``on_divergence`` (default ``"fail"``) is read; an unknown key is a decode
    error so future peer knobs are added deliberately, not silently ignored.
    """
    on_divergence = raw.get("on_divergence", "fail")
    if on_divergence not in ("fail", "accept"):
        raise DecodeError(
            f"peer.on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
        )
    unknown = set(raw) - {"on_divergence"}
    if unknown:
        raise DecodeError(f"unknown peer section keys: {sorted(unknown)!r}")
    return PeerConfig(on_divergence=on_divergence)


def load_table(raw: dict) -> DialogueTable:
    """Parse a raw ``{script, turns[], peer?}`` dict into a :class:`DialogueTable`.

    ``notes`` are carried on each :class:`Turn` (the decoder ignores them) but
    the structural fields are validated for shape here; symbol resolution
    happens later in :func:`decode`. A ``peer`` section (optional) selects peer
    mode and carries its modifiers (spec ``@specs/peer-dialogue.md`` §The Table
    in Peer Mode); its presence is the mode selector — there is no top-level
    ``mode`` field. Unknown keys inside ``peer`` are rejected.
    """
    if "script" not in raw or not isinstance(raw["script"], str):
        raise DecodeError("dialogue table missing string 'script'")
    # ``script`` may be a path to a KScript file or inline source. The two are
    # disambiguated by **path-likeness**, not by existence: a value is treated
    # as a path when it contains a path separator (``/`` or ``\``) or ends in
    # ``.ks``. A path-like value is resolved against the file system — a missing
    # or unreadable file is a hard error (no silent inline fallback, which would
    # mask a typo or a wrong working directory). A non-path-like value is used
    # verbatim as inline KScript (the original behaviour).
    script = raw["script"]
    looks_like_path = ("/" in script) or ("\\" in script) or script.endswith(".ks")
    if looks_like_path:
        script_path = Path(script)
        try:
            script = script_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DecodeError(
                f"dialogue table 'script' path {script!r} could not be read: {exc}"
            ) from exc
    if "turns" not in raw or not isinstance(raw["turns"], list):
        raise DecodeError("dialogue table missing list 'turns'")
    peer_raw = raw.get("peer")
    if peer_raw is not None:
        if not isinstance(peer_raw, dict):
            raise DecodeError("'peer' section must be an object")
        peer = _peer_config_from_dict(peer_raw)
    else:
        peer = None
    turns = tuple(_turn_from_dict(t) for t in raw["turns"])
    return DialogueTable(script=script, turns=turns, peer=peer)
