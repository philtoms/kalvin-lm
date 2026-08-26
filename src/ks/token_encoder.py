"""TokenEncoder — converts symbolic entries into KValue objects.

Final stage of the KScript v3 compilation pipeline. Takes the symbolic
(string) entries produced by ASTEmitter and encodes them into uint64
values via a pluggable tokenizer, wrapping each KLine in a KValue whose
significance is derived from the production op.

Encoding rules:
  - Signature → tokenizer.encode(sig) → uint64 (multi-token results are
    OR-reduced via signature_of()). A compound signature heads its kline
    like any other — including an empty-form UNKNOWN `{compound: []}`.
  - Nodes → each encoded individually via _encode_node(); a multi-token
    word (a resolved word the tokenizer splits into ≥2 subwords) triggers
    compound-word decomposition, which emits a self-referential identity
    whose signature is the OR-reduction of the subword tokens.
  - Canonical encoding: a declared compound identifier's signature is
    computed once at its MTS CANONIZES definition (OR of its resolved
    component node values) and reused by every reference via the
    ``_compound_sigs`` registry; declared compounds are exempt from
    compound-word decomposition (their decomposition is their MTS entry,
    not a re-encoding of the literal string).

Significance levels (compile-time intent) — each emitted KValue carries
kalvin.significance.band_significance(op), computed from the production op at
encode time (never from dbg):
    COUNTERSIGNS → S1    DENOTES → S3    CANONIZES → S2
    CONNOTES → S3      UNKNOWN → S4

Dependencies: kalvin.kline.KLine, kalvin.kvalue.KValue,
              kalvin.significance.band_significance, kalvin.abstract.KTokenizer,
              kalvin.signifier.NLPSignifier, ks.ast_emitter.SymbolicEntry.

Output ordering: compiled source (operator + identity klines from the
script) precedes every decomposition kline — MTS expansions (declared
compounds) and compound-word decompositions (BPE-split words).
See ``encode_entries``.
"""

from __future__ import annotations

import contextlib

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.significance import SIG_S1, band_significance
from kalvin.kline import KDbg, KLine, KNode, using_resolver
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier

from .ast_emitter import SymbolicEntry

__all__ = ["TokenEncoder"]


class TokenEncoder:
    """Converts symbolic entries into encoded KLine objects.

    Args:
        tokenizer: A KTokenizer implementation that converts strings to
            uint64 node values.
        dev: Enable development/diagnostic mode (populates dbg).
    """

    def __init__(
        self,
        tokenizer: KTokenizer,
        *,
        signifier: KSignifier | None = None,
        dev: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._signifier = signifier or NLPSignifier()
        self._dev = dev
        # Track emitted compound-word identity signatures so a word
        # used as a node more than once emits its identity only once.
        self._compound_identity_emitted: set[int] = set()
        # Canonical encoding registry: a declared compound
        # identifier's signature uint64, computed once at its MTS CANONIZES
        # definition as OR of its resolved component node values, then reused
        # by every referencing entry. The ASTEmitter emits definitions before
        # references, so this is populated on demand.
        self._compound_sigs: dict[str, int] = {}
        # Reverse of ``_compound_sigs`` (compound signature → label) so
        # ``_resolve_node`` can label a compound-word node value.
        self._compound_labels: dict[int, str] = {}
        # Single-token node words keyed by their uint64 value, for display:
        # a word like "did" that encodes to one token and never heads an
        # entry would otherwise have no label downstream.
        self.node_labels: dict[int, str] = {}

    # Public API

    def encode_entries(self, symbolic: list[SymbolicEntry]) -> list[KValue]:
        """Encode a list of symbolic entries into compiled KValues.

        Args:
            symbolic: List of SymbolicEntry tuples from ASTEmitter.

        Returns:
            Ordered list of KValue objects (each wrapping a KLine).
            **Compiled source precedes any decomposition entries:** operator
            and identity klines that come from the script appear first,
            followed by every auxiliary decomposition kline — MTS
            expansions (declared compounds) and compound-word
            decompositions (BPE-split words).

            Encoding runs in def-before-ref order internally (so a declared
            compound's canonical signature is registered before any
            reference is encoded); the source-before-decomposition ordering
            is a stable partition applied to the finished output,
            preserving relative order within each group. ``KDbg.scope`` and
            ``KDbg.annotation`` are carried through so downstream consumers
            can group by owning scope regardless of this partition. Every
            KValue carries a band-representative significance derived from
            the production ``op``.
        """
        if not symbolic:
            return []

        tagged: list[tuple[KValue, bool]] = []
        with (using_resolver(self._resolve_node) if self._dev else contextlib.nullcontext()):
            for entry in symbolic:
                for kv, bpe_mts in self._encode_entries_for_entry(entry):
                    tagged.append((kv, entry.is_mts or bpe_mts))

        source = [kv for kv, is_mts in tagged if not is_mts]
        mts = [kv for kv, is_mts in tagged if is_mts]
        return source + mts

    # Per-entry encoding

    def _encode_entries_for_entry(self, entry: SymbolicEntry) -> list[tuple[KValue, bool]]:
        """Process one SymbolicEntry into one or more (KValue, is_bpe_mts) pairs.

        Steps:
          1. Encode signature → uint64 (with compound-word
             decomposition if the sig is a multi-token word).
          2. Encode each node → uint64 (with compound-word
             decomposition if the node is a multi-token word).
          3. Emit the main entry wrapped as a KValue.

        Returns:
            List of (KValue, is_bpe_mts).  ``is_bpe_mts`` marks KValues
            that are compound-word decomposition extras; the main
            entry is tagged ``False``.  The entry-level MTS flag
            (``entry.is_mts``) is combined with this in
            :meth:`encode_entries` so the final output can push every
            decomposition kline after
            compiled source.
        """
        extras: list[tuple[KValue, bool]] = []

        is_compound_def = entry.op == "CANONIZES" and len(entry.sig) > 1
        is_compound_ref = entry.sig in self._compound_sigs

        # Compound refs reuse the registry; compound defs defer
        # to step 3 below; others encode the sig directly (a multi-token
        # sig is a compound signature via signature_of, and heads its
        # kline like any other sig — including an empty-form UNKNOWN).
        if is_compound_ref:
            sig_uint64 = self._compound_sigs[entry.sig]
        elif is_compound_def:
            sig_uint64 = 0  # computed after nodes are encoded
        else:
            sig_tokens = self._tokenizer.encode(entry.sig)
            if entry.op == "IDENTITY" and len(sig_tokens) > 1:
                # A multi-token IDENTITY is its own self-referential form
                # {compound: [compound]}: register the compound signature
                # (so the entry's node resolves to the same value) and
                # mark it emitted so a later node-side use of the same
                # word does not re-emit its identity.
                compound = self._signifier.signature_of(sig_tokens)
                if entry.sig:
                    self._compound_sigs.setdefault(entry.sig, compound)
                    self._compound_labels.setdefault(compound, entry.sig)
                self._compound_identity_emitted.add(compound)
                sig_uint64 = compound
            elif len(sig_tokens) == 1:
                sig_uint64 = KNode(sig_tokens[0], entry.sig)
            else:
                sig_uint64 = self._signifier.signature_of(sig_tokens)
                if entry.sig:
                    self._compound_labels.setdefault(sig_uint64, entry.sig)

        # 2. Encode nodes (compound nodes reuse the registry value).
        node_values: list[int] = []
        for node_str in entry.nodes or []:
            if node_str in self._compound_sigs:
                node_values.append(KNode(self._compound_sigs[node_str], node_str))
            else:
                node_val, node_extras = self._encode_node(
                    node_str, annotation=entry.annotation, scope=entry.scope,
                )
                extras.extend((kv, True) for kv in node_extras)
                node_values.append(node_val)

        # 3. Declared-compound definition: sig = OR of resolved component
        #    node values; register for reuse by references.
        #    Only the DEFINING entry registers — the MTS CANONIZES entry
        #    (declared compound → its declared characters), which is
        #    emitted before any block canon. A block-canon entry
        #    (compound → block operands, e.g. `WDMH => M H W`) is a
        #    REFERENCE: it reuses the registered signature and must NOT
        #    recompute it from its own (possibly partial/misfit) operands,
        #    or it would clobber the compound's true signature with
        #    signature_of(block_nodes) ( signature is a registry
        #    lookup, not a per-entry reduction of nodes).
        if is_compound_def and not is_compound_ref:
            sig_uint64 = self._signifier.signature_of(node_values)
            self._compound_sigs[entry.sig] = sig_uint64
            self._compound_labels.setdefault(sig_uint64, entry.sig)

        # 4. Debug info.
        dbg = KDbg(op=entry.op)
        if self._dev:
            dbg = self._build_dbg(sig_uint64, entry.sig, op=entry.op)
        dbg.annotation = entry.annotation
        dbg.scope = entry.scope

        main = KLine(
            signature=sig_uint64,
            nodes=node_values,
            dbg=dbg,
        )
        # Wrap the main entry as a KValue. Significance comes from the
        # production op (entry.op — the SymbolicEntry field), NEVER read
        # back from main.dbg.op (D3: dbg is unspec'd dev-only provenance).
        extras.append((KValue(main, band_significance(entry.op)), False))
        return extras

    # Node encoding

    def _encode_node(
        self, word: str, *, annotation: str = "", scope: int = 0,
    ) -> tuple[int, list[KValue]]:
        """Encode a single word to a uint64 node value.

        Args:
            word: The string to encode.

        Returns:
            (node_value, extra_entries) — node_value is the uint64 to use
            in the parent kline.  extra_entries are KValue-wrapped MTS
            expansion entries that must appear before the entry that uses
            this node.
        """
        tokens = self._tokenizer.encode(word)

        if len(tokens) == 1:
            self.node_labels.setdefault(tokens[0], word)
            return (KNode(tokens[0], word), [])

        # Multi-token word → compound-word decomposition.
        return self._emit_mts_for_tokens(
            tokens, dbg_label=word, op="UNKNOWN",
            annotation=annotation, scope=scope,
        )

    # compound-word decomposition for multi-token results

    def _emit_mts_for_tokens(
        self,
        tokens: list[int],
        dbg_label: str = "",
        op: str = "UNKNOWN",
        *,
        annotation: str = "",
        scope: int = 0,
    ) -> tuple[int, list[KValue]]:
        """Emit the compound-word identity for a multi-token word.

        A resolved word the external tokenizer splits into ≥2 subwords
        (e.g. ``Mary`` → ``[mar, y]``) is a *compound-word*: one lexical
        item whose decomposition is an encoding artefact, not a declared
        aggregation. The word is represented as a single self-referential
        identity whose signature is the OR-reduction of the subword tokens
        — the subwords live in the signature. No marker token is used.

        Emits exactly one entry: the self-referential identity
        ``{compound → [compound]}`` (S1). No per-subword component entries
        are emitted — the subwords are values inside the signature, not headed
        klines. This mirrors MTS, which emits only the canon.

        Args:
            tokens: List of BPE token uint64 values.
            dbg_label: Debug label for dev mode.
            op: Unused for the identity emission (kept for call-site
                compatibility); the identity always carries SIG_S1.

        Returns:
            (compound_signature, extra_entries).
        """
        # The compound-word signature is the OR-reduction of the subword
        # tokens — the subwords live in the signature. No marker token is
        # involved; the compound value is reused by references (a
        # block-canon under the same word).
        compound = self._signifier.signature_of(tokens)

        # Register the compound-word's signature ( the compound-word
        # DEFINES the signature; a later block-canon entry with the same
        # word id is a REFERENCE that must reuse this value, not recompute it
        # from its own operands). Only register when ``dbg_label`` names the
        # compound-word (it is empty at internal call sites that have no id).
        if dbg_label:
            self._compound_sigs.setdefault(dbg_label, compound)
            self._compound_labels.setdefault(compound, dbg_label)

        # Self-referential identity: compound sig → [compound]. An identity
        # claims S1 (sig_level returns S1 for {S:[S]}). Emitted once per
        # compound-word signature (a word reused as a node does not re-emit
        # its identity).
        extras: list[KValue] = []
        if compound not in self._compound_identity_emitted:
            self._compound_identity_emitted.add(compound)
            id_dbg: KDbg | None = None
            if self._dev:
                id_dbg = self._build_dbg(compound, dbg_label, op="IDENTITY")
            else:
                id_dbg = KDbg(op="IDENTITY")
            # A compound-word identity is a  decomposition extra —
            # scope+1 relative to the entry that triggered it.
            id_dbg.scope = scope + 1
            id_dbg.annotation = annotation
            id_kline = KLine(
                signature=compound,
                nodes=[compound],
                dbg=id_dbg,
            )
            extras.append(KValue(id_kline, SIG_S1))
        return (KNode(compound, dbg_label) if dbg_label else compound, extras)

    # Debug construction

    def _build_dbg(
        self,
        sig_uint64: int,
        label: str,
        op: str = "UNKNOWN",
    ) -> KDbg:
        """Build a KDbg for a compiled signature.

        A single-token signature is decoded defensively only to default an
        empty ``label`` (the kline's label names what the kline *is*, e.g.
        a ``M`` subword, rather than the compound word it was split from);
        ``decoded`` itself is no longer set here — it is populated by
        :func:`kalvin.kline.kline_decode` at the call site. Compound
        signatures always reach here with a non-empty ``label`` (their
        KScript identifier or word), so decode never runs for them.
        """
        if not label:
            try:
                label = self._tokenizer.decode([sig_uint64])
            except Exception:
                label = ""
        type_info = ""
        # type-info is an NLP-specific debug affordance: only type-aware
        # tokenizers expose a node-taking entry lookup. The KTokenizer
        # interface does not, so the path is gated rather than assumed.
        lookup = getattr(self._tokenizer, "lookup_type_entry_for_node", None)
        entry = lookup(sig_uint64) if lookup is not None else None
        if entry:
            # Summarise the entry's non-text string fields generically so
            # core code stays agnostic to whatever generated the dictionary.
            labels = [
                str(v)
                for k, v in entry.items()
                if k != "text" and isinstance(v, str) and v
            ]
            type_info = " ".join(labels)
        return KDbg(op=op, label=label, type_info=type_info)

    # Node resolution for ``kline_decode``

    def _resolve_node(self, node: int) -> KLine | None:
        """Resolve a node value to a labelled KLine from the encoder's indices.

        Backs :func:`kalvin.kline.kline_decode` at compile time: a real model
        is not available (and the entries are still being constructed), so the
        encoder resolves from what it has already registered — single-token
        node words (``node_labels``) and declared compound-word signatures
        (the reverse of ``_compound_sigs``).
        """
        label = self.node_labels.get(node)
        if label is None:
            label = self._compound_labels.get(node)
        if label is None:
            return None
        return KLine(node, [node], dbg=KDbg(label=label))
