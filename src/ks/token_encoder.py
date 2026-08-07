"""TokenEncoder — converts symbolic entries into KValue objects.

Final stage of the KScript v3 compilation pipeline. Takes the symbolic
(string) entries produced by ASTEmitter and encodes them into uint64
values via a pluggable tokenizer, wrapping each KLine in a KValue whose
significance is derived from the production op (KP-1, D3).

Encoding rules (spec §11):
  - Signature → tokenizer.encode(sig) → uint64 (multi-token results are
    OR-reduced via signature_of()).
  - Nodes → each encoded individually via _encode_node(); a multi-token
    word (a resolved word the tokenizer splits into ≥2 subwords) triggers
    §11.3 compound-word decomposition, which emits a self-referential
    identity whose signature is the OR-reduction of the subword tokens.
  - Canonical encoding (§11.4/§11.5): a declared compound identifier's
    signature is computed once at its MTS CANONIZES definition (OR of its
    resolved component node values) and reused by every reference via the
    ``_compound_sigs`` registry; declared compounds are exempt from §11.3
    (their decomposition is their §8 MTS entry, not a re-encoding of the
    literal string); a packed signature never heads an empty-form
    `{S: []}` UNKNOWN kline (CONTEXT.md "Identity"). Packed signatures
    are opaque per §11.5.

Significance levels (compile-time intent) — each emitted KValue carries
kalvin.significance.band_significance(op), computed from the production op at
encode time (never from dbg):
    COUNTERSIGNS → S1    DENOTES → S3    CANONIZES → S2
    CONNOTES → S3      UNKNOWN → S4

Dependencies: kalvin.kline.KLine, kalvin.kvalue.KValue,
              kalvin.significance.band_significance, kalvin.abstract.KTokenizer,
              kalvin.signifier.NLPSignifier, ks.ast_emitter.SymbolicEntry.

Output ordering: compiled source (operator + identity klines from the
script) precedes every decomposition kline — §8 MTS expansions (declared
compounds) and §11.3 compound-word decompositions (BPE-split words).
See ``encode_entries``.
"""

from __future__ import annotations

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.significance import SIG_S1, band_significance
from kalvin.kline import KDbg, KLine
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
        # Track emitted compound-word identity signatures (§11.3) so a word
        # used as a node more than once emits its identity only once.
        self._compound_identity_emitted: set[int] = set()
        # Canonical encoding registry (§11.4): a declared compound
        # identifier's signature uint64, computed once at its MTS CANONIZES
        # definition as OR of its resolved component node values, then reused
        # by every referencing entry. The ASTEmitter emits definitions before
        # references, so this is populated on demand.
        self._compound_sigs: dict[str, int] = {}
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
            followed by every auxiliary decomposition kline — §8 MTS
            expansions (declared compounds) and §11.3 compound-word
            decompositions (BPE-split words).

            Encoding runs in def-before-ref order internally (so a declared
            compound's canonical signature is registered before any
            reference is encoded); the source-before-decomposition ordering
            is a stable partition applied to the finished output,
            preserving relative order within each group. ``KDbg.scope`` and
            ``KDbg.annotation`` are carried through so downstream consumers
            can group by owning scope regardless of this partition. Every
            KValue carries a band-representative significance derived from
            the production ``op`` (KP-1).
        """
        if not symbolic:
            return []

        tagged: list[tuple[KValue, bool]] = []
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
          1. Encode signature → uint64 (with §11.3 compound-word
             decomposition if the sig is a multi-token word).
          2. Encode each node → uint64 (with §11.3 compound-word
             decomposition if the node is a multi-token word).
          3. Emit the main entry wrapped as a KValue.

        Returns:
            List of (KValue, is_bpe_mts).  ``is_bpe_mts`` marks KValues
            that are §11.3 compound-word decomposition extras; the main
            entry is tagged ``False``.  The entry-level §8 MTS flag
            (``entry.is_mts``) is combined with this in
            :meth:`encode_entries` so the final output can push every
            decomposition kline (§8 MTS or §11.3 compound-word) after
            compiled source.
        """
        extras: list[tuple[KValue, bool]] = []

        is_compound_def = entry.op == "CANONIZES" and len(entry.sig) > 1
        is_compound_ref = entry.sig in self._compound_sigs
        sig_is_packed = False

        # Compound refs reuse the registry; compound defs defer
        # to step 3 below; others use §11.3 compound-word decomposition
        # for multi-token sigs.
        if is_compound_ref:
            sig_uint64 = self._compound_sigs[entry.sig]
            sig_is_packed = True
        elif is_compound_def:
            sig_uint64 = 0  # computed after nodes are encoded
        else:
            sig_tokens = self._tokenizer.encode(entry.sig)
            if len(sig_tokens) == 1:
                sig_uint64 = sig_tokens[0]
            else:
                # An IDENTITY entry whose sig multi-token-splits provides
                # its own self-referential identity as a source kline
                # {packed: [packed]}, so its §11.3 decomposition must not
                # emit a second identity. Register the packed signature
                # (so the entry's node resolves to the same packed value)
                # but take no decomposition extras. Other ops take the
                # decomposition's identity as the word's representation.
                if entry.op == "IDENTITY":
                    packed = self._signifier.signature_of(sig_tokens)
                    if entry.sig:
                        self._compound_sigs.setdefault(entry.sig, packed)
                    # The IDENTITY main entry emits {packed:[packed]} as
                    # source; mark it emitted so a later node-side use of
                    # the same word does not re-emit it.
                    self._compound_identity_emitted.add(packed)
                    sig_uint64 = packed
                    sig_is_packed = True
                else:
                    sig_uint64, sig_extras = self._emit_mts_for_tokens(
                        sig_tokens,
                        dbg_label=entry.sig,
                        op="UNKNOWN",
                        annotation=entry.annotation,
                        scope=entry.scope,
                    )
                    extras.extend((kv, True) for kv in sig_extras)
                    sig_is_packed = True

        # 2. Encode nodes (compound nodes reuse the registry value).
        node_values: list[int] = []
        for node_str in entry.nodes or []:
            if node_str in self._compound_sigs:
                node_values.append(self._compound_sigs[node_str])
            else:
                node_val, node_extras = self._encode_node(
                    node_str, annotation=entry.annotation, scope=entry.scope,
                )
                extras.extend((kv, True) for kv in node_extras)
                node_values.append(node_val)

        # 3. Declared-compound definition: sig = OR of resolved component
        #    node values (§11.4); register for reuse by references.
        #    Only the DEFINING entry registers — the MTS CANONIZES entry
        #    (declared compound → its declared characters), which is
        #    emitted before any block canon. A block-canon entry
        #    (compound → block operands, e.g. `WDMH => M H W`) is a
        #    REFERENCE: it reuses the registered signature and must NOT
        #    recompute it from its own (possibly partial/misfit) operands,
        #    or it would clobber the compound's true signature with
        #    signature_of(block_nodes) (§11.4: signature is a registry
        #    lookup, not a per-entry reduction of nodes).
        if is_compound_def and not is_compound_ref:
            sig_uint64 = self._signifier.signature_of(node_values)
            self._compound_sigs[entry.sig] = sig_uint64
            sig_is_packed = True

        # 4. Debug info.
        dbg = KDbg(op=entry.op)
        if self._dev:
            dbg = self._build_dbg(sig_uint64, entry.sig, op=entry.op, packed=sig_is_packed)
        dbg.annotation = entry.annotation
        dbg.scope = entry.scope

        # 5. A packed signature cannot head an empty-form `{S: []}`
        #    UNKNOWN kline (CONTEXT.md "Identity"); the §11.3 compound-word
        #    decomposition (a self-referential identity) or the
        #    §8 MTS entry is the sole representation. Operator entries with a
        #    packed sig are legitimate references and are emitted normally.
        if entry.op == "UNKNOWN" and sig_is_packed:
            return extras

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
            return (tokens[0], [])

        # Multi-token word → §11.3 compound-word decomposition.
        return self._emit_mts_for_tokens(
            tokens, dbg_label=word, op="UNKNOWN",
            annotation=annotation, scope=scope,
        )

    # §11.3 compound-word decomposition for multi-token results

    def _emit_mts_for_tokens(
        self,
        tokens: list[int],
        dbg_label: str = "",
        op: str = "UNKNOWN",
        *,
        annotation: str = "",
        scope: int = 0,
    ) -> tuple[int, list[KValue]]:
        """Emit the §11.3 compound-word identity for a multi-token word.

        A resolved word the external tokenizer splits into ≥2 subwords
        (e.g. ``Mary`` → ``[mar, y]``) is a *compound-word*: one lexical
        item whose decomposition is an encoding artefact, not a declared
        aggregation. The word is represented as a single self-referential
        identity whose signature is the OR-reduction of the subword tokens
        — the subwords live in the signature. No marker token is used.

        Emits exactly one entry: the self-referential identity
        ``{packed → [packed]}`` (S1). No per-subword component entries are
        emitted — the subwords are values inside the signature, not headed
        klines. This mirrors §8 MTS, which emits only the canon.

        Args:
            tokens: List of BPE token uint64 values.
            dbg_label: Debug label for dev mode.
            op: Unused for the identity emission (kept for call-site
                compatibility); the identity always carries SIG_S1.

        Returns:
            (packed_signature, extra_entries).
        """
        # The compound-word signature is the OR-reduction of the subword
        # tokens — the subwords live in the signature. No marker token is
        # involved; ``packed`` is reused by references (a block-canon under
        # the same word).
        packed = self._signifier.signature_of(tokens)

        # Register the compound-word's signature (§11.4: the compound-word
        # DEFINES the signature; a later block-canon entry with the same
        # word id is a REFERENCE that must reuse this value, not recompute it
        # from its own operands). Only register when ``dbg_label`` names the
        # compound-word (it is empty at internal call sites that have no id).
        if dbg_label:
            self._compound_sigs.setdefault(dbg_label, packed)

        # Self-referential identity: packed sig → [packed]. Packed values
        # are opaque per §11.5 — _build_dbg skips decode for them. An
        # identity claims S1 (kline spec KL-21; sig_level returns S1 for
        # {S:[S]}; kscript §11.3). Emitted once per compound-word signature
        # (a word reused as a node does not re-emit its identity).
        extras: list[KValue] = []
        if packed not in self._compound_identity_emitted:
            self._compound_identity_emitted.add(packed)
            id_dbg: KDbg | None = None
            if self._dev:
                id_dbg = self._build_dbg(packed, dbg_label, op="IDENTITY", packed=True)
            else:
                id_dbg = KDbg(op="IDENTITY")
            # A compound-word identity is a §11.3 decomposition extra —
            # scope+1 relative to the entry that triggered it.
            id_dbg.scope = scope + 1
            id_dbg.annotation = annotation
            extras.append(
                KValue(
                    KLine(
                        signature=packed,
                        nodes=[packed],
                        dbg=id_dbg,
                    ),
                    SIG_S1,
                )
            )
        return (packed, extras)

    # Debug construction

    def _build_dbg(
        self,
        sig_uint64: int,
        label: str,
        op: str = "UNKNOWN",
        *,
        packed: bool = False,
    ) -> KDbg:
        """Build a KDbg for a compiled signature.

        A packed signature (§11.3 multi-token word or §11.4 compound) is
        opaque per §11.5: its low-32 bits are a bitwise OR of several
        bpe_ids, so decode/type-lookup are meaningless (decode may
        crash or return an unrelated word). ``label`` carries the
        human-readable name instead. Single tokens are decoded and their
        type-dictionary entry summarised into ``type_info`` (decode is
        defensive — ``decoded`` is purely diagnostic and must not crash
        compilation).

        When ``label`` is empty for a single (non-packed) token, it
        defaults to the token's own decoded text — the kline's label then
        names what the kline *is* (e.g. a ``M`` subword) rather than the
        compound word it was split from (``Mary``).
        """
        if packed:
            return KDbg(op=op, label=label)
        try:
            decoded = self._tokenizer.decode([sig_uint64])
        except Exception:
            decoded = ""
        if not label:
            label = decoded
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
        return KDbg(op=op, label=label, decoded=decoded, type_info=type_info)
