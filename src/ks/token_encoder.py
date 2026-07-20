"""TokenEncoder — converts symbolic entries into KValue objects.

Final stage of the KScript v3 compilation pipeline. Takes the symbolic
(string) entries produced by ASTEmitter and encodes them into uint64
values via a pluggable tokenizer, wrapping each KLine in a KValue whose
significance is derived from the production op (KP-1, D3).

Encoding rules (spec §11):
  - Signature → tokenizer.encode(sig) → uint64 (multi-token results are
    OR-reduced via make_signature()).
  - Nodes → each encoded individually via _encode_node(); a multi-token
    word (a resolved word the tokenizer splits into ≥2 subwords) triggers
    §11.3 compound-word decomposition, which emits a CANONIZES-shaped
    identity carrying the COMPOUND_TOKEN boundary marker.
  - Canonical encoding (§11.4/§11.5): a declared compound identifier's
    signature is computed once at its MTS CANONIZES definition (OR of its
    resolved component node values) and reused by every reference via the
    ``_compound_sigs`` registry; declared compounds are exempt from §11.3
    (their decomposition is their §8 MTS entry, not a re-encoding of the
    literal string); a packed signature never heads an empty-form
    `{S: []}` IDENTITY kline (CONTEXT.md "Identity"). Packed signatures
    are opaque per §11.5.

Significance levels (compile-time intent) — each emitted KValue carries
kalvin.expand.band_significance(op), computed from the production op at
encode time (never from dbg):
    COUNTERSIGNS → S1    DENOTES → S3    CANONIZES → S2
    CONNOTES → S3      IDENTITY → S4

Dependencies: kalvin.kline.KLine, kalvin.kvalue.KValue,
              kalvin.expand.band_significance, kalvin.abstract.KTokenizer,
              kalvin.signifier.NLPSignifier, ks.ast_emitter.SymbolicEntry.

Output ordering: compiled source (operator + identity klines from the
script) precedes every decomposition kline — §8 MTS expansions (declared
compounds) and §11.3 compound-word decompositions (BPE-split words).
See ``encode_entries``.
"""

from __future__ import annotations

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.expand import band_significance
from kalvin.kline import KDbg, KLine
from kalvin.nlp_tokenizer import COMPOUND_TOKEN
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
        # Track already-decomposed multi-token words to avoid duplicate
        # §11.3 compound-word emissions.  Key is tuple of BPE tokens (same
        # word always produces the same BPE tokens).
        self._decomposed: set[tuple[int, ...]] = set()
        # Canonical encoding registry (§11.4): a declared compound
        # identifier's signature uint64, computed once at its MTS CANONIZES
        # definition as OR of its resolved component node values, then reused
        # by every referencing entry. The ASTEmitter emits definitions before
        # references, so this is populated on demand.
        self._compound_sigs: dict[str, int] = {}

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

            Encoding still runs in def-before-ref order internally (so a
            declared compound's canonical signature is registered before
            any reference is encoded); the source-before-decomposition
            ordering is a stable partition applied to the finished output,
            preserving relative order within each group.  Every KValue
            carries a band-representative significance derived from the
            production ``op`` (KP-1).
        """
        if not symbolic:
            return []

        # Encode in emission order (def-before-ref), tagging each output
        # KValue as a decomposition entry (§8 MTS or §11.3 compound-word)
        # or source.
        tagged: list[tuple[KValue, bool]] = []
        for entry in symbolic:
            for kv, bpe_mts in self._encode_entries_for_entry(entry):
                tagged.append((kv, entry.is_mts or bpe_mts))

        # Output ordering: compiled source precedes any decomposition
        # entries (§8 MTS or §11.3 compound-word).
        # Stable partition preserves relative order within each group.
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
                sig_uint64, sig_extras = self._emit_mts_for_tokens(
                    sig_tokens,
                    dbg_label=entry.sig,
                    op="IDENTITY",
                )
                extras.extend((kv, True) for kv in sig_extras)
                sig_is_packed = True

        # 2. Encode nodes (compound nodes reuse the registry value).
        node_values: list[int] = []
        for node_str in entry.nodes or []:
            if node_str in self._compound_sigs:
                node_values.append(self._compound_sigs[node_str])
            else:
                node_val, node_extras = self._encode_node(node_str)
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
        #    make_signature(block_nodes) (§11.4: signature is a registry
        #    lookup, not a per-entry reduction of nodes).
        if is_compound_def and not is_compound_ref:
            sig_uint64 = self._signifier.make_signature(node_values)
            self._compound_sigs[entry.sig] = sig_uint64
            sig_is_packed = True

        # 4. Debug info.
        dbg = KDbg(op=entry.op)
        if self._dev:
            dbg = self._build_dbg(sig_uint64, entry.sig, op=entry.op, packed=sig_is_packed)

        # 5. A packed signature cannot head an empty-form `{S: []}`
        #    IDENTITY kline (CONTEXT.md "Identity"); the §11.3 compound-word
        #    decomposition (CANONIZES-shaped, carrying COMPOUND_TOKEN) or the
        #    §8 MTS entry is the sole representation. Operator entries with a
        #    packed sig are legitimate references and are emitted normally.
        if entry.op == "IDENTITY" and sig_is_packed:
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

    def _encode_node(self, word: str) -> tuple[int, list[KValue]]:
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
            return (tokens[0], [])

        # Multi-token word → §11.3 compound-word decomposition.
        return self._emit_mts_for_tokens(tokens, dbg_label=word, op="IDENTITY")

    # §11.3 compound-word decomposition for multi-token results

    def _emit_mts_for_tokens(
        self,
        tokens: list[int],
        dbg_label: str = "",
        op: str = "IDENTITY",
    ) -> tuple[int, list[KValue]]:
        """Emit §11.3 compound-word decomposition entries for a multi-token word.

        A resolved word the external tokenizer splits into ≥2 subwords
        (e.g. ``Mary`` → ``[mar, y]``) is a *compound-word*: one lexical
        item whose decomposition is an encoding artefact, not a declared
        aggregation. This is orthogonal to §8 MTS (declared compounds),
        which shares the emit shape but produces a canon rather than an
        identity. The two are distinguished structurally by COMPOUND_TOKEN:
        only a compound-word's kline carries it.

        Emits:
          1. One IDENTITY KValue per BPE subword token.
          2. One CANONIZES-shaped KValue whose nodes are the subword tokens
             plus COMPOUND_TOKEN — canon-shaped but an identity (S1)
             because of the marker.

        Deduplicates: if this exact token tuple has been seen before,
        no entries are emitted (but the packed signature is still returned).

        Each emitted KValue carries the band-representative significance for
        its production op (subword IDENTITY entries use ``op`` — always
        "IDENTITY" at call sites; the CANONIZES entry uses "CANONIZES").

        Args:
            tokens: List of BPE token uint64 values.
            dbg_label: Debug label for dev mode.
            op: Operator for the identity subword entries.

        Returns:
            (packed_signature, extra_entries).
        """
        token_key = tuple(tokens)
        # The compound-word identity kline carries COMPOUND_TOKEN as an extra
        # node (e.g. ``Mary: [M, ary, COMPOUND_TOKEN]``). The token
        # participates in the signature algebra like any other node, so the
        # compound's signature is ``make_signature(tokens + [COMPOUND_TOKEN])``
        # — the marker is *encoded* in the signature, not OR'd on as a bit.
        # No masking anywhere: ``packed`` below is this full signature, and it
        # is the value reused by references (a block-canon under the same
        # word). See @kline spec §Structural Predicates.
        compound_nodes = [COMPOUND_TOKEN] + list(tokens)
        packed = self._signifier.make_signature(compound_nodes)

        # Register the compound-word's signature (§11.4: the compound-word
        # CANONIZES — word → its subword tokens + marker — DEFINES the
        # signature; a later block-canon entry with the same word id is a
        # REFERENCE that must reuse this value, not recompute it from its
        # own operands). Without this registration, a block canon
        # (e.g. `had => did have`) falls into the defining branch and
        # clobbers the compound-word's true signature with
        # make_signature(block_nodes). Only register when ``dbg_label``
        # names the compound-word (it is empty at internal call sites that
        # have no id).
        if dbg_label:
            self._compound_sigs.setdefault(dbg_label, packed)

        extras: list[KValue] = []

        if token_key not in self._decomposed:
            self._decomposed.add(token_key)

            for tok in tokens:
                tok_dbg: KDbg | None = None
                if self._dev:
                    tok_dbg = self._build_dbg(tok, dbg_label, op="IDENTITY")
                else:
                    tok_dbg = KDbg(op="IDENTITY")
                extras.append(
                    KValue(
                        KLine(
                            signature=tok,
                            nodes=[],
                            dbg=tok_dbg,
                        ),
                        band_significance(op),
                    )
                )

            # CANONIZES: compound sig → subword tokens + COMPOUND_TOKEN.
            # Packed values are opaque per §11.5 — _build_dbg skips decode
            # for them.
            canon_dbg: KDbg | None = None
            if self._dev:
                canon_dbg = self._build_dbg(packed, dbg_label, op="CANONIZES", packed=True)
            else:
                canon_dbg = KDbg(op="CANONIZES")
            extras.append(
                KValue(
                    KLine(
                        signature=packed,
                        nodes=compound_nodes,
                        dbg=canon_dbg,
                    ),
                    band_significance("CANONIZES"),
                )
            )

        return (packed, extras)

    # Debug construction

    def _build_dbg(
        self,
        sig_uint64: int,
        label: str,
        op: str = "IDENTITY",
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
        """
        if packed:
            return KDbg(op=op, label=label)
        try:
            decoded = self._tokenizer.decode([sig_uint64])
        except Exception:
            decoded = ""
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
