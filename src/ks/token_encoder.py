"""TokenEncoder — converts symbolic entries to encoded KLine objects.

Final stage of the KScript v3 compilation pipeline. Takes the symbolic
(string) entries produced by ASTEmitter and encodes them into uint64
values via a pluggable tokenizer.

Encoding rules (spec §11):
  - Signature → tokenizer.encode(sig) → uint64 (multi-token results are
    OR-reduced via make_signature()).
  - Nodes → each encoded individually via _encode_node(); multi-token
    words trigger §11.3 BPE-level MTS.
  - Canonical encoding (§11.4/§11.5): a compound identifier's signature
    is computed once at its CANONIZED definition (OR of its resolved
    component node values) and reused by every reference via the
    ``_compound_sigs`` registry; compounds are exempt from §11.3; a
    packed signature never heads an IDENTITY kline (CONTEXT.md
    "Identity"). Packed signatures are opaque per §11.5.

Significance levels (compile-time intent):
    COUNTERSIGNED → S1    UNDERSIGNED → S3    CANONIZED → S2
    CONNOTED → S3      IDENTITY → S4

Dependencies: kalvin.kline.KLine, kalvin.abstract.KTokenizer,
              kalvin.signature.make_signature, ks.ast_emitter.SymbolicEntry.
"""

from __future__ import annotations

from kalvin.abstract import KTokenizer
from kalvin.kline import KDbg, KLine
from kalvin.signature import make_signature

from .ast_emitter import SymbolicEntry

__all__ = ["TokenEncoder"]


# Significance level mapping (compile-time intent)
_SIG_LEVELS: dict[str, str] = {
    "COUNTERSIGNED": "S1",
    "UNDERSIGNED": "S3",
    "CANONIZED": "S2",
    "CONNOTED": "S3",
    "IDENTITY": "S4",
}


class TokenEncoder:
    """Converts symbolic entries into encoded KLine objects.

    Args:
        tokenizer: A KTokenizer implementation that converts strings to
            uint64 node values.
        dev: Enable development/diagnostic mode (populates dbg).
    """

    def __init__(self, tokenizer: KTokenizer, dev: bool = False) -> None:
        self._tokenizer = tokenizer
        self._dev = dev
        # Track already-decomposed multi-token words to avoid duplicate
        # MTS emissions.  Key is tuple of BPE tokens (same word always
        # produces the same BPE tokens).
        self._decomposed: set[tuple[int, ...]] = set()
        # Canonical encoding registry (§11.4): a compound identifier's
        # signature uint64, computed once at its CANONIZED definition as
        # OR of its resolved component node values, then reused by every
        # referencing entry. The ASTEmitter emits definitions before
        # references, so this is populated on demand.
        self._compound_sigs: dict[str, int] = {}

    # Public API

    def encode_entries(self, symbolic: list[SymbolicEntry]) -> list[KLine]:
        """Encode a list of symbolic entries into compiled KLines.

        Args:
            symbolic: List of SymbolicEntry tuples from ASTEmitter.

        Returns:
            Ordered list of KLine objects.  MTS expansion entries
            appear before the entries that reference them.
        """
        if not symbolic:
            return []

        result: list[KLine] = []
        for entry in symbolic:
            result.extend(self._encode_entries_for_entry(entry))
        return result

    # Per-entry encoding

    def _encode_entries_for_entry(self, entry: SymbolicEntry) -> list[KLine]:
        """Process one SymbolicEntry into one or more KLine objects.

        Steps:
          1. Encode signature → uint64 (with multi-token MTS if needed).
          2. Encode each node → uint64 (with multi-token MTS if needed).
          3. Emit the main entry.

        Returns:
            List of KLine (extras first, then main entry).
        """
        extras: list[KLine] = []

        is_compound_def = entry.op == "CANONIZED" and len(entry.sig) > 1
        is_compound_ref = entry.sig in self._compound_sigs
        sig_is_packed = False

        # Compound refs reuse the registry; compound defs defer
        # to step 3 below; others use §11.3 MTS for multi-token sigs.
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
                extras.extend(sig_extras)
                sig_is_packed = True

        # 2. Encode nodes (compound nodes reuse the registry value).
        node_values: list[int] = []
        for node_str in entry.nodes or []:
            if node_str in self._compound_sigs:
                node_values.append(self._compound_sigs[node_str])
            else:
                node_val, node_extras = self._encode_node(node_str)
                extras.extend(node_extras)
                node_values.append(node_val)

        # 3. Compound definition: sig = OR of resolved component node
        #    values (§11.4); register for reuse by references.
        if is_compound_def:
            sig_uint64 = make_signature(node_values)
            self._compound_sigs[entry.sig] = sig_uint64
            sig_is_packed = True

        # 4. Debug info.
        dbg = KDbg(op=entry.op)
        if self._dev:
            dbg = self._build_dbg(sig_uint64, entry.sig, op=entry.op, packed=sig_is_packed)

        # 5. A packed signature cannot head an IDENTITY kline (CONTEXT.md
        #    "Identity"); the §11.3/§8 decomposition above is the sole
        #    representation. Operator entries with a packed sig are
        #    legitimate references and are emitted normally.
        if entry.op == "IDENTITY" and sig_is_packed:
            return extras

        main = KLine(
            signature=sig_uint64,
            nodes=node_values,
            dbg=dbg,
        )
        extras.append(main)
        return extras

    # Node encoding

    def _encode_node(self, word: str) -> tuple[int, list[KLine]]:
        """Encode a single word to a uint64 node value.

        Args:
            word: The string to encode.

        Returns:
            (node_value, extra_entries) — node_value is the uint64 to use
            in the parent kline.  extra_entries are MTS expansion entries
            that must appear before the entry that uses this node.
        """
        tokens = self._tokenizer.encode(word)

        if len(tokens) == 1:
            return (tokens[0], [])

        # Multi-token word → MTS at BPE-token level (§11.3)
        return self._emit_mts_for_tokens(tokens, dbg_label=word, op="IDENTITY")

    # MTS emission for multi-token results

    def _emit_mts_for_tokens(
        self,
        tokens: list[int],
        dbg_label: str = "",
        op: str = "IDENTITY",
    ) -> tuple[int, list[KLine]]:
        """Emit MTS entries for a multi-token encoding result.

        Emits:
          1. One IDENTITY KLine per BPE subword token.
          2. One CANONIZE entry with packed signature → subword tokens.

        Deduplicates: if this exact token tuple has been seen before,
        no entries are emitted (but the packed signature is still returned).

        Args:
            tokens: List of BPE token uint64 values.
            dbg_label: Debug label for dev mode.
            op: Operator for the identity subword entries.

        Returns:
            (packed_signature, extra_entries).
        """
        token_key = tuple(tokens)
        packed = make_signature(tokens)

        extras: list[KLine] = []

        if token_key not in self._decomposed:
            self._decomposed.add(token_key)

            for tok in tokens:
                tok_dbg: KDbg | None = None
                if self._dev:
                    tok_dbg = self._build_dbg(tok, dbg_label, op="IDENTITY")
                else:
                    tok_dbg = KDbg(op="IDENTITY")
                extras.append(
                    KLine(
                        signature=tok,
                        nodes=[],
                        dbg=tok_dbg,
                    )
                )

            # CANONIZE: packed sig → subword tokens. Packed values are
            # opaque per §11.5 — _build_dbg skips decode for them.
            canon_dbg: KDbg | None = None
            if self._dev:
                canon_dbg = self._build_dbg(packed, dbg_label, op="CANONIZED", packed=True)
            else:
                canon_dbg = KDbg(op="CANONIZED")
            extras.append(
                KLine(
                    signature=packed,
                    nodes=list(tokens),
                    dbg=canon_dbg,
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
        entry = self._tokenizer.lookup_type_entry(sig_uint64 & 0xFFFFFFFF)
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
