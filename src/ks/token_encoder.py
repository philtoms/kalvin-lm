"""TokenEncoder — converts symbolic entries to encoded KLine objects.

Final stage of the KScript v3 compilation pipeline.  Takes the symbolic
(string) entries produced by ASTEmitter and encodes them into uint64 values
via a pluggable tokenizer.

Encoding rules (spec §11):
  - **Signature** → tokenizer.encode(sig) → uint64.  Multi-token results
    are OR-reduced via make_signature() into a single packed uint64.
  - **Nodes** → each encoded individually via _encode_node().  Multi-token
    words trigger full MCS at the BPE-token level (§11.4).
  - **Mod32 fallback** (§11.3): unbound characters are naturally handled by
    the tokenizer — no special code needed in the encoder.
  - **Signatures** are full unmasked uint64 — no masking or truncation.

Multi-token word MCS (§11.4):
  When a word BPE-encodes to multiple tokens, the TokenEncoder:
    1. Emits one IDENTITY KLine per BPE subword token.
    2. OR-reduces all subword tokens into a single packed signature.
    3. Emits one CANONIZE entry mapping the packed sig to the subword tokens.
    4. The packed signature becomes the single node value used in the parent
       kline — preserving the node-count invariant (one word = one node).

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


# ── Significance level mapping (compile-time intent) ──────────────────
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
        # MCS emissions.  Key is tuple of BPE tokens (same word always
        # produces the same BPE tokens).
        self._decomposed: set[tuple[int, ...]] = set()

    # ── Public API ────────────────────────────────────────────────────

    def encode_entries(self, symbolic: list[SymbolicEntry]) -> list[KLine]:
        """Encode a list of symbolic entries into compiled KLines.

        Args:
            symbolic: List of SymbolicEntry tuples from ASTEmitter.

        Returns:
            Ordered list of KLine objects.  MCS expansion entries
            appear before the entries that reference them.
        """
        if not symbolic:
            return []

        result: list[KLine] = []
        for entry in symbolic:
            result.extend(self._encode_entries_for_entry(entry))
        return result

    # ── Per-entry encoding ────────────────────────────────────────────

    def _encode_entries_for_entry(self, entry: SymbolicEntry) -> list[KLine]:
        """Process one SymbolicEntry into one or more KLine objects.

        Steps:
          1. Encode signature → uint64 (with multi-token MCS if needed).
          2. Encode each node → uint64 (with multi-token MCS if needed).
          3. Emit the main entry.

        Returns:
            List of KLine (extras first, then main entry).
        """
        extras: list[KLine] = []

        # 1. Encode signature
        sig_tokens = self._tokenizer.encode(entry.sig)
        sig_uint64: int
        if len(sig_tokens) == 1:
            sig_uint64 = sig_tokens[0]
        else:
            # Multi-token signature → MCS + packed sig
            sig_uint64, sig_extras = self._emit_mcs_for_tokens(
                sig_tokens,
                dbg_label=entry.sig,
                op="IDENTITY",
            )
            extras.extend(sig_extras)

        # 2. Encode nodes
        node_values: list[int] = []
        for node_str in entry.nodes or []:
            node_val, node_extras = self._encode_node(node_str)
            extras.extend(node_extras)
            node_values.append(node_val)

        # 3. Build debug info (op is always populated)
        dbg = KDbg(op=entry.op)
        if self._dev:
            dbg = self._build_dbg(sig_uint64, entry.sig, op=entry.op)

        # 4. Emit main entry
        main = KLine(
            signature=sig_uint64,
            nodes=node_values,
            dbg=dbg,
        )
        extras.append(main)
        return extras

    # ── Node encoding ─────────────────────────────────────────────────

    def _encode_node(self, word: str) -> tuple[int, list[KLine]]:
        """Encode a single word to a uint64 node value.

        Args:
            word: The string to encode.

        Returns:
            (node_value, extra_entries) — node_value is the uint64 to use
            in the parent kline.  extra_entries are MCS expansion entries
            that must appear before the entry that uses this node.
        """
        tokens = self._tokenizer.encode(word)

        if len(tokens) == 1:
            # Single token — no MCS needed
            return (tokens[0], [])

        # Multi-token word → MCS at BPE-token level (§11.4)
        return self._emit_mcs_for_tokens(tokens, dbg_label=word, op="IDENTITY")

    # ── MCS emission for multi-token results ──────────────────────────

    def _emit_mcs_for_tokens(
        self,
        tokens: list[int],
        dbg_label: str = "",
        op: str = "IDENTITY",
    ) -> tuple[int, list[KLine]]:
        """Emit MCS entries for a multi-token encoding result.

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

        # Compute packed signature via OR-reduction
        packed = make_signature(tokens)

        extras: list[KLine] = []

        if token_key not in self._decomposed:
            self._decomposed.add(token_key)

            # One IDENTITY per subword token
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

            # One CANONIZE: packed sig → subword tokens
            canon_dbg: KDbg | None = None
            if self._dev:
                canon_dbg = self._build_dbg(packed, dbg_label, op="CANONIZED")
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

    # ── Debug construction ──────────────────────────────────────────────

    def _build_dbg(self, sig_uint64: int, label: str, op: str = "IDENTITY") -> KDbg:
        """Build a KDbg for a compiled signature.

        Looks up NLP grammar info from the tokenizer's grammar dict
        and decodes the BPE token to get the actual subword text.

        Args:
            sig_uint64: The packed NLP-BPE signature.
            label: Origin label (the word/operator being compiled).
            op: The originating operator.

        Returns:
            Populated KDbg instance.
        """
        decoded = self._tokenizer.decode([sig_uint64])
        pos = ""
        dep = ""
        morph = ""
        bpe_id = sig_uint64 & 0xFFFFFFFF
        grammar_entry = self._tokenizer.lookup_grammar(bpe_id)
        if grammar_entry:
            pos = grammar_entry.get("pos", "")
            dep = grammar_entry.get("dep", "")
            morph = grammar_entry.get("morph", "")
        return KDbg(
            op=op,
            label=label,
            decoded=decoded,
            pos=pos,
            dep=dep,
            morph=morph,
        )
