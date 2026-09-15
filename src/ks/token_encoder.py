"""TokenEncoder — converts symbolic entries into KValue objects.

Final stage of the KScript v3 compilation pipeline. Takes the symbolic
(string) entries produced by ASTEmitter and encodes them into uint64
values via a standard BPE tokenizer (upper 32 bits zero on raw tokens),
wrapping each KLine in a KValue whose significance is derived from the
production op.

Node layout (the compiler's packing, distinct from the raw tokenizer)::

    node = (word_bit << 32) | bpe_token_id

- ``word_bit`` (upper 32 bits) — the **word word**: one bit per distinct
  word, assigned on a first-encountered basis at bits 0-30 (bit 31 is
  reserved for ``ASK_BPE_TOKEN``). The 32nd distinct word is a system
  error: word size overflow.
- ``bpe_token_id`` (lower 32 bits) — the OR-reduction of the word's BPE
  subword tokens. A multi-subword word (``Mary`` → ``[mar, y]``) is still
  ONE word and ONE bit — the subword ids OR together, so a multi-subword
  word needs no decomposition kline; the shared word bit does the job.

A compound signature (MTS, e.g. ``MHALL``) is not a word: its signature
is the OR-reduction of its component words' values — one bit per word
(5 words → 5 bits). A CONNOTES concatenation signature (``SubjectM`` =
``Subject`` + ``M`` — the compound sitting in the sig's slot) is a
compound the same way: its value is the OR of its component words — it
never takes a word bit. The entry's ``concat`` field carries the
components in identifier order.

Encoding rules:
  - Signature → the encoded word value, or the registered compound
    signature for compound refs/defs.
  - Nodes → each encoded individually via ``_encode_word``.
  - Canonical encoding: a declared compound identifier's signature is
    computed once at its MTS CANONICALISES definition (OR of its resolved
    component node values) and reused by every reference via the
    ``_compound_sigs`` registry.

Significance levels (compile-time intent) — each emitted KValue carries
kalvin.significance.band_significance(op), computed from the production op at
encode time (never from dbg):
    COUNTERSIGNS → S2    CONNOTES → S2    CANONICALISES → S2
    DENOTES → S3      UNKNOWN → S4      MTS → S1

Dependencies: kalvin.kline.KLine, kalvin.kvalue.KValue,
              kalvin.significance.band_significance, kalvin.abstract.KTokenizer,
              kalvin.signifier.NLPSignifier, ks.ast_emitter.SymbolicEntry.

Output ordering: compiled source (operator + identity klines from the
script) precedes MTS expansion klines. See ``encode_entries``.
"""

from __future__ import annotations

import contextlib

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.significance import SIG_S1, band_significance
from kalvin.kline import KDbg, KLine, KNode, using_resolver
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier, ASK_BPE_TOKEN

from .ast_emitter import SymbolicEntry

__all__ = ["TokenEncoder"]

# The word word occupies the upper 32 bits of a node: word_bit << 32.
TOP_WORD_SHIFT = 32
# Word size: one bit per distinct word, bits 0-30. Bit 31 is ASK_BPE_TOKEN.
WORD_SIZE = 31


class TokenEncoder:
    """Converts symbolic entries into encoded KLine objects.

    Args:
        tokenizer: A KTokenizer implementation that converts strings to
            uint64 token values (upper 32 bits zero).
        dev: Enable development/diagnostic mode (populates dbg).
    """

    def __init__(
        self,
        tokenizer: KTokenizer,
        *,
        signifier: KSignifier | None = None,
        dev: bool = False,
        word_bits: dict[str, int] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._signifier = signifier or NLPSignifier()
        self._dev = dev
        # Word word: distinct word → its bit (first-encountered basis).
        # A caller-supplied table is adopted in place (shared, mutated): the
        # word→bit mapping must stay stable across compiles and sessions so
        # persisted node values keep meaning the same words.
        self._word_bits: dict[str, int] = word_bits if word_bits is not None else {}
        self._next_word_bit = (
            max(self._word_bits.values(), default=0).bit_length()
        )
        # Canonical encoding registry: a declared compound
        # identifier's signature uint64, computed once at its MTS CANONICALISES
        # definition as OR of its resolved component node values, then reused
        # by every referencing entry. The ASTEmitter emits definitions before
        # references, so this is populated on demand.
        self._compound_sigs: dict[str, int] = {}
        # Reverse of ``_compound_sigs`` (compound signature → label) so
        # ``_resolve_node`` can label a compound node value.
        self._compound_labels: dict[int, str] = {}
        # Word values keyed by their uint64 value, for display:
        # a word that never heads an entry would otherwise have no label
        # downstream.
        self.node_labels: dict[int, str] = {}

    # Word word

    def _word_bit(self, word: str) -> int:
        """The word's bit in the word word, assigned on first encounter.

        Raises:
            SystemError: word size overflow — the 32nd distinct word.
        """
        bit = self._word_bits.get(word)
        if bit is not None:
            return bit
        if self._next_word_bit >= WORD_SIZE:
            raise SystemError(
                f"word size overflow: more than {WORD_SIZE} distinct words "
                f"('{word}' is one too many)"
            )
        bit = 1 << self._next_word_bit
        self._word_bits[word] = bit
        self._next_word_bit += 1
        return bit

    # Public API

    def encode_entries(self, symbolic: list[SymbolicEntry]) -> list[KValue]:
        """Encode a list of symbolic entries into compiled KValues.

        Args:
            symbolic: List of SymbolicEntry tuples from ASTEmitter.

        Returns:
            Ordered list of KValue objects (each wrapping a KLine).
            **Compiled source precedes MTS expansion entries:** operator
            and identity klines that come from the script appear first,
            followed by MTS expansions. ``KDbg.scope`` and
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
                for kv, is_mts in self._encode_entries_for_entry(entry):
                    tagged.append((kv, is_mts))

        source = [kv for kv, is_mts in tagged if not is_mts]
        mts = [kv for kv, is_mts in tagged if is_mts]
        return source + mts

    # Per-entry encoding

    def _encode_entries_for_entry(self, entry: SymbolicEntry) -> list[tuple[KValue, bool]]:
        """Process one SymbolicEntry into one or more (KValue, is_mts) pairs.

        Steps:
          1. Encode signature → uint64 (word value, or the registered
             compound signature for compound refs/defs).
          2. Encode each node → uint64 word value.
          3. Emit the entry wrapped as a KValue.
        """
        is_compound_def = entry.op == "CANONICALISES" and len(entry.sig) > 1
        is_compound_ref = entry.sig in self._compound_sigs
        # A multi-char uppercase sig that no MTS entry registered (e.g. a
        # sigless annotation's synthesized initials `WW...`) is still a
        # compound: its signature composes from its nodes — it is not a
        # word and never takes a word bit.
        is_compound_sig = len(entry.sig) > 1 and entry.sig.isupper()

        # Compound refs reuse the registry; compound defs and unregistered
        # compound sigs defer to step 3 below; everything else encodes the
        # sig as a word — a multi-subword sig is still one word (one bit),
        # and heads its kline like any other sig — including an empty-form
        # UNKNOWN and the DENOTES head word.
        if is_compound_ref:
            sig_uint64 = self._compound_sigs[entry.sig]
        elif entry.concat is not None or is_compound_def or is_compound_sig:
            sig_uint64 = 0  # computed after nodes are encoded
        else:
            sig_uint64 = self._encode_word(entry.sig)

        # 2. Encode nodes — each reuses the registry or encodes as a word.
        node_values: list[KNode] = []
        for node_str in entry.nodes or []:
            if node_str in self._compound_sigs:
                node_values.append(KNode(self._compound_sigs[node_str], node_str))
            else:
                node_values.append(self._encode_word(node_str))

        # 3. Declared-compound definition: sig = OR of resolved component
        #    node values; register for reuse by references.
        #    Only the DEFINING entry registers — the MTS CANONICALISES entry
        #    (declared compound → its declared characters). A block-canon
        #    entry (compound → block operands, e.g. `WDMH => M H W`) is a
        #    REFERENCE: it reuses the registered signature and must NOT
        #    recompute it from its own (possibly partial/misfit) operands,
        #    or it would clobber the compound's true signature
        #    (the signature is a registry lookup, not a per-entry
        #    reduction of nodes).
        if entry.concat is not None and not is_compound_ref:
            # Synthesized compound signature (CONNOTES concat — the compound
            # in the sig's slot, e.g. AB:[B]): composes from its components
            # like any compound, never takes a word bit, and registers
            # under its identifier so later references reuse the value.
            sig_uint64 = self._compose_concat(entry.concat, entry.sig)
        elif is_compound_sig and not is_compound_ref:
            sig_uint64 = self._signifier.signature_of(node_values).with_label(entry.sig)
            if is_compound_def:
                self._compound_sigs[entry.sig] = sig_uint64
                self._compound_labels.setdefault(sig_uint64, entry.sig)

        # 4. Ask bit: an ask keeps its original canonical signature with
        #    the ASK_BPE_TOKEN flag OR-ed in — any signature can be an ask.
        if entry.is_ask:
            sig_uint64 = KNode(int(sig_uint64) | ASK_BPE_TOKEN, entry.sig)

        # 5. Debug info.
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
        # Wrap as a KValue. Significance comes from the production op
        # (entry.op — the SymbolicEntry field), NEVER read back from
        # main.dbg.op (D3: dbg is unspec'd dev-only provenance). MTS
        # emissions are asserted at S1; an MTS ask keeps its ask band.
        band = (
            band_significance("MTS")
            if entry.is_mts and not entry.is_ask
            else band_significance(entry.op)
        )
        return [(KValue(main, band), entry.is_mts)]

    # Word encoding

    def _compose_concat(self, components: list[str], label: str) -> KNode:
        """Compose a synthesized compound signature from its component words.

        Each component resolves to a registered compound signature or an
        encoded word; the compound is the OR-reduction, registered so
        later references (as a component of a larger concat, or a plain
        node ref) reuse the same value. Never takes a word bit.
        """
        parts: list[KNode] = []
        for c in components:
            if c in self._compound_sigs:
                parts.append(KNode(self._compound_sigs[c], c))
            else:
                parts.append(self._encode_word(c))
        value = self._signifier.signature_of(parts).with_label(label)
        self._compound_sigs[label] = value
        self._compound_labels.setdefault(value, label)
        return value

    def _encode_word(self, word: str) -> KNode:
        """Encode a word to its uint64 node value.

        One word, one bit: ``(word_bit << 32) | OR(bpe_token_ids)``. A
        multi-subword word ORs its subword ids into the lower half and
        still carries a single word bit — no decomposition kline.
        """
        if not word:
            return KNode(0, word)
        tokens = self._tokenizer.encode(word)
        bit = self._word_bit(word)
        bpe = 0
        for t in tokens:
            bpe |= t & 0xFFFFFFFF
        value = KNode((bit << TOP_WORD_SHIFT) | bpe, word)
        self.node_labels.setdefault(value, word)
        return value

    # Debug construction

    def _build_dbg(
        self,
        sig_uint64: int,
        label: str,
        op: str = "UNKNOWN",
    ) -> KDbg:
        """Build a KDbg for a compiled signature.

        A single-token signature is decoded defensively only to default an
        empty ``label``; ``decoded`` itself is no longer set here — it is
        populated by :func:`kalvin.kline.kline_decode` at the call site.
        Compound signatures always reach here with a non-empty ``label``
        (their KScript identifier or word), so decode never runs for them.
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
        encoder resolves from what it has already registered — encoded words
        (``node_labels``) and declared compound signatures (the reverse of
        ``_compound_sigs``).
        """
        label = self.node_labels.get(node)
        if label is None:
            label = self._compound_labels.get(node)
        if label is None:
            return None
        return KLine(node, [node], dbg=KDbg(label=label))
