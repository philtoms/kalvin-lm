"""NLPSignifier — the signature bit-algebra.

This module is the NLP interpretation of the signature algebra defined by
:class:`kalvin.abstract.KSignifier`. It is the **sole concrete Signifier**
and the production implementation.
Nodes pack a word word into the upper 32 bits and a BPE token ID into
the lower 32 (assigned by the ks compiler's TokenEncoder; the standard
BPE tokenizer returns tokens with the upper 32 bits zero — see
:mod:`kalvin.bpe_tokenizer`)::

    node = (word_bit << 32) | bpe_token_id

The word word gives one bit per distinct word (bits 0-30,
first-encountered basis); bit 31 carries the ASK marker.

``NLPSignifier`` understands this packing. Its operations:

- :meth:`signature_of` — bitwise OR-reduce over the full 64-bit node
  values.
- :meth:`signifies` — masked overlap: AND the two values restricted to the
  upper 32 bits (the word word); non-zero means overlap.
- :meth:`residual` — masked set-difference: the word-word bits of *a* not
  in *b*.

The lower 32 bits (BPE token IDs) are masked off in :meth:`signifies` so two
values are compared by word-bit overlap, not by token-ID collision. The ASK
marker (bit 31 of the word word — :data:`kalvin.kline.ASK_SIG`) is masked
off in :meth:`signifies` and :meth:`residual` alike: it marks identity, not
content, so it never weighs as an atom and never appears as a gap.

Misfit classification of a kline's signature against its nodes is a
structural concern: see :func:`kalvin.kline.classify_misfit`, which
orchestrates this class's :meth:`residual` (the residual representation and
its masking stay here).
"""

from __future__ import annotations

from collections.abc import Sequence

from kalvin.abstract import KSignifier
from kalvin.kline import ASK_SIG, KNode, KSig

# The word word occupies the upper 32 bits of a node; signifies() compares
# only that half — the BPE component (lower 32) is masked off so two values
# signify each other based on word-bit overlap, not token identity. The ASK
# marker is masked off with it: measurement reads content only.
_TYPE_MASK = 0xFFFF_FFFF_0000_0000 & ~ASK_SIG


class NLPSignifier(KSignifier):
    """The production Signifier: the masked bit-algebra.

    Operationally:

    - :meth:`signature_of` OR-reduces the full 64-bit node values.
    - :meth:`signifies` is ``(a & b & _TYPE_MASK) != 0``.
    - :meth:`residual` is ``(a & ~b) & _TYPE_MASK``.
    """

    def signature_of(self, nodes: Sequence[KNode]) -> KSig:
        """Produce a signature by OR-reducing the full node values.

        Every node contributes its entire 64-bit value; the result
        accumulates the word words of all nodes. Lossy of order and
        multiplicity (``{A, B}`` and ``{A, A, B}`` reduce identically).
        The returned KNode is labelled with the whole label of the single
        node, or the first character of each node's label uppercased and
        concatenated.
        """
        sig = 0
        for node in nodes:
            sig |= node
        labels = [getattr(n, "label", "") for n in nodes]
        if len(labels) == 1:
            label = labels[0]
        else:
            label = "".join(l for l in labels)
        return KSig(sig, label)

    def signifies(self, a: KSig, b: KSig) -> bool:
        """Test whether two values share a word-word bit.

        The lower 32 bits (BPE token IDs) and the ASK marker are masked
        off; only the word word (bits 0-30) participates.
        """
        s = (a & b & _TYPE_MASK) != 0
        return s

    def residual(self, a: KSig, b: KSig) -> KSig:
        """Return the masked word-word bits of *a* not in *b*.

        ``(a & ~b) & _TYPE_MASK`` — consistent with :meth:`signifies`,
        BPE-token-id residuals and the ASK marker are excluded so the
        residual captures word-dimension claims, not token-id differences
        nor ask-marked distinctiveness. The label is the mask expression
        ``a.label & ~b.label``.
        """
        mask = (a & ~b) & _TYPE_MASK
        label = f"{getattr(a, 'label', '')} & ~{getattr(b, 'label', '')}"
        return KSig(mask, label)

    def node_in(self, node: KNode, signature: KSig) -> bool:
        """Does ``node``'s bit pattern sit inside ``signature``?"""
        return (node & signature) == node
