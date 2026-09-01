"""NLPSignifier — the NLP signature bit-algebra.

This module is the NLP interpretation of the signature algebra defined by
:class:`kalvin.abstract.KSignifier`. It is the **sole concrete Signifier**
and the production implementation.

NLP nodes pack an NLP type word into the upper 32 bits and a BPE token ID
into the lower 32 (see :mod:`kalvin.nlp_tokenizer` for the type-word
layout)::

    node = (nlp_type32 << 32) | bpe_token_id

``NLPSignifier`` understands this packing — the peer coupling permitted
between the NLP Tokenizer and the NLP Signifier. Its operations:

- :meth:`signature_of` — bitwise OR-reduce over the full 64-bit node
  values.
- :meth:`signifies` — masked overlap: AND the two values restricted to the
  upper 32 bits (the NLP type word); non-zero means overlap.
- :meth:`residual` — masked set-difference: the type-word bits of *a* not
  in *b*.

The lower 32 bits (BPE token IDs) are masked off in :meth:`signifies` so two
values are compared by NLP type-word overlap, not by token-ID collision.

Misfit classification of a kline's signature against its nodes is a
structural concern: see :func:`kalvin.kline.classify_misfit`, which
orchestrates this class's :meth:`residual` (the residual representation and
its masking stay here).
"""

from __future__ import annotations

from collections.abc import Sequence

from kalvin.abstract import KSignifier
from kalvin.kline import KNode

# ASK is a type-word flag, not a token: bit 31 of the NLP type word (the
# type dictionary allocates bits 0-29). OR-ed into a kline signature, it
# marks the kline as an ask regardless of its signature — any signature
# can be an ask. Compiled asks read ``sig|ASK_BPE_TOKEN:[nodes]``.
ASK_BPE_TOKEN = 1 << 31


# The NLP type word occupies the upper 32 bits of a node; signifies() compares
# only that half — the BPE component (lower 32) is masked off so two values
# signify each other based on type-word overlap, not token identity.
_TYPE_MASK = 0xFFFF_FFFF_0000_0000


class NLPSignifier(KSignifier):
    """The production Signifier: the NLP masked bit-algebra.

    Operationally:

    - :meth:`signature_of` OR-reduces the full 64-bit node values.
    - :meth:`signifies` is ``(a & b & _TYPE_MASK) != 0``.
    - :meth:`residual` is ``(a & ~b) & _TYPE_MASK``.
    """

    def signature_of(self, nodes: Sequence[KNode]) -> KNode:
        """Produce a signature by OR-reducing the full node values.

        Every node contributes its entire 64-bit value; the result
        accumulates the NLP type words of all nodes. Lossy of order and
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
            label = "".join(l[:1].upper() for l in labels)
        return KNode(sig, label)

    def signifies(self, a: KNode, b: KNode) -> bool:
        """Test whether two values share an NLP type-word bit.

        The lower 32 bits (BPE token IDs) are masked off; only the upper 32
        (the NLP type word) participate.
        """
        return (a & b & _TYPE_MASK) != 0

    def residual(self, a: KNode, b: KNode) -> KNode:
        """Return the masked type-word bits of *a* not in *b*.

        ``(a & ~b) & _TYPE_MASK`` — consistent with :meth:`signifies`,
        BPE-token-id residuals are excluded so the residual captures
        type-dimension claims, not token-id differences. The label is the
        mask expression ``a.label & ~b.label``.
        """
        mask = (a & ~b) & _TYPE_MASK
        label = f"{getattr(a, 'label', '')} & ~{getattr(b, 'label', '')}"
        return KNode(mask, label)

    def bit_in(self, node: KNode, signature: KNode) -> bool:
        """Does ``node``'s bit pattern sit inside ``signature``?"""
        return (node & signature) == node

    def is_ask(self, signature: KNode) -> bool:
        """Does ``signature`` carry the ASK_BPE_TOKEN flag?"""
        return (signature & ASK_BPE_TOKEN) != 0
