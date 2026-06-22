"""NLPSignifier — the NLP signature bit-algebra.

This module is the NLP interpretation of the signature algebra defined by
:class:`kalvin.abstract.KSignifier`. It is the **sole concrete Signifier**
and the production implementation.

NLP nodes pack an NLP type word into the upper 32 bits and a BPE token ID
into the lower 32 (see :mod:`kalvin.nlp_tokenizer` for the type-word
layout)::

    node = (nlp_type32 << 32) | bpe_token_id

``NLPSignifier`` understands this packing — the peer coupling permitted
between the NLP Tokenizer and the NLP Signifier (see specs/signifier.md
§NLPSignifier). Its two operations:

- :meth:`make_signature` — bitwise OR-reduce over the full 64-bit node
  values.
- :meth:`signifies` — masked overlap: AND the two values restricted to the
  upper 32 bits (the NLP type word); non-zero means overlap.

The lower 32 bits (BPE token IDs) are masked off in :meth:`signifies` so two
values are compared by NLP type-word overlap, not by token-ID collision.

See specs/signifier.md for the full specification, including the
NLPSignifier-specific properties (determinism, commutativity, the empty→0
convention) that are consequences of this bit-algebra and are **not**
required by the ``KSignifier`` interface.
"""

from __future__ import annotations

from collections.abc import Sequence

from kalvin.abstract import KSignifier

# The NLP type word occupies the upper 32 bits of a node; signifies() compares
# only that half — the BPE component (lower 32) is masked off so two values
# signify each other based on type-word overlap, not token identity.
_TYPE_MASK = 0xFFFF_FFFF_0000_0000


class NLPSignifier(KSignifier):
    """The production Signifier: the NLP masked bit-algebra.

    See specs/signifier.md §NLPSignifier. Operationally:

    - :meth:`make_signature` OR-reduces the full 64-bit node values.
    - :meth:`signifies` is ``(a & b & _TYPE_MASK) != 0``.
    """

    def make_signature(self, nodes: Sequence[int]) -> int:
        """Produce a signature by OR-reducing the full node values.

        Every node contributes its entire 64-bit value; the result
        accumulates the NLP type words of all nodes. Lossy of order and
        multiplicity (``{A, B}`` and ``{A, A, B}`` reduce identically).
        """
        sig = 0
        for node in nodes:
            sig |= node
        return sig

    def signifies(self, a: int, b: int) -> bool:
        """Test whether two values share an NLP type-word bit.

        The lower 32 bits (BPE token IDs) are masked off; only the upper 32
        (the NLP type word) participate.
        """
        return (a & b & _TYPE_MASK) != 0
