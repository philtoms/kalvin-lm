from __future__ import annotations

from dataclasses import dataclass

from kalvin.kline import KSig

@dataclass(frozen=True)
class KPath:
    left: KSig
    right: KSig
    hops: int
