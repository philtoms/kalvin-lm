"""Instrument Derivation.run: dump every derivation whose witness ever
contains the SVO canon head atom (0xe000007ffe), move by move, with
composed bridges."""
from __future__ import annotations

import sys

sys.path.insert(0, "dev/dialogue")

import kalvin.derivation as D
from harness import make_rationaliser, _sig_to_label  # type: ignore
from kalvin.bpe_tokenizer import BPETokenizer

SVO = 0xE000007FFE
NAMES = {
    0x2100004DFD: "MarySubject", 0x420000017E: "hadVerb",
    0x18000003FFF: "QueryObject", 0x11C00001FFF: "ALLQuery",
    0xE000007FFE: "SVO", 0x1F00001FFD: "MHALL", 0x1C00001EFD: "ALL",
    0x1000005FD: "Mary", 0x200000178: "had", 0x10000010FD: "lamb",
    0x400000061: "a", 0x800000E7D: "little",
    0x2000004DDC: "Subject", 0x400000017E: "Verb", 0x800000E7D & 0xFFFFFFFF: "?",
    137438973404: "Subject", 274877907326: "Verb", 549755830254: "Object",
    4294968829: "Mary", 8589934968: "had", 17179869281: "a",
    34359742077: "little", 68719481085: "lamb",
}


def nm(v):
    v = int(v)
    if v in NAMES:
        return NAMES[v]
    if (v & 0xFFFFFFFF) and (v >> 32):
        lo = NAMES.get(v & 0xFFFFFFFF, hex(v & 0xFFFFFFFF))
        return f"{lo}·t{v >> 32}"
    return NAMES.get(v, hex(v))


import kalvin.memory as MM
_orig_ext = MM.Memory.extend_stm

def patched_ext(self, klines):
    for k in klines:
        vals = [int(n) for n in k.nodes] + [int(k.signature)]
        if SVO in vals:
            print(f"  STM⟵ {nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}]")
    return _orig_ext(self, klines)

MM.Memory.extend_stm = patched_ext

_orig_ground = D.Derivation._ground_composed

def patched_ground(self, composed):
    if SVO in [int(n) for n in composed.nodes] or SVO == int(composed.signature):
        print(f"  ⟂ COMPOSED {nm(composed.signature)}:"
              f"[{', '.join(nm(n) for n in composed.nodes)}]  "
              f"queued={nm(self.queued.signature)} goal={nm(self.goal.signature)}")
    return _orig_ground(self, composed)

D.Derivation._ground_composed = patched_ground

_orig_run = D.Derivation.run


def patched_run(self):
    r = _orig_run(self)
    if any(SVO == int(n) for step in r.trace for n in step):
        print(f"\n== queued={nm(self.queued.signature)} "
              f"goal={nm(self.goal.signature)} ending={r.ending} "
              f"j1={r.j1:.3f}")
        for i, st in enumerate(r.trace):
            print(f"   t{i}: [{', '.join(nm(n) for n in st)}]")
        for c in self.composed:
            print(f"   composed: {nm(c.signature)}:"
                  f"[{', '.join(nm(n) for n in c.nodes)}]")
    return r


D.Derivation.run = patched_run

src = open("data/scripts/mhall.ks").read()
tok = BPETokenizer()
h = make_rationaliser(tok)
h.run(src)
print("\ndone")
