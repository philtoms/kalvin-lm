"""Show γ(WDMH, MHALL) atom-by-atom with the actual compiled values."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from kalvin.significance import gamma_to_byte, word_atom_count, BandLayout

layout = BandLayout()
tok = BPETokenizer()
sig = NLPSignifier()
bits: dict[str, int] = {}

for path in ("data/scripts/mhall.ks", "data/scripts/wdmh-underfit.ks"):
    entries = compile_source(open(path).read(), tokenizer=tok, signifier=sig,
                             dev=True, word_bits=bits)
    for e in entries:
        d = e.kline.dbg
        if d and d.goal:
            ask = e
            goal = next(g for g in entries
                        if g.kline.dbg and g.kline.dbg.label == d.goal
                        and g.kline.nodes)
            a, b = int(ask.kline.signature), int(goal.kline.signature)
            word_mask = 0x7FFFFFFF << 32  # word-bit half
            a_atoms = a & word_mask
            b_atoms = b & word_mask
            inter, union = a_atoms & b_atoms, a_atoms | b_atoms
            names = {}
            for e2 in entries:
                for n in ([e2.kline.signature, *e2.kline.nodes]):
                    names.setdefault(int(n) & word_mask, getattr(n, "label", hex(int(n))))
            def show(v):
                return sorted(names.get(bit, hex(bit)) for bit in
                              (i for i in range(64) if v & (1 << i)))
            j = word_atom_count(inter) / word_atom_count(union)
            print(f"{path}:")
            print(f"  ask  {d.label} atoms: {show(a_atoms)}")
            print(f"  goal {d.goal} atoms: {show(b_atoms)}")
            print(f"  shared: {show(inter)}  ({word_atom_count(inter)} of "
                  f"{word_atom_count(union)} union)")
            print(f"  J = {j:.3f}  γ = J·δ^0 = {j:.3f}  "
                  f"byte 0x{gamma_to_byte(j):02x} band {layout.classify(gamma_to_byte(j))}")
            print()
