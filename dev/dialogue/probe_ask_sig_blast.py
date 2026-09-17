"""Blast probe: the proposed ask form WDMH|ASK_SIG:[what,did,Mary,have].

Compares structural readings, grounding, γ pollution, Def 22 candidates,
and self-exclusion between the three forms:
  (1) current  WDMH:[]                      (empty ask)
  (2) noded    WDMH:[what,did,Mary,have]    (no bit — identical to the canon)
  (3) proposal WDMH|ASK_SIG:[what,did,Mary,have]  (bit 63 of the word word)
"""
import sys
sys.path.insert(0, "src")

from pathlib import Path
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import KLine, is_terminal, is_canon, is_exact, sig_level, classify_misfit
from kalvin.significance import word_atom_count, misfit_mass
from kalvin.hop import candidate_goals
from dialogue.harness import load_engine

ASK_SIG = 1 << 63  # bit 31 of the word word

tok = BPETokenizer()
h = load_engine(Path("data/dialogue/wdmh-underfit.json"), tok)
st, sig = h.state, h.signifier

def name(v):
    return getattr(v, "label", "") or hex(int(v))

def render(k):
    return f"{name(k.signature)}:[{', '.join(name(n) for n in k.nodes)}]"

# The withheld canon gives us node values for what/did/Mary/have.
from ks.compiler import compile_source
entries = compile_source(open("data/scripts/wdmh-underfit.ks").read(),
                         tokenizer=tok, signifier=sig, dev=True,
                         word_bits=dict(st.word_bits or {}))
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.label == "WDMH"
             and e.kline.dbg.op == "CANONICALISES" and e.kline.dbg.scope == 1)
nodes = list(canon.kline.nodes)
base_sig = int(canon.kline.signature)

empty = KLine(base_sig, [])
noded = KLine(base_sig, nodes)
bitd = KLine(base_sig | ASK_SIG, nodes, dbg=canon.kline.dbg)

print(f"canon: {render(canon.kline)}  sig={base_sig:#x}")
for label, k in (("empty WDMH:[]", empty), ("noded (no bit)", noded), ("WDMH|ASK_SIG:[...]", bitd)):
    exact = is_exact(k, sig)
    print(f"\n-- {label} --")
    print(f"  sig_level      : {sig_level(k, sig)}")
    print(f"  is_terminal    : {is_terminal(k)}   is_canon: {is_canon(k, sig)}   is_exact: {exact}")
    if k.nodes:
        u, o = classify_misfit(k, sig)
        print(f"  classify_misfit: underfit={u} overfit={o}")
    print(f"  is_groundable  : {st.is_groundable(k)}   is_grounded: {st.is_grounded(k)}")
    print(f"  atoms in sig   : {word_atom_count(int(k.signature))}")
    goals = candidate_goals(st, k, sig)
    print(f"  Def22 candidates ({len(goals)}): {[(render(g), ) for g in goals[:4]]}")

# γ pollution: J between the ask signature and the goal MHALL signature.
goal = next(k for k in st.where(lambda x: getattr(x.signature, "label", "") == "MHALL"
                                and x.nodes))
gb = int(goal.signature)
for label, a in (("empty", base_sig), ("bitd", base_sig | ASK_SIG)):
    union = word_atom_count(a | gb)
    j = word_atom_count(a & gb) / union if union else 1.0
    print(f"\nJ({label}, MHALL) = {word_atom_count(a & gb)}/{union} = {j:.3f}   misfit_mass={misfit_mass(a, gb)}")

# Self-exclusion: is the pooled WDMH canon itself a candidate of the ask?
canon_in = any(g.signature == canon.kline.signature and g.nodes == canon.kline.nodes
               for g in candidate_goals(st, bitd, sig))
print(f"\nself-exclusion: WDMH canon is candidate of bit'd ask: {canon_in}")
print(f"                 WDMH canon is candidate of noded ask: "
      f"{any(g is canon or (g.signature == canon.kline.signature and g.nodes == canon.kline.nodes) for g in candidate_goals(st, noded, sig))}")
