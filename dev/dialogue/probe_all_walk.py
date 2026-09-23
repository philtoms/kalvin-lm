"""The ALL slot's walk under the single-node-witness rule: which path
delivers Object, at what depth?"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.derivation import Derivation
from kalvin.hop import trawl
from kalvin.kline import KLine, using_resolver
from kalvin.kvalue import KValue
from ks.compiler import compile_source
from dev.dialogue.harness import make_rationaliser

tok = BPETokenizer()
h = make_rationaliser(tok)
st = h.state
sig = h.signifier
bits: dict[str, int] = {}

entries = compile_source(Path("data/scripts/mhall.ks").read_text(),
                         tokenizer=tok, signifier=sig, dev=True, word_bits=bits)
ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
goal = next(e for e in entries if e.kline.dbg and e.kline.dbg.label == "SVO"
            and e.kline.dbg.op == "CANONICALISES")
scaffolding = [e for e in entries if e is not ask]

def name(v):
    return getattr(v, "label", "") or hex(int(v))

def render(k):
    return f"{name(k.signature)}:[{', '.join(name(n) for n in k.nodes)}]"

with using_resolver(st.find):
    h.rationaliser.rationalise(scaffolding)
    while True:
        size = len(st.work_list)
        cogitate(st)
        if len(st.work_list) == size:
            break
    # The re-entered state after Mary and had bridges: [Subject, Verb, ALL]
    h.rationaliser.rationalise([KValue(ask.kline, 0x00)])
    from kalvin.cogitator import cogitate as cg
    # run the ask once to write the Mary/had bridges, then inspect ALL's walk
    cg(st)

a = KLine(ask.kline.signature, [next(int(n) for n in scaffolding[0].kline.nodes if True)] if False else None)

# build the [Subject, Verb, ALL] re-entry directly
subject = next(int(n) for k in st.where(lambda k: True, True) for n in k.nodes if name(n) == "Subject")
verb = next(int(n) for k in st.where(lambda k: True, True) for n in k.nodes if name(n) == "Verb")
allv = next(int(k.signature) for k in st.where(lambda k: True, True) if name(k.signature) == "ALL")
reentry = KLine(ask.kline.signature, [subject, verb, allv], dbg=ask.kline.dbg)

scope = trawl(st, reentry.nodes, goal.kline.nodes, sig)
d = Derivation(scope, reentry, goal.kline, sig)
print(f"queued {render(reentry)}  gap={d.gap():#x} excess={d.excess():#x}")
print(f"a_starts: {[name(n) or hex(int(n)) for n in reentry.nodes if sig.signifies(n, d.gap())]}")
path_a = d.descend([n for n in reentry.nodes if sig.signifies(n, d.gap())])
for v, (depth, key, start) in sorted(path_a.items(), key=lambda t: t[1][0]):
    print(f"  A {name(v) or hex(v):12s} @{depth} via {name(key[0]) or hex(key[0])}:"
          f"[{', '.join(name(n) or hex(int(n)) for n in key[1])}] from {name(start) or hex(start)}")
