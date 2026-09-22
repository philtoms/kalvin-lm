"""Round 2c: instrument Derivation.run's policy loop on the ask-vs-SVO
derivation — which move fired, what was offered at each stage, why stuck."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.derivation import Derivation
from kalvin.hop import trawl
from kalvin.kline import is_ask, using_resolver
from kalvin.kvalue import KValue
from kalvin.memory import Memory
from kalvin.rationaliser import Rationaliser
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from dev.dialogue.harness import Harness

S1 = 0xFF
S4 = 0x00


class AskAttends(Rationaliser):
    def _fast_route(self, query: KValue) -> bool:
        if is_ask(query.kline.signature):
            return False
        return super()._fast_route(query)


class AskNeverAnswered(Memory):
    def is_answered(self, kline) -> bool:
        if is_ask(kline.signature):
            return False
        return super().is_answered(kline)


def turn(h, feeds):
    with using_resolver(h.state.find):
        h.rationaliser.rationalise(feeds)
        while True:
            size = len(h.state.work_list)
            cogitate(h.state)
            if len(h.state.work_list) == size:
                break


def name(v):
    return getattr(v, "label", "") or hex(int(v))


def render(k):
    return f"{name(k.signature)}:[{', '.join(name(n) for n in k.nodes)}]"


tok = BPETokenizer()
entries = compile_source(open("data/scripts/mhall.ks").read(), tokenizer=tok,
                         signifier=None, dev=True, word_bits={})
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
             and e.kline.dbg.label == "MHALL")
compiled_ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
svo = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
           and e.kline.dbg.label == "SVO")

state = AskNeverAnswered(NLPSignifier())
h = Harness(tok, AskAttends(state))
sig = h.signifier
scaffolds = [e for e in entries if e is not canon and e is not compiled_ask]
turn(h, scaffolds)
turn(h, [KValue(canon.kline, S1)])
turn(h, [KValue(compiled_ask.kline, S4)])
ask_kline = next(k for k in state.work_list if is_ask(k.signature))

scope = trawl(state, ask_kline.nodes, svo.kline.nodes, sig)
d = Derivation(scope, ask_kline, svo.kline, sig)

print(f"relationship band: {d.relationship_band()}")
print(f"gap={hex(d.gap())} excess={hex(d.excess())} mismatch={d.mismatch()}")

print("\n== usable evidence ==")
for k in d.memory:
    print(f"  {'OK ' if d.usable(k) else 'NO '} {render(k)}")

print("\n== canonicalisations offered ==")
for k, group, new in d.canonicalisations():
    print(f"  canon {render(k)} group={[name(g) for g in group]} "
          f"exposes={d._exposes(new)} new={[name(x) for x in new]}")

print("\n== targetings offered ==")
for k, kind, new, d0, d1 in d.targetings():
    print(f"  {kind:8s} {render(k)}  misfit {d0}->{d1}  new={[name(x) for x in new]}")

print("\n== run with per-step logging ==")
d2 = Derivation(scope, ask_kline, svo.kline, sig)
import kalvin.derivation as D

for step in range(6):
    print(f"step {step}: nodes={[name(n) for n in d2.nodes]} "
          f"done={d2.done()} misfit={d2.mismatch()}")
    if d2.done():
        break
    offered_c = list(d2.canonicalisations())
    if any(d2._exposes(new) for _, _, new in offered_c):
        d2._canonicalise()
        print("   -> canonicalised")
        continue
    t = next(d2.targetings(), None)
    if t is not None:
        d2._target()
        print(f"   -> targeted via {render(t[0])} {t[1]}")
        continue
    moved = d2._walk()
    print(f"   -> walk={moved}")
    if not moved:
        print("   STUCK")
        break
