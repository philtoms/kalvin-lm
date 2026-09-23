"""Round 2b: with the ask surviving, bypass Def 22 selection and derive the
riding ask directly toward the SVO canon (the ==-declared goal). Is
selection the only blocker, or does the move set block too?"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.derivation import Derivation
from kalvin.hop import Hop, trawl
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
    nodes = ", ".join(name(n) for n in k.nodes)
    ask = "|ASK" if is_ask(k.signature) else ""
    return f"{name(k.signature)}{ask}:[{nodes}]"


tok = BPETokenizer()
entries = compile_source(open("data/scripts/mhall.ks").read(), tokenizer=tok,
                         signifier=None, dev=True, word_bits={})
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
             and e.kline.dbg.label == "MHALL")
compiled_ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
svo = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
           and e.kline.dbg.label == "SVO")
print(f"declared goal (KDbg.goal on ask): {compiled_ask.kline.dbg.goal}")

state = AskNeverAnswered(NLPSignifier())
h = Harness(tok, AskAttends(state))
sig = h.signifier

scaffolds = [e for e in entries if e is not canon and e is not compiled_ask]
turn(h, scaffolds)
turn(h, [KValue(canon.kline, S1)])
turn(h, [KValue(compiled_ask.kline, S4)])

ask_kline = next(k for k in state.work_list if is_ask(k.signature))
print(f"\nask attending: {render(ask_kline)}")

scope = trawl(state, ask_kline.nodes, svo.kline.nodes, sig)
print(f"scope (ask roots + SVO roots): {len(scope)}")
for k in scope:
    print(f"  {render(k)}")

d = Derivation(scope, ask_kline, svo.kline, sig)
print(f"\nusable evidence: {[render(k) for k in d.memory]}")
r = d.run()
print(f"derivation vs SVO canon: ending={r.ending} j1={r.j1:.3f} "
      f"composed={len(r.composed)} hbar={r.hbar:.2f}")
for step_i, tr_ in enumerate(getattr(r, "trace", []) or []):
    print(f"  trace[{step_i}]: {[name(n) for n in tr_]}")

# Also try the full Hop chain with SVO forced onto the goal list.
hop = Hop(state, ask_kline, sig).run()
print(f"\nnatural hop: ending={hop.ending}")
