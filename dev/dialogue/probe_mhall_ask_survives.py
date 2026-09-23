"""Probe round 2: canon S1 + empty ask S4, asks attend (fast_route ignores
ask-marked feeds) AND is_answered ignores ask-marked klines. The ask must
now survive to _propose. Variants: riding ask, riding ask + scaffolds."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.hop import Hop, candidate_goals, trawl
from kalvin.kline import KLine, is_ask, mark_ask, using_resolver
from kalvin.kvalue import KValue
from kalvin.memory import Memory
from kalvin.rationaliser import Rationaliser
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from dev.dialogue.harness import Harness

S1 = 0xFF
S4 = 0x00


class AskAttends(Rationaliser):
    """_fast_route ignores ask-marked feeds: questions attend, never refuse."""

    def _fast_route(self, query: KValue) -> bool:
        if is_ask(query.kline.signature):
            return False
        return super()._fast_route(query)


class AskNeverAnswered(Memory):
    """is_answered ignores ask-marked klines: no inspection discharge."""

    def is_answered(self, kline: KLine) -> bool:
        if is_ask(kline.signature):
            return False
        return super().is_answered(kline)


def fresh():
    state = AskNeverAnswered(NLPSignifier())
    return Harness(tok, AskAttends(state))


def turn(h, feeds):
    with using_resolver(h.state.find):
        h.rationaliser.rationalise(feeds)
        batch = []
        while True:
            size = len(h.state.work_list)
            batch.extend(cogitate(h.state))
            if len(h.state.work_list) == size:
                break
        return batch


def name(v):
    return getattr(v, "label", "") or hex(int(v))


def render(k):
    nodes = ", ".join(name(n) for n in k.nodes)
    ask = "|ASK" if is_ask(k.signature) else ""
    return f"{name(k.signature)}{ask}:[{nodes}]"


def dump(h, batch, label):
    st = h.state
    print(f"\n== {label} ==")
    print(f"emissions: {len(batch)}")
    for v in batch:
        print(f"  -> {render(v.kline)} 0x{v.significance:02x}")
    print(f"work_list: {[render(k) for k in st.work_list]}")
    print(f"refused: {sorted(st.refused)}  stm: {len(st.stm)}")
    print(f"frame: {[render(k) for b in st.frame.values() for k in b]}")
    for k in st.work_list:
        if not is_ask(k.signature):
            continue
        goals = candidate_goals(st, k, h.signifier)
        print(f"  ask hop on {render(k)}:")
        print(f"    goals={[render(g) for g in goals]}")
        for g in goals[:4]:
            scope = trawl(st, k.nodes, g.nodes, h.signifier)
            print(f"    vs goal {render(g)}: scope={len(scope)} "
                  f"{[render(x) for x in scope[:6]]}")
        if goals:
            hop = Hop(st, k, h.signifier).run()
            print(f"    ending={hop.ending} writes={len(hop.writes)}")
            for r in hop.results:
                tail = " > ".join(name(n) for n in r.trace[-1][:6])
                print(f"      {r.ending:10s} j1={r.j1:.3f} tail=[{tail}]")


tok = BPETokenizer()
entries = compile_source(open("data/scripts/mhall.ks").read(), tokenizer=tok,
                         signifier=None, dev=True, word_bits={})
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
             and e.kline.dbg.label == "MHALL")
compiled_ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
empty_ask = KValue(KLine(mark_ask(canon.kline.signature), []), S4)
canon_s1 = KValue(canon.kline, S1)

# 1. The spec: canon S1 -> empty ask S4, both discharges off.
h1 = fresh()
turn(h1, [canon_s1])
b1 = turn(h1, [empty_ask])
dump(h1, b1, "1. canon S1 -> empty ask S4 (attends, never answered)")

# 2. Riding ask instead of empty, both discharges off.
h2 = fresh()
turn(h2, [canon_s1])
b2 = turn(h2, [KValue(compiled_ask.kline, S4)])
dump(h2, b2, "2. canon S1 -> riding ask S4 (attends, never answered)")

# 3. Full lesson: scaffolds -> canon S1 -> riding ask S4.
h3 = fresh()
scaffolds = [e for e in entries if e is not canon and e is not compiled_ask]
turn(h3, scaffolds)
turn(h3, [canon_s1])
b3 = turn(h3, [KValue(compiled_ask.kline, S4)])
dump(h3, b3, "3. scaffolds -> canon S1 -> riding ask S4")
print("\n(non-ask work list items: "
      f"{[render(k) for k in h3.state.work_list if not is_ask(k.signature)]})")
