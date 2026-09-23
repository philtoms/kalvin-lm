"""Probe: canon S1 + empty ask S4, with _fast_route letting ask-marked
feeds attend. Variants: no-canon control, riding-nodes ask, plus scaffolds."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.hop import Hop, candidate_goals
from kalvin.kline import KLine, is_ask, mark_ask, using_resolver
from kalvin.kvalue import KValue
from kalvin.rationaliser import Rationaliser
from ks.compiler import compile_source
from dev.dialogue.harness import make_rationaliser

S1 = 0xFF
S4 = 0x00


class AskAttends(Rationaliser):
    """_fast_route ignores ask-marked feeds: questions attend, never refuse."""

    def _fast_route(self, query: KValue) -> bool:
        if is_ask(query.kline.signature):
            return False
        return super()._fast_route(query)


def fresh():
    h = make_rationaliser(tok)
    h._rationaliser = AskAttends(h.state)
    return h


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
    print(f"refused: {len(st.refused)}  stm: {len(st.stm)}")
    grounded = [render(k) for b in st.frame.values() for k in b]
    print(f"frame: {grounded}")
    for k in st.work_list:
        goals = candidate_goals(st, k, h.signifier)
        print(f"  hop on {render(k)}: goals={[render(g) for g in goals[:4]]}")
        if goals:
            hop = Hop(st, k, h.signifier).run()
            print(f"    ending={hop.ending} writes={len(hop.writes)}")


tok = BPETokenizer()
entries = compile_source(open("data/scripts/mhall.ks").read(), tokenizer=tok,
                         signifier=None, dev=True, word_bits={})
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
             and e.kline.dbg.label == "MHALL")
compiled_ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
mhall_sig = canon.kline.signature
empty_ask = KValue(KLine(mark_ask(mhall_sig), []), S4)
canon_s1 = KValue(canon.kline, S1)
print(f"canon: {render(canon.kline)}  compiled byte 0x{canon.significance:02x}")
print(f"compiled ask: {render(compiled_ask.kline)}  byte 0x{compiled_ask.significance:02x}")
print(f"empty ask: {render(empty_ask.kline)}  byte 0x{empty_ask.significance:02x}")

# 1. The user's exact spec: canon S1, then empty ask S4.
h1 = fresh()
turn(h1, [canon_s1])
b1 = turn(h1, [empty_ask])
dump(h1, b1, "1. canon S1 -> empty ask S4 (ask attends)")

# 2. Control: empty ask alone (no canon fed).
h2 = fresh()
b2 = turn(h2, [empty_ask])
dump(h2, b2, "2. empty ask S4 alone (control, no canon)")

# 3. Riding-nodes ask at S4 with canon grounded first.
h3 = fresh()
turn(h3, [canon_s1])
b3 = turn(h3, [KValue(compiled_ask.kline, S4)])
dump(h3, b3, "3. canon S1 -> riding ask S4")

# 4. Scaffolds + canon + empty ask (the full lesson shape).
h4 = fresh()
scaffolds = [e for e in entries if e is not canon and e is not compiled_ask]
turn(h4, scaffolds)
turn(h4, [canon_s1])
b4 = turn(h4, [empty_ask])
dump(h4, b4, "4. scaffolds -> canon S1 -> empty ask S4")
