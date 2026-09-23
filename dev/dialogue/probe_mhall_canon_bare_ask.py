"""Feed MHALL:[Mary, had, a, little, lamb] S1 (the canon) + MHALL:[] S4
(the bare ask / Unknown shape). Variants isolate each factor:
  A. canon S1, then bare ask S4        (the asked pair)
  B. canon S1, then bare ask S3        (attends instead of refusing)
  C. bare ask S3 alone                 (no canon in memory)
  D. bare ask S4 alone                 (no canon, refused)"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.kline import KLine, using_resolver, mark_ask
from kalvin.kvalue import KValue
from ks.compiler import compile_source
from dev.dialogue.harness import make_rationaliser

tok = BPETokenizer()
bits: dict[str, int] = {}
sig = make_rationaliser(tok).signifier

entries = compile_source(open("data/scripts/mhall.ks").read(), tokenizer=tok,
                         signifier=sig, dev=True, word_bits=bits)
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
             and e.kline.dbg.label == "MHALL")
ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
bare = KLine(ask.kline.signature, [])  # same marked signature, no riding canon


def name(v):
    return getattr(v, "label", "") or hex(int(v))


def run(label, feeds):
    h = make_rationaliser(tok)
    st = h.state
    out = []
    with using_resolver(st.find):
        for feed in feeds:
            h.rationaliser.rationalise([feed])
            while True:
                size = len(st.work_list)
                out.extend(cogitate(st))
                if len(st.work_list) == size:
                    break
    print(f"\n== {label} ==")
    for v in out:
        nodes = ", ".join(name(n) for n in v.kline.nodes) or "(empty)"
        print(f"  emits {name(v.kline.signature)}:[{nodes}]  0x{v.significance:02x}")
    if not out:
        print("  emits (nothing)")
    b = bare
    print(f"  bare ask: refused={st.is_refused(b)}  "
          f"grounded={st.is_grounded(b)}  "
          f"answered={st.is_answered(b)}  "
          f"attending={any(e.signature == b.signature for e in st.work_list)}")
    print(f"  canon: grounded={st.is_grounded(canon.kline)}")
    for k in st.work_list:
        nodes = ", ".join(name(n) for n in k.nodes) or "(empty)"
        print(f"  work: {name(k.signature)}:[{nodes}]")


S1, S3, S4 = 0xFF, 0x40, 0x00
canon_v = KValue(canon.kline, S1)
run("A: canon S1, then bare ask S4", [canon_v, KValue(bare, S4)])
run("B: canon S1, then bare ask S3", [canon_v, KValue(bare, S3)])
run("C: bare ask S3 alone", [KValue(bare, S3)])
run("D: bare ask S4 alone", [KValue(bare, S4)])
