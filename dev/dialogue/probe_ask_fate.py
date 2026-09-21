"""Feed the compiled WDMH ask directly to the rationaliser; inspect its fate."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.cogitator import cogitate
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from dev.dialogue.harness import make_rationaliser
from kalvin.kvalue import KValue

tok = BPETokenizer()
h = make_rationaliser(tok)
sig = h.signifier
bits: dict[str, int] = {}

mhall = compile_source(open("data/scripts/mhall.ks").read(), tokenizer=tok,
                       signifier=sig, dev=True, word_bits=bits)
wdmh = compile_source(open("data/scripts/wdmh-underfit.ks").read(), tokenizer=tok,
                      signifier=sig, dev=True, word_bits=bits)

# Ground the mhall material first (all entries, batch-style).
h.rationaliser.rationalise(mhall)
batch = cogitate(h.state)
print(f"mhall batch emissions: {len(batch)}")

ask = next(e for e in wdmh if e.kline.dbg and e.kline.dbg.op == "ASK")
print(f"ask entry: sig={ask.kline.dbg.label} nodes={ask.kline.nodes} band_byte=0x{ask.significance:02x}")
h.rationaliser.rationalise([ask])
out = cogitate(h.state)
print(f"emissions from ask feed: {len(out)}")
st = h.state
print(f"ask in work_list: {any(e.signature == ask.kline.signature for e in st.work_list)}")
print(f"ask refused: {st.is_refused(ask.kline)}")
print(f"work_list size: {len(st.work_list)}")

# And the queued misfit form the algebra uses: wdmh:[w,d,m,h]-shaped MTS canon.
mts = next(e for e in wdmh if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
           and e.kline.dbg.label == "WDMH" and e.kline.dbg.scope == 1)
print(f"\nMTS canon entry: {mts.kline.dbg.label} nodes={[str(n) for n in mts.kline.nodes]}")
h.rationaliser.rationalise([mts])
out = cogitate(h.state)
print(f"emissions from MTS-canon feed: {len(out)}")
print(f"MTS canon in work_list: {any(e.signature == mts.kline.signature and e.nodes == mts.kline.nodes for e in st.work_list)}")
print(f"MTS canon grounded: {st.is_grounded(mts.kline)}")
