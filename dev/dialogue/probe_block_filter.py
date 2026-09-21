
from kalvin.engine import Engine
from kalvin.engine_state import EngineState
from kalvin.kvalue import KValue
from kalvin.significance import SIG_S1, SIG_S3
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from kalvin.bpe_tokenizer import BPETokenizer

src = open("data/scripts/mhall.ks").read()
tok = BPETokenizer(); sign = NLPSignifier()
entries = compile_source(src, tokenizer=tok, signifier=sign, dev=True)
from kalvin.kline import is_identity, is_canon

def run(filter_primed):
    st = EngineState(NLPSignifier())
    en = Engine(st)
    priming = [e for e in entries if (is_identity(e.kline) or is_canon(e.kline, st.signifier)) and not st.is_grounded(e.kline)]
    en.rationalise([KValue(e.kline, SIG_S1) for e in priming])
    primed = {(e.kline.signature, tuple(e.kline.nodes)) for e in priming if st.is_grounded(e.kline)}
    g1 = entries[:12]
    feed = [e for e in g1 if (e.kline.signature, tuple(e.kline.nodes)) not in primed] if filter_primed else list(g1)
    b1 = en.rationalise(feed)
    # find Mary:[Subject] entry
    ms = next(e for e in entries if e.kline.nodes and len(e.kline.nodes)==1 and e.kline.dbg and e.kline.dbg.annotation=="Subject" and e.kline.dbg.scope==0)
    b2 = en.rationalise([ms])
    print("filtered" if filter_primed else "unfiltered", "| step2 batch:", len(b2), "| grounded Mary:[Subject]:", st.is_grounded(ms.kline))
    print("  ltm Mary sig:", [str(k) for k in st.ltm.get(ms.kline.signature, [])])

run(False); run(True)
