from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
tok = NLPTokenizer(); sigf = NLPSignifier()
entries = compile_source(open("data/scripts/wdmh-underfit.ks").read(), tokenizer=tok, signifier=sigf, dev=True)
for e in entries:
     print("   ", e.kline.signature.label, [n.label for n in e.kline.nodes], e.kline.dbg.op if e.kline.dbg else "?", "ann:", repr(e.kline.dbg.annotation if e.kline.dbg else None))
