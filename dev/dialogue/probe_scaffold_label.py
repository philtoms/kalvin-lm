import sys
sys.path.insert(0, 'src')
from pathlib import Path
from dialogue.harness import make_engine
from kalvin.bpe_tokenizer import BPETokenizer
from ks.compiler import Compiler
from ks.lexer import Lexer
from ks.parser import Parser

tok = BPETokenizer()
h = make_engine(tok)
source = Path("data/scripts/wdmh-underfit.ks").read_text()
c = Compiler(tok, signifier=h.signifier, dev=True, word_bits=h.word_bits)
entries = c.compile(Parser(Lexer(source).tokenize()).parse())
for e in entries:
    sig_lbl = getattr(e.kline.signature, "label", "")
    print(f"sig={e.kline.dbg.label!r:>10} signode={sig_lbl!r:>10} nodes={[getattr(n,'label',None) for n in e.kline.nodes]} op={e.kline.dbg.op}")
