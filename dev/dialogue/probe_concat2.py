import sys
sys.path.insert(0, 'src')
from pathlib import Path
from dialogue.harness import make_engine
from kalvin.bpe_tokenizer import BPETokenizer
from ks.compiler import Compiler
from ks.lexer import Lexer
from ks.parser import Parser
import ks.token_encoder as te

orig = te.TokenEncoder._encode_entries_for_entry
def patched(self, entry):
    out = orig(self, entry)
    if entry.sig == 'DHhad':
        kv = out[0][0]
        print("CONCAT sig repr:", repr(kv.kline.signature), "nodes:", kv.kline.nodes)
    return out
te.TokenEncoder._encode_entries_for_entry = patched

tok = BPETokenizer()
h = make_engine(tok)
source = Path("data/scripts/wdmh-underfit.ks").read_text()
c = Compiler(tok, signifier=h.signifier, dev=True, word_bits=h.word_bits)
c.compile(Parser(Lexer(source).tokenize()).parse())

orig_so = te.NLPSignifier.signature_of
def so(self, nodes):
    out = orig_so(self, nodes)
    print("signature_of labels:", [getattr(n,'label','') for n in nodes], "->", out)
    return out
te.NLPSignifier.signature_of = so
c2 = Compiler(tok, signifier=h.signifier, dev=True, word_bits=h.word_bits)
c2.compile(Parser(Lexer(source).tokenize()).parse())
