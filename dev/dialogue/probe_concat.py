import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
from pathlib import Path
from dev.dialogue.harness import make_engine
from kalvin.bpe_tokenizer import BPETokenizer
from ks.lexer import Lexer
from ks.parser import Parser
from ks.ast_emitter import ASTEmitter
import ks.ast_emitter as ae

tok = BPETokenizer()
h = make_engine(tok)
source = Path("data/scripts/wdmh-underfit.ks").read_text()
tree = Parser(Lexer(source).tokenize()).parse()
emitted = ASTEmitter().emit(tree)
for e in emitted:
    print(e.op, repr(e.sig), e.nodes)
