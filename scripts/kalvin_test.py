import sys
sys.path.insert(0, 'src')

from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler
from kalvin import Kalvin
from kalvin.mod_tokenizer import Mod32Tokenizer

source = '''
MHALL == SVO =>
  S < M
  V < H
  O < ALL =>
    A > D
    L > M
    L > O < BS =>
      B = "baby"
      S = "sheep"
'''

tokens = Lexer(source).tokenize()
kast = Parser(tokens).parse()
compiler = Compiler(Mod32Tokenizer(), dev=True)
klines = compiler.compile(kast)
kalvin = Kalvin()
for k in klines:
    print(f'{k.dbg_text}')
    kalvin.rationalise(k)
