import sys
sys.path.insert(0, 'src')

from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler
from kscript.decompiler import Decompiler
from kalvin import Kalvin
from kalvin.mod_tokenizer import Mod32Tokenizer

source = '''
MHALL = SVO =>
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
tokenizer = Mod32Tokenizer()
compiler = Compiler(tokenizer, dev=True)
decompiler = Decompiler(tokenizer)
klines = compiler.compile(kast)
kalvin = Kalvin(tokenizer)
for k in klines:
    print(f'{k.dbg_text}')
    rks = kalvin.rationalise(k)
    # for rk in decompiler.decompile(rks):
    #     print(rk.to_kscript())

print("Done!")