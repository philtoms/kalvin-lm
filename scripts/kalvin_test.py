import sys
sys.path.insert(0, 'src')

from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler
from kscript.decompiler import Decompiler
from kalvin import Model
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
model = Model(tokenizer)

results = []
model.events.subscribe(lambda e: results.append(e))

for k in klines:
    # print(f'{k.dbg_text}')
    model.rationalise(k)
    for e in results:
        print(f'  {e.kind}: {e.kline.dbg_text}')
    results.clear()

print("Done!")