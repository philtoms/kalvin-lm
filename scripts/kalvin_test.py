import sys
sys.path.insert(0, 'src')

from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler
from kscript.decompiler import Decompiler
from kalvin.model import Model
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
ast = Parser(tokens).parse()
tokenizer = Mod32Tokenizer()
klines = Compiler(tokenizer, dev=True).compile(ast)
# decompiler = Decompiler(tokenizer)
model = Model(tokenizer, dev=True)

results = []
model.events.subscribe(lambda e: results.append(e))

for k in klines:
    print(f'{k.dbg_text}')
    model.rationalise(k)
    for e in results:
        print(f'  {e.kind}: {e.kline.dbg_text}')
    results.clear()

print("Done!")