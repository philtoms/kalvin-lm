import sys
sys.path.insert(0, 'src')

from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler
from kscript.decompiler import Decompiler
from kalvin.agent import Agent
from kalvin.mod_tokenizer import Mod32Tokenizer

source = '''
AB = C => A B
(MHALL = SVO =>
  S < M
  V < H
  O < ALL =>
    A > D
    L > M
    L > O < BS =>
      B = "baby"
      S = "sheep")
'''

tokens = Lexer(source).tokenize()
ast = Parser(tokens).parse()
tokenizer = Mod32Tokenizer()
klines = Compiler(tokenizer, dev=True).compile(ast)
# decompiler = Decompiler(tokenizer)
agent = Agent(tokenizer, dev=True)

results = []
agent.events.subscribe(lambda e: results.append(e))

for k in klines:
    print(f'{k.dbg_text}')
    agent.rationalise(k)
    for e in results:
        print(f'  {e.kind}: {e.query.dbg_text} -> {e.value.dbg_text}, {e.significance:#018x}')
    results.clear()

print("Done!")