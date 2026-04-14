import sys
import threading

sys.path.insert(0, 'src')

from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.compiler import Compiler
from kscript.decompiler import Decompiler
from kalvin.agent import Agent
from kalvin.mod_tokenizer import Mod32Tokenizer

source = '''
(S3 ~> S1)
A > 1 < B
A = B
(A = BC => A B C
 AB > B
 C = 2)

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
decompiler = Decompiler(tokenizer)
agent = Agent(tokenizer, dev=True)
sig = agent.significance

done_event = threading.Event()

agent.events.subscribe(lambda e:
    (print(f'  {e.kind}: {e.query.dbg_text} -> {e.value.dbg_text}, {sig.get_level(e.significance)}')
     if e.kind != 'done' else done_event.set())
)

for k in klines:
    print(f'{k.dbg_text}')
    agent.rationalise(k)

done_event.wait()
agent.cogitate_join()
print("Done!")