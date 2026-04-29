import sys
import threading

sys.path.insert(0, 'src')

from kscript.compiler import compile_source
from kalvin.agent import Agent
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.significance import D_MAX

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


def significance_level(sig: int) -> str:
    """Map agent event significance to a display level.

    The agent emits:
      - D_MAX - 1 for grounded / all-literal (S1)
      - 0 for novel (S4)
      - pipeline significance for assessed results (S1–S3)
    """
    if sig == D_MAX - 1:
        return "S1"
    if sig == 0:
        return "S4"
    return f"0x{sig:016x}"


tokenizer = Mod32Tokenizer()
klines = compile_source(source, tokenizer, dev=True)
agent = Agent(tokenizer)

done_event = threading.Event()


def on_event(e):
    if e.kind == 'done':
        done_event.set()
    else:
        level = significance_level(e.significance)
        print(f'  {e.kind}: {e.query.dbg_text} -> {e.value.dbg_text}, {level}')


agent.events.subscribe(on_event)

for k in klines:
    print(f'{k.dbg_text}')
    agent.rationalise(k)

done_event.wait()
agent.cogitate_join()
print("Done!")
