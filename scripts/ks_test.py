import importlib, sys

# Prevent kscript __init__ from loading (it imports compiler which imports old parser types)
# We'll import lexer and parser directly
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

load_module('kscript.token', 'src/kscript/token.py')
load_module('kscript.lexer', 'src/kscript/lexer.py')
load_module('kscript.parser', 'src/kscript/parser.py')

from kscript.lexer import Lexer
from kscript.parser import Block, Parser

# Load compiler_v2 directly
spec = importlib.util.spec_from_file_location('kscript.compiler_v2', 'src/kscript/compiler_v2.py')
compiler_v2 = importlib.util.module_from_spec(spec)
sys.modules['kscript.compiler_v2'] = compiler_v2
spec.loader.exec_module(compiler_v2)

source = '''
A =>
  B
  C <= D
  E
(Mary had a little lamb)
MHALL == SVO =>
  S(ubject) = M
  V = H
  O = ALL =>
    A = D
    L = M < MOD => A B
    L > O < BS =>
      B = "baby"
      S = "sheep"
  EXT =>
    E
    X
    T <= EXTRA => E X T R A > O
'''

tokens = Lexer(source).tokenize()
print('=== Tokens ===')
for t in tokens:
    print(f'  {t.type.name:16s} {t.value!r}')
print()

kfile = Parser(tokens).parse()

def show(node, indent=0):
    pad = '  ' * indent
    if hasattr(node, 'scripts'):
        print(f'{pad}KScriptFile:')
        for s in node.scripts:
            show(s, indent+1)
    elif hasattr(node, 'constructs'):
        print(f'{pad}{type(node).__name__}:')
        for c in node.constructs:
            show(c, indent+1)
    elif hasattr(node, 'inner'):
        print(f'{pad}Construct:')
        if isinstance(node.inner, Block):
            print(f'{pad}  Block:')
            for c in node.inner.constructs:
                show(c, indent+2)
        else:  # list[PrimaryConstruct]
            for pc in node.inner:
                show(pc, indent+1)
        if node.chain_op:
            print(f'{pad}  chain_op: {node.chain_op}')
            show(node.chain_right, indent+2)
    elif hasattr(node, 'sig'):
        parts = [f'sig={node.sig.id}']
        if node.op:
            nid = node.node.id if hasattr(node.node, 'id') else node.node
            parts.append(f'op={str(node.op).replace("TokenType.", "")} node={nid}')
        print(f'{pad}PrimaryConstruct({" ".join(parts)})')
    elif hasattr(node, 'id'):
        print(f'{pad}{type(node).__name__}({node.id})')

show(kfile)

print('\n=== Compiler V2 Output ===\n')
compiler = compiler_v2.CompilerV2(dev=True)
entries = compiler.compile(kfile)
for e in entries:
    print(f'{e.op:14s} {e.dbg}')
