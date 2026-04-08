import sys
sys.path.insert(0, 'src')

from kscript.lexer import Lexer
from kscript.parser import Block, Parser
from kscript.compiler import Compiler

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

print('\n=== Compiler Output ===\n')
from kalvin.mod_tokenizer import Mod32Tokenizer
compiler = Compiler(Mod32Tokenizer(), dev=True)
entries = compiler.compile(kfile)
for e in entries:
    print(f'{e.dbg_text}')
