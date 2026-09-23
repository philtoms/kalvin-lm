"""The ask kline's exact lifecycle: compile -> feed -> work list -> leave."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import ASK_SIG, canon_key, is_ask
from ks.compiler import compile_source
from dev.dialogue.harness import make_rationaliser

tok = BPETokenizer()
h = make_rationaliser(tok)
sig = h.signifier
bits: dict[str, int] = {}

entries = compile_source(Path("data/scripts/mhall.ks").read_text(),
                         tokenizer=tok, signifier=sig, dev=True, word_bits=bits)
ask = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK")
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
             and e.kline.dbg.label == "MHALL")

a, c = ask.kline, canon.kline
print(f"ask  entry: sig={int(a.signature):#x}  ASK-set={bool(int(a.signature) & ASK_SIG)}"
      f"  nodes={[hex(int(n)) for n in a.nodes]}  byte=0x{ask.significance:02x}")
print(f"canon twin: sig={int(c.signature):#x}  ASK-set={bool(int(c.signature) & ASK_SIG)}"
      f"  nodes={[hex(int(n)) for n in c.nodes]}  byte=0x{canon.significance:02x}")
print(f"canon_key(ask sig) == twin sig: {canon_key(a.signature) == c.signature}")
print(f"same nodes: {[int(n) for n in a.nodes] == [int(n) for n in c.nodes]}")

h.run(Path("data/scripts/mhall.ks").read_text())
st = h.state
print(f"\nafter run:")
print(f"  ask-marked kline grounded anywhere: "
      f"{any(k.signature == a.signature for b in list(st.frame.values()) + list(st.ltm.values()) for k in b)}")
print(f"  canon twin ({int(c.signature):#x}) held: "
      f"{any(k.signature == c.signature and [int(n) for n in k.nodes] == [int(n) for n in c.nodes] for b in list(st.frame.values()) + list(st.ltm.values()) for k in b)}")
print(f"  ask still in work list: {any(k.signature == a.signature for k in st.work_list)}")
held = [k for b in st.frame.values() for k in b if k.signature == canon_key(a.signature)]
print(f"  held at canon key {canon_key(a.signature):#x}: "
      f"{[(hex(int(k.signature)), [getattr(n, 'label', '') for n in k.nodes]) for k in held]}")
