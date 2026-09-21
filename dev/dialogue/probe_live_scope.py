"""Instrument the live wdmh run: print each hop's goal and its Def 23 scope."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path
from kalvin import hop as hop_mod
from kalvin.kline import is_ask, is_terminal

orig_run = hop_mod.Hop.run
orig_trawl = hop_mod.trawl

def render(k):
    lab = getattr(k.signature, "label", "") or hex(int(k.signature))
    m = "?" if is_ask(k.signature) else " "
    return f"{m}{lab}:[{', '.join(getattr(n, 'label', '') or hex(int(n)) for n in k.nodes)}]"

def traced_trawl(state, a_nodes, b_nodes, *, max_depth=4):
    scope = orig_trawl(state, a_nodes, b_nodes, max_depth=max_depth)
    print(f"      trawl(a={[getattr(n, 'label', n) for n in a_nodes]}, "
          f"b={[getattr(n, 'label', n) for n in b_nodes]}) -> {len(scope)}:")
    for k in scope:
        print(f"        {render(k)}")
    return scope

def traced_run(self):
    goals = hop_mod.candidate_goals(self.state, self.queued, self.signifier)
    print(f"\n>> hop on {render(self.queued)}  goals[:4]={[render(g) for g in goals[:4]]}")
    if is_ask(self.queued.signature):
        mem = self.state.where(lambda k: not is_terminal(k), True)
        print(f"   memory ({len(self.state.stm)} stm writes):")
        for k in mem:
            print(f"     {render(k)}")
    return orig_run(self)

hop_mod.trawl = traced_trawl
hop_mod.Hop.run = traced_run

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin import engine as eng_mod

eng_mod.Hop = hop_mod.Hop  # engine imported Hop by name

sys.argv = ["harness", "data/scripts/wdmh-underfit.ks", "-p", "data/dialogue/mhall.json"]
from dev.dialogue.harness import main
main()
