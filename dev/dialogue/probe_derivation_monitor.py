"""Monitor the ask's derivations: memory prior, per-goal endings + traces."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path
from kalvin import hop as hop_mod
from kalvin.kline import is_ask, is_terminal
from kalvin.derivation import Derivation, DerivationResult

def nm(v):
    return getattr(v, "label", "") or hex(int(v))

def render(k):
    m = "?" if is_ask(k.signature) else " "
    return f"{m}{nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}]"

def render_state(nodes):
    return f"[{', '.join(nm(n) for n in nodes)}]"

orig_trawl = hop_mod.trawl
orig_derivation_run = Derivation.run

def monitored_run(self):
    r = orig_derivation_run(self)
    q = self.queued if hasattr(self, "queued") else None
    print(f"      derivation ending={r.ending:9s} j0={r.j0:.3f} j1={r.j1:.3f} "
          f"γ={r.gamma:.3f} steps={len(r.trace)}")
    for t in r.trace:
        print(f"        {render_state(t)}")
    if r.composed:
        print(f"      composed: {[render(k) for k in r.composed]}")
    return r

Derivation.run = monitored_run

def monitored_hop_run(self):
    if is_ask(self.queued.signature):
        print(f"\n>> hop on {render(self.queued)}")
        print("   memory prior to derivation:")
        for k in self.state.where(lambda k: not is_terminal(k), True):
            print(f"     {render(k)}")
        goals = hop_mod.candidate_goals(self.state, self.queued, self.signifier)
        print(f"   goal list: {[render(g) for g in goals[:5]]}")
    return monitored_hop_run.__wrapped__(self) if False else orig_hop_run(self)

orig_hop_run = hop_mod.Hop.run
hop_mod.Hop.run = monitored_hop_run

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin import rationaliser as rat_mod
rat_mod.Hop = hop_mod.Hop

sys.argv = ["harness", "data/scripts/wdmh.ks", "-p", "data/dialogue/mhall.json"]
from dev.dialogue.harness import main
main()
