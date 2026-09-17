"""Value-level trace: hex values in the memory tiers, trawl rounds for the
MHALL goal, and the fate of had:[did,have] after its feed."""
import sys
sys.path.insert(0, "src")

from pathlib import Path
from kalvin import hop as hop_mod
from kalvin.kline import ASK_SIG, is_ask, is_terminal
from kalvin.bpe_tokenizer import BPETokenizer
from dialogue import engine as eng_mod
from dialogue.engine_state import EngineState

def nm(v):
    lab = getattr(v, "label", "")
    return f"{lab}({int(v):#x})" if lab else f"{int(v):#x}"

def render(k):
    m = "?" if is_ask(k.signature) else " "
    return f"{m}{nm(k.signature)}:[{', '.join(nm(n) for n in k.nodes)}]"

# 1. trace the ask-marked kline + had:[did,have] through the engine state
orig_ground = EngineState.ground
orig_add = EngineState.add_work
orig_propose = eng_mod.Engine._propose

def traced_ground(self, kline, store=None):
    r = orig_ground(self, kline, store) if store is not None else orig_ground(self, kline)
    if r and getattr(kline.signature, "label", "") in ("had", "ALL", "MHALL", "did"):
        print(f"    [ground] {render(kline)}")
    return r

def traced_propose(self, kline):
    held = self._state.where(lambda k: not is_terminal(k), True)
    had_mem = [k for k in held if getattr(k.signature, "label", "") == "had"]
    print(f"\n[propose] queued={render(kline)}  had-klines in held: "
          f"{[render(k) for k in had_mem]}  work_list has had: "
          f"{[render(k) for k in self._state.work_list if getattr(k.signature, 'label', '') == 'had']}")
    return orig_propose(self, kline)

EngineState.ground = traced_ground
eng_mod.Engine._propose = traced_propose

# 2. value-level trawl rounds for the MHALL goal hop
def traced_run(self):
    if is_ask(self.queued.signature):
        print(f"\n>> hop on {render(self.queued)}  memory (incl. STM):")
        for k in self.state.where(lambda k: not is_terminal(k), True):
            print(f"     {render(k)}")
        goals = hop_mod.candidate_goals(self.state, self.queued, self.signifier)
        if goals and getattr(goals[0].signature, "label", "") == "MHALL":
            reach = {int(n) for n in self.queued.nodes} | {int(n) for n in goals[0].nodes}
            pending = self.state.where(lambda k: not is_terminal(k), True)
            for rnd in range(1, 5):
                hit = [k for k in pending
                       if int(k.signature) in reach or any(int(n) in reach for n in k.nodes)]
                if not hit:
                    print(f"      round {rnd}: none")
                    break
                print(f"      round {rnd}: {[render(k) for k in hit]}")
                for k in hit:
                    reach.add(int(k.signature))
                    reach.update(int(n) for n in k.nodes)
                ids = {id(k) for k in hit}
                pending = [k for k in pending if id(k) not in ids]
    return hop_mod.Hop.run.__wrapped__(self) if hasattr(hop_mod.Hop.run, "__wrapped__") else None

# don't replace run; do rounds passively via a wrapper that still calls original
orig_run = hop_mod.Hop.run
def passive_run(self):
    traced_run(self)
    return orig_run(self)
hop_mod.Hop.run = passive_run
eng_mod.Hop = hop_mod.Hop

sys.argv = ["harness", "data/scripts/wdmh-underfit.ks", "-p", "data/dialogue/mhall.json"]
from dialogue.harness import main
main()
