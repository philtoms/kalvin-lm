"""Probe why Cogitator.expand yields nothing for WDMH=>MHALL.

Loads mhall state, runs wdmh-underfit.ks with Cogitator.cogitate and
.expand instrumented to log entries, queries, candidates, and expansions.
"""
import sys
sys.path.insert(0, "src")
from pathlib import Path
from dialogue.harness import load_engine
from kalvin.bpe_tokenizer import BPETokenizer
from dialogue.cogitator import Cogitator
from kalvin.kline import is_misfit

orig_cogitate = Cogitator.cogitate
orig_expand = Cogitator.expand

def logged_expand(self, underfit, overfit, fit):
    prop, dist = orig_expand(self, underfit, overfit, fit)
    print(f"  [expand] u={underfit} o={overfit} f={fit} -> {prop} d={dist}")
    return prop, dist

def logged_cogitate(self, entry):
    queries = [entry] if self.signifier.is_ask(entry.signature) else self.state.find_canons(entry.signature)
    cands = self._candidates(entry)
    print(f"[cogitate] entry={entry.signature}:{entry.nodes} is_ask={self.signifier.is_ask(entry.signature)} "
          f"queries={[q.signature for q in queries]} candidates={[c.signature for c in cands]}")
    return orig_cogitate(self, entry)

Cogitator.expand = logged_expand
Cogitator.cogitate = logged_cogitate

tok = BPETokenizer()
h = load_engine(Path("data/dialogue/mhall.json"), tok)
h.run(Path("data/scripts/wdmh-underfit.ks").read_text())
