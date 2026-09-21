"""Trace denotate/expand for the WDMH<->MHALL candidate pair specifically."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from pathlib import Path
from dev.dialogue.harness import load_rationaliser
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.work_runner import WorkRunner
from kalvin.kline import is_terminal, is_identity

tok = BPETokenizer()
h = load_rationaliser(Path("data/dialogue/mhall.json"), tok)
h.run(Path("data/scripts/wdmh-underfit.ks").read_text())
state = h.state
cog = WorkRunner(state)

def by_label(label):
    return [k for k in state.where(lambda k: getattr(k.signature, "label", None) == label)]

print("== klines under key labels ==")
for label in ("WDMH", "MHALL", "DH", "hadDH", "SVO"):
    for k in by_label(label):
        print(f"  {label}: {[str(n) for n in k.nodes]} terminal={is_terminal(k)} identity={is_identity(k)}")

wdmh_k = [k for k in by_label("WDMH") if k.nodes and not is_terminal(k)]
for w in wdmh_k:
    print("WDMH non-terminal kline:", [str(n) for n in w.nodes])
    print("  denotate:")
    for kl, hops in cog.denotate(w.signature):
        print(f"    hops={hops} sig={kl.signature} nodes={[str(n) for n in kl.nodes]}")
    for m in by_label("MHALL"):
        q_set, c_set = set(w.nodes), set(m.nodes)
        u, o, f = list(q_set-c_set), list(c_set-q_set), list(q_set & c_set)
        print(f"  vs MHALL{[str(n) for n in m.nodes]}: u={[str(x) for x in u]} o={[str(x) for x in o]} f={[str(x) for x in f]}")
        prop, dist = cog.expand(u, o, f)
        print(f"    proposal={[str(p) for p in prop]} dist={dist}")
