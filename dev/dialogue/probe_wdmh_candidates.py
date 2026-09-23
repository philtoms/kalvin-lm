"""What candidates does WDMH generate? Inspect the post-run state and
compute Def 22 candidate lists for both forms of the question."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pathlib import Path
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import KLine, is_terminal
from kalvin.hop import candidate_goals
from dev.dialogue.harness import load_rationaliser

tok = BPETokenizer()
h = load_rationaliser(Path("data/dialogue/wdmh-underfit.json"), tok)
st = h.state
sig = h.signifier

def name(v):
    return getattr(v, "label", "") or hex(int(v))

def render(k):
    return f"{name(k.signature)}:[{', '.join(name(n) for n in k.nodes)}]"

print("== memory layers ==")
print(f"frame: {sum(len(b) for b in st.frame.values())} klines, "
      f"ltm: {sum(len(b) for b in st.ltm.values())}, "
      f"work_list: {len(st.work_list)}")

wdmh_val = None
for e in st.where(lambda k: True):
    pass
# Find the WDMH ask in the work list and any held kline under WDMH.
ask = next((k for k in st.work_list if getattr(k.signature, "label", "") == "WDMH"
            and not k.nodes), None)
held_wdmh = [k for k in st.where(lambda k: getattr(k.signature, "label", "") == "WDMH")]
print(f"\nask in work list: {render(ask) if ask else None}")
print(f"held klines under WDMH: {[render(k) for k in held_wdmh]}")

# Word identities — are all four question words grounded?
for w in ("what", "did", "Mary", "have"):
    kl = KLine(next((int(n) for k in st.where(lambda k: True)
                     for n in [k.signature] + list(k.nodes)
                     if getattr(n, "label", "") == w), 0), [])
    found = [k for k in st.where(lambda k: getattr(k.signature, "label", "") == w)]
    grounded = any(st.is_grounded(k) for k in found)
    print(f"  word {w!r}: held={[render(k) for k in found]} grounded={grounded}")

# The withheld canon (never fed): what would it be, and is it groundable?
from ks.compiler import compile_source
entries = compile_source(open("data/scripts/wdmh-underfit.ks").read(),
                         tokenizer=tok, signifier=sig, dev=True,
                         word_bits=dict(st.word_bits or {}))
canon = next(e for e in entries if e.kline.dbg and e.kline.dbg.label == "WDMH"
             and e.kline.dbg.op == "CANONICALISES" and e.kline.dbg.scope == 1)
print(f"\nwithheld canon: {render(canon.kline)}")
print(f"  is_grounded: {st.is_grounded(canon.kline)}  "
      f"is_groundable: {st.is_groundable(canon.kline)}  "
      f"in work list: {any(k.signature == canon.kline.signature and k.nodes == canon.kline.nodes for k in st.work_list)}")

# Def 22 candidates for both forms.
memory = st.where(lambda k: not is_terminal(k))
print(f"\n== Def 22 candidates for the ask (empty form) ==")
for g in candidate_goals(memory, ask, sig):
    print(f"  {render(g)}")
print("  (none)" if not candidate_goals(memory, ask, sig) else "")

print("== Def 22 candidates for the canon form (counterfactual A0) ==")
for g in candidate_goals(memory, canon.kline, sig):
    from kalvin.significance import word_atom_count
    a = int(sig.signature_of(canon.kline.nodes))
    b = int(sig.signature_of(g.nodes))
    u = word_atom_count(a | b)
    j = word_atom_count(a & b) / u if u else 1.0
    print(f"  γ={j:.3f}  {render(g)}")
