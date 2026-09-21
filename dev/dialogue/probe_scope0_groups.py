# Verify scope-0 delineation: isolated annotation shorthand, multi-root scripts.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kalvin.bpe_tokenizer import BPETokenizer
from dev.dialogue.harness import make_rationaliser

CASES = {
    "isolated annotation (shorthand for (ann)compound)":
        "(mary had lamb)\n",
    "two root constructs":
        "(mary had lamb)\nM == L =>\n  a < b\n(hello world)\n",
    "nested only":
        "A == B =>\n  C < D =>\n    E = F\n",
}

for name, src in CASES.items():
    h = make_rationaliser(BPETokenizer())
    results = h.run(src)
    print(f"── {name}: {len(results)} step(s)")
    for r in results:
        d = r.entry.kline.dbg
        print(f"   opener scope={d.scope} op={d.op:<8} {d.label}:{[n.label for n in r.entry.kline.nodes]}")
