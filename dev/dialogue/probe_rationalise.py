"""Probe the engine turn-by-turn on the MHALL opening.

Feeds the first T queries by hand through the dialogue harness's engine and
prints K's batch + observations, so we can verify the expected behaviour:
  - After `MHALL COUNTERSIGNS SVO` (S2 proposal), K emits identity asks
    for the unrecognised signatures.
  - After `MHALL CANONIZES [Mary, had, a, little, lamb]` (an S2-stamped
    canon), the engine ignores the subjective S2 stamp, treats it as the
    canon it structurally is, and the next batch is a run of identity
    asks for the unrecognised canon nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SYS_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

from dialogue.harness import make_engine  # noqa: E402
from kalvin.kline import sig_level  # noqa: E402
from kalvin.nlp_tokenizer import NLPTokenizer  # noqa: E402
from kalvin.significance import BandLayout  # noqa: E402
from ks.compiler import compile_source  # noqa: E402

_LAYOUT = BandLayout()
_BAND_ORDER = ("S1", "S2", "S3", "S4")


def _label_map(entries):
    out = {}
    for e in entries:
        d = e.kline.dbg
        if d and (d.label or d.decoded):
            out.setdefault(e.kline.signature, d.label or d.decoded)
    return out


def _find(entries, *, op, label):
    for e in entries:
        d = e.kline.dbg
        if d and d.op == op and (d.label == label or d.decoded == label):
            return e
    raise LookupError(f"no entry op={op!r} label={label!r}")


def _show_batch(tag, batch, labels):
    print(f"\n=== {tag} ===")
    if not batch:
        print("  (empty batch)")
    for v in batch:
        lab = labels.get(v.kline.signature, v.kline.signature.label or v.kline.signature)
        nodes = [labels.get(n, n.label or n) for n in v.kline.nodes]
        band = _LAYOUT.classify(v.significance)
        kind = "asks" if not v.kline.nodes else "proposes"
        print(f"  {kind:<8s} {lab!s:20s} nodes={nodes}  band={band} ({v.significance & 0xFF})")


def _show_obs(obs, labels):
    if not obs:
        return
    print("  observations:")
    for v in obs:
        lab = labels.get(v.kline.signature,
                   v.kline.signature.label or v.kline.signature)
        nodes = [labels.get(n, n.label or n) for n in v.kline.nodes]
        print(f"    S1 ground: {lab!s:20s} nodes={nodes}")


def main() -> None:
    tok = NLPTokenizer()
    harness = make_engine(tok)
    entries = compile_source(
        "(Mary had a little lamb)MHALL == SVO",
        tokenizer=tok, signifier=harness.signifier, dev=True,
    )
    labels = _label_map(entries)
    engine, state = harness.engine, harness.state

    def step(tag, query):
        print(f"\n----- T query: {tag} -----")
        d = query.kline.dbg
        op = d.op if d else "?"
        lab = labels.get(query.kline.signature, query.kline.signature)
        nodes = [labels.get(n, n.label or n) for n in query.kline.nodes]
        struct = sig_level(query.kline, harness.signifier)
        print(f"  incoming: {op:12s} {lab!s:20s} nodes={nodes}  "
              f"declared={_LAYOUT.classify(query.significance)}  "
              f"structural={struct}")
        batch, obs = engine.rationalise([query])
        _show_batch("K batch", batch, labels)
        _show_obs(obs, labels)
        print(f"  stm depth: {len(state.stm)}")
        for e in state.stm:
            el = labels.get(e.signature, e.signature.label or e.signature)
            print(f"    - {el!s:16s} nodes={[labels.get(n, n.label or n) for n in e.nodes]}")
        if state.ltm:
            print(f"  ltm depth: {sum(len(v) for v in state.ltm.values())}")
            for sig, bucket in state.ltm.items():
                fl = labels.get(sig, sig.label or sig)
                for kl in bucket:
                    knodes = [labels.get(n, n.label or n) for n in kl.nodes]
                    print(f"    - {fl!s:16s} nodes={knodes}")
        return batch

    # Drive the full T-query sequence from the script (T turns only).
    t_sequence = [
        ("MHALL COUNTERSIGNS SVO",         "COUNTERSIGNS", "MHALL"),
        ("MHALL CANONIZES [M had a little lamb]", "CANONIZES",  "MHALL"),
        ("Mary IDENTITY [M ary]",          "IDENTITY",    "Mary"),
        ("had IDENTITY [h ad]",            "IDENTITY",    "had"),
        ("a CONNOTES [Det]",               "CONNOTES",    "a"),
        ("Det IDENTITY [D et]",            "IDENTITY",    "Det"),
        ("little IDENTITY [l ittle]",      "IDENTITY",    "little"),
        ("lamb IDENTITY [l amb]",          "IDENTITY",    "lamb"),
        ("SVO CANONIZES [Subject Verb Object]", "CANONIZES", "SVO"),
        ("Subject IDENTITY [Sub ject]",    "IDENTITY",    "Subject"),
        ("Verb IDENTITY [V er b]",         "IDENTITY",    "Verb"),
        ("Object IDENTITY [Ob ject]",      "IDENTITY",    "Object"),
    ]
    for tag, op, label in t_sequence:
        try:
            q = _find(entries, op=op, label=label)
        except LookupError:
            print(f"\n----- (skip {tag}: no compiled entry) -----")
            continue
        step(tag, q)


if __name__ == "__main__":
    main()
