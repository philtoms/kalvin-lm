"""Probe the Rationaliser turn-by-turn on the MHALL opening.

Feeds the first T queries by hand and prints K's batch + observations,
so we can verify the expected behaviour:
  - After `MHALL COUNTERSIGNS SVO` (S2 proposal), K emits two identity
    asks (MHALL, SVO).
  - After `MHALL CANONIZES [Mary, had, a, little, lamb]` (an S2-stamped
    canon), the rationaliser ignores the subjective S2 stamp, treats it
    as the canon it structurally is, and the next batch is a run of
    identity asks for the unrecognised canon nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SYS_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4, structural_significance
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from training.dialogue.rationalise import Rationaliser, RationaliserState

_BAND = {SIG_S1: "S1", SIG_S2: "S2", SIG_S3: "S3", SIG_S4: "S4"}


def _label_map(entries):
    out = {}
    for e in entries:
        d = e.kline.dbg
        if d and (d.label or d.decoded):
            out.setdefault(e.kline.signature, d.label or d.decoded)
    return out


def _key(kline):
    return (kline.signature, tuple(kline.nodes))


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
        d = v.kline.dbg
        op = d.op if d else "?"
        lab = labels.get(v.kline.signature, v.kline.signature)
        nodes = [labels.get(n, n) for n in v.kline.nodes]
        print(f"  {op:12s} {lab!s:20s} nodes={nodes}  band={_BAND.get(v.significance, v.significance)}")


def _show_obs(obs, labels):
    if not obs:
        return
    print("  observations:")
    for v in obs:
        d = v.kline.dbg
        op = d.op if d else "?"
        lab = labels.get(v.kline.signature, v.kline.signature)
        nodes = [labels.get(n, n) for n in v.kline.nodes]
        print(f"    S1 ground: {op:12s} {lab!s:20s} nodes={nodes}")


def main() -> None:
    tok = NLPTokenizer()
    sigf = NLPSignifier()
    entries = compile_source(
        "(Mary had a little lamb)\nMHALL == SVO",
        tokenizer=tok, signifier=sigf, dev=True,
    )
    labels = _label_map(entries)

    engine = Rationaliser(sigf)
    state = RationaliserState()

    def step(tag, query):
        print(f"\n----- T query: {tag} -----")
        d = query.kline.dbg
        op = d.op if d else "?"
        lab = labels.get(query.kline.signature, query.kline.signature)
        nodes = [labels.get(n, n) for n in query.kline.nodes]
        struct = structural_significance(query.kline, sigf)
        print(f"  incoming: {op:12s} {lab!s:20s} nodes={nodes}  "
              f"declared={_BAND.get(query.significance,'?')}  "
              f"structural={_BAND.get(struct,'?')}")
        batch, obs = engine.rationalise(state, [query])
        _show_batch("K batch", batch, labels)
        _show_obs(obs, labels)
        print(f"  work_list depth: {len(state.work_list)}")
        for e in state.work_list:
            ed = e.dbg
            el = labels.get(e.signature, e.signature)
            print(f"    - {ed.op if ed else '?':10s} {el!s:16s} nodes={[labels.get(n,n) for n in e.nodes]}")
        if state.frame:
            print(f"  frame depth: {sum(len(v) for v in state.frame.values())}")
            for sig, bucket in state.frame.items():
                fl = labels.get(sig, sig)
                for kl in bucket:
                    print(f"    - {fl!s:16s} nodes={[labels.get(n,n) for n in kl.nodes]}")
        return batch

    # Drive the full T-query sequence from the script (T turns only).
    # Each T query in declared order; observe K's batch + grounding.
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
