"""Trace every K batch + grounding across the full decoded MHALL script.

Drives the real decoded script through the RationalisingTrainee the same way
the runner does, but prints each K batch and each S1 grounding observation
so we can see exactly when identity asks / grounds happen.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SYS_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from training.dialogue import decode, load_script
from training.dialogue.actors import RationalisingTrainee

_BAND = {SIG_S1: "S1", SIG_S2: "S2", SIG_S3: "S3", SIG_S4: "S4"}


def _labels(script, tok, sigf):
    entries = compile_source(script.source, tokenizer=tok, signifier=sigf, dev=True)
    out = {}
    for e in entries:
        d = e.kline.dbg
        if d and (d.label or d.decoded):
            out.setdefault(e.kline.signature, d.label or d.decoded)
    return out


class _Sink(list):
    """A minimal event sink that just records emitted events."""

    def publish(self, event):
        self.append(event)


def main() -> None:
    tok = NLPTokenizer()
    sigf = NLPSignifier()
    script = load_script(__import__("json").loads(
        Path("scripts/dialogue-mhall.json").read_text()))
    decoded = decode(script, tokenizer=tok, signifier=sigf)
    labels = _labels(script, tok, sigf)

    sink = _Sink()
    trainee = RationalisingTrainee(sigf, sink=sink)

    def lab(s):
        return labels.get(s, s)

    def show(tag, kvalues):
        for v in kvalues:
            d = v.kline.dbg
            op = d.op if d else "?"
            print(f"    {tag:14s} {op:12s} {lab(v.kline.signature)!s:18s} "
                  f"nodes={[lab(n) for n in v.kline.nodes]}  band={_BAND.get(v.significance, v.significance)}")

    # The decoded list interleaves T and K turns. We feed each T turn's
    # proposal into the trainee and read out its batch + observations, in order.
    t_turns = [t for t in decoded if t.role == "T" and t.value is not None]
    print(f"{len(t_turns)} T turns to feed.\n")
    for i, turn in enumerate(t_turns, 1):
        incoming = turn.value
        d = incoming.kline.dbg
        op = d.op if d else "?"
        print(f"--- T#{i}: {op:12s} {lab(incoming.kline.signature)!s:18s} "
              f"nodes={[lab(n) for n in incoming.kline.nodes]}  "
              f"scripted={_BAND.get(incoming.significance,'?')}")
        # next_events takes a list of incoming events; wrap the KValue.
        from kalvin.events import RationaliseEvent
        print(f"    [work-list before: {len(trainee._state.work_list)} entries]")
        for e in trainee._state.work_list:
            print(f"      - {lab(e.signature)!s:16s} nodes={[lab(n) for n in e.nodes]}")
        grounded_sigs = list(trainee._state.grounded.keys())
        print(f"    [grounded sigs: {[lab(s) for s in grounded_sigs]}]")
        events = list(trainee.next_events([RationaliseEvent(
            kind="frame", query=incoming, proposal=incoming, role="T")]))
        show("K emit", [e.proposal for e in events])
        obs = trainee.drain_observations()
        show("K ground", obs)
        print()

    # Dump final grounded state to confirm the MHALL<->SVO countersignature.
    print("=== final grounded (signature -> node-sets) ===")
    g = trainee._state.grounded
    for s, klines in sorted(g.items()):
        lab_s = lab(s)
        for kl in klines:
            print(f"    {lab_s:18s} -> {[lab(n) for n in kl.nodes]}")


if __name__ == "__main__":
    main()
