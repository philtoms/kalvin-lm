"""T2 bounds in the derivation: the canonicalisation survey is
witness-driven (not subset enumeration over a grown node list), and the
slot walk enforces its expanded-state bound.

The subset enumeration the survey replaced was exponential: a node list
grown to ~40 by expansions made one canonicalise step enumerate ~10^29
subsets — an effective hang inside Hop.run. Witness-driven matching is
the same relation (a group contracts iff a held canon counter-witnesses
exactly it) at polynomial cost.
"""

from __future__ import annotations

from kalvin.derivation import Derivation
from kalvin.kline import KLine, KNode
from kalvin.signifier import NLPSignifier

SIG = NLPSignifier()


def bit(n: int) -> KNode:
    return KNode(1 << (32 + n))


def test_canonicalisation_survey_survives_a_grown_node_list():
    # 45 nodes, a canon whose witness sits among them: the survey must
    # yield promptly (subset enumeration would hang at this size).
    pad = [bit(i % 20) for i in range(42)]
    x, y = bit(21), bit(22)
    canon = KLine(int(x) | int(y), [x, y])
    d = Derivation([canon], KLine(0, pad + [x, y]), KLine(0, [bit(30)]), SIG)
    found = list(d.canonicalisations())
    assert found and found[0][0] is canon
    new = found[0][2]
    assert len(new) == 43 and new[42] == int(canon.signature)


def test_canonicalisation_places_each_disjoint_occurrence():
    a, b = bit(0), bit(1)
    canon = KLine(int(a) | int(b), [a, b])
    d = Derivation([canon], KLine(0, [a, b, a, b]), KLine(0, [bit(9)]), SIG)
    groups = [g for _, g, _ in d.canonicalisations()]
    assert (int(a), int(b)) in groups and (int(b), int(a)) in groups


def test_slot_walk_enforces_the_state_bound():
    # a lattice of interchangeable denotations: the BFS would expand
    # unboundedly without the T2 state bound.
    mem = [KLine(bit(i), [bit(i + 1)]) for i in range(40)]
    d = Derivation(mem, KLine(0, [bit(0)]), KLine(0, [bit(60)]), SIG,
                   max_walk_states=8)
    assert d.slot_walk(bit(0), end_mask=1 << (32 + 60)) is None
