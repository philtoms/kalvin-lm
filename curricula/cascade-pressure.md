## Objective
Stress-test intra-lesson reactive cascade by creating high candidate density. Many identity atoms are established first, then compound entries that share signature patterns create overlapping candidate pools. The final lessons submit entries where the query signature matches many candidates simultaneously.

## Approach
1. Build up a dense model with many identities and relationships
2. Submit compound misfits whose compound signature signifies the established pair klines
3. The goal is to observe how the reactor and cogitator handle high candidate counts within a single lesson

The misfit lessons use the `>` CONNOTED operator (not `=>` CANONIZED): a CANONIZED
compound definition sets its signature to `make_signature(nodes)` by construction
(@kscript §11.4), so such a kline is canonical, never a misfit, and resolves S1 via
the self-grounded short-circuit (@agent AGT-14) before any candidate is retrieved
(rule 47, @curriculum). With `>`, the signature stays the intended compound while the
nodes are disjoint identities, so each kline is a genuine misfit whose signature
differs from `make_signature(nodes)`. Under the settled compiler contracts a genuine
(non-canonical) misfit kline is always single-node, so a multi-node intent is
expressed as several single-node misfits — one per node — each of which retrieves
the established klines because `model.where(sig)` is bit-overlap based.

## Lessons

### 1
Introduce ten identity atoms to create a dense signature space.

```
A(lpha)
B(eta)
C(harlie)
D(elta)
E(cho)
F(oxtrot)
G(olf)
H(otel)
I(ndia)
J(uliet)
```

### 2
Establish bidirectional countersigns between pairs. This populates the kline space with structural relationships.

```
A(lpha) == B(eta)
C(harlie) == D(elta)
E(cho) == F(oxtrot)
G(olf) == H(otel)
I(ndia) == J(uliet)
```

### 3
Submit cross-group connoted misfits that bridge the pairs. Each entry is a
single-node overfit misfit whose compound signature (e.g. `AB` = Alpha|Beta)
signifies the established pair countersigns, while its single node (e.g. Charlie)
is disjoint from both — so it routes S2/S3 and produces expansion proposals.

```
(Alpha Beta Charlie)
AB > C
(Charlie Delta Echo)
CD > E
(Echo Foxtrot Golf)
EF > G
(Golf Hotel India)
GH > I
```

### 4
Submit compound misfits that combine multiple groups. Each compound signature
(e.g. `ABCD` = Alpha|Beta|Charlie|Delta) signifies many established klines at
once, so each single-node misfit retrieves a large candidate pool. The final
entry (`ACEGI`, every other atom) is the densest — its signature overlaps every
pair group.

```
(Alpha Beta Charlie Delta Echo Foxtrot)
ABCD > E F
(Echo Foxtrot Golf Hotel India Juliet)
EFGH > I J
(Alpha Charlie Echo Golf India Beta Delta Foxtrot Hotel Juliet)
ACEGI > B D F H J
```

### 5
The pressure test: submit misfits whose signature spans all ten atoms. The
compound `ABCDEFGHIJ` signifies every established identity and countersign, so
each single-node misfit creates maximum candidate overlap. The connoted nodes
span four of the five groups (Alpha, Charlie, Echo, Golf); this is deliberately
fewer than lesson 4's node set because the overlap comes from the ten-atom
signature, not the node count, and a full ten-node decomposition would generate
a pathological number of expansion events.

```
(Alpha Beta Charlie Delta Echo Foxtrot Golf Hotel India Juliet)
ABCDEFGHIJ > A C E G
```
