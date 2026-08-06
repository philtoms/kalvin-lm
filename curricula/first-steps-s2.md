## Objective

Teach Kalvin identities Mark and Halo, establish their countersign, add a new
identity Alpha, then submit a misfit whose signature differs from its
nodes so it routes S2/S3 against the countersign and triggers expansion.

## Approach

Introduce identities, countersign, add a new identity, then submit a kline
whose signature (the Mark|Halo compound) does not match its nodes. The
mismatched kline partially matches the countersignature kline, routing S2/S3
and producing expansion proposals that the supervisor must ratify.

## Terminating Goal

Kalvin proposes the following canonical kline once for each misfit submission:

```
  {MH: [M, H]
```

## Lessons

### 1

Introduce the identity Mark.

```
M(ark)
```

### 2

Introduce the identity Halo.

```
H(alo)
```

### 3

Establish a bidirectional countersign between Mark and Halo.

```
M(ark) == H(alo)
```

### 4

Introduce the identity Alpha.

```
A(lpha)
```

### 5

Submit an underfit that routes S2/S3 against the {Mark: [Halo]}
countersign established in lesson 3.

- {MH: [Halo]} — an underfit: the signature promises Mark|Halo but the
  node delivers only Halo. It routes S2 against {Mark: [Halo]} because the Halo
  node overlaps, producing expansion proposals.

The block annotation binds M->Mark, H->Halo, A->Alpha so the connoted nodes
resolve to the same identities the earlier lessons introduced.

```
(Mark Halo Alpha)
MH => H
```

### 6

Submit an overfit that routes S2/S3 against the {Mark: [Halo]}
countersign established in lesson 3.

- {MH: [Mark, Halo, Alpha]} — an overfit (overfit Alpha): the Alpha
  node is disjoint from the Mark|Halo signature. Its expansion proposals reshape
  the kline and, not matching any loaded lesson entry, request ratification.

The block annotation binds M->Mark, H->Halo, A->Alpha so the connoted nodes
resolve to the same identities the earlier lessons introduced.

```
(Mark Halo Alpha)
MH => H A
```

### 7

Submit a bad-fit that routes S2/S3 against the {Mark: [Halo]}
countersign established in lesson 3.

- {MH: [Alpha]} — a bad-fit (underfit Mark|Halo, overfit Alpha): the Alpha
  node is disjoint from the Mark|Halo signature. Its expansion proposals reshape
  the kline and, not matching any loaded lesson entry, request ratification.

The block annotation binds M->Mark, H->Halo, A->Alpha so the connoted nodes
resolve to the same identities the earlier lessons introduced.

```
(Mark Halo Alpha)
MH => A
```
