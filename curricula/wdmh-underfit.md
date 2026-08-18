## Objective

Exercise the S2 underfit gap-fill: locate the denotations of a query word (W) through a connotation chain, graded by hop distance, and complete a multi-bit underfit.

## Approach

Ground the Mary-had-a-little-lamb world first (identities, roles, the SVO canon, the object chain). Then author the query `what` connoting the object chain, and finally feed the underfit entries: a single-value underfit (WDMH missing W) and a multi-value underfit (WDMH missing W and M).

## Goal

`WDMH:[Mary, DH, lamb]` and `WDMH:[Mary, DH, pig]` are proposed at 2 hops (0xfb), ahead of the mid-chain `ALL` (0xf9). The multi-value underfit proposes gap-completing node sets.

## Lessons

### 1

Ground the world: a pig Object connotation and the MHALL canon and its Subject/Verb/Object roles.

```
(pig)
P > O(bject)

(Mary had a little lamb)
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = Q(uery) = ALL =>
     A > D(et)
     L > M(od)
     L > O
```

### 2

Ground a second object chain `what` connotes Query which connotes Object.

```
(what)
W > Q(uery) > O
```

### 3

Single-value underfit: WDMH is authored with Mary and DH, missing only W.
Expect a stall at `asks  WDMH:[Mary, DH, pig]` (pig is grounded before lamb)

```
(what did Mary have)
WDMH =>
  M
  DH > h(ad)
```

### 4

Multi-value underfit: WDMH is authored with only DH, missing both W and M.

```
(what did Mary have)
WDMH =>
  DH > h(ad)
```
