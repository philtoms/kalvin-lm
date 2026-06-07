## Objective
Stress-test intra-lesson reactive cascade by creating high candidate density. Many identity atoms are established first, then compound entries that share signature patterns create overlapping candidate pools. The final lesson submits entries where the query signature matches dozens of candidates simultaneously.

## Approach
1. Build up a dense model with many identities and relationships
2. Submit compound canonizes that create misfits against multiple established klines
3. The goal is to observe how the reactor and cogitator handle high candidate counts within a single lesson

## Lessons

### 1
Introduce ten identity atoms to create a dense signature space.

```
A
B
C
D
E
F
G
H
I
J
```

### 2
Establish bidirectional countersigns between pairs. This populates the kline space with structural relationships.

```
A == B
C == D
E == F
G == H
I == J
```

### 3
Submit cross-group canonizes that bridge the pairs. Each entry creates a misfit against the established pair klines.

```
AB => C
CD => E
EF => G
GH => I
```

### 4
Submit compound entries that combine multiple groups. These entries should match many candidates simultaneously due to the accumulated model density.

```
ABCD => E F
EFGH => I J
ACEGI => B D F H J
```

### 5
The pressure test: submit a highly compound entry whose nodes span many groups, creating maximum candidate overlap.

```
ABCDEFGHIJ => A B C D E F G H I J
```
