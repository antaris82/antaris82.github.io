# RESULTS — Operator Tower Gluing Test

## Status

Python diagnostic completed successfully to **level 9** using the shell-normalized ancestor kernel:

```text
K(d) = 1 / (3^(d-1) d²)
```

Final size:

```text
29524 nodes
9841 completed local axes / sibling triples
9840 vertical parent-child gluing pairs
3280 sibling-subtower gluing triples
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

The previous test showed that local derived `J`-planes exist and that their axes are highly coherent.

This test checks whether those local axes/J-planes are compatible across the tower:

```text
parent local axis  vs  child local axis
parent local J     vs  child local J
child-subtower axes inside each sibling triple
```

## Critical limitation

The provenance tree itself has no closed loops. Therefore this test checks **vertical compatibility**, not genuine loop holonomy.

True global holonomy/frustration requires additional effective geometry closure/gluing edges.

## Raw summary

```text
OPERATOR TOWER GLUING TEST
  final level=9, nodes=29524, completed local axes=9841
  vertical parent-child gluing pairs=9840
  sibling-subtower gluing triples=3280
  global mean axis=(0.516566646, 0.289319796, 0.805886441)
  global axis mean norm=0.999995639
  global axis coherence=0.999995639

VERTICAL GLUING ALL
  mean |dot|=0.999996197
  min |dot|=0.999909504
  mean angle deg=0.128991460
  max angle deg=0.770826471
  mean plane distance=2.251319481e-03
  mean J mismatch=3.183795286e-03

SIBLING-SUBTOWER GLUING
  mean child-axis |dot|=0.999999199
  min child-axis |dot|=0.999998152
  mean child angle deg=0.066919346
  max child angle deg=0.110162934
  mean parent-to-child-mean angle deg=0.128848378

VERTICAL BY PARENT LEVEL
  parent_level=0: count=3, mean |dot|=0.999997734, max angle=0.155466888, mean J mismatch=2.928751191e-03
  parent_level=1: count=9, mean |dot|=0.999930528, max angle=0.734075426, mean J mismatch=1.663224424e-02
  parent_level=2: count=27, mean |dot|=0.999925404, max angle=0.770826471, mean J mismatch=1.723416478e-02
  parent_level=3: count=81, mean |dot|=0.999955262, max angle=0.637748111, mean J mismatch=1.331957954e-02
  parent_level=4: count=243, mean |dot|=0.999978619, max angle=0.480600051, mean J mismatch=9.162564448e-03
  parent_level=5: count=729, mean |dot|=0.999990818, max angle=0.352444768, mean J mismatch=5.931987579e-03
  parent_level=6: count=2187, mean |dot|=0.999996182, max angle=0.261281141, mean J mismatch=3.710611845e-03
  parent_level=7: count=6561, mean |dot|=0.999998337, max angle=0.200711783, mean J mismatch=2.280114397e-03
```

## Vertical gluing by parent level

|   count |   max_angle_deg |   mean_J_mismatch |   mean_angle_deg |   mean_axis_abs_dot |   mean_plane_distance |   min_axis_abs_dot |   parent_level |
|--------:|----------------:|------------------:|-----------------:|--------------------:|----------------------:|-------------------:|---------------:|
|       3 |        0.155467 |        0.00292875 |        0.118657  |            0.999998 |            0.00207094 |           0.999996 |              0 |
|       9 |        0.734075 |        0.0166322  |        0.673847  |            0.999931 |            0.0117606  |           0.999918 |              1 |
|      27 |        0.770826 |        0.0172342  |        0.698233  |            0.999925 |            0.0121862  |           0.99991  |              2 |
|      81 |        0.637748 |        0.0133196  |        0.539635  |            0.999955 |            0.00941826 |           0.999938 |              3 |
|     243 |        0.4806   |        0.00916256 |        0.371215  |            0.999979 |            0.00647888 |           0.999965 |              4 |
|     729 |        0.352445 |        0.00593199 |        0.24033   |            0.999991 |            0.00419454 |           0.999981 |              5 |
|    2187 |        0.261281 |        0.00371061 |        0.150333  |            0.999996 |            0.00262381 |           0.99999  |              6 |
|    6561 |        0.200712 |        0.00228011 |        0.0923804 |            0.999998 |            0.00161234 |           0.999994 |              7 |

## Main findings

### 1. Local axes glue extremely well through the tower

Across all vertical parent-child pairs:

```text
mean |dot|       = 0.999996197
min |dot|        = 0.999909504
mean angle       = 0.128991°
max angle        = 0.770826°
mean plane dist  = 0.002251
mean J mismatch  = 0.003184
```

Interpretation:

```text
The local derived J-planes are almost parallel across parent-child growth.
```

This is much stronger than merely saying each local triple has its own arbitrary J-plane.

### 2. Axis coherence is essentially global

Global axis statistics:

```text
axis mean norm = 0.999995639
axis coherence = 0.999995639
```

The mean axis is:

```text
(0.516566646, 0.289319796, 0.805886441)
```

The axis field is not random and not locally arbitrary.

### 3. Sibling-subtower gluing is even tighter

For the three child-subtriples inside each completed parent triple:

```text
mean child-axis |dot| = 0.999999199
min child-axis |dot|  = 0.999998152
mean child angle      = 0.066919°
max child angle       = 0.110163°
```

Interpretation:

```text
The three sibling branches inherit almost the same local J-axis.
```

### 4. Vertical mismatch decreases again at deeper parent levels

The largest mismatch occurs around parent levels 1–2:

```text
parent_level 1 max angle ≈ 0.734°
parent_level 2 max angle ≈ 0.771°
```

Then it decreases:

```text
parent_level 7 max angle ≈ 0.201°
```

This is consistent with the idea that early transient asymmetry stabilizes into a coherent tower pattern.

## Interpretation

This is the strongest tower-level positive result so far:

```text
shell-controlled growth
+ local birth/backreaction operator
→ stable local J-planes
→ parent-child compatible axes
→ coherent tower axis field
```

But it is still not a global CNNA `J` theorem.

## What this shows

It supports the hypothesis that the local complex sectors are not independent artifacts. They glue coherently along the growth tree.

## What it does not show

Still open:

1. true loop holonomy/frustration in a closed effective geometry;
2. a global Hilbert/pre-Hilbert space;
3. a derived global metric/weight `G`;
4. a single global operator `J` with `J² = -I`;
5. Lean formalization;
6. vN/Type-III/modular/nuclearity structure.

## Next recommended step

There are now two possible branches.

### Branch A — formalize the secured local/tower coherence

Lean target:

```text
ShellControlledGrowthHasStableLocalAxisCriterion
```

This should only register:

```text
shell-controlled kernel
local directed response
stable axis/plane coherence criterion
```

No global `J` claim.

### Branch B — test true holonomy/frustration

Python target:

```text
test_closed_geometry_holonomy_frustration.py
```

This must add actual effective closure/gluing edges and test whether going around a closed loop produces:

```text
zero holonomy
nonzero but trivial/gauge holonomy
nontrivial coherent holonomy
frustrated incompatible J-plane gluing
```

My recommendation: do **Branch A first** as a compact formal checkpoint, then Branch B.
