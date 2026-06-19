# RESULTS — Closed Geometry Holonomy / Frustration Test

## Status

Python diagnostic completed successfully to **level 9** with the shell-normalized kernel:

```text
K(d) = 1 / (3^(d-1) d²)
```

Final size:

```text
29524 nodes
9841 local axes
17501 tested closure loops
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

The previous tower test showed that local `J`-axes glue coherently along the provenance tree.

This test adds actual closed loops in the effective closure graph and checks whether those loops produce:

```text
zero holonomy
nonzero coherent holonomy
or frustrated/incompatible J-plane gluing
```

## Closure modes tested

```text
sibling_cycle:
  c1 -> c2 -> c3 -> c1

parent_fan_triangle:
  p -> c_i -> c_j -> p

parent_child_ring:
  p -> c1 -> c2 -> c3 -> p

level_birth_ring:
  artificial same-level birth-order ring

level_birth_ring_chunk9:
  artificial same-level local chunks of length 9
```

The two level-ring modes are stress tests, not CNNA claims.

## Transport used

For each edge between local axes, the test uses the **minimal SO(3) rotation** sending one local axis to the next.

This is the most conservative flat-connection test. It checks whether axis-field gluing itself creates unavoidable holonomy/frustration.

## Raw summary

```text
CLOSED GEOMETRY HOLONOMY / FRUSTRATION TEST
  final level=9, nodes=29524, local axes=9841, loops=17501

BY MODE
  level_birth_ring: count=8, mean|phase|=0.000000000 deg, p95|phase|=0.000000000, max|phase|=0.000000000, mean axis residual=2.460e-09, mean J mismatch=3.479e-09, mean len=1230.00
  level_birth_ring_chunk9: count=1093, mean|phase|=0.000000000 deg, p95|phase|=0.000000000, max|phase|=0.000000000, mean axis residual=1.800e-11, mean J mismatch=2.545e-11, mean len=9.00
  parent_child_ring: count=3280, mean|phase|=0.000000000 deg, p95|phase|=0.000000000, max|phase|=0.000000000, mean axis residual=8.000e-12, mean J mismatch=1.131e-11, mean len=4.00
  parent_fan_triangle: count=9840, mean|phase|=0.000000000 deg, p95|phase|=0.000000000, max|phase|=0.000000000, mean axis residual=6.000e-12, mean J mismatch=8.485e-12, mean len=3.00
  sibling_cycle: count=3280, mean|phase|=0.000000000 deg, p95|phase|=0.000000000, max|phase|=0.000000000, mean axis residual=6.000e-12, mean J mismatch=8.485e-12, mean len=3.00

SELECTED BY LEVEL
  parent_child_ring L=0: count=1, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_child_ring L=1: count=3, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_child_ring L=2: count=9, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_child_ring L=3: count=27, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_child_ring L=4: count=81, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_child_ring L=5: count=243, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_child_ring L=6: count=729, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_child_ring L=7: count=2187, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=1.131e-11
  parent_fan_triangle L=0: count=3, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  parent_fan_triangle L=1: count=9, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  parent_fan_triangle L=2: count=27, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  parent_fan_triangle L=3: count=81, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  parent_fan_triangle L=4: count=243, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  parent_fan_triangle L=5: count=729, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  parent_fan_triangle L=6: count=2187, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  parent_fan_triangle L=7: count=6561, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  sibling_cycle L=1: count=1, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.484e-12
  sibling_cycle L=2: count=3, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  sibling_cycle L=3: count=9, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  sibling_cycle L=4: count=27, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  sibling_cycle L=5: count=81, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  sibling_cycle L=6: count=243, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  sibling_cycle L=7: count=729, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
  sibling_cycle L=8: count=2187, mean|phase|=0.000000000, max|phase|=0.000000000, mean Jmis=8.485e-12
```

## By mode

|   count |   max_J_loop_mismatch |   max_abs_phase_deg |   max_axis_closure_residual |   max_holonomy_angle_deg |   mean_J_loop_mismatch |   mean_abs_phase_deg |   mean_axis_closure_residual |   mean_holonomy_angle_deg |   mean_loop_len | mode                    |   p95_abs_phase_deg |
|--------:|----------------------:|--------------------:|----------------------------:|-------------------------:|-----------------------:|---------------------:|-----------------------------:|--------------------------:|----------------:|:------------------------|--------------------:|
|       8 |           1.85564e-08 |         9.64937e-14 |                 1.31213e-08 |              0.00928169  |            3.47879e-09 |          1.74433e-14 |                  2.45988e-09 |               0.0027112   |            1230 | level_birth_ring        |         6.80386e-14 |
|    1093 |           2.54562e-11 |         8.97577e-15 |                 1.80002e-11 |              0.000343775 |            2.54546e-11 |          1.8612e-15  |                  1.79991e-11 |               0.000343766 |               9 | level_birth_ring_chunk9 |         4.48386e-15 |
|    3280 |           1.13144e-11 |         5.66849e-15 |                 8.00032e-12 |              0.000229187 |            1.13131e-11 |          1.2361e-15  |                  7.9996e-12  |               0.000229177 |               4 | parent_child_ring       |         3.03215e-15 |
|    9840 |           8.48611e-12 |         5.32171e-15 |                 6.00043e-12 |              0.000198487 |            8.48486e-12 |          1.09089e-15 |                  5.9997e-12  |               0.000198474 |               3 | parent_fan_triangle     |         2.67806e-15 |
|    3280 |           8.48598e-12 |         4.80617e-15 |                 6.00046e-12 |              0.000198487 |            8.48486e-12 |          1.12046e-15 |                  5.9997e-12  |               0.000198474 |               3 | sibling_cycle           |         2.80468e-15 |

## Main finding

The result is flat to numerical precision:

```text
mean |phase| = 0
max |phase|  = 0
axis residual ~ 1e-11 to 1e-9
J mismatch    ~ 1e-11 to 1e-9
```

This holds for all tested closure modes, including the artificial same-level rings.

## Interpretation

This is a **negative holonomy result** for the minimal axis connection:

```text
The coherent local J-axis field glues without frustration.
Closure edges alone do not generate an additional phase.
```

That is not a failure. It means:

```text
The shell-controlled birth/backreaction model produces a nearly flat coherent J-axis bundle under minimal transport.
```

So the previous positive tower result survives closure testing, but no new curvature/holonomy appears from these closure loops alone.

## What this shows

It shows:

```text
1. local axes are compatible around tested closed loops;
2. no loop-level J-plane frustration appears;
3. the closure graph does not destroy the coherent tower axis;
4. minimal axis transport is essentially flat.
```

## What it does not show

It does **not** show:

```text
1. nontrivial gauge curvature;
2. nonzero phase holonomy;
3. global physical J;
4. metric/weight G;
5. operator-algebra tower;
6. Type III/modular/nuclearity behavior.
```

## Consequence

The next test cannot use only minimal axis transport. That connection is too flat by construction once the axis field is highly coherent.

To test for genuine phase curvature, the next test must use a **response-derived connection**, for example:

```text
test_response_weighted_holonomy.py
```

where loop transport is built from the local Markov/response operators themselves, not from minimal rotations between axes.

Candidate transports:

```text
1. project local Markov operator to derived J-plane and compare phases;
2. use polar decomposition of the 2D local response block;
3. transport local frames via parent-child response maps, not minimal SO(3);
4. test whether loop products have nonzero U(1)-phase after removing gauge.
```

## Current CNNA interpretation

We now have:

```text
positive:
  stable local operator J-plane
  coherent tower gluing
  no closure frustration

negative/flat:
  no new holonomy from minimal axis connection
```

So the model currently points toward a coherent, almost-flat local complex-axis bundle. To get nontrivial curvature, the curvature must come from the response/operator connection or from a richer closure/gluing rule, not from axis mismatch alone.
