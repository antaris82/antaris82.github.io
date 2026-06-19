# RESULTS — Response-Weighted Holonomy Test

## Status

Python diagnostic completed successfully to **level 9** with the shell-normalized kernel:

```text
K(d) = 1 / (3^(d-1) d²)
```

Final size:

```text
29524 nodes
9841 local response operators
17501 tested closure loops
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

The previous closure test used only minimal SO(3) transport between local axes and found a flat connection.

This test asks the sharper question:

```text
Does the response/operator connection carry nontrivial phase
even when the axis connection is flat?
```

## Method

For each local completed sibling triple:

1. build full local Markov response `P`;
2. derive the local skew axis and axis-orthogonal plane;
3. project the response to the derived 2D plane:

```text
R2 = Bᵀ P B
```

4. extract the SO(2) polar rotation `U2`;
5. multiply around closure loops:

```text
x_{i+1} = G_{i→i+1} U_i x_i
```

where `G` is the minimal frame/gauge transport and `U_i` is the local response rotation.

## Raw summary

```text
RESPONSE-WEIGHTED HOLONOMY TEST
  final level=9, nodes=29524, local operators=9841, loops=17501
  mean local theta=167.727636213 deg
  mean |local theta|=167.727636213 deg
  local polar reflection fraction=0.000000

BY MODE
  level_birth_ring: count=8, mean|excess|=102.327720517 deg, p95|excess|=158.539564474, max|excess|=165.305257135, mean gauge phase=3.945e-13, mean local theta per step=165.713295524, mean cond=1.000000, mean len=1230.00
  level_birth_ring_chunk9: count=1093, mean|excess|=69.581947035 deg, p95|excess|=70.668596393, max|excess|=70.797744987, mean gauge phase=9.004e-15, mean local theta per step=167.731327448, mean cond=1.000000, mean len=9.00
  parent_child_ring: count=3280, mean|excess|=49.279323281 deg, p95|excess|=50.672416738, max|excess|=85.400133335, mean gauge phase=5.883e-15, mean local theta per step=167.680169180, mean cond=1.000000, mean len=4.00
  parent_fan_triangle: count=9840, mean|excess|=142.991652170 deg, p95|excess|=143.545003441, max|excess|=143.689041031, mean gauge phase=4.945e-15, mean local theta per step=167.663884057, mean cond=1.000000, mean len=3.00
  sibling_cycle: count=3280, mean|excess|=143.187073647 deg, p95|excess|=143.600425745, max|excess|=143.670192573, mean gauge phase=5.059e-15, mean local theta per step=167.729024549, mean cond=1.000000, mean len=3.00

SELECTED BY LEVEL
  parent_child_ring L=0: count=1, mean|excess|=85.400133335, max|excess|=85.400133335, mean theta/step=158.649966666
  parent_child_ring L=1: count=3, mean|excess|=69.249194950, max|excess|=70.118473488, mean theta/step=162.687701262
  parent_child_ring L=2: count=9, mean|excess|=60.256259154, max|excess|=61.224569429, mean theta/step=164.935935212
  parent_child_ring L=3: count=27, mean|excess|=55.137779118, max|excess|=56.011440356, mean theta/step=166.215555220
  parent_child_ring L=4: count=81, mean|excess|=52.178687380, max|excess|=52.927542185, mean theta/step=166.955328155
  parent_child_ring L=5: count=243, mean|excess|=50.449381277, max|excess|=51.088896760, mean theta/step=167.387654681
  parent_child_ring L=6: count=729, mean|excess|=49.431552510, max|excess|=49.986441181, mean theta/step=167.642111872
  parent_child_ring L=7: count=2187, mean|excess|=48.829781054, max|excess|=49.322884171, mean theta/step=167.792554736
  parent_fan_triangle L=0: count=3, mean|excess|=114.422048595, max|excess|=114.764279457, mean theta/step=158.140682865
  parent_fan_triangle L=1: count=9, mean|excess|=127.226476056, max|excess|=127.988857009, mean theta/step=162.408825352
  parent_fan_triangle L=2: count=27, mean|excess|=134.337270229, max|excess|=135.077803432, mean theta/step=164.779090076
  parent_fan_triangle L=3: count=81, mean|excess|=138.376970794, max|excess|=139.010682519, mean theta/step=166.125656931
  parent_fan_triangle L=4: count=243, mean|excess|=140.709291776, max|excess|=141.238607196, mean theta/step=166.903097259
  parent_fan_triangle L=5: count=729, mean|excess|=142.071086097, max|excess|=142.517662346, mean theta/step=167.357028699
  parent_fan_triangle L=6: count=2187, mean|excess|=142.872142535, max|excess|=143.258147697, mean theta/step=167.624047512
  parent_fan_triangle L=7: count=6561, mean|excess|=143.345580949, max|excess|=143.689041031, mean theta/step=167.781860316
  sibling_cycle L=1: count=1, mean|excess|=120.533454208, max|excess|=120.533454208, mean theta/step=160.177818069
  sibling_cycle L=2: count=3, mean|excess|=130.572986980, max|excess|=131.006083633, mean theta/step=163.524328993
  sibling_cycle L=3: count=9, mean|excess|=136.219411853, max|excess|=136.683702421, mean theta/step=165.406470618
  sibling_cycle L=4: count=27, mean|excess|=139.455750264, max|excess|=139.859525268, mean theta/step=166.485250088
  sibling_cycle L=5: count=81, mean|excess|=141.336062532, max|excess|=141.669157296, mean theta/step=167.112020844
  sibling_cycle L=6: count=243, mean|excess|=142.438597879, max|excess|=142.712130271, mean theta/step=167.479532626
  sibling_cycle L=7: count=729, mean|excess|=143.088914863, max|excess|=143.317432005, mean theta/step=167.696304954
  sibling_cycle L=8: count=2187, mean|excess|=143.473913991, max|excess|=143.670192573, mean theta/step=167.824637997
```

## By mode

|   count |   max_abs_excess_phase_deg |   max_abs_raw_wrapped_phase_deg |   max_response_condition |   mean_abs_excess_phase_deg |   mean_abs_raw_wrapped_phase_deg |   mean_gauge_phase_deg |   mean_local_theta_deg |   mean_loop_len |   mean_response_condition | mode                    |   p95_abs_excess_phase_deg |   p95_abs_raw_wrapped_phase_deg |
|--------:|---------------------------:|--------------------------------:|-------------------------:|----------------------------:|---------------------------------:|-----------------------:|-----------------------:|----------------:|--------------------------:|:------------------------|---------------------------:|--------------------------------:|
|       8 |                   165.305  |                        165.305  |                        1 |                    102.328  |                         102.328  |            3.94522e-13 |                165.713 |            1230 |                         1 | level_birth_ring        |                   158.54   |                        158.54   |
|    1093 |                    70.7977 |                         70.7977 |                        1 |                     69.5819 |                          69.5819 |            9.00363e-15 |                167.731 |               9 |                         1 | level_birth_ring_chunk9 |                    70.6686 |                         70.6686 |
|    3280 |                    85.4001 |                         85.4001 |                        1 |                     49.2793 |                          49.2793 |            5.8826e-15  |                167.68  |               4 |                         1 | parent_child_ring       |                    50.6724 |                         50.6724 |
|    9840 |                   143.689  |                        143.689  |                        1 |                    142.992  |                         142.992  |            4.94514e-15 |                167.664 |               3 |                         1 | parent_fan_triangle     |                   143.545  |                        143.545  |
|    3280 |                   143.67   |                        143.67   |                        1 |                    143.187  |                         143.187  |            5.0587e-15  |                167.729 |               3 |                         1 | sibling_cycle           |                   143.6    |                        143.6    |

## Main finding

The response connection carries a large nonzero loop phase, while the gauge/minimal-axis part remains numerically flat.

Gauge phase:

```text
mean gauge phase ~ 1e-14 degrees
```

Response excess phase:

```text
sibling_cycle:
  mean |excess| ≈ 143.19°

parent_fan_triangle:
  mean |excess| ≈ 142.99°

parent_child_ring:
  mean |excess| ≈ 49.28°

level_birth_ring_chunk9:
  mean |excess| ≈ 69.58°
```

So the earlier flat result was only the flatness of the **axis connection**. It did not remove the response phase.

## Local response phase

At level 9:

```text
mean local theta ≈ 167.73°
local polar reflection fraction = 0
```

So each local response block has a robust SO(2) rotation part on the derived plane.

The loop phases are essentially accumulated/wrapped local response phases:

```text
sibling 3-cycle:
  3 × 167.7° ≈ 503.1° ≡ 143.1° mod 360°

parent-child ring:
  4 × 167.7° ≈ 670.8° ≡ -49.2° mod 360°
```

This matches the measured values.

## Interpretation

This is a strong positive result for the response-layer connection:

```text
minimal axis connection:
  flat

response-weighted connection:
  nontrivial U(1)-like phase
```

Therefore:

```text
Nontrivial phase is not produced by axis mismatch.
It is carried by the local response operator itself.
```

## What this shows

The model now supports:

```text
1. stable local J-plane,
2. coherent tower gluing,
3. flat axis connection,
4. nonzero response-weighted loop phase.
```

That is exactly the expected separation between:

```text
geometry of the J-axis bundle
vs.
operator/response phase connection.
```

## What this does not show

Still open:

1. gauge-invariant continuum/tower limit of the response phase;
2. derived physical metric/weight `G`;
3. global Hilbert/pre-Hilbert structure;
4. vN/Type-III/modular/nuclearity;
5. Lean proof;
6. physical interpretation as actual `i`/time/phase.

## Important caution

The loop transport uses local SO(2) polar rotation from the response block. That is a natural diagnostic, but it is not yet derived as the unique CNNA connection.

The next test should check robustness under different response-connection definitions.

## Recommended next test

```text
test_response_connection_robustness.py
```

It should compare:

```text
1. polar rotation phase of R2;
2. eigenphase of the complex pair of P;
3. skew-normalized phase from A2;
4. metric-weighted phase using candidate G = -S2;
5. phase after subtracting local mean/background phase.
```

The key question:

```text
Is the response holonomy robust under reasonable operator connection choices,
or is it an artifact of the polar-extraction convention?
```
