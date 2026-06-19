# RESULTS — Response Connection Robustness Test

## Status

Python diagnostic completed successfully to **level 9** with the shell-normalized ancestor kernel:

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

The previous response-weighted holonomy test found large loop phases using the polar SO(2) rotation of the local 2D response block.

This robustness test asks whether that phase is stable under other phase extractions, or whether it is an artifact of the polar convention.

## Phase extractions compared

```text
polar:
  SO(2) polar rotation of R2 = Bᵀ P B

eigen:
  argument of the complex eigenvalue pair of the full local Markov operator P

skew_iso:
  atan2(skew coefficient, isotropic trace coefficient) of R2

G_weighted:
  polar phase after similarity transform using candidate metric G = -S2
  when -S2 is positive definite
```

For each method, the test also subtracts the global mean local phase and measures the centered loop phase.

## Raw summary

```text
RESPONSE CONNECTION ROBUSTNESS TEST
  final level=9, nodes=29524, local operators=9841, loops=17501
  global mean local phase:
    polar: 167.727636213 deg
    eigen: 172.403402463 deg
    skew_iso: 167.727636213 deg
    G_weighted: 167.589711739 deg

BY METHOD
  G_weighted: count=17501, mean|excess|=120.568253680, p95|excess|=143.146218676, mean|centered|=0.526079572, p95|centered|=1.422998852, mean theta step=167.544286442
  eigen: count=17501, mean|excess|=130.524132711, p95|excess|=157.400930203, mean|centered|=0.265965990, p95|centered|=0.717029496, mean theta step=172.380472924
  polar: count=17501, mean|excess|=120.861626716, p95|excess|=143.557929860, mean|centered|=0.523515456, p95|centered|=1.414972019, mean theta step=167.682465106
  skew_iso: count=17501, mean|excess|=120.861626716, p95|excess|=143.557929860, mean|centered|=0.523515456, p95|centered|=1.414972019, mean theta step=167.682465106

SELECTED METHOD × MODE
  G_weighted / parent_child_ring: count=3280, mean|excess|=49.832089239, mean|centered|=0.545623283, mean theta=167.541977690
  G_weighted / parent_fan_triangle: count=9840, mean|excess|=142.576810559, mean|centered|=0.425618231, mean theta=167.525603520
  G_weighted / sibling_cycle: count=3280, mean|excess|=142.773300605, mean|centered|=0.390337739, mean theta=167.591100202
  eigen / parent_child_ring: count=3280, mean|excess|=30.482763827, mean|centered|=0.276135724, mean theta=172.379309043
  eigen / parent_fan_triangle: count=9840, mean|excess|=157.113037123, mean|centered|=0.215433027, mean theta=172.371012374
  eigen / sibling_cycle: count=3280, mean|excess|=157.212597149, mean|centered|=0.197353982, mean theta=172.404199050
  polar / parent_child_ring: count=3280, mean|excess|=49.279323281, mean|centered|=0.542617833, mean theta=167.680169180
  polar / parent_fan_triangle: count=9840, mean|excess|=142.991652170, mean|centered|=0.423276140, mean theta=167.663884057
  polar / sibling_cycle: count=3280, mean|excess|=143.187073647, mean|centered|=0.388178372, mean theta=167.729024549
  skew_iso / parent_child_ring: count=3280, mean|excess|=49.279323281, mean|centered|=0.542617833, mean theta=167.680169180
  skew_iso / parent_fan_triangle: count=9840, mean|excess|=142.991652170, mean|centered|=0.423276140, mean theta=167.663884057
  skew_iso / sibling_cycle: count=3280, mean|excess|=143.187073647, mean|centered|=0.388178372, mean theta=167.729024549
```

## By method

|   count |   max_abs_centered_phase_deg |   max_abs_excess_phase_deg |   mean_abs_centered_phase_deg |   mean_abs_excess_phase_deg |   mean_gauge_phase_abs_deg |   mean_loop_len |   mean_theta_step_deg | method     |   p95_abs_centered_phase_deg |   p95_abs_excess_phase_deg |
|--------:|-----------------------------:|---------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------:|----------------------:|:-----------|-----------------------------:|---------------------------:|
|   17501 |                      178.114 |                    153.554 |                      0.52608  |                     120.568 |                5.57367e-15 |         4.12302 |               167.544 | G_weighted |                     1.423    |                    143.146 |
|   17501 |                       91.608 |                    157.468 |                      0.265966 |                     130.524 |                5.57367e-15 |         4.12302 |               172.38  | eigen      |                     0.717029 |                    157.401 |
|   17501 |                      179.132 |                    165.305 |                      0.523515 |                     120.862 |                5.57367e-15 |         4.12302 |               167.682 | polar      |                     1.41497  |                    143.558 |
|   17501 |                      179.132 |                    165.305 |                      0.523515 |                     120.862 |                5.57367e-15 |         4.12302 |               167.682 | skew_iso   |                     1.41497  |                    143.558 |

## Selected method × mode

|   count |   max_abs_centered_phase_deg |   max_abs_excess_phase_deg |   mean_abs_centered_phase_deg |   mean_abs_excess_phase_deg |   mean_gauge_phase_abs_deg |   mean_loop_len |   mean_theta_step_deg | method     | mode                |   p95_abs_centered_phase_deg |   p95_abs_excess_phase_deg |
|--------:|-----------------------------:|---------------------------:|------------------------------:|----------------------------:|---------------------------:|----------------:|----------------------:|:-----------|:--------------------|-----------------------------:|---------------------------:|
|    3280 |                      36.4084 |                    86.0496 |                      0.545623 |                     49.8321 |                5.8826e-15  |               4 |               167.542 | G_weighted | parent_child_ring   |                     1.59193  |                    51.2331 |
|    9840 |                      29.1116 |                   143.278  |                      0.425618 |                    142.577  |                4.94514e-15 |               3 |               167.526 | G_weighted | parent_fan_triangle |                     1.29811  |                   143.133  |
|    3280 |                      22.7459 |                   143.259  |                      0.390338 |                    142.773  |                5.0587e-15  |               3 |               167.591 | G_weighted | sibling_cycle       |                     0.879607 |                   143.189  |
|    3280 |                      19.7962 |                    50.1826 |                      0.276136 |                     30.4828 |                5.8826e-15  |               4 |               172.379 | eigen      | parent_child_ring   |                     0.802071 |                    31.1885 |
|    9840 |                      15.9727 |                   157.468  |                      0.215433 |                    157.113  |                4.94514e-15 |               3 |               172.371 | eigen      | parent_fan_triangle |                     0.654064 |                   157.394  |
|    3280 |                      11.9578 |                   157.458  |                      0.197354 |                    157.213  |                5.0587e-15  |               3 |               172.404 | eigen      | sibling_cycle       |                     0.443088 |                   157.423  |
|    3280 |                      36.3107 |                    85.4001 |                      0.542618 |                     49.2793 |                5.8826e-15  |               4 |               167.68  | polar      | parent_child_ring   |                     1.58296  |                    50.6724 |
|    9840 |                      29.046  |                   143.689  |                      0.423276 |                    142.992  |                4.94514e-15 |               3 |               167.664 | polar      | parent_fan_triangle |                     1.29079  |                   143.545  |
|    3280 |                      22.6495 |                   143.67   |                      0.388178 |                    143.187  |                5.0587e-15  |               3 |               167.729 | polar      | sibling_cycle       |                     0.874678 |                   143.6    |
|    3280 |                      36.3107 |                    85.4001 |                      0.542618 |                     49.2793 |                5.8826e-15  |               4 |               167.68  | skew_iso   | parent_child_ring   |                     1.58296  |                    50.6724 |
|    9840 |                      29.046  |                   143.689  |                      0.423276 |                    142.992  |                4.94514e-15 |               3 |               167.664 | skew_iso   | parent_fan_triangle |                     1.29079  |                   143.545  |
|    3280 |                      22.6495 |                   143.67   |                      0.388178 |                    143.187  |                5.0587e-15  |               3 |               167.729 | skew_iso   | sibling_cycle       |                     0.874678 |                   143.6    |

## Main findings

### 1. The local response phase is robust

The local mean phases are:

```text
polar:       167.727636°
skew_iso:    167.727636°
G_weighted:  167.589712°
eigen:       172.403402°
```

So the polar and skew/isotropic extractions agree exactly in this model, and the G-weighted extraction is very close. The eigenphase is different by about `4.68°`, but it is still coherent and stable.

Interpretation:

```text
The existence of a large local response phase is not a polar-artifact.
```

### 2. Raw loop phase is robust, but method-dependent in value

Mean absolute loop excess phases:

```text
polar:       120.86°
skew_iso:    120.86°
G_weighted:  120.57°
eigen:       130.52°
```

The exact value depends on the connection extraction, but the fact that the response connection carries a large wrapped phase is robust.

### 3. The loop phase is mostly uniform background accumulation

After subtracting the global mean local phase, centered loop phases become small:

```text
polar:
  mean |centered| ≈ 0.524°
  p95  |centered| ≈ 1.415°

skew_iso:
  same as polar

G_weighted:
  mean |centered| ≈ 0.526°
  p95  |centered| ≈ 1.423°

eigen:
  mean |centered| ≈ 0.266°
  p95  |centered| ≈ 0.717°
```

Interpretation:

```text
Most of the nonzero loop phase is a coherent local response phase density,
not irregular loop curvature.
```

This is crucial. The model has a robust phase connection, but its curvature after subtracting the uniform background is small.

### 4. The centered residual is also robust

For the three main local closure modes:

```text
sibling_cycle:
  centered residual ~0.39° polar / ~0.20° eigen

parent_fan_triangle:
  centered residual ~0.42° polar / ~0.22° eigen

parent_child_ring:
  centered residual ~0.54° polar / ~0.28° eigen
```

So the small residual is not mode-specific.

## Interpretation

The correct conclusion is two-level:

```text
Positive:
  A robust local response phase exists.
  It survives polar, skew/isotropic, eigenphase, and G-weighted extraction.

Caution:
  Loop holonomy is dominated by a nearly uniform background phase.
  After subtracting that background, curvature/residual holonomy is small.
```

So the model currently looks like:

```text
flat J-axis bundle
+ coherent local phase density
+ small residual curvature
```

rather than:

```text
large irregular loop curvature
```

## What this shows

The response phase is **not** merely a polar-extraction artifact.

The stronger claim, however, is not “large curvature.” The stronger accurate claim is:

```text
The system generates a stable local phase density on the derived J-plane.
```

## What this does not show

Still open:

1. physical metric/weight `G`;
2. continuum/tower limit of the phase density;
3. interpretation of the uniform phase as time-step, frequency, mass/energy, or modular parameter;
4. whether residual centered curvature survives under different gluing rules;
5. Lean proof;
6. vN/Type-III/modular/nuclearity.

## Recommended next test

The next test should stop asking only “is there loop phase?” and ask:

```text
What is the scaling limit of the local phase density and residual curvature?
```

Suggested script:

```text
test_phase_density_scaling.py
```

It should measure by level and by parent depth:

```text
1. local phase mean θ_L
2. local phase variance Var(θ_L)
3. centered residual curvature on loops
4. convergence of θ_L as L grows
5. dependence on kernel choice
6. whether residual curvature decays, stabilizes, or grows
```

This is the right next decision point before introducing any physical interpretation.
