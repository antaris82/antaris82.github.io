# RESULTS — Phase Density Scaling Test

## Status

Python diagnostic completed successfully to **level 8** for five ancestor/backreaction kernels.

Final size per kernel:

```text
9841 nodes
3280 completed local response operators
5465 closure-loop residual records per method at level 8
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

The previous robustness test showed:

```text
large raw loop phase
small centered residual phase
```

This test measures how the local phase density and residual curvature scale with level and with kernel choice.

## Kernels tested

```text
inverse_square:
  K(d) = 1/d²

critical_exp_1over3:
  K(d) = (1/3)^(d-1)

exp_0p40:
  K(d) = 0.40^(d-1)

exp_0p25:
  K(d) = 0.25^(d-1)

shell_norm_inverse_square:
  K(d) = 1 / (3^(d-1) d²)
```

## Level-8 local phase density

| kernel                    |   count |   polar_mean_linear_deg |   polar_std_linear_deg |   eigen_mean_linear_deg |   eigen_std_linear_deg |   G_weighted_mean_linear_deg |   G_weighted_std_linear_deg |
|:--------------------------|--------:|------------------------:|-----------------------:|------------------------:|-----------------------:|-----------------------------:|----------------------------:|
| inverse_square            |    3280 |                 169.751 |               0.900587 |                 173.426 |               0.470912 |                      169.623 |                    0.903768 |
| critical_exp_1over3       |    3280 |                 168.913 |               0.633212 |                 172.986 |               0.332225 |                      168.782 |                    0.63552  |
| exp_0p40                  |    3280 |                 169.635 |               0.729715 |                 173.354 |               0.383523 |                      169.507 |                    0.73204  |
| exp_0p25                  |    3280 |                 168.269 |               0.561048 |                 172.663 |               0.295029 |                      168.135 |                    0.563233 |
| shell_norm_inverse_square |    3280 |                 167.534 |               0.500503 |                 172.305 |               0.264429 |                      167.395 |                    0.502571 |

## Level-8 loop residuals

| kernel                    | method     |   count |   mean_abs_raw_wrapped_deg |   mean_abs_centered_global_deg |   mean_abs_centered_level_deg |   p95_abs_centered_level_deg |   mean_theta_step_deg |
|:--------------------------|:-----------|--------:|---------------------------:|-------------------------------:|------------------------------:|-----------------------------:|----------------------:|
| critical_exp_1over3       | G_weighted |    5465 |                    125.887 |                       1.01043  |                     0.415126  |                     0.978781 |               168.679 |
| critical_exp_1over3       | eigen      |    5465 |                    132.693 |                       0.517946 |                     0.212867  |                     0.503247 |               172.933 |
| critical_exp_1over3       | polar      |    5465 |                    126.097 |                       1.00628  |                     0.413484  |                     0.975034 |               168.81  |
| exp_0p25                  | G_weighted |    5465 |                    124.875 |                       0.83536  |                     0.279419  |                     0.659335 |               168.047 |
| exp_0p25                  | eigen      |    5465 |                    132.19  |                       0.425588 |                     0.142017  |                     0.334646 |               172.619 |
| exp_0p25                  | polar      |    5465 |                    125.09  |                       0.831382 |                     0.278095  |                     0.65614  |               168.181 |
| exp_0p40                  | G_weighted |    5465 |                    127.008 |                       1.25719  |                     0.570721  |                     1.36623  |               169.38  |
| exp_0p40                  | eigen      |    5465 |                    133.261 |                       0.651827 |                     0.297006  |                     0.708774 |               173.288 |
| exp_0p40                  | polar      |    5465 |                    127.214 |                       1.25313  |                     0.569054  |                     1.36178  |               169.508 |
| inverse_square            | G_weighted |    5465 |                    127.069 |                       1.79979  |                     0.718674  |                     1.75879  |               169.418 |
| inverse_square            | eigen      |    5465 |                    133.313 |                       0.931826 |                     0.374189  |                     0.91735  |               173.32  |
| inverse_square            | polar      |    5465 |                    127.275 |                       1.79348  |                     0.716439  |                     1.75399  |               169.547 |
| shell_norm_inverse_square | G_weighted |    5465 |                    123.705 |                       0.720879 |                     0.143222  |                     0.348064 |               167.316 |
| shell_norm_inverse_square | eigen      |    5465 |                    131.624 |                       0.365904 |                     0.0723304 |                     0.175715 |               172.265 |
| shell_norm_inverse_square | polar      |    5465 |                    123.928 |                       0.716932 |                     0.142421  |                     0.346109 |               167.455 |

## Raw level trend

```text
PHASE DENSITY SCALING TEST
  max_level=8

KERNEL inverse_square
  L=1: triples=1, polar_mean=154.066412456, polar_std=0.000000000, centered_loop_mean=nan
  L=2: triples=4, polar_mean=158.756244343, polar_std=2.723177259, centered_loop_mean=0.174420110
  L=3: triples=13, polar_mean=162.336724723, polar_std=2.836766836, centered_loop_mean=0.549480901
  L=4: triples=40, polar_mean=164.894777319, polar_std=2.412787512, centered_loop_mean=0.609218428
  L=5: triples=121, polar_mean=166.678121409, polar_std=1.882916346, centered_loop_mean=0.619390681
  L=6: triples=364, polar_mean=167.949265517, polar_std=1.426052661, centered_loop_mean=0.629296424
  L=7: triples=1093, polar_mean=168.921267939, polar_std=1.097436381, centered_loop_mean=0.658616138
  L=8: triples=3280, polar_mean=169.750893640, polar_std=0.900587221, centered_loop_mean=0.716439163

KERNEL critical_exp_1over3
  L=1: triples=1, polar_mean=154.066412456, polar_std=0.000000000, centered_loop_mean=nan
  L=2: triples=4, polar_mean=158.809168609, polar_std=2.757202007, centered_loop_mean=0.193097328
  L=3: triples=13, polar_mean=162.447982197, polar_std=2.881679285, centered_loop_mean=0.578842166
  L=4: triples=40, polar_mean=165.011285440, polar_std=2.432980936, centered_loop_mean=0.608457952
  L=5: triples=121, polar_mean=166.713858960, polar_std=1.852286004, centered_loop_mean=0.557396846
  L=6: triples=364, polar_mean=167.804439851, polar_std=1.329344632, centered_loop_mean=0.496461168
  L=7: triples=1093, polar_mean=168.488672889, polar_std=0.923033584, centered_loop_mean=0.447529181
  L=8: triples=3280, polar_mean=168.913121847, polar_std=0.633211745, centered_loop_mean=0.413484207

KERNEL exp_0p40
  L=1: triples=1, polar_mean=154.066412456, polar_std=0.000000000, centered_loop_mean=nan
  L=2: triples=4, polar_mean=158.851404765, polar_std=2.784575021, centered_loop_mean=0.207997343
  L=3: triples=13, polar_mean=162.579467705, polar_std=2.944253632, centered_loop_mean=0.652830457
  L=4: triples=40, polar_mean=165.263978001, polar_std=2.523998153, centered_loop_mean=0.725120952
  L=5: triples=121, polar_mean=167.099051982, polar_std=1.958580117, centered_loop_mean=0.700479095
  L=6: triples=364, polar_mean=168.317244112, polar_std=1.438924754, centered_loop_mean=0.651430736
  L=7: triples=1093, polar_mean=169.114917092, polar_std=1.028007307, centered_loop_mean=0.605020935
  L=8: triples=3280, polar_mean=169.634889113, polar_std=0.729715252, centered_loop_mean=0.569053970

KERNEL exp_0p25
  L=1: triples=1, polar_mean=154.066412456, polar_std=0.000000000, centered_loop_mean=nan
  L=2: triples=4, polar_mean=158.756244343, polar_std=2.723177259, centered_loop_mean=0.174420110
  L=3: triples=13, polar_mean=162.293704546, polar_std=2.810518560, centered_loop_mean=0.498619317
  L=4: triples=40, polar_mean=164.733687467, polar_std=2.338533984, centered_loop_mean=0.490732518
  L=5: triples=121, polar_mean=166.315903336, polar_std=1.751632546, centered_loop_mean=0.421813016
  L=6: triples=364, polar_mean=167.303193621, polar_std=1.234434081, centered_loop_mean=0.356361286
  L=7: triples=1093, polar_mean=167.905724766, polar_std=0.839277482, centered_loop_mean=0.308826369
  L=8: triples=3280, polar_mean=168.269021240, polar_std=0.561048404, centered_loop_mean=0.278095206

KERNEL shell_norm_inverse_square
  L=1: triples=1, polar_mean=154.066412456, polar_std=0.000000000, centered_loop_mean=nan
  L=2: triples=4, polar_mean=158.649966666, polar_std=2.655825087, centered_loop_mean=0.136892345
  L=3: triples=13, polar_mean=162.024525200, polar_std=2.694171160, centered_loop_mean=0.385962622
  L=4: triples=40, polar_mean=164.307338357, polar_std=2.209588535, centered_loop_mean=0.355452909
  L=5: triples=121, polar_mean=165.765279268, polar_std=1.634544393, centered_loop_mean=0.280956939
  L=6: triples=364, polar_mean=166.664340265, polar_std=1.138370523, centered_loop_mean=0.217305724
  L=7: triples=1093, polar_mean=167.208050449, polar_std=0.763599552, centered_loop_mean=0.171757124
  L=8: triples=3280, polar_mean=167.533603072, polar_std=0.500503112, centered_loop_mean=0.142420786

```

## Main findings

### 1. Local phase density stabilizes, but the limiting value is kernel-dependent

At level 8, the polar local mean phase is:

```text
inverse_square:             169.7509°
critical_exp_1over3:         168.9131°
exp_0p40:                    169.6349°
exp_0p25:                    168.2690°
shell_norm_inverse_square:   167.5336°
```

So stronger remote damping gives a lower local phase density. The shell-normalized kernel is the most local and gives the lowest level-8 phase.

### 2. Phase variance decreases with level

For the shell-normalized kernel:

```text
polar std:
L2: 2.6558°
L4: 2.2096°
L6: 1.1384°
L8: 0.5005°
```

So the local phase density becomes more homogeneous as the tower grows.

This is a strong positive scaling result.

### 3. Centered residual curvature decreases for shell-normalized and subcritical kernels

Polar centered loop mean at level 8:

```text
inverse_square:             0.7164°
critical_exp_1over3:         0.4135°
exp_0p40:                    0.5691°
exp_0p25:                    0.2781°
shell_norm_inverse_square:   0.1424°
```

The shell-normalized kernel gives the smallest residual curvature by a large margin.

### 4. Raw loop phase remains large, but that is mostly background phase density

For shell-normalized kernel at level 8:

```text
polar raw wrapped loop phase mean ≈ 123.93°
polar centered level residual     ≈ 0.142°
```

So again:

```text
large raw loop phase
= accumulated local phase density

small centered phase
= residual curvature / nonuniformity
```

### 5. Eigenphase residuals are even smaller

For shell-normalized kernel at level 8:

```text
eigen centered residual mean ≈ 0.0723°
p95 ≈ 0.1757°
```

So the eigenphase connection sees an even cleaner homogeneous phase density than the polar connection.

## Interpretation

The model is not producing large irregular curvature. It is producing:

```text
stable local response phase density
+ increasingly homogeneous phase field
+ small residual curvature
```

The shell-normalized kernel is currently the best candidate because it simultaneously gives:

```text
1. bounded old-root backreaction,
2. coherent local J-axis tower,
3. robust local response phase,
4. smallest centered residual curvature,
5. decreasing phase variance.
```

## Important implication

This is now closer to a **phase-density / clock-step / frequency-like** structure than to a raw curvature/holonomy structure.

The correct cautious language is:

```text
A coherent local phase density emerges in the response connection.
```

not yet:

```text
physical time or physical i is derived.
```

## What remains open

1. Does the phase density converge as `L -> ∞`?
2. Can a continuum/tower limit be extracted?
3. Is the residual curvature asymptotically zero or merely small?
4. Is the phase density tied to a metric/weight `G`?
5. Can this be formalized in Lean without importing complex numbers?
6. Is there any route from this phase density to modular flow/vN structure?

## Next recommended Python test

The next test should fit the scaling curves:

```text
test_phase_density_extrapolation.py
```

It should estimate:

```text
theta_infty
decay exponent for phase variance
decay exponent for centered residual curvature
kernel-dependent asymptotic class
```

Especially for:

```text
shell_norm_inverse_square
exp_0p25
critical_exp_1over3
```

These are the useful candidate regimes.
