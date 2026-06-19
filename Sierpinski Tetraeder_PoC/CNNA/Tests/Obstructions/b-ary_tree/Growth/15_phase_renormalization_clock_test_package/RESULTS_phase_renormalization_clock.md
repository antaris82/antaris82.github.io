# RESULTS — Phase Renormalization / Clock Subtraction Test

## Status

Clock-renormalization diagnostic completed to **level 10**.

The level-11 variant is heavier because this script computes several loop background subtractions for every level; level 10 already contains:

```text
88573 nodes
29524 completed local response operators
```

This is a numerical/model diagnostic, not a Lean theorem and not a physical-time claim.

## Fixed background clock phases

The test used the current high-L extrapolated values:

```text
polar:       theta_inf = 168.027421321°
eigen:       theta_inf = 172.552820014°
G_weighted:  theta_inf = 167.890861027°
```

For a loop of length `n`, the clock-renormalized residual is:

```text
sum(theta_i) - n * theta_inf
```

wrapped to `[-180°, 180°]`.

## Raw summary

```text
PHASE RENORMALIZATION / CLOCK SUBTRACTION TEST
  max_level=10
  final nodes=88573
  final completed=29524
  theta_inf:
    polar: 168.027421321 deg
    eigen: 172.552820014 deg
    G_weighted: 167.890861027 deg

  L=1: nodes=4, completed=1, polar_mean=154.066412456, gap=-13.961008865, std=0.000000000, clock_res=nan, level_res=nan, heff_loop=nan
  L=2: nodes=13, completed=4, polar_mean=158.649966666, gap=-9.377454655, std=2.655825087, clock_res=30.007854896, level_res=0.136892345, heff_loop=1.785890342e-01
  L=3: nodes=40, completed=13, polar_mean=162.024525200, gap=-6.002896121, std=2.694171160, clock_res=20.317291864, level_res=0.385962622, heff_loop=1.209165248e-01
  L=4: nodes=121, completed=40, polar_mean=164.307338357, gap=-3.720082964, std=2.209588535, clock_res=13.100305185, level_res=0.355452909, heff_loop=7.796528139e-02
  L=5: nodes=364, completed=121, polar_mean=165.765279268, gap=-2.262142053, std=1.634544393, clock_res=8.171229962, level_res=0.280956939, heff_loop=4.863033604e-02
  L=6: nodes=1093, completed=364, polar_mean=166.664340265, gap=-1.363081056, std=1.138370523, clock_res=4.997816092, level_res=0.217305724, heff_loop=2.974405042e-02
  L=7: nodes=3280, completed=1093, polar_mean=167.208050449, gap=-0.819370872, std=0.763599552, clock_res=3.028072216, level_res=0.171757124, heff_loop=1.802129791e-02
  L=8: nodes=9841, completed=3280, polar_mean=167.533603072, gap=-0.493818249, std=0.500503112, clock_res=1.830803448, level_res=0.142420786, heff_loop=1.089586112e-02
  L=9: nodes=29524, completed=9841, polar_mean=167.727636213, gap=-0.299785108, std=0.324339634, clock_res=1.111206852, level_res=0.124682857, heff_loop=6.613247070e-03
  L=10: nodes=88573, completed=29524, polar_mean=167.843070223, gap=-0.184351098, std=0.210759333, clock_res=0.681150781, level_res=0.114290571, heff_loop=4.053807261e-03

FITS
  polar theta_gap_decay: last=-1.843510975e-01, model=C_r_pow_L, r=0.604988, rmse=5.446975e-03
  polar phase_std_decay: last=2.107593329e-01, model=C_r_pow_L, r=0.672576, rmse=7.376551e-02
  polar clock_residual_decay: last=6.811507805e-01, model=C_r_pow_L, r=0.609475, rmse=8.279224e-02
  polar level_centered_decay: last=1.142905707e-01, model=C_N_minus_alpha, alpha=0.177134, rmse=1.219678e-02
  polar heff_loop_abs_decay: last=4.053807261e-03, model=C_r_pow_L, r=0.609475, rmse=4.927305e-04
  eigen theta_gap_decay: last=-9.073726726e-02, model=C_N_minus_alpha, alpha=0.464309, rmse=2.001342e-03
  eigen phase_std_decay: last=1.098009777e-01, model=C_r_pow_L, r=0.663724, rmse=3.495143e-02
  eigen clock_residual_decay: last=3.366615074e-01, model=C_r_pow_L, r=0.605149, rmse=3.735331e-02
  eigen level_centered_decay: last=5.800526022e-02, model=C_N_minus_alpha, alpha=0.182282, rmse=7.106926e-03
  eigen heff_loop_abs_decay: last=1.951063491e-03, model=C_r_pow_L, r=0.605149, rmse=2.164746e-04
  G_weighted theta_gap_decay: last=-1.850800290e-01, model=C_r_pow_L, r=0.604897, rmse=5.429302e-03
  G_weighted phase_std_decay: last=2.117381085e-01, model=C_r_pow_L, r=0.672967, rmse=7.449173e-02
  G_weighted clock_residual_decay: last=6.839914071e-01, model=C_r_pow_L, r=0.609399, rmse=8.524402e-02
  G_weighted level_centered_decay: last=1.149220370e-01, model=C_N_minus_alpha, alpha=0.176997, rmse=1.219403e-02
  G_weighted heff_loop_abs_decay: last=4.074024059e-03, model=C_r_pow_L, r=0.609399, rmse=5.077347e-04
```

## Level table

|   global_level |   nodes |   completed |   polar_mean |   polar_mean_minus_theta_inf |   polar_std |   polar_clock_residual_mean |   polar_level_centered_mean |   polar_heff_loop_abs_mean |   eigen_clock_residual_mean |   eigen_level_centered_mean |
|---------------:|--------:|------------:|-------------:|-----------------------------:|------------:|----------------------------:|----------------------------:|---------------------------:|----------------------------:|----------------------------:|
|              1 |       4 |           1 |      154.066 |                   -13.961    |    0        |                  nan        |                  nan        |               nan          |                  nan        |                 nan         |
|              2 |      13 |           4 |      158.65  |                    -9.37745  |    2.65583  |                   30.0079   |                    0.136892 |                 0.178589   |                   16.3151   |                   0.0782304 |
|              3 |      40 |          13 |      162.025 |                    -6.0029   |    2.69417  |                   20.3173   |                    0.385963 |                 0.120917   |                   10.7116   |                   0.210725  |
|              4 |     121 |          40 |      164.307 |                    -3.72008  |    2.20959  |                   13.1003   |                    0.355453 |                 0.0779653  |                    6.78268  |                   0.187417  |
|              5 |     364 |         121 |      165.765 |                    -2.26214  |    1.63454  |                    8.17123  |                    0.280957 |                 0.0486303  |                    4.18519  |                   0.145244  |
|              6 |    1093 |         364 |      166.664 |                    -1.36308  |    1.13837  |                    4.99782  |                    0.217306 |                 0.0297441  |                    2.54229  |                   0.111156  |
|              7 |    3280 |        1093 |      167.208 |                    -0.819371 |    0.7636   |                    3.02807  |                    0.171757 |                 0.0180213  |                    1.53226  |                   0.0874117 |
|              8 |    9841 |        3280 |      167.534 |                    -0.493818 |    0.500503 |                    1.8308   |                    0.142421 |                 0.0108959  |                    0.921424 |                   0.0723304 |
|              9 |   29524 |        9841 |      167.728 |                    -0.299785 |    0.32434  |                    1.11121  |                    0.124683 |                 0.00661325 |                    0.555235 |                   0.0632821 |
|             10 |   88573 |       29524 |      167.843 |                    -0.184351 |    0.210759 |                    0.681151 |                    0.114291 |                 0.00405381 |                    0.336662 |                   0.0580053 |

## Fit table

|         C |      alpha | column                          | kind                 |        last | method     | model           |          r |        rmse |
|----------:|-----------:|:--------------------------------|:---------------------|------------:|:-----------|:----------------|-----------:|------------:|
| 27.7561   | nan        | polar_mean_minus_theta_inf      | theta_gap_decay      | -0.184351   | polar      | C_r_pow_L       |   0.604988 | 0.00544698  |
| 11.6786   | nan        | polar_std                       | phase_std_decay      |  0.210759   | polar      | C_r_pow_L       |   0.672576 | 0.0737655   |
| 96.3944   | nan        | polar_clock_residual_mean       | clock_residual_decay |  0.681151   | polar      | C_r_pow_L       |   0.609475 | 0.0827922   |
|  0.778257 |   0.177134 | polar_level_centered_mean       | level_centered_decay |  0.114291   | polar      | C_N_minus_alpha | nan        | 0.0121968   |
|  0.573682 | nan        | polar_heff_loop_abs_mean        | heff_loop_abs_decay  |  0.00405381 | polar      | C_r_pow_L       |   0.609475 | 0.000492731 |
| 17.8411   |   0.464309 | eigen_mean_minus_theta_inf      | theta_gap_decay      | -0.0907373  | eigen      | C_N_minus_alpha | nan        | 0.00200134  |
|  6.8886   | nan        | eigen_std                       | phase_std_decay      |  0.109801   | eigen      | C_r_pow_L       |   0.663724 | 0.0349514   |
| 51.2619   | nan        | eigen_clock_residual_mean       | clock_residual_decay |  0.336662   | eigen      | C_r_pow_L       |   0.605149 | 0.0373533   |
|  0.415805 |   0.182282 | eigen_level_centered_mean       | level_centered_decay |  0.0580053  | eigen      | C_N_minus_alpha | nan        | 0.00710693  |
|  0.29708  | nan        | eigen_heff_loop_abs_mean        | heff_loop_abs_decay  |  0.00195106 | eigen      | C_r_pow_L       |   0.605149 | 0.000216475 |
| 27.9194   | nan        | G_weighted_mean_minus_theta_inf | theta_gap_decay      | -0.18508    | G_weighted | C_r_pow_L       |   0.604897 | 0.0054293   |
| 11.67     | nan        | G_weighted_std                  | phase_std_decay      |  0.211738   | G_weighted | C_r_pow_L       |   0.672967 | 0.0744917   |
| 96.9541   | nan        | G_weighted_clock_residual_mean  | clock_residual_decay |  0.683991   | G_weighted | C_r_pow_L       |   0.609399 | 0.085244    |
|  0.781558 |   0.176997 | G_weighted_level_centered_mean  | level_centered_decay |  0.114922   | G_weighted | C_N_minus_alpha | nan        | 0.012194    |
|  0.577483 | nan        | G_weighted_heff_loop_abs_mean   | heff_loop_abs_decay  |  0.00407402 | G_weighted | C_r_pow_L       |   0.609399 | 0.000507735 |

## Main findings

### 1. Fixed clock subtraction works, but is dominated by remaining theta-gap

For polar phase:

```text
L8 clock residual:  1.8308°
L9 clock residual:  1.1112°
L10 clock residual: 0.6812°
```

This decreases quickly:

```text
polar clock residual decay r ≈ 0.6095 per level
```

The normalized effective loop generator residual also decreases:

```text
polar h_eff loop abs:
L8:  1.09e-2
L9:  6.61e-3
L10: 4.05e-3
```

So using `theta_inf` as a background clock is meaningful.

### 2. But level-centered subtraction is still smaller at finite L

At L10:

```text
polar clock residual:       0.6812°
polar level-centered resid: 0.1143°
```

That is expected. The fixed `theta_inf` subtraction still includes the finite-level gap:

```text
theta_L - theta_inf
```

At L10:

```text
polar theta gap = -0.18435°
```

For a 3- or 4-step loop this alone contributes about `0.55°` to `0.74°`, which explains most of the clock residual.

### 3. The theta-gap decays with the same clock-residual rate

For polar:

```text
theta gap decay r     ≈ 0.6050
clock residual decay r ≈ 0.6095
```

So the clock residual is mostly controlled by the approach of `theta_L` to `theta_inf`.

### 4. Eigen convention remains the cleanest

At L10:

```text
eigen clock residual:       0.3367°
eigen level-centered resid: 0.0580°
```

Fits:

```text
eigen clock residual decay r ≈ 0.6051
eigen h_eff loop decay r     ≈ 0.6051
```

The eigenphase has about half the residual of polar/G-weighted in this diagnostic.

### 5. Residual curvature after local level-background subtraction is much smaller

Level-centered residual is the better curvature diagnostic at finite L:

```text
polar L10 level-centered residual ≈ 0.1143°
eigen L10 level-centered residual ≈ 0.0580°
G-weighted L10 level-centered residual ≈ 0.1149°
```

It decays slowly as a power in total node count:

```text
polar ~ N^(-0.177)
eigen ~ N^(-0.182)
G-weighted ~ N^(-0.177)
```

## Interpretation

The test supports a two-part decomposition:

```text
theta_i = theta_inf + finite_level_gap(L) + small_local_curvature_i
```

where:

```text
finite_level_gap(L) decays quickly, about 0.61^L
local curvature residual decays slowly, roughly N^(-0.18)
```

So `theta_inf` behaves like a plausible background clock/frequency phase density, but at finite levels the best curvature diagnostic still subtracts the current local level-background.

## Current best cautious statement

```text
The shell-normalized model produces a stable background response-clock phase theta_inf.
After removing this clock phase, the remaining loop residual decreases.
After additionally removing the finite-level background, the residual curvature is very small.
```

Still not allowed:

```text
physical time is derived
physical i is proven
modular flow is established
```

## Next useful test

The next step should test whether the residual can be tied to geometry/gluing depth rather than just level size:

```text
test_residual_curvature_locality.py
```

It should measure residuals by:

```text
parent level
loop mode
birth-order position
subtree depth
distance to newest frontier
```

The key question:

```text
Is residual curvature concentrated near active growth/frontier regions,
or does it persist uniformly through the tower?
```
