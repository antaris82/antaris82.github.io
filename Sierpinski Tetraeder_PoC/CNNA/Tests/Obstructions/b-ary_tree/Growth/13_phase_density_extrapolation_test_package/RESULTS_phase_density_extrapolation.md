# RESULTS — Phase Density Extrapolation Test

## Status

Extrapolation completed from the existing `test_phase_density_scaling.py` level data.

Input dataset:

```text
phase_density_scaling_out_L8
levels 1..8
five ancestor/backreaction kernels
```

Fit window:

```text
tail_start = 4
```

The tail-only fit is deliberate: levels 1–3 contain visible transients, so extrapolating from all levels overestimates several limiting values.

This is a numerical extrapolation, not a Lean theorem.

## Raw summary

```text
PHASE DENSITY EXTRAPOLATION TEST

KEY KERNELS — best polar theta fits
  shell_norm_inverse_square: theta_inf=168.056215644 deg, last=167.533603072, model=theta_inf_plus_A_r_pow_L, rmse=3.024160e-03, regime=shell-normalized local/subcritical candidate
  exp_0p25: theta_inf=168.862843813 deg, last=168.269021240, model=theta_inf_plus_A_r_pow_L, rmse=3.853056e-03, regime=subcritical exponential remote kernel (3ρ < 1)
  critical_exp_1over3: theta_inf=169.643995909 deg, last=168.913121847, model=theta_inf_plus_A_r_pow_L, rmse=4.783666e-03, regime=critical remote kernel (3ρ ≈ 1)

KEY KERNELS — best polar phase-std decay
  shell_norm_inverse_square: last_std=0.500503112, model=C_times_r_pow_L, r=0.688598, rmse=5.319964e-02
  exp_0p25: last_std=0.561048404, model=C_times_r_pow_L, r=0.698325, rmse=5.461599e-02
  critical_exp_1over3: last_std=0.633211745, model=C_times_r_pow_L, r=0.712579, rmse=5.349726e-02

KEY KERNELS — best polar centered residual decay
  shell_norm_inverse_square: last_residual=0.142420786, model=C_times_N_minus_alpha, exponent=0.211120, rmse=3.306415e-03
  exp_0p25: last_residual=0.278095206, model=C_times_L_minus_p, exponent=0.837055, rmse=5.280547e-03
  critical_exp_1over3: last_residual=0.413484207, model=C_times_r_pow_L, r=0.905548, rmse=4.025184e-03

ALL KERNELS — best polar theta_inf
  critical_exp_1over3: theta_inf=169.643995909 deg, last=168.913121847, model=theta_inf_plus_A_r_pow_L, rmse=4.784e-03
  exp_0p25: theta_inf=168.862843813 deg, last=168.269021240, model=theta_inf_plus_A_r_pow_L, rmse=3.853e-03
  exp_0p40: theta_inf=170.664563208 deg, last=169.634889113, model=theta_inf_plus_A_r_pow_L, rmse=3.468e-03
  inverse_square: theta_inf=172.133906480 deg, last=169.750893640, model=theta_inf_plus_A_r_pow_L, rmse=2.829e-02
  shell_norm_inverse_square: theta_inf=168.056215644 deg, last=167.533603072, model=theta_inf_plus_A_r_pow_L, rmse=3.024e-03
```

## Best polar theta∞ fits

| kernel                    |   theta_inf |   last_observed | model                    |     r |       rmse | regime                                                   |
|:--------------------------|------------:|----------------:|:-------------------------|------:|-----------:|:---------------------------------------------------------|
| shell_norm_inverse_square |     168.056 |         167.534 | theta_inf_plus_A_r_pow_L | 0.61  | 0.00302416 | shell-normalized local/subcritical candidate             |
| exp_0p25                  |     168.863 |         168.269 | theta_inf_plus_A_r_pow_L | 0.615 | 0.00385306 | subcritical exponential remote kernel (3ρ < 1)           |
| critical_exp_1over3       |     169.644 |         168.913 | theta_inf_plus_A_r_pow_L | 0.63  | 0.00478367 | critical remote kernel (3ρ ≈ 1)                          |
| exp_0p40                  |     170.665 |         169.635 | theta_inf_plus_A_r_pow_L | 0.66  | 0.00346836 | supercritical exponential remote kernel (3ρ > 1)         |
| inverse_square            |     172.134 |         169.751 | theta_inf_plus_A_r_pow_L | 0.76  | 0.0282944  | locally decaying but globally accumulating/supercritical |

## Best polar phase-std decay fits

| kernel                    |   last_observed | model                 |          r |   decay_exponent |      rmse |
|:--------------------------|----------------:|:----------------------|-----------:|-----------------:|----------:|
| shell_norm_inverse_square |        0.500503 | C_times_r_pow_L       |   0.688598 |       nan        | 0.0531996 |
| exp_0p25                  |        0.561048 | C_times_r_pow_L       |   0.698325 |       nan        | 0.054616  |
| critical_exp_1over3       |        0.633212 | C_times_r_pow_L       |   0.712579 |       nan        | 0.0534973 |
| exp_0p40                  |        0.729715 | C_times_r_pow_L       |   0.731507 |       nan        | 0.0492215 |
| inverse_square            |        0.900587 | C_times_N_minus_alpha | nan        |         0.228353 | 0.0231321 |

## Best polar centered-residual decay fits

| kernel                    |   last_observed | model                 |          r |   decay_exponent |       rmse |
|:--------------------------|----------------:|:----------------------|-----------:|-----------------:|-----------:|
| shell_norm_inverse_square |        0.142421 | C_times_N_minus_alpha | nan        |         0.21112  | 0.00330642 |
| exp_0p25                  |        0.278095 | C_times_L_minus_p     | nan        |         0.837055 | 0.00528055 |
| critical_exp_1over3       |        0.413484 | C_times_r_pow_L       |   0.905548 |       nan        | 0.00402518 |
| exp_0p40                  |        0.569054 | C_times_r_pow_L       |   0.938828 |       nan        | 0.00682895 |
| inverse_square            |        0.716439 | C_times_r_pow_L       |   1.03932  |       nan        | 0.0135584  |

## Main findings

### 1. Shell-normalized kernel gives the lowest fitted phase-density limit

Tail fit estimate:

```text
shell_norm_inverse_square:
  theta_inf ≈ 168.0562°
  last observed L8 ≈ 167.5336°
```

The candidate regimes order by fitted polar `theta_inf` is:

```text
shell_norm_inverse_square  ≈ 168.06°
exp_0p25                   ≈ 168.86°
critical_exp_1over3        ≈ 169.64°
exp_0p40                   ≈ 170.66°
inverse_square             ≈ 172.13°
```

So stronger locality / better remote-shell control lowers the limiting phase-density.

### 2. Phase variance decays exponentially in level over the tail window

For the key kernels:

```text
shell_norm_inverse_square:
  std decay r ≈ 0.6886 per level

exp_0p25:
  std decay r ≈ 0.6983 per level

critical_exp_1over3:
  std decay r ≈ 0.7126 per level
```

So the phase field becomes more homogeneous with depth.

### 3. Centered residual curvature also decays

Best tail fits for polar centered residual:

```text
shell_norm_inverse_square:
  best fit ~ N^(-0.211)
  L8 residual ≈ 0.1424°

exp_0p25:
  best fit ~ L^(-0.837)
  L8 residual ≈ 0.2781°

critical_exp_1over3:
  best fit ~ 0.9055^L
  L8 residual ≈ 0.4135°
```

The precise decay model is not yet robust with only levels 4–8, but the ordering is robust:

```text
shell_norm_inverse_square < exp_0p25 < critical_exp_1over3
```

### 4. The shell-normalized kernel remains the best current default

It simultaneously gives:

```text
lowest theta_inf estimate
fastest/cleanest phase homogenization
smallest residual loop curvature
bounded old-root backreaction
stable local J-axis tower
```

## Interpretation

The model is converging toward:

```text
a coherent local phase-density limit
+ vanishing or very small residual curvature
```

not toward:

```text
large chaotic loop curvature.
```

For the shell-normalized candidate, the emerging asymptotic picture is:

```text
theta_infty ≈ 168.06°
phase std → 0
centered residual curvature → 0 or very small
```

## Caution

The theta∞ estimates use only levels 4–8. They are good enough for model-selection, not good enough for a final asymptotic claim.

The residual-decay model is especially underdetermined: exponential, power-in-level, and power-in-node fits are close with this small data window.

## Next recommended test

The next useful step is a **focused high-level run for only the shell-normalized kernel**, not all five kernels.

Suggested script:

```text
test_shell_phase_limit_highL.py
```

Goal:

```text
run shell_norm_inverse_square to the highest feasible level
fit theta_inf again
fit residual decay again
check whether theta_inf ≈ 168.06° remains stable
```

This is computationally cheaper than re-running all kernels and directly tests the current best candidate.
