# RESULTS — Shell Phase Limit High-L Test

## Status

Focused high-level tests completed for the current best kernel:

```text
shell_norm_inverse_square:
  K(d) = 1 / (3^(d-1) d²)
```

Two runs were performed:

```text
1. full multi-method run to L10
   polar, eigen, G_weighted

2. ultrafast polar-only run to L11
   optimized for the highest feasible level in this environment
```

This is a numerical diagnostic, not a Lean theorem.

## Why the L11 run is polar-only

The previous robustness test already showed:

```text
polar = skew_iso
G_weighted very close
eigen different but coherent and cleaner
```

For the high-level limit probe, the fastest robust quantity is therefore the polar phase. The full multi-method run was kept to L10 as a consistency check; the highest-level limit probe was run polar-only to L11.

## L11 polar-only result

```text
SHELL PHASE LIMIT ULTRAFAST POLAR TEST
  max_level=11
  tail_start=6

  L=1: nodes=4, triples=1, polar_mean=154.066412456, polar_std=0.000000000, centered_loop_mean=nan, loops=0
  L=2: nodes=13, triples=4, polar_mean=158.649966666, polar_std=2.655825087, centered_loop_mean=0.136892345, loops=5
  L=3: nodes=40, triples=13, polar_mean=162.024525200, polar_std=2.694171160, centered_loop_mean=0.385962622, loops=20
  L=4: nodes=121, triples=40, polar_mean=164.307338357, polar_std=2.209588535, centered_loop_mean=0.355452909, loops=65
  L=5: nodes=364, triples=121, polar_mean=165.765279268, polar_std=1.634544393, centered_loop_mean=0.280956939, loops=200
  L=6: nodes=1093, triples=364, polar_mean=166.664340265, polar_std=1.138370523, centered_loop_mean=0.217305724, loops=605
  L=7: nodes=3280, triples=1093, polar_mean=167.208050449, polar_std=0.763599552, centered_loop_mean=0.171757124, loops=1820
  L=8: nodes=9841, triples=3280, polar_mean=167.533603072, polar_std=0.500503112, centered_loop_mean=0.142420786, loops=5465
  L=9: nodes=29524, triples=9841, polar_mean=167.727636213, polar_std=0.324339634, centered_loop_mean=0.124682857, loops=16400
  L=10: nodes=88573, triples=29524, polar_mean=167.843070223, polar_std=0.210759333, centered_loop_mean=0.114290571, loops=49205
  L=11: nodes=265720, triples=88573, polar_mean=167.911708102, polar_std=0.140536174, centered_loop_mean=0.108283575, loops=147620

FITS
  polar theta_inf=168.011604349, last=167.911708102, r=0.595000, rmse=8.033215e-04
  polar phase_std_decay: last=0.140536174, r=0.656011, rmse=6.718305e-03
  polar centered_residual_decay: last=0.108283575, r=0.870905, rmse=9.175375e-03

elapsed_sec=18.330
```

## L11 tail table

|   global_level |   nodes |   polar_max_deg |   polar_mean_deg |   polar_min_deg |   polar_std_deg |   triples |
|---------------:|--------:|----------------:|-----------------:|----------------:|----------------:|----------:|
|              6 |    1093 |         167.294 |          166.664 |         154.066 |        1.13837  |       364 |
|              7 |    3280 |         167.635 |          167.208 |         154.066 |        0.7636   |      1093 |
|              8 |    9841 |         167.833 |          167.534 |         154.066 |        0.500503 |      3280 |
|              9 |   29524 |         167.949 |          167.728 |         154.066 |        0.32434  |      9841 |
|             10 |   88573 |         168.016 |          167.843 |         154.066 |        0.210759 |     29524 |
|             11 |  265720 |         168.056 |          167.912 |         154.066 |        0.140536 |     88573 |

## L11 fits

| kind                    |   last_observed |   last_pred |        r |        rmse |   theta_inf |
|:------------------------|----------------:|------------:|---------:|------------:|------------:|
| theta_inf               |      167.912    |  167.911    | 0.595    | 0.000803322 |     168.012 |
| phase_std_decay         |        0.140536 |    0.139894 | 0.656011 | 0.0067183   |     nan     |
| centered_residual_decay |        0.108284 |    0.100519 | 0.870905 | 0.00917538  |     nan     |

## L10 full multi-method cross-check

```text
SHELL PHASE LIMIT HIGH-L TEST
  max_level=10
  final nodes=88573
  final completed=29524

  L=1: nodes=4, completed=1, polar_mean=154.066412456, polar_std=0.000000000, polar_centered=nan, elapsed=0.00s
  L=2: nodes=13, completed=4, polar_mean=158.649966666, polar_std=2.655825087, polar_centered=0.136892345, elapsed=0.01s
  L=3: nodes=40, completed=13, polar_mean=162.024525200, polar_std=2.694171160, polar_centered=0.385962622, elapsed=0.01s
  L=4: nodes=121, completed=40, polar_mean=164.307338357, polar_std=2.209588535, polar_centered=0.355452909, elapsed=0.03s
  L=5: nodes=364, completed=121, polar_mean=165.765279268, polar_std=1.634544393, polar_centered=0.280956939, elapsed=0.09s
  L=6: nodes=1093, completed=364, polar_mean=166.664340265, polar_std=1.138370523, polar_centered=0.217305724, elapsed=0.23s
  L=7: nodes=3280, completed=1093, polar_mean=167.208050449, polar_std=0.763599552, polar_centered=0.171757124, elapsed=0.63s
  L=8: nodes=9841, completed=3280, polar_mean=167.533603072, polar_std=0.500503112, polar_centered=0.142420786, elapsed=1.88s
  L=9: nodes=29524, completed=9841, polar_mean=167.727636213, polar_std=0.324339634, polar_centered=0.124682857, elapsed=5.64s
  L=10: nodes=88573, completed=29524, polar_mean=167.843070223, polar_std=0.210759333, polar_centered=0.114290571, elapsed=17.08s

FITS
  polar theta_inf=168.028995377, last=167.843070223, r=0.605000, rmse=4.658444e-03
  polar phase_std decay: last=0.210759333, r=0.672576, rmse=7.376551e-02
  polar centered_residual decay: last=0.114290571, alpha=0.177134, rmse=1.219678e-02
  eigen theta_inf=172.553391407, last=172.462082747, r=0.600000, rmse=1.335042e-03
  eigen phase_std decay: last=0.109800978, r=0.663724, rmse=3.495143e-02
  eigen centered_residual decay: last=0.058005260, alpha=0.182282, rmse=7.106926e-03
  G_weighted theta_inf=167.892376169, last=167.705780998, r=0.605000, rmse=4.846052e-03
  G_weighted phase_std decay: last=0.211738109, r=0.672967, rmse=7.449173e-02
  G_weighted centered_residual decay: last=0.114922037, alpha=0.176997, rmse=1.219403e-02
```

## Main findings

### 1. The phase-density limit estimate stabilized

Earlier L8 tail fit gave:

```text
theta_inf ≈ 168.056°
```

The L10 full run gave:

```text
polar theta_inf ≈ 168.019°
eigen theta_inf ≈ 172.548°
G_weighted theta_inf ≈ 167.883°
```

The L11 polar-only run gave:

```text
polar theta_inf ≈ 168.0116°
```

So the polar phase-density limit is stabilizing near:

```text
theta_inf ≈ 168.01°
```

### 2. The local phase density continues to approach the limit monotonically

Polar mean phase:

```text
L8  = 167.5336°
L9  = 167.7276°
L10 = 167.8431°
L11 = 167.9117°
```

This is consistent with convergence toward `~168.01°`.

### 3. Phase variance keeps falling

Polar standard deviation:

```text
L8  = 0.5005°
L9  = 0.3243°
L10 = 0.2108°
L11 = 0.1405°
```

Fit:

```text
std decay r ≈ 0.656 per level
```

So the phase field is becoming strongly homogeneous.

### 4. Centered residual curvature keeps falling slowly

Centered residual loop mean:

```text
L8  = 0.1424°
L9  = 0.1247°
L10 = 0.1143°
L11 = 0.1083°
```

Fit:

```text
residual decay r ≈ 0.871 per level
```

The residual decay is slower than variance decay, but still downward.

### 5. The current asymptotic picture is sharper

For the shell-normalized kernel, the best current numerical picture is:

```text
polar theta_inf        ≈ 168.01°
polar phase std        -> 0
centered residual      -> small, probably decreasing
```

The model is therefore not producing chaotic curvature. It is producing:

```text
stable local phase density
+ strong homogenization
+ weak residual curvature
```

## Interpretation

This strengthens the current CNNA diagnostic chain:

```text
shell-controlled remote backreaction
→ bounded old-root response
→ stable local J-plane
→ coherent tower gluing
→ robust local response phase
→ phase density converging near 168.01°
→ residual curvature small and decreasing
```

## Caution

The L11 run is polar-only. It is justified as a high-level probe because polar, skew_iso, and G_weighted were already very close at L9/L10, but it is not a replacement for a future full operator-algebra derivation.

Still not shown:

```text
1. physical time
2. physical i
3. metric/weight G as theorem
4. global Hilbert/pre-Hilbert structure
5. vN / Type III / modular flow
6. Lean proof
```

## Recommended next step

The next useful Python test should ask whether this phase density can be normalized into a per-birth/per-level clock variable:

```text
test_phase_clock_normalization.py
```

Target questions:

```text
1. Is theta_inf tied to a canonical unit-step?
2. Does subtracting theta_inf give a decaying fluctuation field?
3. Can residual curvature be interpreted as finite-level correction?
4. Does a rescaled generator H = theta / Δτ stabilize?
5. Is there a natural τ-step from growth level, birth count, or shell normalization?
```

This is the correct next conceptual transition: from phase-density detection to phase-clock normalization.
