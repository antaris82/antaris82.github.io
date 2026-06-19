# RESULTS — Conductance Scaling and Growth-Generalization Test

## Status

Python diagnostic completed successfully to **level 8**.

```text
9841 nodes
6561 births in the last level
ternary branching b = 3
dynamic mode = log response
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

This test checks the user's expectation:

1. changes to early ancestors should become smaller for later/deeper births;
2. the largest conductance changes should occur locally where new children are born;
3. a high-level extrapolation needs a generalized/renormalized growth law;
4. local operator-sector behavior should remain complex while remote ancestor effects attenuate.

## Critical distinction

There are two different statements:

```text
A. Per individual birth:
   root/early-ancestor update should decrease with depth.

B. Aggregated over a whole level:
   total root/early-ancestor update may grow, stay constant, or decay,
   depending on whether the ancestor kernel beats exponential shell growth.
```

For branching `b = 3`, births per level grow like `3^L`. Therefore an ancestor kernel `K(d)` must satisfy approximately:

```text
3^L K(L) not growing
```

for aggregate root updates to stay bounded.

For an exponential kernel `K(d) = ρ^(d-1)` this means:

```text
3ρ < 1       convergent aggregate remote backreaction
3ρ = 1       critical aggregate remote backreaction
3ρ > 1       growing aggregate remote backreaction
```

The old `1/d²` kernel attenuates each individual birth, but it does **not** beat the exponential shell count.

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

## Level-8 final table

| kernel                    |   nodes |   root_g |   root_delta_sum |   root_delta_mean_per_birth |   immediate_parent_delta_mean_per_birth |   root_to_local_mean_ratio |   local_fraction_of_ancestor_delta |   birth_g_mean |
|:--------------------------|--------:|---------:|-----------------:|----------------------------:|----------------------------------------:|---------------------------:|-----------------------------------:|---------------:|
| inverse_square            |    9841 | 12.9618  |       6.4852     |                 0.000988447 |                               0.0632606 |                0.015625    |                           0.654698 |        1.40579 |
| critical_exp_1over3       |    9841 |  2.46349 |       0.188475   |                 2.87265e-05 |                               0.0628249 |                0.000457247 |                           0.666768 |        1.39611 |
| exp_0p40                  |    9841 |  4.06331 |       0.678746   |                 0.000103452 |                               0.0631419 |                0.0016384   |                           0.600393 |        1.40315 |
| exp_0p25                  |    9841 |  1.64315 |       0.0250482  |                 3.81775e-06 |                               0.0625499 |                6.10352e-05 |                           0.750011 |        1.39    |
| shell_norm_inverse_square |    9841 |  1.26367 |       0.00291782 |                 4.44722e-07 |                               0.0622468 |                7.14449e-06 |                           0.910219 |        1.38326 |

## Raw summary

```text
KERNEL inverse_square
  final level=8, nodes=9841, births=6561
  root_g=12.961842
  root_delta_sum=6.485198e+00
  root_delta_mean_per_birth=9.884467e-04
  immediate_parent_delta_mean_per_birth=6.326059e-02
  root/local mean ratio=1.562500e-02
  local fraction of ancestor delta=0.654698
  last ratio root_delta_sum L/L-1=2.308426
  last ratio root_delta_mean_per_birth L/L-1=0.769475
  last ratio local_mean_per_birth L/L-1=1.005029
  deepest parent_level=7: mean leakage=2.471751e-02, axis_align=0.936102, frac complex=1.000

KERNEL critical_exp_1over3
  final level=8, nodes=9841, births=6561
  root_g=2.463492
  root_delta_sum=1.884746e-01
  root_delta_mean_per_birth=2.872650e-05
  immediate_parent_delta_mean_per_birth=6.282486e-02
  root/local mean ratio=4.572474e-04
  local fraction of ancestor delta=0.666768
  last ratio root_delta_sum L/L-1=1.001949
  last ratio root_delta_mean_per_birth L/L-1=0.333983
  last ratio local_mean_per_birth L/L-1=1.001949
  deepest parent_level=7: mean leakage=2.761156e-02, axis_align=0.933788, frac complex=1.000

KERNEL exp_0p40
  final level=8, nodes=9841, births=6561
  root_g=4.063313
  root_delta_sum=6.787463e-01
  root_delta_mean_per_birth=1.034517e-04
  immediate_parent_delta_mean_per_birth=6.314188e-02
  root/local mean ratio=1.638400e-03
  local fraction of ancestor delta=0.600393
  last ratio root_delta_sum L/L-1=1.203034
  last ratio root_delta_mean_per_birth L/L-1=0.401011
  last ratio local_mean_per_birth L/L-1=1.002528
  deepest parent_level=7: mean leakage=2.545189e-02, axis_align=0.935648, frac complex=1.000

KERNEL exp_0p25
  final level=8, nodes=9841, births=6561
  root_g=1.643145
  root_delta_sum=2.504823e-02
  root_delta_mean_per_birth=3.817746e-06
  immediate_parent_delta_mean_per_birth=6.254995e-02
  root/local mean ratio=6.103516e-05
  local fraction of ancestor delta=0.750011
  last ratio root_delta_sum L/L-1=0.751210
  last ratio root_delta_mean_per_birth L/L-1=0.250403
  last ratio local_mean_per_birth L/L-1=1.001613
  deepest parent_level=7: mean leakage=2.951991e-02, axis_align=0.932214, frac complex=1.000

KERNEL shell_norm_inverse_square
  final level=8, nodes=9841, births=6561
  root_g=1.263668
  root_delta_sum=2.917818e-03
  root_delta_mean_per_birth=4.447215e-07
  immediate_parent_delta_mean_per_birth=6.224678e-02
  root/local mean ratio=7.144490e-06
  local fraction of ancestor delta=0.910219
  last ratio root_delta_sum L/L-1=0.766725
  last ratio root_delta_mean_per_birth L/L-1=0.255575
  last ratio local_mean_per_birth L/L-1=1.001436
  deepest parent_level=7: mean leakage=3.167238e-02, axis_align=0.930483, frac complex=1.000

```

## Main findings

### 1. The old inverse-square kernel is locally plausible but globally nonconvergent

For `inverse_square`:

```text
root_delta_mean_per_birth decreases:
  level 7 -> 8 ratio ≈ 0.769

but root_delta_sum grows:
  level 7 -> 8 ratio ≈ 2.308
```

Interpretation:

```text
A single deep birth affects the root less.
But there are exponentially many deep births.
The level-summed root update grows.
```

So `1/d²` is not enough for an infinite-level extrapolation.

### 2. Critical exponential damping gives almost constant root update per shell

For `critical_exp_1over3`:

```text
root_delta_sum level 8 ≈ 0.188475
last ratio root_delta_sum L/L-1 ≈ 1.001949
root_delta_mean_per_birth ratio ≈ 0.333983
```

Interpretation:

```text
Per-birth root effect decays like 1/3 per level.
Level-summed root effect is approximately critical/constant.
```

This is the cleanest critical scaling law.

### 3. Supercritical exponential damping still grows

For `exp_0p40`:

```text
last ratio root_delta_sum L/L-1 ≈ 1.203
```

Since `3*0.40 = 1.20`, this matches the expected aggregate growth.

### 4. Subcritical exponential damping decays

For `exp_0p25`:

```text
last ratio root_delta_sum L/L-1 ≈ 0.751
```

Since `3*0.25 = 0.75`, this matches the expected aggregate decay.

### 5. Shell-normalized inverse-square is strongly local and extrapolatable

For `shell_norm_inverse_square`:

```text
root_delta_sum level 8 ≈ 0.002918
root_delta_mean_per_birth ≈ 4.45e-7
root/local mean ratio ≈ 7.14e-6
local fraction of ancestor delta ≈ 0.910
```

Interpretation:

```text
Most backreaction stays local.
Remote ancestor effects decay even after summing over an entire shell.
```

This kernel best matches the user's locality expectation.

## Operator-sector observation

At the deepest completed parent level, all tested kernels still have:

```text
frac complex = 1.000
```

with leakage roughly:

```text
inverse_square:             0.0247
critical_exp_1over3:         0.0276
exp_0p40:                    0.0255
exp_0p25:                    0.0295
shell_norm_inverse_square:   0.0317
```

So the local complex response sector survives the more local kernels.

## Interpretation for CNNA

The growth law needs two components:

```text
1. local birth response:
   new child senses parent line + older siblings;
   newborn backreacts most strongly on immediate parent and older siblings.

2. remote ancestor kernel:
   controls how much the newborn affects the old parent line up to the root.
```

The old unnormalized `1/d²` remote kernel is not suitable for an infinite-level limit because it does not beat branching growth.

A viable generalized law should use either:

```text
K(d) = ρ^(d-1), with ρ < 1/b
```

or

```text
K(d) = 1 / (b^(d-1) d^p), p > 0
```

For `b=3`, the tested shell-normalized inverse-square kernel:

```text
K(d) = 1 / (3^(d-1) d²)
```

is a strong candidate.

## What this shows

The user's intuition is correct **per birth** and for suitably renormalized kernels:

```text
Remote ancestor updates become small.
The largest updates remain local.
The local complex response sector survives.
```

But the raw inverse-square model violates the aggregate infinite-level expectation:

```text
It is locally decaying but globally accumulating.
```

## What this does not show

Still open:

1. exact derived kernel from CNNA/NGF provenance,
2. proof that `ρ < 1/b` or shell normalization is forced,
3. full effective operator tower,
4. global compatible `J`,
5. Lean formalization.

## Next recommended formal target

Before more numerical testing, the next conceptual/Lean target should be a small criterion:

```text
AncestorBackreactionKernelCriterion
```

with a theorem-like design goal:

```text
If branching = b and remote kernel K(d) satisfies
  sum_L b^L K(L) < ∞
or an appropriate critical/renormalized bound,
then aggregate old-root backreaction is finite/bounded.
```

Numerically, the candidate kernels to keep are:

```text
critical_exp_1over3          boundary case
exp_0p25                     subcritical exponential case
shell_norm_inverse_square    local shell-normalized case
```

The recommended CNNA default for the next operator-tower test is:

```text
shell_norm_inverse_square
```

because it preserves local response while preventing uncontrolled global accumulation.
