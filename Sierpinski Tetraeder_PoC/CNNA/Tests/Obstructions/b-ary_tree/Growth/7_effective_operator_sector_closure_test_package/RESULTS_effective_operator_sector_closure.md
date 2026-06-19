# RESULTS — Effective Operator Sector-Closure Test

## Status

Python diagnostic completed successfully to **level 9** with the shell-normalized ancestor kernel:

```text
K(d) = 1 / (3^(d-1) d²)
```

Final size:

```text
29524 nodes
9841 completed sibling triples
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

The previous operator test showed a projected local `J² ≈ -I` on the standard sibling sector, but also showed leakage: the full skew operator did not leave the standard `sum=0` sector exactly invariant.

This test asks whether the full skew part has a **derived invariant plane** and whether that plane is stable enough to be a serious candidate for a later operator tower.

## Critical guardrail

Any nonzero `3×3` skew matrix has an exact invariant plane: the plane orthogonal to its axial vector.

So the positive fact

```text
derived-plane J² ≈ -I
```

is not by itself the breakthrough. The nontrivial tests are:

```text
1. Is the derived axis stable across the tower?
2. Does it align with the canonical sibling constant vector?
3. Does standard-sector leakage decrease or stabilize?
4. Is the symmetric part sign-definite on the derived plane?
5. Is its anisotropy controlled?
```

## Final level-9 table

|   global_level |   nodes |   completed_triples |   frac_complex_pair |   mean_std_leakage |   mean_axis_align_const |   axis_coherence_to_mean |   mean_derived_J2_resid |   mean_derived_plane_leakage |   frac_Sder_sign_definite |   mean_Sder_anisotropy |
|---------------:|--------:|--------------------:|--------------------:|-------------------:|------------------------:|-------------------------:|------------------------:|-----------------------------:|--------------------------:|-----------------------:|
|              9 |   29524 |                9841 |                   1 |          0.0315844 |                0.930553 |                 0.999996 |             1.20035e-16 |                  3.37307e-18 |                         1 |               0.301999 |

## Level trend

|   global_level |   nodes |   completed_triples |   mean_std_leakage |   mean_axis_align_const |   axis_coherence_to_mean |   mean_Sder_anisotropy |
|---------------:|--------:|--------------------:|-------------------:|------------------------:|-------------------------:|-----------------------:|
|              1 |       4 |                   1 |          0.0714281 |                0.922785 |                 1        |               0.221386 |
|              2 |      13 |                   4 |          0.0578035 |                0.922483 |                 1        |               0.258684 |
|              3 |      40 |                  13 |          0.0479467 |                0.923958 |                 0.999986 |               0.277239 |
|              4 |     121 |                  40 |          0.0413513 |                0.925871 |                 0.999968 |               0.287268 |
|              5 |     364 |                 121 |          0.0371696 |                0.927551 |                 0.999966 |               0.293244 |
|              6 |    1093 |                 364 |          0.0346038 |                0.928802 |                 0.999974 |               0.297027 |
|              7 |    3280 |                1093 |          0.0330575 |                0.929655 |                 0.999984 |               0.299463 |
|              8 |    9841 |                3280 |          0.0321339 |                0.930207 |                 0.999991 |               0.301019 |
|              9 |   29524 |                9841 |          0.0315844 |                0.930553 |                 0.999996 |               0.301999 |

## Raw summary

```text
SHELL-NORMALIZED EFFECTIVE OPERATOR SECTOR CLOSURE
  final level=9, nodes=29524, completed triples=9841
  frac complex pair=1.000
  mean standard-sector leakage=3.158444e-02
  mean standard-sector J2 residual=1.226331e-16
  mean axis alignment with constant=0.930553
  axis mean norm=0.999996
  axis coherence to mean=0.999996
  mean derived-plane J2 residual=1.200349e-16
  mean derived-plane leakage=3.373067e-18
  frac S-derived-plane sign-definite=1.000
  mean S-derived-plane anisotropy=0.301999

LEVEL TREND
  L=1: nodes=4, triples=1, std_leak=7.142813e-02, axis_align=0.922785, axis_coh=1.000000, S_aniso=0.221386
  L=2: nodes=13, triples=4, std_leak=5.780349e-02, axis_align=0.922483, axis_coh=1.000000, S_aniso=0.258684
  L=3: nodes=40, triples=13, std_leak=4.794665e-02, axis_align=0.923958, axis_coh=0.999986, S_aniso=0.277239
  L=4: nodes=121, triples=40, std_leak=4.135134e-02, axis_align=0.925871, axis_coh=0.999968, S_aniso=0.287268
  L=5: nodes=364, triples=121, std_leak=3.716960e-02, axis_align=0.927551, axis_coh=0.999966, S_aniso=0.293244
  L=6: nodes=1093, triples=364, std_leak=3.460376e-02, axis_align=0.928802, axis_coh=0.999974, S_aniso=0.297027
  L=7: nodes=3280, triples=1093, std_leak=3.305750e-02, axis_align=0.929655, axis_coh=0.999984, S_aniso=0.299463
  L=8: nodes=9841, triples=3280, std_leak=3.213395e-02, axis_align=0.930207, axis_coh=0.999991, S_aniso=0.301019
  L=9: nodes=29524, triples=9841, std_leak=3.158444e-02, axis_align=0.930553, axis_coh=0.999996, S_aniso=0.301999
```

## Main findings

### 1. The complex local sector survives with shell-normalized scaling

At level 9:

```text
frac complex pair = 1.000
```

So the local complex response sector survives even after replacing the old remote kernel by the shell-controlled kernel.

### 2. The standard sibling sector is not exact, but leakage decreases

Standard-sector leakage:

```text
L1: 0.0714
L6: 0.0346
L9: 0.0316
```

It decreases strongly at first and then appears to approach a small nonzero asymptote around `~0.03`.

Interpretation:

```text
The naive sum-zero sibling sector is a good approximation,
but probably not the exact invariant sector.
```

### 3. The derived invariant plane is exact numerically

At level 9:

```text
mean derived-plane J2 residual     ≈ 1.2e-16
mean derived-plane leakage         ≈ 3.4e-18
```

This is expected for the axis-orthogonal plane of a `3×3` skew matrix, but it confirms that the exact local invariant plane is available from the response operator itself.

### 4. The derived axis is extremely coherent

At level 9:

```text
axis mean norm       = 0.999996
axis coherence       = 0.999996
axis alignment const = 0.930553
```

This is the strongest nontrivial result of this test.

Interpretation:

```text
The derived local axes are almost all pointing in the same birth-order direction.
The axis is not random or arbitrary across the tower.
```

This makes the derived plane much less suspicious than a per-triple projection artifact.

### 5. The symmetric part is sign-definite on the derived plane

At level 9:

```text
frac S-derived-plane sign-definite = 1.000
```

So the symmetric part remains a plausible local energy/metric candidate after restriction to the derived plane.

### 6. But the symmetric part is anisotropic

At level 9:

```text
mean S-derived-plane anisotropy ≈ 0.302
```

This is not zero and appears to stabilize around `~0.30`.

Interpretation:

```text
The derived plane supports a local J-like skew structure,
but the compatible metric/weight is not automatically isotropic.
```

A physical metric `G` still has to be derived. We must not claim a full Kähler/Hilbert structure yet.

## Interpretation

This test significantly strengthens the operator-level picture:

```text
shell-controlled growth
+ local birth/backreaction response
→ stable local complex response sector
→ exact derived invariant plane from skew response
→ highly coherent axis field across levels
→ sign-definite symmetric part on that plane
```

But it still does not close the full CNNA J theorem because:

```text
1. the metric/weight G is not derived;
2. the standard sibling sector is only approximate;
3. local derived planes still need compatible gluing;
4. the result is numerical, not Lean-formalized.
```

## What this shows

A serious local operator candidate exists.

The result is stronger than:

```text
selected Z3 cycle has complex eigenvalues
```

because it uses the full local response matrix and finds a stable derived axis/plane across nearly ten thousand completed triples.

## What it does not show

Still open:

1. exact Lean formalization;
2. derived global/tower Hilbert or pre-Hilbert structure;
3. compatible metric `G`;
4. inter-triple gluing of local `J` planes;
5. algebra tower / Type III / modular theory.

## Recommended next step

Now the next formal/numerical target is no longer basic local sector existence. It should be:

```text
test_operator_tower_gluing.py
```

with the shell-normalized kernel.

It should test:

```text
1. parent-child compatibility of local axes;
2. whether local derived planes glue coherently across adjacent triples;
3. whether a block/tower operator preserves the direct sum of local J-planes;
4. whether the anisotropy is stable or renormalizable;
5. whether a candidate metric G can be assembled from the sign-definite symmetric blocks.
```

The corresponding Lean target should remain modest:

```text
ShellControlledGrowthHasStableLocalAxisCriterion
```

not yet:

```text
J derived
```
