# RESULTS — Effective Operator J-Candidate Test

## Status

Python diagnostic completed successfully to **level 6**.

The attempted level-8 run was stopped by runtime limits. Level 6 already contains:

```text
1093 nodes
364 completed sibling triples
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

This test moves beyond graph/cochain/topology diagnostics.

For each completed sibling triple `(1,2,3)` it builds the full local directed response transport matrix `M` from dynamic birth/backreaction weights. Then it decomposes:

```text
S = (M + Mᵀ)/2
A = (M - Mᵀ)/2
```

and checks whether the skew part `A` carries a local complex-structure candidate on the 2D sibling sector orthogonal to `(1,1,1)`.

## Guardrails

- Conductance/response weights are not geometric edge lengths.
- The test does not derive the physical metric `G`.
- The Euclidean/sum-zero sector is used only as a first operator screen.
- A projected `J² = -I` result is not yet a CNNA theorem.
- Type III, vN algebra, nuclearity, and modular flow are not tested here.

## Level-6 final table

| mode       |   nodes |   completed_triples |   frac_markov_complex |   frac_markov_J2_ok |   mean_markov_J2_resid |   mean_markov_sumzero_leakage |   mean_markov_axis_align_const |   frac_markov_kappa_flip_ok |   frac_path_no_closure_complex |   frac_sym_raw_complex |   frac_markov_S2_sign_definite |
|:-----------|--------:|--------------------:|----------------------:|--------------------:|-----------------------:|------------------------------:|-------------------------------:|----------------------------:|-------------------------------:|-----------------------:|-------------------------------:|
| linear     |    1093 |                 364 |                     1 |                   1 |            9.85814e-17 |                     0.0619311 |                       0.848852 |                           1 |                              0 |                      0 |                              1 |
| log        |    1093 |                 364 |                     1 |                   1 |            1.07056e-16 |                     0.03094   |                       0.931491 |                           1 |                              0 |                      0 |                              1 |
| saturating |    1093 |                 364 |                     1 |                   1 |            1.12018e-16 |                     0.1331    |                       0.868945 |                           1 |                              0 |                      0 |                              1 |

## Raw summary

```text
MODE linear
  final nodes=1093, completed triples=364
  mean neutral norm=0.274849
  frac Markov complex=1.000
  frac Markov J2 ok=1.000
  mean Markov J2 residual=9.858145e-17
  mean Markov sum-zero leakage=6.193109e-02
  mean Markov axis alignment with constant=0.848852
  frac Markov kappa flip ok=1.000
  control: frac selected forward J2 ok=1.000
  control: path no-closure complex=0.000
  control: sym raw complex=0.000
  root child g=4.07827492541 4.37041099007 4.7393881029
  root Markov eigs=1+0j -0.5+0.132389657j -0.5-0.132389657j
  root Markov alpha=0.184722920
  root Markov J2 residual=3.142722e-16
  root Markov axis alignment=0.896878862
  root Markov kappa flip residual=2.452182e-16
  root Markov S2 eigs=-0.628823662 -0.371176338
  root kappa preserves birth order=False

MODE log
  final nodes=1093, completed triples=364
  mean neutral norm=0.007699
  frac Markov complex=1.000
  frac Markov J2 ok=1.000
  mean Markov J2 residual=1.070557e-16
  mean Markov sum-zero leakage=3.093996e-02
  mean Markov axis alignment with constant=0.931491
  frac Markov kappa flip ok=1.000
  control: frac selected forward J2 ok=1.000
  control: path no-closure complex=0.000
  control: sym raw complex=0.000
  root child g=2.66312503161 2.72491751176 2.75743649686
  root Markov eigs=1+0j -0.5+0.138051716j -0.5-0.138051716j
  root Markov alpha=0.171062068
  root Markov J2 residual=1.765745e-16
  root Markov axis alignment=0.922785039
  root Markov kappa flip residual=1.550322e-16
  root Markov S2 eigs=-0.601014627 -0.398985373
  root kappa preserves birth order=False

MODE saturating
  final nodes=1093, completed triples=364
  mean neutral norm=0.026327
  frac Markov complex=1.000
  frac Markov J2 ok=1.000
  mean Markov J2 residual=1.120182e-16
  mean Markov sum-zero leakage=1.331002e-01
  mean Markov axis alignment with constant=0.868945
  frac Markov kappa flip ok=1.000
  control: frac selected forward J2 ok=1.000
  control: path no-closure complex=0.000
  control: sym raw complex=0.000
  root child g=3.38976999179 3.528453305 3.5547911978
  root Markov eigs=1+0j -0.5+0.212180858j -0.5-0.212180858j
  root Markov alpha=0.300969116
  root Markov J2 residual=4.142670e-17
  root Markov axis alignment=0.893893500
  root Markov kappa flip residual=2.346084e-16
  root Markov S2 eigs=-0.71345185 -0.28654815
  root kappa preserves birth order=False

```

## Main positive findings

### 1. Full local directed Markov response has a complex pair

For all tested update modes:

```text
frac Markov complex = 1.000
```

So the **full local directed response operator**, not only the selected ideal forward `Z3` cycle, has complex eigenvalues on every completed sibling triple.

### 2. Controls work

The controls stay negative:

```text
path without closure complex = 0.000
symmetrized raw complex      = 0.000
```

Interpretation:

```text
Birth-order path alone:
  no complex sector

Symmetrized response:
  no complex sector

Full directed birth/backreaction response:
  complex sector
```

This is exactly the separation we needed.

### 3. Projected local skew part gives `J² ≈ -I`

For all modes:

```text
frac Markov J2 ok = 1.000
mean Markov J2 residual ~ 1e-16
```

This means that the projected skew part on the local 2D sibling sector behaves like a normalized complex structure.

### 4. κ-test is positive at the projected level

For all modes:

```text
frac Markov kappa flip ok = 1.000
frac kappa preserves birth order = 0.000
```

So, in this surrogate:

```text
κ J κ⁻¹ = -J
```

while κ does not preserve the irreversible birth-time history.

## Main caution

The full skew operator does **not** leave the `sum=0` sector exactly invariant.

The mean leakage values at level 6 are:

```text
linear:      0.0619
log:         0.0309
saturating:  0.1331
```

The skew-axis alignment with the constant vector is high but not perfect:

```text
linear:      0.8489
log:         0.9315
saturating:  0.8689
```

Therefore the honest interpretation is:

```text
The projected local J test is positive,
but a closed full-sector J operator is not yet proven.
```

This matters because any 2x2 nonzero skew matrix can be normalized to a `J`.
The nontrivial extra requirement is that the relevant 2D sector be canonically selected and invariant under the full operator. That is not fully closed yet.

## Symmetric part / metric status

The symmetric part on the 2D sector is sign-definite in all tested triples:

```text
frac Markov S2 sign-definite = 1.000
```

For the root examples, `S2` is negative definite:

```text
linear:      -0.6288, -0.3712
log:         -0.6010, -0.3990
saturating:  -0.7135, -0.2865
```

This suggests that `-S2` could act as a positive local metric candidate. But this sign choice is not derived here and must not be declared physical yet.

## Interpretation

This is the strongest positive operator-level surrogate so far:

```text
sequential birth
+ ancestor/sibling backreaction
+ full directed local response
→ complex local Markov operator
→ projected skew J with J² ≈ -I
→ κ flips J while κ violates birth history
```

But the result is still conditional because the full invariant sector and physical metric are not yet derived.

## What this test shows

It supports the hypothesis that the dynamic response layer can carry a local operator-level rotation candidate, not merely scalar circulation or topological H1.

## What it does not show

Still open:

1. exact invariant subspace of the full effective operator,
2. derived metric/weight `G`,
3. `J = G⁻¹ A` with `J² = -I` without projection artifact,
4. compatibility between local triples across levels,
5. Lean formalization,
6. algebra tower / Type III / modular flow.

## Next required step

No more graph/cochain-only tests should be added before this is addressed.

The next test should be:

```text
test_effective_operator_sector_closure.py
```

It must test:

1. whether the full local operator has a canonical invariant 2D sector,
2. whether the leakage goes to zero or stabilizes under level growth,
3. whether `-S2` can be used as a derived positive metric candidate,
4. whether `J = G⁻¹ A` satisfies `J² ≈ -I` without arbitrary projection,
5. whether adjacent triples glue compatibly into a tower operator.

Only after that should Lean receive a minimal formal statement, probably not yet `J derived`, but:

```text
DirectedBirthResponseOperatorHasProjectedRotationalSector
```

or, if the next test succeeds:

```text
DirectedBirthResponseOperatorHasInvariantLocalJPlane
```
