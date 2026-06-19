# RESULTS — Effective Geometry Closure Test

## Status

Python diagnostic completed successfully to **level 7**.

This is a model/surrogate test, not a Lean theorem and not a physical/operator-algebra claim.

## Purpose

The previous dynamic birth-response test showed that sequential birth with ancestor and sibling backreaction creates:

- unequal sibling response weights,
- nonzero neutral phasor,
- nonzero local log-circulation,
- complex local Markov monodromy.

This test asks whether that response-layer monodromy also appears in an **extracted effective simplicial geometry**, and if so whether it appears as:

1. topological `H1`,
2. local 2-form curvature on filled faces,
3. or only as response-layer transport without geometric closure.

## Critical guardrails

- Conductance/response weights are **not** geometric edge lengths.
- The effective simplicial geometry still uses equal combinatorial edge lengths.
- A nonzero response curvature is not yet `J`.
- A nonzero Betti number is not yet a complex structure.
- This test does not reach Type III, vN algebra, nuclearity, or modular theory.

## Geometry extractors tested

For each dynamic mode, five extraction modes were tested:

1. `radial_tree`: parent-child edges only.
2. `sibling_cycle_unfilled`: parent-child edges plus sibling 3-cycle edges, no sibling face.
3. `sibling_triangle_filled`: sibling 3-cycle plus filled sibling triangle.
4. `parent_fan_filled`: parent-child edges, sibling cycle edges, and parent fan faces `(parent, child_i, child_j)`.
5. `full_local_surface`: parent fan faces plus sibling face.

## Level-7 primary output

```text
MODE linear
  radial_tree: V=3280 E=3279 F=0 b1=0 b2=0 mean|face_curv|=0.000000 frac face curv!=0=0.000 mean|sibling_curv|=0.665658 full-local complex=1.000
  sibling_cycle_unfilled: V=3280 E=6558 F=0 b1=3279 b2=0 mean|face_curv|=0.000000 frac face curv!=0=0.000 mean|sibling_curv|=0.665658 full-local complex=1.000
  sibling_triangle_filled: V=3280 E=6558 F=1093 b1=2186 b2=0 mean|face_curv|=0.665658 frac face curv!=0=1.000 mean|sibling_curv|=0.665658 full-local complex=1.000
  parent_fan_filled: V=3280 E=6558 F=3279 b1=0 b2=0 mean|face_curv|=0.744179 frac face curv!=0=1.000 mean|sibling_curv|=0.665658 full-local complex=1.000
  full_local_surface: V=3280 E=6558 F=4372 b1=0 b2=1093 mean|face_curv|=0.724549 frac face curv!=0=1.000 mean|sibling_curv|=0.665658 full-local complex=1.000
  root g=7.10429108 7.45467425 7.89091754
  root sibling curvature=1.379981
  root neutral |Z|=0.682590, phase=-146.394 deg
  root full-local Markov=complex_pair, eigs=1+0j -0.5+0.132389657j -0.5-0.132389657j

MODE log
  radial_tree: V=3280 E=3279 F=0 b1=0 b2=0 mean|face_curv|=0.000000 frac face curv!=0=0.000 mean|sibling_curv|=0.510948 full-local complex=1.000
  sibling_cycle_unfilled: V=3280 E=6558 F=0 b1=3279 b2=0 mean|face_curv|=0.000000 frac face curv!=0=0.000 mean|sibling_curv|=0.510948 full-local complex=1.000
  sibling_triangle_filled: V=3280 E=6558 F=1093 b1=2186 b2=0 mean|face_curv|=0.510948 frac face curv!=0=1.000 mean|sibling_curv|=0.510948 full-local complex=1.000
  parent_fan_filled: V=3280 E=6558 F=3279 b1=0 b2=0 mean|face_curv|=0.692698 frac face curv!=0=1.000 mean|sibling_curv|=0.510948 full-local complex=1.000
  full_local_surface: V=3280 E=6558 F=4372 b1=0 b2=1093 mean|face_curv|=0.647261 frac face curv!=0=1.000 mean|sibling_curv|=0.510948 full-local complex=1.000
  root g=3.93661981 3.99956319 4.03314615
  root sibling curvature=1.238490
  root neutral |Z|=0.084873, phase=-159.960 deg
  root full-local Markov=complex_pair, eigs=1+0j -0.5+0.138051716j -0.5-0.138051716j

MODE saturating
  radial_tree: V=3280 E=3279 F=0 b1=0 b2=0 mean|face_curv|=0.000000 frac face curv!=0=0.000 mean|sibling_curv|=1.912531 full-local complex=1.000
  sibling_cycle_unfilled: V=3280 E=6558 F=0 b1=3279 b2=0 mean|face_curv|=0.000000 frac face curv!=0=0.000 mean|sibling_curv|=1.912531 full-local complex=1.000
  sibling_triangle_filled: V=3280 E=6558 F=1093 b1=2186 b2=0 mean|face_curv|=1.912531 frac face curv!=0=1.000 mean|sibling_curv|=1.912531 full-local complex=1.000
  parent_fan_filled: V=3280 E=6558 F=3279 b1=0 b2=0 mean|face_curv|=2.109853 frac face curv!=0=1.000 mean|sibling_curv|=1.912531 full-local complex=1.000
  full_local_surface: V=3280 E=6558 F=4372 b1=0 b2=1093 mean|face_curv|=2.060523 frac face curv!=0=1.000 mean|sibling_curv|=1.912531 full-local complex=1.000
  root g=5.00968884 5.14906798 5.17597572
  root sibling curvature=2.782875
  root neutral |Z|=0.154599, phase=-171.331 deg
  root full-local Markov=complex_pair, eigs=1+0j -0.5+0.212180858j -0.5-0.212180858j

```

## Interpretation

### 1. Radial tree

`radial_tree` has:

```text
b1 = 0
b2 = 0
```

So the pure parent-child provenance tree has no geometric cycles. It still has nonzero response sibling curvature because the response layer is stored independently of geometric faces.

Conclusion:

```text
Pure radial provenance alone gives no topological holonomy.
```

### 2. Sibling cycle, unfilled

`sibling_cycle_unfilled` has large `b1`:

```text
level 7: b1 = 3279
```

This is expected because every completed sibling triple contributes an unfilled 3-cycle. This produces topological `H1`, but this is a deliberately unfilled geometry. It is a useful control, not yet the target geometry.

Conclusion:

```text
Unfilled sibling cycles create H1, but this may be an artifact of not filling local simplexes.
```

### 3. Sibling triangle filled

`sibling_triangle_filled` reduces `H1` but does not remove all `H1`:

```text
level 7: b1 = 2186
```

The sibling face carries nonzero response curvature:

```text
frac face curvature != 0 = 1.000
```

Conclusion:

```text
The local sibling monodromy survives as 2-form curvature on filled sibling faces.
Residual H1 remains due to radial-parent cycles not filled by this extractor.
```

### 4. Parent fan filled

`parent_fan_filled` has:

```text
b1 = 0
b2 = 0
frac face curvature != 0 = 1.000
```

This is structurally important. It kills topological 1-cycles but retains nonzero local face curvature.

Conclusion:

```text
The response monodromy does not require global H1 in this extractor.
It appears as local curvature on filled 2-simplices.
This is closer to a gauge-curvature picture than to pure H1 holonomy.
```

### 5. Full local surface

`full_local_surface` has:

```text
b1 = 0
b2 = 1093
```

The extra sibling face plus fan faces creates local 2-cycles/surface closures. Face curvature remains nonzero.

Conclusion:

```text
The effective geometry can carry nontrivial H2-like local surface structure,
but this is not yet a proof of physical/operator J.
```

## Main finding

The response-layer complex monodromy is not only a selected Z3 artifact. In all tested update modes, the **full local Markov response matrix** has a complex pair for all completed triples:

```text
frac full-local Markov complex = 1.000
```

At the same time, the topological interpretation depends strongly on the geometry extractor:

```text
radial tree:
  no H1/H2

unfilled sibling cycles:
  H1

filled triangles / fan:
  response appears as local face curvature

full local surface:
  H2-like closure plus curvature
```

## What this shows

The dynamic birth/backreaction model creates a robust local curvature/monodromy signal.

It is not erased by extracting effective simplicial geometry, but it may appear as:

- topological H1 if cycles are unfilled,
- local 2-form curvature if faces are filled,
- H2/surface closure in the full local surface extractor.

## What this does not show

Still not shown:

1. a Lean theorem,
2. a unique derived geometry extractor,
3. `J² = -I` for a full network operator,
4. a compatible metric/symmetric form `G` and skew form `A`,
5. Type III, vN algebra, nuclearity, or modular flow.

## Next required step

The next diagnostic must stop being only graph/cochain geometry and move toward **effective operators**, as requested.

Recommended next test:

```text
test_effective_operator_J_candidate.py
```

For each geometry extractor:

1. build an operator from the response cochain/curvature,
2. split symmetric and skew parts:
   S = (M + Mᵀ)/2
   A = (M - Mᵀ)/2
3. restrict to the nontrivial local/soft sector,
4. test whether a normalized candidate
   J = G⁻¹ A
   satisfies
   J² ≈ -α² I
5. run κ-test:
   κ J κ⁻¹ = -J?
6. check whether the result is stable with level.

This is the last stage before any operator-algebra/vN discussion. Type-III claims remain forbidden until an algebra tower and compatible state/weight are defined.
