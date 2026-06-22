# SUMMARY

Package: `apply_top_ranked_non_shelling_move_and_reaudit_pkg`

This package applies top-ranked shelling/cap/handle/quotient candidates and then recomputes Betti numbers and K exactness/harmonic projection.

Parameters:

```json
{
  "max_boundary_faces": 20,
  "max_pair_candidates": 100,
  "max_rows": 600
}
```

Base real-growth topology:

```json
{
  "vertices": 40,
  "edges": 114,
  "faces": 112,
  "tets": 37,
  "beta0": 1,
  "beta1": 0,
  "beta2": 0,
  "beta3": 0,
  "boundary_fraction": 0.6785714285714225,
  "K_mean": 0.9095255451534915,
  "exact_residual_ratio": 0.26722530405981093,
  "harmonic_ratio": 0.0
}
```

Main result:

- Shelling/cap moves do not create a harmonic sector.
- The top-ranked real-growth handle creates `β2=1` and `harmonic_ratio≈0.00796`.
- A topology-ranked quotient creates `β2=1` and `harmonic_ratio≈0.0215`.
- Symmetrized/no-backreaction controls also create β2 under non-shelling moves, so the move class is necessary, but the selection rule is not yet uniquely derived from CNNA.
