# SUMMARY — directed antisymmetric birth transport through pairing/inter-fan gate

This package combines the missing pieces from the recent sequence:

```text
51: asymmetry-gated beta2 carrier + pairing transport + harmonic projection
55: local birth-order-derived antisymmetric transport operator without final sym(M)
```

The tested operator is real and directed:

```text
M = metric_part + eta * strength * (q⊗h - h⊗q)
```

No final `sym(M)` is applied in the tested path.  The antisymmetric term is derived from ternary birth order; it is not an input `J`, orientation, Hodge star, positivity, norm, spin, or complex scalar.

| variant | beta auto | pairings | raw K harm | raw local coh | raw 3face coh | pair harm | pair kappa | pair birth kappa | pair H coh | interfan residual | used Δβ? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth | (1,0,2,0) | 2 | 1.63004e-16 | 0.202816 | 0.589549 | 0.229731 | 0.0318445 | 0.0318445 | 0.0567699 | 0.785714 | False |
| strict_symmetrized_control | (1,0,0,0) | 0 | 0 | 0.345397 | 0.634491 | 0 | 0 | 0 | 0 | 0 | False |
| no_backreaction | (1,0,2,0) | 2 | 1.71245e-16 | 0.21965 | 0.583538 | 0.223795 | 0.0293466 | 0.0293466 | 0.0535042 | 0.9295 | False |

Interpretation must remain conservative: a positive pair harmonic component is not a J-derivation.  The gate determines whether the local directed birth orientation survives pairing transport and inter-face/3-face coherence checks.
