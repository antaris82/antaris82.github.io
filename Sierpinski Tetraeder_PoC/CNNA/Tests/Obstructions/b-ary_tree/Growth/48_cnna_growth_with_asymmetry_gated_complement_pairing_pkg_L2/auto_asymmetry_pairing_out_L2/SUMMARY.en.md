# SUMMARY — growth with asymmetry-gated complement pairing

This package tests the previously missing step directly within primal geometry growth:

```text
ordinary outward birth
-> boundary scan
-> if a nonlocal boundary-face pair passes the asymmetry-invariant complement gate
   and the resulting move is manifold-legal:
      apply the non-shelling complement pairing immediately
-> otherwise continue ordinary outward NGF/CQNM-like attachment
```

The decision gate uses A-invariant provenance/response quantities and transverse complementarity. It deliberately does **not** use delta beta in the decision. Delta beta is logged only after application for audit purposes.

| variant | baseline beta | auto beta | pairings | auto harmonic | opened beta2 | nonexact K |
|---|---:|---:|---:|---:|---:|---:|
| real_growth | (1,0,0,0) | (1,0,2,0) | 1 | 0.172295 | True | True |
| historical_symmetrized_birth | (1,0,0,0) | (1,0,1,0) | 1 | 0.12891 | True | True |
| strict_symmetrized_control | (1,0,0,0) | (1,0,0,0) | 0 | 0 | False | False |
| no_backreaction | (1,0,0,0) | (1,0,2,0) | 1 | 0.148596 | True | True |

Interpretation: A positive result requires beta2 to open and the K-field to acquire a nonzero harmonic component during the automatic growth run itself. If beta2 opens only during actual growth but not during strict symmetry control, the selective complement-pairing mechanism is no longer merely a post-hoc candidate explanation. If strict controls also cause beta2 to open, the move class is still too generic.
