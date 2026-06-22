# SUMMARY — harmonic K orientation / kappa gate

This package builds on the result regarding nonlinear asymmetry-cascade growth and addresses the following question:

```text
beta2 opened -> harmonic K exists;
but does the harmonic K-component carry an oriented kappa/J-like bias?
```

It computes a vector-valued harmonic projection of the skew operator field.  The scalar harmonic ratio from the previous package used Frobenius norms on faces; this package also projects the axial vectors of the skew matrices into the harmonic 2-cochain space.

| variant | beta auto | pairings | scalar harmonic | axial harmonic | Hdim | kappa out | kappa birth | signed out | signed birth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth | (1,0,2,0) | 2 | 0.121617 | 1.61921e-16 | 2 | 0.0611243 | 0.0611243 | -4.69181e-05 | -4.69181e-05 |
| strict_symmetrized_control | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| no_backreaction | (1,0,2,0) | 2 | 0.128934 | 1.69633e-16 | 2 | 0.0397181 | 0.0397181 | -2.77285e-05 | -2.77285e-05 |

A positive result requires β₂ > 0, scalar and axial harmonic components > 0, strict symmetry to be negative, and preferably a nonzero signed orientation bias. The signed κ metrics are diagnostic only; the face orientation convention is still a model choice and not yet a theorem.
