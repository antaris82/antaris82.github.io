# SUMMARY — candidate sharp-closure obstruction gate

## Model label

CNNA deterministic growing primal simplicial complex; provenance tree as birth history; candidate sharp-closure obstruction diagnostic on the real C2 carrier; NGF/CQNM used only for comparison, not as a source of derivation

## Anti-smuggling constraint

No i, J, Hodge star, C*-adjoint, positivity axiom, norm axiom, spin structure, Fourier sign, branch cut, upper/lower half-plane convention, or external orientation package is used as input. The map A# = R A^T R is only a real finite-coefficient reversal diagnostic. It is not claimed to be a derived star operation.

## Gate question

The previous package found that the generated real product algebra on the C² carrier is product-closed after adding products, but is not stable under the candidate reversal

```text
A# = R_pair A^T R_pair.
```

This package localizes the obstruction by splitting generator families:

```text
{I,R,M_closed,M_harmonic}
cap maps only
pair maps only
signed vs unsigned R
record/live/full
forced #-closure
```

## Focus table: real_growth_live_pair_plus_response with signed R

| family | gen | init dim | init prod resid | init # resid | prod alg dim | full M? | prod alg # resid | forced dim | extra dim | forced closed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diag_core | 4 | 4 | 0.499154 | 0.0224152 | 36 | True | 1.53496e-15 | 36 | 0 | True |
| cap_only | 6 | 6 | 0.433013 | 0.9398 | 7 | False | 0.959983 | 17 | 10 | True |
| pair_only | 4 | 4 | 0.707107 | 8.61732e-16 | 5 | False | 1.57532e-15 | 5 | 0 | True |
| all_seed | 10 | 10 | 0.893809 | 0.918548 | 36 | True | 2.2629e-15 | 36 | 0 | True |

## Conservative reading

- `diag_core` is the smallest direct test of the closed/harmonic diagonal 2-sector plus pair-reversal.
- `cap_only` isolates cap-boundary rank-one maps.
- `pair_only` isolates pair transport/swap maps.
- `forced_sharp_product_dim` measures the additional algebraic material required if one insists on # stability.
- A small forced extra dimension is a hint of compatibility; a large jump indicates that the candidate # is not native to the seed family.
- Full-matrix saturation is not a criterion for success: it means that # compatibility was achieved by expanding to the entire finite carrier algebra, which is too nonselective for a derived minimal structure.

No row is a `*` derivation.  The only legitimate claim is compatibility with or obstruction of this finite real reversal diagnostic.
