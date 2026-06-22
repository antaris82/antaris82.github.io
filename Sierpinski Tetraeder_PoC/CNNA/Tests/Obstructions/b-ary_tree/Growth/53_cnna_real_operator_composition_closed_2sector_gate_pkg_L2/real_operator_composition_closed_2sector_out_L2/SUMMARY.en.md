# SUMMARY — real operator composition from a closed 2-sector gate

## Model label

CNNA deterministic growing primal simplicial complex; provenance tree as birth history; real closed/harmonic 2-sector operator composition diagnostic; NGF/CQNM used only for comparison

## Anti-smuggling constraint

No i, J, Hodge star, positivity axiom, C*-norm, spin structure, Fourier sign, branch cut, upper/lower half-plane convention, or external orientation package is used as input. The candidate sharp map is only a real coefficient-dual plus cap/pair reversal diagnostic; it is not claimed to be a derived C*-adjoint.

## Gate question

The previous package found a real 2-cochain `K ∈ C²` with a strongly closed component, a nonzero harmonic residual, and a controlled tetrahedral defect `δK ∈ C³`.  This package asks the next, narrower question:

```text
Can the closed/harmonic real 2-sector plus cap/pair transport logs generate a small real operator family that is stable under composition and under a candidate involution-like reversal?
```

The candidate sharp map is:

```text
A^# := R_pair A^T R_pair
```

where `R_pair` is the real cap/pair reversal map on face-cochains and `A^T` is coefficient-dualization in the finite face basis.  This is a diagnostic anti-involution, not a C*-adjoint and not a positivity/norm claim.

## Comparative table

| variant | beta | H2 | K harm | K defect | pairings | carrier | R nontriv | init dim | init prod resid | init sharp resid | alg dim | alg sharp resid | used Δβ? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth_live_pair_plus_response | (1,0,2,0) | 2 | 0.227627 | 0.117849 | 2 | 6 | True | 8 | 0.919411 | 0.930752 | 28 | 1 | False |
| real_growth_record_only_pair_plus_response | (1,0,2,0) | 2 | 0.227353 | 0.117707 | 2 | 6 | True | 8 | 0.919502 | 0.930759 | 28 | 1 | False |
| real_growth_record_plus_live_pair_plus_response | (1,0,2,0) | 2 | 0.227491 | 0.117778 | 2 | 6 | True | 8 | 0.919458 | 0.930756 | 28 | 1 | False |
| real_growth_live_pair_only | (1,0,2,0) | 2 | 0.96576 | 0.5 | 2 | 5 | True | 7 | 0.741492 | 0.924818 | 11 | 0.936743 | False |
| strict_symmetrized_control | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | False |
| no_backreaction_record_pair_plus_response | (1,0,2,0) | 2 | 0.227353 | 0.117707 | 2 | 6 | True | 8 | 0.919502 | 0.930759 | 28 | 1 | False |

## Conservative reading

- `real_growth` variants have β₂ = 2 and a nontrivial pair-reversal map.
- The initial hand-built generator span is generally not product-closed; this is a real obstacle to claiming an immediate small operator system.
- The finitely generated real algebra closes after adding products, but its dimension is larger than that of the initial seed span.  This is expected and should not be overinterpreted.
- Stability under `#` is merely a compatibility diagnostic.  In the current L2 output, the generated product algebra is not `#`-stable, so this is an obstruction rather than a positive `*` result.
