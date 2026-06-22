# SUMMARY — 2-form closure and 3-form defect gate

## Model label

CNNA deterministic growing primal simplicial complex with provenance/birth-history tracking.  The tree-like birth process is a provenance register, not space.  NGF/CQNM is only a comparison frame.  This package is not SG/ST geometry, not a finished CQNM model, and not a complex/J derivation.

## Gate

This test ensures that the next object is real:

```text
K ∈ C² on triangular faces
δK ∈ C³ on tetrahedra
```

It determines whether the pairing-carried real 2-cochain has a closed/harmonic carrier or whether it has a large local 3-form defect on filled tetrahedra.

## Comparative result

| variant | source | mode | beta | pairings | K closed | K exact | K harmonic | δK defect | H² dim | |K| harmonic | used Δβ? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth_live_pair_plus_response | live | pair_plus_response | (1,0,2,0) | 2 | 0.998129 | 0.971826 | 0.227627 | 0.117849 | 2 | 0.155663 | False |
| real_growth_record_only_pair_plus_response | record | pair_plus_response | (1,0,2,0) | 2 | 0.998133 | 0.971895 | 0.227353 | 0.117707 | 2 | 0.156418 | False |
| real_growth_record_plus_live_pair_plus_response | full | pair_plus_response | (1,0,2,0) | 2 | 0.998131 | 0.971861 | 0.227491 | 0.117778 | 2 | 0.156045 | False |
| real_growth_live_pair_only | live | pair_only | (1,0,2,0) | 2 | 0.96576 | 3.43175e-16 | 0.96576 | 0.5 | 2 | 0.0696944 | False |
| strict_symmetrized_control | record | pair_plus_response | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| no_backreaction_record_pair_plus_response | record | pair_plus_response | (1,0,2,0) | 2 | 0.998133 | 0.971895 | 0.227353 | 0.117707 | 2 | 0.156418 | False |

## Conservative reading

A positive β₂ and a positive harmonic K² support imply only that the generated primal complex carries a real 2-cochain sector.  They do not imply `J`, `i`, a sign of `J`, a Hodge star, a norm, positivity, spin, or a complex structure.

The new structure-package warning is encoded as a test condition: it does not assert that a sign flip is a convention unless all dependent structures are transformed.  This package therefore does not use branch choices, upper/lower half-planes, Fourier sign conventions, positivity, or analytic square-root/logarithm data.
