# RESULTS — 2-form closure and 3-form defect gate

## Comparative table

| variant | source | mode | beta | pairings | K closed | K exact | K harmonic | δK defect | H² dim | |K| harmonic | used Δβ? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth_live_pair_plus_response | live | pair_plus_response | (1,0,2,0) | 2 | 0.998129 | 0.971826 | 0.227627 | 0.117849 | 2 | 0.155663 | False |
| real_growth_record_only_pair_plus_response | record | pair_plus_response | (1,0,2,0) | 2 | 0.998133 | 0.971895 | 0.227353 | 0.117707 | 2 | 0.156418 | False |
| real_growth_record_plus_live_pair_plus_response | full | pair_plus_response | (1,0,2,0) | 2 | 0.998131 | 0.971861 | 0.227491 | 0.117778 | 2 | 0.156045 | False |
| real_growth_live_pair_only | live | pair_only | (1,0,2,0) | 2 | 0.96576 | 3.43175e-16 | 0.96576 | 0.5 | 2 | 0.0696944 | False |
| strict_symmetrized_control | record | pair_plus_response | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| no_backreaction_record_pair_plus_response | record | pair_plus_response | (1,0,2,0) | 2 | 0.998133 | 0.971895 | 0.227353 | 0.117707 | 2 | 0.156418 | False |

## Interpretation by gate

```text
strict_symmetrized_control:
  should kill β₂ and K-support.

real_growth_*:
  checks whether nonlinear asymmetry-gated cap/pair carriers produce β₂ and H² support.

no_backreaction_record:
  tests whether the carrier depends on live backreaction or already follows from sequential provenance asymmetry.

δK defect:
  if large, K is not a clean closed 2-form; the obstruction lives in the tetrahedral 3-form defect.
```

## Anti-smuggling checks

- `decision_used_delta_beta_any` must remain false.
- Orientation is treated only as a finite cochain gauge needed to compute incidence matrices.
- Sign-flip diagnostics compare scalar ratios under `K -> -K`.
- Relabel diagnostics compare the same scalar ratios after a deterministic vertex relabeling.
- No `i`, `J`, Hodge star, positivity, norm-as-axiom, branch cut, logarithm, square root, spin, or Fourier convention is used as input.

## Files

- `comparative_2form_closure_3form_defect_summary.csv`
- `comparative_summary.json`
- per-variant `variant_2form_closure_3form_defect_summary.json`
- per-variant `top_2form_faces.csv`
- per-variant `top_3form_defects.csv`
- per-variant `pairing_cap_log.csv`

## Next test

`test_real_operator_composition_from_closed_2sector_gate.py`: Construct the smallest family of real operators from the closed/harmonic 2-sector and pair/cap transport records, then test for closure under composition and for a candidate involution without introducing `*`, positivity, or a C*-norm.
