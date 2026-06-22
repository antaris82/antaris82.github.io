# SUMMARY

Package: `test_cqnm_response_forced_face_saturation.py`

Purpose: Replace the previous heuristic closure pass with a response-forced active-face/saturation rule. The tree retains its provenance, not its space. The root is located inside; vertices grow outward with a transverse offset based on sibling order. Boundary faces are selected based on local response quantities: live DtN-like K-norm, directed imbalance, aging/backreaction, conductance spread, parent-face coupling, and candidate operator mismatch.

## Model status

This is still a Python diagnostic surrogate, not a Lean proof and not a derived CNNA theorem. The crucial improvement is that the closure choice is no longer purely geometric or random; it is forced by locally available response and provenance data.

## Primary real-growth/live/full results

- parent_fan_tetra: V/E/F/T=13/24/16/4, boundary=1.000, saturated=0.000, beta=(1,0,0,0), edge_link_cycles=0.000, K_mean=0.440488, harmonic=3.47723e-16, exact_res=3.81955e-16, closed_res=1.03372e-16
- response_forced_outward_ngf: V/E/F/T=13/33/31/10, boundary=0.710, saturated=0.290, beta=(1,0,0,0), edge_link_cycles=0.000, K_mean=0.977064, harmonic=1.16238e-15, exact_res=1.28858e-15, closed_res=1.06269e-16
- response_forced_sminus1_saturation: V/E/F/T=13/42/50/20, boundary=0.400, saturated=0.600, beta=(1,0,0,0), edge_link_cycles=0.353, K_mean=0.927179, harmonic=2.18244e-15, exact_res=2.17124e-15, closed_res=1.02406e-16
- random_saturation_control: V/E/F/T=13/47/59/24, boundary=0.373, saturated=0.627, beta=(1,0,0,0), edge_link_cycles=0.368, K_mean=0.992767, harmonic=2.51281e-15, exact_res=2.47729e-15, closed_res=1.23392e-16

## Gate evaluation

- boundary_reduced_vs_outward_ngf: True
- creates_more_edge_link_cycles: True
- operator_K_nonzero: True
- harmonic_rest_nonzero_threshold_1e_minus_3: False
- nontrivial_global_betti: False
- cleaner_or_equal_boundary_than_random_control: False
- real_growth_K_ge_sym_no_backreaction: True
- stage4_candidate_strong: False
- interpretation: Positive local/saturation gate only if boundary/link/operator conditions hold. Strong Stage-4 requires nontrivial Betti plus harmonic rest; otherwise, the closure law remains insufficient.

## Growth controls

- real_growth: nodes=13, completed=4, directed_edges=66, neutral_current=0.537191, cycle_log_bias=1.14152
- symmetrized_birth: nodes=13, completed=4, directed_edges=54, neutral_current=0.0585746, cycle_log_bias=24.6062
- no_backreaction: nodes=13, completed=4, directed_edges=66, neutral_current=0.590827, cycle_log_bias=26.0346
