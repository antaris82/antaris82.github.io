# RESULTS

## Verdict

The test improves upon the previous package by making the boundary-face saturation decision depend on local Response/DtN/backreaction data rather than on a generic closure heuristic.

The test is positive only in a limited sense if response-forced saturation reduces the boundary and creates more edge-link cycles while preserving a nonzero operator K. It is a strong Stage-4 candidate only if it also creates nontrivial global Betti support and a non-negligible harmonic/non-exact K component.

## Numerical result

```json
{
  "test_name": "test_cqnm_response_forced_face_saturation",
  "max_level": 2,
  "mode": "linear",
  "response_forced_saturation_passes": 1,
  "growth": [
    {
      "control": "real_growth",
      "nodes": 13,
      "birth_events": 12,
      "completed_parents": 4,
      "directed_edges": 66,
      "mean_abs_neutral_current": 0.537191262144539,
      "mean_abs_neutral_birth": 0.6522710744236213,
      "mean_abs_cycle_log_bias": 1.1415193272507314
    },
    {
      "control": "symmetrized_birth",
      "nodes": 13,
      "birth_events": 12,
      "completed_parents": 4,
      "directed_edges": 54,
      "mean_abs_neutral_current": 0.05857458171392886,
      "mean_abs_neutral_birth": 0.026258202919723164,
      "mean_abs_cycle_log_bias": 24.60620557235653
    },
    {
      "control": "no_backreaction",
      "nodes": 13,
      "birth_events": 12,
      "completed_parents": 4,
      "directed_edges": 66,
      "mean_abs_neutral_current": 0.5908272712745323,
      "mean_abs_neutral_birth": 0.5908272712745323,
      "mean_abs_cycle_log_bias": 26.03462955613689
    }
  ],
  "primary": [
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "live",
      "reduction": "full",
      "vertices": 13,
      "edges": 24,
      "faces": 16,
      "tets": 4,
      "euler": 1,
      "beta0": 1,
      "beta1": 0,
      "beta2": 0,
      "beta3": 0,
      "boundary_faces": 16,
      "saturated_faces": 0,
      "overfull_faces": 0,
      "boundary_fraction": 0.9999999999999376,
      "saturated_fraction": 0.0,
      "manifold_face_fraction": 0.9999999999999376,
      "edge_links_checked": 0,
      "edge_link_cycle_count": 0,
      "edge_link_cycle_fraction": 0.0,
      "mean_edge_link_cycle_length": 0.0,
      "face_count": 16,
      "K_mean": 0.44048791004617927,
      "K_p95": 0.758610369231475,
      "exact_residual_ratio": 3.819547757476847e-16,
      "closed_residual_ratio": 1.0337200626516613e-16,
      "harmonic_ratio": 3.4772275771570904e-16,
      "link_flux_mean": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "live",
      "reduction": "full",
      "vertices": 13,
      "edges": 33,
      "faces": 31,
      "tets": 10,
      "euler": 1,
      "beta0": 1,
      "beta1": 0,
      "beta2": 0,
      "beta3": 0,
      "boundary_faces": 22,
      "saturated_faces": 9,
      "overfull_faces": 0,
      "boundary_fraction": 0.7096774193548159,
      "saturated_fraction": 0.29032258064515193,
      "manifold_face_fraction": 0.9999999999999678,
      "edge_links_checked": 15,
      "edge_link_cycle_count": 0,
      "edge_link_cycle_fraction": 0.0,
      "mean_edge_link_cycle_length": 0.0,
      "face_count": 31,
      "K_mean": 0.9770644971078658,
      "K_p95": 2.1304021144874694,
      "exact_residual_ratio": 1.2885778510050482e-15,
      "closed_residual_ratio": 1.0626905251738204e-16,
      "harmonic_ratio": 1.1623840508043762e-15,
      "link_flux_mean": 2.334310989221746
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "live",
      "reduction": "full",
      "vertices": 13,
      "edges": 42,
      "faces": 50,
      "tets": 20,
      "euler": 1,
      "beta0": 1,
      "beta1": 0,
      "beta2": 0,
      "beta3": 0,
      "boundary_faces": 20,
      "saturated_faces": 30,
      "overfull_faces": 0,
      "boundary_fraction": 0.399999999999992,
      "saturated_fraction": 0.599999999999988,
      "manifold_face_fraction": 0.99999999999998,
      "edge_links_checked": 34,
      "edge_link_cycle_count": 12,
      "edge_link_cycle_fraction": 0.3529411764705778,
      "mean_edge_link_cycle_length": 3.9166666666666665,
      "face_count": 50,
      "K_mean": 0.9271790660003245,
      "K_p95": 2.0505732150959606,
      "exact_residual_ratio": 2.1712396766787033e-15,
      "closed_residual_ratio": 1.0240611409141045e-16,
      "harmonic_ratio": 2.182442584421088e-15,
      "link_flux_mean": 1.9597994534113132
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "live",
      "reduction": "full",
      "vertices": 13,
      "edges": 47,
      "faces": 59,
      "tets": 24,
      "euler": 1,
      "beta0": 1,
      "beta1": 0,
      "beta2": 0,
      "beta3": 0,
      "boundary_faces": 22,
      "saturated_faces": 37,
      "overfull_faces": 0,
      "boundary_fraction": 0.37288135593219707,
      "saturated_fraction": 0.627118644067786,
      "manifold_face_fraction": 0.999999999999983,
      "edge_links_checked": 38,
      "edge_link_cycle_count": 14,
      "edge_link_cycle_fraction": 0.3684210526315692,
      "mean_edge_link_cycle_length": 4.0,
      "face_count": 59,
      "K_mean": 0.992767467562472,
      "K_p95": 2.0616274612741665,
      "exact_residual_ratio": 2.4772936611640315e-15,
      "closed_residual_ratio": 1.2339198254477965e-16,
      "harmonic_ratio": 2.512806214725903e-15,
      "link_flux_mean": 2.178947115469769
    }
  ],
  "gate": {
    "boundary_reduced_vs_outward_ngf": true,
    "creates_more_edge_link_cycles": true,
    "operator_K_nonzero": true,
    "harmonic_rest_nonzero_threshold_1e_minus_3": false,
    "nontrivial_global_betti": false,
    "cleaner_or_equal_boundary_than_random_control": false,
    "real_growth_K_ge_sym_no_backreaction": true,
    "stage4_candidate_strong": false,
    "interpretation": "Positive local/saturation gate only if boundary/link/operator conditions hold. Strong Stage-4 requires nontrivial Betti plus harmonic rest; otherwise the closure law remains insufficient."
  },
  "decision_row_count": 134,
  "kill_controls": [
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "parent_fan_tetra",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_outward_ngf",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "real_growth",
      "geometry": "random_saturation_control",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "parent_fan_tetra",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_outward_ngf",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "response_forced_sminus1_saturation",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "symmetrized_birth",
      "geometry": "random_saturation_control",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "parent_fan_tetra",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_outward_ngf",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "response_forced_sminus1_saturation",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "record",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "record",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "live",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "live",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "handoff",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "handoff",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "aging",
      "reduction": "diagonal",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    },
    {
      "control": "no_backreaction",
      "geometry": "random_saturation_control",
      "source": "aging",
      "reduction": "trace_scalar",
      "K_remaining_fraction": 0.0,
      "harmonic_remaining_fraction": 0.0,
      "exact_residual_remaining_fraction": 0.0
    }
  ]
}
```

## Interpretation discipline

Interpret `stage4_candidate_strong=false` as a genuine obstacle, not as a failure of the test. It means that the response-forced local saturation law is not yet sufficient to produce a non-exact global carrier. This would localize the next missing ingredient to the closure dynamics or topological growth rule, rather than to the tree-provenance mechanism.

Interpret `stage4_candidate_strong=true` only as a candidate gate at the Python level, not as a CNNA theorem. It would still require replacing the surrogate DtN operators with the full dynamic DtN pipeline and, later, a Lean formalization of the discrete growth law.
