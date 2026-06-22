# RESULTS

## real_local_only

- growth_rule: `real_growth`
- interfan_strength: `0.0`
- interfan_edge_count: `0`
- interfan_weight_total: `0`
- base beta: `(1,0,0,0)`
- base boundary_fraction: `0.709677`
- base K_mean: `1.17139`
- base harmonic_ratio: `0`

Best handle candidates:

```json
{
  "top_response": {
    "K_pair_norm": 3.2153236474552953,
    "address_similarity": 0.4,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": 4,
    "directed_coupling": 1.4336787734342589,
    "directed_imbalance": 0.6760603330060956,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 1,
    "response_score": 7.7610106763705655,
    "transverse_complementarity": 0.34967541746063685
  },
  "top_topology": {
    "K_pair_norm": 3.2153236474552953,
    "address_similarity": 0.4,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": 4,
    "directed_coupling": 1.4336787734342589,
    "directed_imbalance": 0.6760603330060956,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 1,
    "response_score": 7.7610106763705655,
    "transverse_complementarity": 0.34967541746063685
  }
}
```

Best quotient candidates:

```json
{
  "top_response": {
    "K_pair_norm": 3.2418061609784816,
    "address_similarity": 0.0,
    "candidate_id": 106,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_boundary_faces": -6,
    "directed_coupling": 0.5510511130635062,
    "directed_imbalance": 0.2502940818853022,
    "new_beta1": 0,
    "new_beta2": 0,
    "response_rank_legal": 16,
    "response_score": 5.4953502318419005,
    "transverse_complementarity": 0.7381486494315066
  },
  "top_topology": {
    "K_pair_norm": 1.0343015155169701,
    "address_similarity": 0.0,
    "candidate_id": 274,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": -2,
    "directed_coupling": 0.0,
    "directed_imbalance": 0.0,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 137,
    "response_score": 1.6355606173735298,
    "transverse_complementarity": 0.45136582572959516
  }
}
```

Applied moves:

```json
[
  {
    "after_K_mean": 1.288271395636364,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.23503207939903267,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 2,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 7 9",
    "move_class": "shelling_disk_move",
    "selector": "response",
    "selector_response_rank": 49,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "real_local_only"
  },
  {
    "after_K_mean": 1.2161149283014197,
    "after_beta0": 1,
    "after_beta1": 1,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2680280718682928,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 9,
    "delta_beta1": 1,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "5 7 8 10",
    "move_class": "shelling_disk_move",
    "selector": "topology",
    "selector_response_rank": 110,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "real_local_only"
  },
  {
    "after_K_mean": 1.2419105135297908,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.26564111931091694,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 3,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "response",
    "selector_response_rank": 44,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "real_local_only"
  },
  {
    "after_K_mean": 1.2419105135297908,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.26564111931091694,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 3,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "topology",
    "selector_response_rank": 44,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "real_local_only"
  },
  {
    "after_K_mean": 1.1093274126320307,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666495,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.25944058654372176,
    "after_harmonic_ratio": 0.06305961613225637,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": 4,
    "encoded_move": "0 3 4 7|0 1 4 7|0 1 2 7",
    "move_class": "handle_candidate",
    "selector": "response",
    "selector_response_rank": 1,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "real_local_only"
  },
  {
    "after_K_mean": 1.1093274126320307,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666495,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.25944058654372176,
    "after_harmonic_ratio": 0.06305961613225637,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": 4,
    "encoded_move": "0 3 4 7|0 1 4 7|0 1 2 7",
    "move_class": "handle_candidate",
    "selector": "topology",
    "selector_response_rank": 1,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "real_local_only"
  },
  {
    "after_K_mean": 1.2842569451478447,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7272727272726943,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.40328085708868194,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 106,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": -6,
    "encoded_move": "{\"1\": 7, \"2\": 10, \"4\": 8}",
    "move_class": "quotient_candidate",
    "selector": "response",
    "selector_response_rank": 16,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "real_local_only"
  },
  {
    "after_K_mean": 1.1529473707932538,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666445,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.28768500242613004,
    "after_harmonic_ratio": 0.1494659644599931,
    "candidate_id": 274,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": -2,
    "encoded_move": "{\"1\": 9, \"5\": 8, \"6\": 7}",
    "move_class": "quotient_candidate",
    "selector": "topology",
    "selector_response_rank": 137,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "real_local_only"
  }
]
```

## real_interfan_transport

- growth_rule: `real_growth`
- interfan_strength: `0.16`
- interfan_edge_count: `36`
- interfan_weight_total: `4.5261478`
- base beta: `(1,0,0,0)`
- base boundary_fraction: `0.709677`
- base K_mean: `1.17139`
- base harmonic_ratio: `0`

Best handle candidates:

```json
{
  "top_response": {
    "K_pair_norm": 4.463654607297307,
    "address_similarity": 0.4,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 1,
    "delta_boundary_faces": 2,
    "directed_coupling": 1.3444762358664106,
    "directed_imbalance": 0.5677339137930086,
    "new_beta1": 0,
    "new_beta2": 1,
    "response_rank_legal": 1,
    "response_score": 8.669529273384466,
    "transverse_complementarity": 0.2891184979677439
  },
  "top_topology": {
    "K_pair_norm": 3.2153236474552953,
    "address_similarity": 0.4,
    "candidate_id": 75,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": 4,
    "directed_coupling": 1.4336787734342589,
    "directed_imbalance": 0.6760603330060956,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 3,
    "response_score": 7.7610106763705655,
    "transverse_complementarity": 0.34967541746063685
  }
}
```

Best quotient candidates:

```json
{
  "top_response": {
    "K_pair_norm": 2.436666478687135,
    "address_similarity": 0.0,
    "candidate_id": 88,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_boundary_faces": -6,
    "directed_coupling": 1.6701254872665559,
    "directed_imbalance": 0.35664313570666617,
    "new_beta1": 0,
    "new_beta2": 0,
    "response_rank_legal": 7,
    "response_score": 6.959861826130285,
    "transverse_complementarity": 0.593594254560297
  },
  "top_topology": {
    "K_pair_norm": 1.0343015155169701,
    "address_similarity": 0.0,
    "candidate_id": 266,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": -2,
    "directed_coupling": 0.724597973080927,
    "directed_imbalance": 0.10742603376829585,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 118,
    "response_score": 3.2458956141878277,
    "transverse_complementarity": 0.45136582572959516
  }
}
```

Applied moves:

```json
[
  {
    "after_K_mean": 1.2327137571301654,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2529418674390664,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 7,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "7 8 10 12",
    "move_class": "shelling_disk_move",
    "selector": "response",
    "selector_response_rank": 73,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "real_interfan_transport"
  },
  {
    "after_K_mean": 1.1887816973874288,
    "after_beta0": 1,
    "after_beta1": 1,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.24989346125847742,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 15,
    "delta_beta1": 1,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 5 9",
    "move_class": "shelling_disk_move",
    "selector": "topology",
    "selector_response_rank": 110,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "real_interfan_transport"
  },
  {
    "after_K_mean": 1.2419105135297908,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.26564111931091694,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 3,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "response",
    "selector_response_rank": 62,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "real_interfan_transport"
  },
  {
    "after_K_mean": 1.2419105135297908,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.26564111931091694,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 3,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "topology",
    "selector_response_rank": 62,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "real_interfan_transport"
  },
  {
    "after_K_mean": 1.2960750977083102,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 1,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6315789473684044,
    "after_edge_link_cycle_fraction": 0.045454545454543395,
    "after_exact_residual_ratio": 0.23867823793867618,
    "after_harmonic_ratio": 0.007918564418249273,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 1,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 7 11|4 7 8 11|7 8 10 11",
    "move_class": "handle_candidate",
    "selector": "response",
    "selector_response_rank": 1,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "real_interfan_transport"
  },
  {
    "after_K_mean": 1.1093274126320307,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666495,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.25944058654372176,
    "after_harmonic_ratio": 0.06305961613225637,
    "candidate_id": 75,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": 4,
    "encoded_move": "0 3 4 7|0 1 4 7|0 1 2 7",
    "move_class": "handle_candidate",
    "selector": "topology",
    "selector_response_rank": 3,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "real_interfan_transport"
  },
  {
    "after_K_mean": 1.046897222945521,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7272727272726943,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2879549594705912,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 88,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": -6,
    "encoded_move": "{\"10\": 2, \"11\": 8, \"12\": 3}",
    "move_class": "quotient_candidate",
    "selector": "response",
    "selector_response_rank": 7,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "real_interfan_transport"
  },
  {
    "after_K_mean": 1.1529473707932538,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666445,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.28768500242613004,
    "after_harmonic_ratio": 0.1494659644599931,
    "candidate_id": 266,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": -2,
    "encoded_move": "{\"1\": 9, \"5\": 8, \"6\": 7}",
    "move_class": "quotient_candidate",
    "selector": "topology",
    "selector_response_rank": 118,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "real_interfan_transport"
  }
]
```

## sym_interfan_transport

- growth_rule: `symmetrized_birth`
- interfan_strength: `0.16`
- interfan_edge_count: `36`
- interfan_weight_total: `3.4937045`
- base beta: `(1,0,0,0)`
- base boundary_fraction: `0.709677`
- base K_mean: `0.675647`
- base harmonic_ratio: `0`

Best handle candidates:

```json
{
  "top_response": {
    "K_pair_norm": 2.9334075519939034,
    "address_similarity": 0.4,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 1,
    "delta_boundary_faces": 2,
    "directed_coupling": 1.0384230792553093,
    "directed_imbalance": 0.3929307732177187,
    "new_beta1": 0,
    "new_beta2": 1,
    "response_rank_legal": 1,
    "response_score": 6.264971193995924,
    "transverse_complementarity": 0.2891184979677439
  },
  "top_topology": {
    "K_pair_norm": 2.109517739167553,
    "address_similarity": 0.4,
    "candidate_id": 71,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": 4,
    "directed_coupling": 1.205873938058854,
    "directed_imbalance": 0.5983613893781115,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 2,
    "response_score": 6.083046681890037,
    "transverse_complementarity": 0.34967541746063685
  }
}
```

Best quotient candidates:

```json
{
  "top_response": {
    "K_pair_norm": 1.000672120176067,
    "address_similarity": 0.0,
    "candidate_id": 92,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_boundary_faces": -6,
    "directed_coupling": 1.4083030747582899,
    "directed_imbalance": 0.43067792783085457,
    "new_beta1": 0,
    "new_beta2": 0,
    "response_rank_legal": 9,
    "response_score": 5.111274830788968,
    "transverse_complementarity": 0.593594254560297
  },
  "top_topology": {
    "K_pair_norm": 0.6830258384589467,
    "address_similarity": 0.0,
    "candidate_id": 258,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": -2,
    "directed_coupling": 0.567289018431697,
    "directed_imbalance": 0.0752496501470063,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 101,
    "response_score": 2.53173745239941,
    "transverse_complementarity": 0.45136582572959516
  }
}
```

Applied moves:

```json
[
  {
    "after_K_mean": 0.7146110688661641,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.25844742295968764,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 2,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 7 9",
    "move_class": "shelling_disk_move",
    "selector": "response",
    "selector_response_rank": 99,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "sym_interfan_transport"
  },
  {
    "after_K_mean": 0.6724175733387748,
    "after_beta0": 1,
    "after_beta1": 1,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2705984045978025,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 18,
    "delta_beta1": 1,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 5 9",
    "move_class": "shelling_disk_move",
    "selector": "topology",
    "selector_response_rank": 117,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "sym_interfan_transport"
  },
  {
    "after_K_mean": 0.6878351969021538,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.278086309378277,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 10,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "response",
    "selector_response_rank": 102,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "sym_interfan_transport"
  },
  {
    "after_K_mean": 0.6878351969021538,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.278086309378277,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 10,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "topology",
    "selector_response_rank": 102,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "sym_interfan_transport"
  },
  {
    "after_K_mean": 0.7760504960472308,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 1,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6315789473684044,
    "after_edge_link_cycle_fraction": 0.045454545454543395,
    "after_exact_residual_ratio": 0.24028177148977867,
    "after_harmonic_ratio": 0.05501415942714737,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 1,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 7 11|4 7 8 11|7 8 10 11",
    "move_class": "handle_candidate",
    "selector": "response",
    "selector_response_rank": 1,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "sym_interfan_transport"
  },
  {
    "after_K_mean": 0.6825293831888232,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666495,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2599217229044788,
    "after_harmonic_ratio": 0.03707990672214233,
    "candidate_id": 71,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": 4,
    "encoded_move": "0 3 4 7|0 1 4 7|0 1 2 7",
    "move_class": "handle_candidate",
    "selector": "topology",
    "selector_response_rank": 2,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "sym_interfan_transport"
  },
  {
    "after_K_mean": 0.6471415872753761,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7272727272726943,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2748474373292416,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 92,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": -6,
    "encoded_move": "{\"10\": 2, \"11\": 8, \"12\": 3}",
    "move_class": "quotient_candidate",
    "selector": "response",
    "selector_response_rank": 9,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "sym_interfan_transport"
  },
  {
    "after_K_mean": 0.6110903128954349,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666445,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2877838245042594,
    "after_harmonic_ratio": 0.13614194307146604,
    "candidate_id": 258,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": -2,
    "encoded_move": "{\"1\": 9, \"5\": 8, \"6\": 7}",
    "move_class": "quotient_candidate",
    "selector": "topology",
    "selector_response_rank": 101,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "sym_interfan_transport"
  }
]
```

## no_backreaction_interfan_transport

- growth_rule: `no_backreaction`
- interfan_strength: `0.16`
- interfan_edge_count: `36`
- interfan_weight_total: `4.3243252`
- base beta: `(1,0,0,0)`
- base boundary_fraction: `0.709677`
- base K_mean: `0.909364`
- base harmonic_ratio: `0`

Best handle candidates:

```json
{
  "top_response": {
    "K_pair_norm": 3.4207418650081407,
    "address_similarity": 0.4,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 1,
    "delta_boundary_faces": 2,
    "directed_coupling": 1.112155451176575,
    "directed_imbalance": 0.7921782125030833,
    "new_beta1": 0,
    "new_beta2": 1,
    "response_rank_legal": 1,
    "response_score": 7.498641409780741,
    "transverse_complementarity": 0.2891184979677439
  },
  "top_topology": {
    "K_pair_norm": 2.2953446197815945,
    "address_similarity": 0.4,
    "candidate_id": 79,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": 4,
    "directed_coupling": 1.023813119999563,
    "directed_imbalance": 1.023813119999563,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 5,
    "response_score": 6.542929522317674,
    "transverse_complementarity": 0.34967541746063685
  }
}
```

Best quotient candidates:

```json
{
  "top_response": {
    "K_pair_norm": 2.1444504406746843,
    "address_similarity": 0.0,
    "candidate_id": 82,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_boundary_faces": -6,
    "directed_coupling": 1.3661812857451787,
    "directed_imbalance": 0.6174913697279639,
    "new_beta1": 0,
    "new_beta2": 0,
    "response_rank_legal": 6,
    "response_score": 6.451029736107027,
    "transverse_complementarity": 0.593594254560297
  },
  "top_topology": {
    "K_pair_norm": 0.9728858722765169,
    "address_similarity": 0.0,
    "candidate_id": 262,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_boundary_faces": -2,
    "directed_coupling": 0.6912621883624261,
    "directed_imbalance": 0.10172051825559286,
    "new_beta1": 0,
    "new_beta2": 2,
    "response_rank_legal": 112,
    "response_score": 3.1092501282413183,
    "transverse_complementarity": 0.45136582572959516
  }
}
```

Applied moves:

```json
[
  {
    "after_K_mean": 0.9750940244350872,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2576042374549955,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 4,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "7 8 10 12",
    "move_class": "shelling_disk_move",
    "selector": "response",
    "selector_response_rank": 75,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "no_backreaction_interfan_transport"
  },
  {
    "after_K_mean": 0.9293692742050462,
    "after_beta0": 1,
    "after_beta1": 1,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7058823529411556,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.26176331652726326,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 11,
    "delta_beta1": 1,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 5 9",
    "move_class": "shelling_disk_move",
    "selector": "topology",
    "selector_response_rank": 113,
    "status": "ok",
    "target_class": "shelling_disk_move",
    "variant": "no_backreaction_interfan_transport"
  },
  {
    "after_K_mean": 0.9825653333566636,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.2992209702210422,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 7,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "response",
    "selector_response_rank": 91,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "no_backreaction_interfan_transport"
  },
  {
    "after_K_mean": 0.9825653333566636,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666464,
    "after_edge_link_cycle_fraction": 0.049999999999997505,
    "after_exact_residual_ratio": 0.2992209702210422,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 7,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": 0,
    "encoded_move": "3 8 11 12",
    "move_class": "cap_move",
    "selector": "topology",
    "selector_response_rank": 91,
    "status": "ok",
    "target_class": "cap_move",
    "variant": "no_backreaction_interfan_transport"
  },
  {
    "after_K_mean": 1.0186399220533557,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 1,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6315789473684044,
    "after_edge_link_cycle_fraction": 0.045454545454543395,
    "after_exact_residual_ratio": 0.24202434087401475,
    "after_harmonic_ratio": 0.030623941156988452,
    "candidate_id": 69,
    "delta_beta1": 0,
    "delta_beta2": 1,
    "delta_beta3": 0,
    "delta_boundary_faces": 2,
    "encoded_move": "3 4 7 11|4 7 8 11|7 8 10 11",
    "move_class": "handle_candidate",
    "selector": "response",
    "selector_response_rank": 1,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "no_backreaction_interfan_transport"
  },
  {
    "after_K_mean": 0.8341772120925101,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666495,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2801956478994305,
    "after_harmonic_ratio": 0.06698787903349132,
    "candidate_id": 79,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": 4,
    "encoded_move": "0 3 4 7|0 1 4 7|0 1 2 7",
    "move_class": "handle_candidate",
    "selector": "topology",
    "selector_response_rank": 5,
    "status": "ok",
    "target_class": "handle_candidate",
    "variant": "no_backreaction_interfan_transport"
  },
  {
    "after_K_mean": 0.8138761833786327,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 0,
    "after_beta3": 0,
    "after_boundary_fraction": 0.7272727272726943,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.2999860407378826,
    "after_harmonic_ratio": 0.0,
    "candidate_id": 82,
    "delta_beta1": 0,
    "delta_beta2": 0,
    "delta_beta3": 0,
    "delta_boundary_faces": -6,
    "encoded_move": "{\"10\": 2, \"11\": 8, \"12\": 3}",
    "move_class": "quotient_candidate",
    "selector": "response",
    "selector_response_rank": 6,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "no_backreaction_interfan_transport"
  },
  {
    "after_K_mean": 0.9228229995739944,
    "after_beta0": 1,
    "after_beta1": 0,
    "after_beta2": 2,
    "after_beta3": 0,
    "after_boundary_fraction": 0.6666666666666445,
    "after_edge_link_cycle_fraction": 0.0,
    "after_exact_residual_ratio": 0.29054071703715323,
    "after_harmonic_ratio": 0.15352689838572703,
    "candidate_id": 262,
    "delta_beta1": 0,
    "delta_beta2": 2,
    "delta_beta3": 0,
    "delta_boundary_faces": -2,
    "encoded_move": "{\"1\": 9, \"5\": 8, \"6\": 7}",
    "move_class": "quotient_candidate",
    "selector": "topology",
    "selector_response_rank": 112,
    "status": "ok",
    "target_class": "quotient_candidate",
    "variant": "no_backreaction_interfan_transport"
  }
]
```
