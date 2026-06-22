# RESULTS — interfan phase propagation obstruction gate

## Comparative table

| variant | mode | beta | pairings | residual pre | residual post | reduction | graph edges | conflict frac | pair harm | pair kappa | pair H coh | used Δβ? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth | none | (1,0,2,0) | 2 | 0.785714 | 0.785714 | 0 | 0 | 0 | 0.229731 | 0.0318445 | 0.0567699 | False |
| real_growth | birth_sum | (1,0,2,0) | 2 | 0.785714 | 0.53008 | 0.255634 | 0 | 0 | 0.239997 | 0.067 | 0.0518262 | False |
| real_growth | pair_graph_abs | (1,0,2,0) | 2 | 0.785714 | 0.415596 | 0.370118 | 2 | 0 | 0.212864 | 0.0410577 | 0.0413221 | False |
| real_growth | pair_graph_signed | (1,0,2,0) | 2 | 0.785714 | 0.785714 | 0 | 2 | 0 | 0.229731 | 0.0318445 | 0.0567699 | False |
| real_growth | face_graph_abs | (1,0,2,0) | 2 | 0.785714 | 0.415596 | 0.370118 | 197 | 0.395939 | 0.212864 | 0.0410577 | 0.0413221 | False |
| real_growth | face_graph_signed | (1,0,2,0) | 2 | 0.785714 | 0.785714 | 0 | 197 | 0.395939 | 0.229731 | 0.0318445 | 0.0567699 | False |
| strict_symmetrized_control | none | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| strict_symmetrized_control | birth_sum | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| strict_symmetrized_control | pair_graph_abs | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| strict_symmetrized_control | pair_graph_signed | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| strict_symmetrized_control | face_graph_abs | (1,0,0,0) | 0 | 0 | 0 | 0 | 99 | 0.505051 | 0 | 0 | 0 | False |
| strict_symmetrized_control | face_graph_signed | (1,0,0,0) | 0 | 0 | 0 | 0 | 99 | 0.424242 | 0 | 0 | 0 | False |
| no_backreaction | none | (1,0,2,0) | 2 | 0.9295 | 0.9295 | 0 | 0 | 0 | 0.223795 | 0.0293466 | 0.0535042 | False |
| no_backreaction | birth_sum | (1,0,2,0) | 2 | 0.9295 | 0.623318 | 0.306182 | 0 | 0 | 0.235372 | 0.0647243 | 0.0520727 | False |
| no_backreaction | pair_graph_abs | (1,0,2,0) | 2 | 0.9295 | 0.538591 | 0.390909 | 2 | 0 | 0.209102 | 0.0400721 | 0.046163 | False |
| no_backreaction | pair_graph_signed | (1,0,2,0) | 2 | 0.9295 | 0.9295 | 0 | 2 | 0 | 0.223795 | 0.0293466 | 0.0535042 | False |
| no_backreaction | face_graph_abs | (1,0,2,0) | 2 | 0.9295 | 0.538591 | 0.390909 | 197 | 0.416244 | 0.209102 | 0.0400721 | 0.046163 | False |
| no_backreaction | face_graph_signed | (1,0,2,0) | 2 | 0.9295 | 0.9295 | 0 | 197 | 0.370558 | 0.223795 | 0.0293466 | 0.0535042 | False |

## Gate reading

- `residual pre/post` measures actual pair-edge interfan residual before/after the Z3 phase propagation.
- `graph edges` and `conflict frac` measure whether the propagated labels are compatible over the chosen graph.
- `pair harm` is H2 projection of the actual pair-transport field after propagation.
- `pair kappa` and `pair H coh` are the oriented/coherent harmonic diagnostics.
- `decision_used_delta_beta_any` must remain false.

A strong positive gate would require: residual reduction, strict-sym kill, and increased pair kappa / pair H coherence without using beta or H2 in the phase assignment.
