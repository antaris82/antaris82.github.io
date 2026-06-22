# RESULTS — directed antisymmetric birth transport through pairing/inter-fan gate

## Comparative table

| variant | beta auto | pairings | raw K harm | raw local coh | raw 3face coh | pair harm | pair kappa | pair birth kappa | pair H coh | interfan residual | used Δβ? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth | (1,0,2,0) | 2 | 1.63004e-16 | 0.202816 | 0.589549 | 0.229731 | 0.0318445 | 0.0318445 | 0.0567699 | 0.785714 | False |
| strict_symmetrized_control | (1,0,0,0) | 0 | 0 | 0.345397 | 0.634491 | 0 | 0 | 0 | 0 | 0 | False |
| no_backreaction | (1,0,2,0) | 2 | 1.71245e-16 | 0.21965 | 0.583538 | 0.223795 | 0.0293466 | 0.0293466 | 0.0535042 | 0.9295 | False |

## Gate reading

- `raw_K_harmonic_ratio`: harmonic projection of the local directed face-K field.
- `raw_local_orientation_coherence`: pre-H2 coherence of local axial K vectors.
- `raw_3face_coherence`: coherence of triples of incident faces around shared edges.
- `pair_transport_harmonic_ratio`: H2 projection of the actual asymmetry-gated pairing transport field.
- `pair_kappa_orientation_ratio` / `pair_kappa_birth_orientation_ratio`: signed normal/birth-normal bias of the harmonic transported field.
- `interfan_phase_transport_residual`: 1 - |cos| averaged over transported actual pairings.
- `decision_used_delta_beta_any` must remain false.

## Conservative status

This is not a complex-structure derivation and not a real `*`-structure.  It is a falsifiable diagnostic for the missing combination: local directed birth orientation + beta2 carrier + pairing/inter-fan transport.
