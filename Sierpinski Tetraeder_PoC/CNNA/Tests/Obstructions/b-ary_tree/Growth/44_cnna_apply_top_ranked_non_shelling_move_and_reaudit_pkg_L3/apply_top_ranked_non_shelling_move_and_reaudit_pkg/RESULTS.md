# RESULTS

## Purpose

Apply the non-shelling candidates found by the previous audit instead of only listing them.

## Real-growth applied moves

| selection | move | rank | Δβ1 | Δβ2 | boundary | harmonic | exact_res |
|---|---:|---:|---:|---:|---:|---:|---:|
| shelling_disk_move:top_response | shelling_disk_move | 100 | 0 | 0 | 0.6783 | 0 | 0.2596 |
| shelling_disk_move:top_topology_gain | shelling_disk_move | 173 | 1 | 0 | 0.6783 | 0 | 0.2747 |
| cap_move:top_response | cap_move | 125 | 0 | 0 | 0.6667 | 0 | 0.2593 |
| handle_candidate:top_response | handle_candidate | 3 | 0 | 1 | 0.6555 | 0.007962 | 0.2472 |
| quotient_candidate:top_response | quotient_candidate | 1 | 0 | 0 | 0.68 | 0 | 0.2779 |
| quotient_candidate:top_topology_gain | quotient_candidate | 39 | 0 | 1 | 0.6545 | 0.0215 | 0.2488 |
| non_shelling:top_response | quotient_candidate | 1 | 0 | 0 | 0.68 | 0 | 0.2779 |
| non_shelling:top_topology_gain | quotient_candidate | 39 | 0 | 1 | 0.6545 | 0.0215 | 0.2488 |

## Interpretation

The base outward complex is still ball-like: `β=(1,0,0,0)` and `harmonic_ratio=0`.

Local shelling/cap moves can reduce boundary or create small local link cycles, but they do not open a harmonic K-sector.

The applied handle move is the first operation in this test that turns the carrier into a nontrivial 2-homology carrier: `Δβ2=+1`, with a small but nonzero harmonic K projection. A topology-ranked quotient move also yields `Δβ2=+1` with a larger harmonic projection in this run.

This confirms the main diagnosis:

```text
The missing ingredient is not local K and not outward/transverse birth itself.
The missing ingredient is a non-shelling complement-pairing/gluing operation.
```

But the controls are critical:

```text
symmetrized_birth and no_backreaction also produce β2-positive non-shelling moves.
```

Therefore this package does **not** prove that real backreaction uniquely selects the correct move. It proves only that once a handle/quotient operation is admitted, the previously empty harmonic sector can open.

## About Script 12 shell mode

Script 12's best result was the `shell_norm_inverse_square` kernel. That was a shell-normalized ancestor/backreaction weighting kernel, not a statement that shelling-type topology is correct. It improved phase-density locality/residual curvature, while the present tests concern whether the simplicial carrier is ball-like or has nontrivial homology.

## Next test

`test_cqnm_provenance_forced_pairing_vs_controls.py`

Goal: construct a stricter score using only derived provenance/response quantities and require that real-growth ranks the β2-positive handle/quotient higher than symmetrized/no-backreaction controls. If the controls remain equally good, the gluing rule is still underderived.
