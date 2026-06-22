# SUMMARY — pairing-transport harmonic kappa gate

This package tests the next anti-smuggling step following nonlinear asymmetry-gated complement pairing.

Previous result: beta2 opens and the scalar |K| field has a harmonic component.  This package investigates whether the **actual applied pairings** carry a transported axial K-flow:

```text
K_pair(face_a, face_b) = K_face_a - orientation_reversed_transport(K_face_b)
```

The pairing logs are used only after the nonlinear growth has already selected and applied moves.  The transport/harmonic/kappa projection is therefore diagnostic and not part of the move decision.

| variant | beta auto | pairings | scalar K harm | pair axial harm | pair scalar harm | Hdim | pair kappa | pair birth kappa | delta beta2 sum | used Δβ? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth | (1,0,2,0) | 2 | 0.121617 | 0.220125 | 0.208125 | 2 | 0.0339795 | 0.0339795 | 2 | False |
| strict_symmetrized_control | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| no_backreaction | (1,0,2,0) | 2 | 0.128934 | 0.216054 | 0.20887 | 2 | 0.0296715 | 0.0296715 | 2 | False |

Interpret the result conservatively: a positive beta2 value indicates that the carrier is present; a positive pair-transport axial harmonic ratio is evidence that the pairing process itself carries an oriented skew flow into H2.
