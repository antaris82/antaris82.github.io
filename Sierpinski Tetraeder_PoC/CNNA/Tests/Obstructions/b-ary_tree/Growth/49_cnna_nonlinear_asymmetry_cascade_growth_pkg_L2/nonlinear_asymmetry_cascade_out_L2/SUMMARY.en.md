# SUMMARY — nonlinear asymmetry-cascade growth

This package implements the correction that growth need not proceed linearly as "one birth, one local closure, next level."

The tested loop is event-driven and nonlinear:

```text
ordinary outward birth
-> local/nonlocal boundary scan
-> asymmetry-gated complement pairing
-> if a pairing fires, rescan the updated complex immediately
-> allow a bounded cascade before the next ordinary birth
```

The decision rule uses provenance/response invariants and a nonlinear reinforcement of the A-gate. It does not examine delta beta when selecting a move.

| variant | response_mode | baseline beta | nonlinear beta | pairings | harmonic | opened beta2 | nonexact K |
|---|---|---:|---:|---:|---:|---:|---:|
| real_growth | power_saturating | (1,0,0,0) | (1,0,3,0) | 2 | 0.13524 | True | True |
| strict_symmetrized_control | power_saturating | (1,0,0,0) | (1,0,0,0) | 0 | 0 | False | False |
| no_backreaction | power_saturating | (1,0,0,0) | (1,0,3,0) | 2 | 0.139537 | True | True |

Interpretation: A positive result indicates that the complex opens beta2 and that the K-sector acquires a nonzero harmonic component during the nonlinear growth itself, not following a post-hoc candidate application. A strict symmetry control must remain negative for the mechanism to be selective rather than merely a proposed topological move.
