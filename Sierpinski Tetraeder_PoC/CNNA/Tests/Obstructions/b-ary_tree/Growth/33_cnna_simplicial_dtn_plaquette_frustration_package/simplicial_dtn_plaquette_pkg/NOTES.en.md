# Notes

## Mathematical gate

For a face `(a,b,c)` with local vertex operators `S_a,S_b,S_c`:

```text
A_ab = S_b - S_a
A_bc = S_c - S_b
A_ca = S_a - S_c
```

Then

```text
A_ab + A_bc + A_ca = 0
```

by construction. This eliminates first-order height/gradient circulation.

The object under consideration is the second-order operatorial plaquette part:

```text
[A_ab,A_bc]
```

and the finite holonomy surrogate

```text
P_abc = exp(eps A_ca) exp(eps A_bc) exp(eps A_ab).
```

Nonzero skew or complex eigenpairs of `P_abc` do not prove `J`. They merely indicate that the provenance-glued 2-simplex carries noncommuting response data.

## Main controls

- Diagonal-only and trace-scalar reductions must collapse the commutator.
- Symmetrized-birth should strongly suppress sibling-triangle signals.
- No-backreaction should eliminate handoff/aging signals but may retain live parent-fan response from sequential birth environments.
- Random same-level triangles serve as a baseline, not a proof control.

## Interpretation

A positive result moves the chain from

```text
provenience pair/operator closure obstruction
```

to

```text
face-level operatorial plaquette frustration candidate.
```

Subsequent tests should compare orientation signs and then project the plaquette skew sector onto local `J` candidate planes only after this face-level gate has stabilized.
