# Summary: boundary-resolved face correspondence transport

## Gate

The test examines whether Double-History suffix gluing results in a nontrivial face correspondence between the three parent-fan faces when the matching is inferred from boundary/port DtN data rather than from labels.

## L5 quick-suite primary result

```text
real_growth | live | full | boundary_ports
count=13, valid=1.000, K=0.00156178
identity_match=1.000, directed_match=0.000, cyclic_match=0.000, reflection_match=0.000
identity_cost=0.003439, directed_cost=0.2466
cycle_identity_product=1.000, cycle_nontrivial_product=0.000
```

## L6 quick-suite primary result

```text
real_growth | live | full | boundary_ports
count=40, valid=1.000, K=0.00186993
identity_match=1.000, directed_match=0.000, cyclic_match=0.000, reflection_match=0.000
identity_cost=0.003000, directed_cost=0.2453
cycle_identity_product=1.000, cycle_nontrivial_product=0.000
```

## Controls

- Identical-history clones prefer identity and exhibit zero or nontrivial transport.
- Diagonal and trace-scalar reductions eliminate the noncommutative K-sector (`valid=0`, `K=0`) but still carry trivial boundary identity labels.
- Symmetrized birth strongly suppresses K while still preferring identity.
- Random same-level cycles also prefer identity.

## Interpretation

The local parent-fan Plaquette/J sector is still present in full live DtN data, but boundary-resolved Double-History matching does not produce nontrivial cyclic, reflection, or directed transport. The best data-inferred correspondence is same-label identity.

This is a negative Level 4 transport test, not a refutation of the local Level 2/3 parent-fan sector.
