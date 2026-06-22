# SOURCE AUDIT — why this test exists

The regression is motivated by package 50's `vertex_operator` structure:

```text
order_phase -> q -> h
M = a rr + b qq + c hh + scalar I
return sym(M)
```

This makes the birth phase available, but encodes it in symmetric metric tensors and then forces a symmetric output.  That is a legitimate metric/DtN-response choice, but it is not a directed growth-transport operator.

This package therefore treats the old operator as the `legacy_sym_metric` branch and tests a separate real transport branch.  The transport branch is not declared ontic truth; it is a regression gate.  It is derived only in the narrow sense that its direction is computed from existing real birth-order fan data.  It is not a J, not a Hodge star, not a complex phase and not a C*-adjoint.

Next test if positive:

```text
test_directed_transport_operator_closure_gate.py
```

Goal: test whether the antisymmetric birth-transport operator family closes under composition on the H² carrier without saturating to the full matrix algebra and without importing an adjoint/positivity package.


## Next test after this run

Because the antisymmetric birth-transport branch changes the total axial field but still projects to an approximately zero harmonic axial ratio, the next test should not assume operator closure yet.  The next falsification/localization gate is:

```text
test_directed_transport_harmonic_obstruction_localization_gate.py
```

Goal: separate whether the harmonic cancellation is caused by (1) the commutator definition `skew([A_ab,A_bc])`, (2) the H² carrier/cap geometry, (3) face-normal/birth-normal projection, or (4) global exact/coboundary cancellation of a locally directed transport field.
