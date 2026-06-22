# RESULTS — Inter-fan transport from asymmetry invariants

## Question

Claude's useful correction was that the missing complement should not be assigned to every child or every face. It should be a selective relation between boundary faces that are:

```text
1. nonlocal / not a shelling cap,
2. transversely complementary,
3. marked by invariant directed birth/backreaction asymmetry,
4. topology-effective when applied as a handle/quotient candidate.
```

This script implements that test. The asymmetry score does **not** use `delta_beta2` for ranking, so the test checks whether the provenance/response data select the handle before examining the topological result.

## Comparative result

| variant | A-gated pairs | gated topology-effective | fan A mean | top gated class | top gated Δβ2 | top gated rank | after β2 | harmonic | max-Δβ2 rank |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| real_growth | 71 | 47 | 1.54765 | handle_candidate | 1 | 1 | 1 | 0.08246868001731612 | 2 |
| historical_symmetrized_birth | 69 | 49 | 2.29558 | handle_candidate | 1 | 1 | 1 | 0.028789744611320695 | 11 |
| strict_symmetrized_control | 0 | 0 | 0 | missing |  |  |  |  |  |
| no_backreaction | 72 | 47 | 1.56275 | handle_candidate | 1 | 1 | 1 | 0.08298906209837992 | 3 |

## Interpretation

The strong positive aspect is:

```text
real_growth:
  top gated candidate = handle_candidate
  top gated delta_beta2 = +1
  applying it opens beta2 and produces nonzero harmonic K projection.

strict_symmetrized_control:
  A_gated_count = 0
  no move passes the selective invariant complement gate.
```

This is the first test in this branch where the selective complement gate collapses under a strict symmetry control while selecting a topology-effective handle under real sequential growth.

The concerning aspect is equally important:

```text
historical_symmetrized_birth still passes the gate.
no_backreaction still passes the gate.
```

This means that the previously generated `symmetrized_birth` control is not strict enough for this specific question, and `no_backreaction` is not a no-asymmetry control. The sequential older-sibling environment already creates a directed imprint. Therefore, the current evidence supports:

```text
selective complement pairing from sequential provenance asymmetry
```

but not yet:

```text
selective complement pairing specifically forced by backreaction alone.
```

## What remains undetermined

The handle is still a proposed candidate that is then applied. The next test must integrate the rule into the growth loop:

```text
if A_gate(face_a, face_b) passes and the pair is manifold-legal,
execute the non-shelling complement-pairing during growth;
otherwise continue ordinary outward NGF/CQNM attachment.
```

Then compare actual growth against strict symmetrized/no-sibling/no-backreaction controls before any candidate is applied retroactively.
