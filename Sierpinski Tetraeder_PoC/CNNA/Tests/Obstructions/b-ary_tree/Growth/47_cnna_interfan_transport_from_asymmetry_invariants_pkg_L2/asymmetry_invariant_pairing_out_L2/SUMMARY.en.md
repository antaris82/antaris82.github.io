# SUMMARY — Asymmetry-invariant complement pairing gate

This package implements the refined test requested following Claude's critique:

```text
not every child/face gets a complement;
only distant, transversely complementary boundary-face pairs with measurable
sequential birth/backreaction asymmetry are allowed through the complement gate.
```

The new ranking uses only provenance/response invariants, not `delta_beta2`:

```text
A_fan = directed sibling imbalance
      + parent-line live/record aging gradient
      + descendant-shell UV-tail difference
      + nonreciprocal residue
      + conductance update asymmetry
```

| variant | A-gated pairs | gated topology-effective | fan A mean | top gated class | top gated Δβ2 | top gated rank | after β2 | harmonic | max-Δβ2 rank |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| real_growth | 71 | 47 | 1.54765 | handle_candidate | 1 | 1 | 1 | 0.08246868001731612 | 2 |
| historical_symmetrized_birth | 69 | 49 | 2.29558 | handle_candidate | 1 | 1 | 1 | 0.028789744611320695 | 11 |
| strict_symmetrized_control | 0 | 0 | 0 | missing |  |  |  |  |  |
| no_backreaction | 72 | 47 | 1.56275 | handle_candidate | 1 | 1 | 1 | 0.08298906209837992 | 3 |

Critical readout:

- `real_growth` passes the selective gate, and its top gated move is a topology-effective handle.
- `strict_symmetrized_control` has `A_gated_count = 0`; the selective invariant gate collapses there. This is the important positive control.
- `historical_symmetrized_birth` does **not** collapse. That older control still retains sequential residues in this lightweight core, so it is not a strict symmetry control for this question.
- `no_backreaction` also does not collapse, because the older-sibling birth-environment feed-forward alone already creates a directed asymmetry. This means the active signal is not purely backreaction; it is sequential birth-environment plus optional backreaction.

The result supports a selective complement-pairing hypothesis, but it still does not prove that the growth rule must execute the proposed mechanism. The next step must establish gated non-shelling pairing as an in-growth event and test whether controls fail before applying the mechanism.
