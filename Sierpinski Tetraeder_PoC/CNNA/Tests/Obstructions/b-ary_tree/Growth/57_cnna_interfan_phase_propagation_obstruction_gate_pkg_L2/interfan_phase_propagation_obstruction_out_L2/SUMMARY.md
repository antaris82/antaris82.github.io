# SUMMARY — interfan phase propagation obstruction gate

Question:

```text
Can a Z3-compatible phase propagation, derived only from birth order, local response vectors, and the actual face/pair graph, reduce the interfan residual and make the pair-transported H2 sector oriented/coherent?
```

The tested local operator remains real and directed:

```text
M = metric_part + eta * strength * (q⊗h - h⊗q)
```

No final `sym(M)` is used in the tested operator.  No `i`, `J`, Hodge star, positivity, norm axiom, spin, or complex scalar is introduced.

Propagation modes:

```text
none              baseline from package 56
birth_sum         face label = sum(birth_order-1) mod 3
pair_graph_abs    Z3 labels from actual pairing edges, maximizing |local cosine|
pair_graph_signed Z3 labels from actual pairing edges, maximizing signed local cosine
face_graph_abs    shared-edge face graph + actual pair graph, maximizing |local cosine|
face_graph_signed shared-edge face graph + actual pair graph, maximizing signed local cosine
```

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

Interpret conservatively: pair-graph/signed modes are diagnostics for whether the obstruction is phase-propagation-compatible; they are not yet a theorem that such a propagation is forced by CNNA.
