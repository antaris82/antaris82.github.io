# Dynamic birth monodromy test — RESULTS

## Status

This is a Python diagnostic surrogate, not a Lean theorem and not a physical claim.  
It extends the dynamic birth-conductance test by asking whether the directed response dynamics produces a genuine rotational/complex monodromy, not just a neutral-current imbalance or a real flow.

The full control run was performed to level `L = 7` for three update modes:

1. `linear`
2. `log`
3. `saturating`

Each mode reaches:

```text
3280 nodes
3279 birth events
1093 completed sibling triples
```

The script records event-level snapshots after each birth and level summaries after each completed level.

## Model assumptions

At each parent, children are born sequentially in order `1,2,3`.  
The newborn has no own UV-tail, but immediately acts as UV-tail/backreaction source for:

```text
parent line up to the global root
already-born older siblings
```

The directed influence graph therefore contains two kinds of directed response edges:

```text
old -> newborn        environment influence
newborn -> old        UV-tail/backreaction
```

For a completed sibling triple `(1,2,3)`, the test compares:

```text
forward cycle: 1 -> 2, 2 -> 3, 3 -> 1
reverse cycle: 1 -> 3, 3 -> 2, 2 -> 1
```

The diagnostic distinguishes four objects:

```text
1. neutral phasor Z
   imbalance of sibling conductances

2. log circulation
   log((w12*w23*w31)/(w13*w32*w21))

3. selected forward-cycle transport
   weighted Z3 closure candidate

4. full local directed Markov transport
   uses all six directed sibling influence weights
```

Two controls are included:

```text
symmetrized local matrix   -> should have real spectrum
path without closure       -> should have real/degenerate spectrum
```

## Level summary

```text
      mode  level  nodes  completed_triples  undirected_H1_support_rank  mean_log_circulation  mean_neutral_norm_current  frac_forward_cycle_complex  frac_full_markov_complex  frac_sym_raw_complex  frac_path_raw_complex
    linear      0      1                  0                           0              0.000000                   0.000000                    0.000000                  0.000000              0.000000               0.000000
    linear      1      4                  1                           3              1.379981                   0.284248                    1.000000                  1.000000              0.000000               0.000000
    linear      2     13                  4                          21              1.141519                   0.278346                    1.000000                  1.000000              0.000000               0.000000
    linear      3     40                 13                         102              0.975414                   0.276033                    1.000000                  1.000000              0.000000               0.000000
    linear      4    121                 40                         426              0.859013                   0.275131                    1.000000                  1.000000              0.000000               0.000000
    linear      5    364                121                        1641              0.775827                   0.274856                    1.000000                  1.000000              0.000000               0.000000
    linear      6   1093                364                        6015              0.714044                   0.274849                    1.000000                  1.000000              0.000000               0.000000
    linear      7   3280               1093                       21324              0.665658                   0.274921                    1.000000                  1.000000              0.000000               0.000000
       log      0      1                  0                           0              0.000000                   0.000000                    0.000000                  0.000000              0.000000               0.000000
       log      1      4                  1                           3              1.238490                   0.061755                    1.000000                  1.000000              0.000000               0.000000
       log      2     13                  4                          21              0.981801                   0.037880                    1.000000                  1.000000              0.000000               0.000000
       log      3     40                 13                         102              0.805414                   0.024049                    1.000000                  1.000000              0.000000               0.000000
       log      4    121                 40                         426              0.687094                   0.015829                    1.000000                  1.000000              0.000000               0.000000
       log      5    364                121                        1641              0.607774                   0.010845                    1.000000                  1.000000              0.000000               0.000000
       log      6   1093                364                        6015              0.552581                   0.007699                    1.000000                  1.000000              0.000000               0.000000
       log      7   3280               1093                       21324              0.510948                   0.005599                    1.000000                  1.000000              0.000000               0.000000
saturating      0      1                  0                           0              0.000000                   0.000000                    0.000000                  0.000000              0.000000               0.000000
saturating      1      4                  1                           3              2.782875                   0.090393                    1.000000                  1.000000              0.000000               0.000000
saturating      2     13                  4                          21              2.456621                   0.031147                    1.000000                  1.000000              0.000000               0.000000
saturating      3     40                 13                         102              2.242462                   0.019145                    1.000000                  1.000000              0.000000               0.000000
saturating      4    121                 40                         426              2.104468                   0.020594                    1.000000                  1.000000              0.000000               0.000000
saturating      5    364                121                        1641              2.015530                   0.023692                    1.000000                  1.000000              0.000000               0.000000
saturating      6   1093                364                        6015              1.955975                   0.026327                    1.000000                  1.000000              0.000000               0.000000
saturating      7   3280               1093                       21324              1.912531                   0.028381                    1.000000                  1.000000              0.000000               0.000000
```

## Root-level monodromy snapshots

### linear

```text
current child conductances: 1.33751143505 1.56547270505 1.8569915728
log circulation fwd/rev: 1.379981
neutral norm current: 0.284248
full Markov eigenvalues: 1+0j -0.5+0.132389657j -0.5-0.132389657j
full Markov class: complex_pair
sym raw class: real_or_degenerate
path raw class: real_or_degenerate
kappa flips selected forward J: True
kappa preserves birth order: False
```

### log

```text
current child conductances: 1.24314425592 1.30287749303 1.33386125657
log circulation fwd/rev: 1.238490
neutral norm current: 0.061755
full Markov eigenvalues: 1+0j -0.5+0.138051716j -0.5-0.138051716j
full Markov class: complex_pair
sym raw class: real_or_degenerate
path raw class: real_or_degenerate
kappa flips selected forward J: True
kappa preserves birth order: False
```

### saturating

```text
current child conductances: 1.56808524621 1.70451889247 1.72989148702
log circulation fwd/rev: 2.782875
neutral norm current: 0.090393
full Markov eigenvalues: 1+0j -0.5+0.212180858j -0.5-0.212180858j
full Markov class: complex_pair
sym raw class: real_or_degenerate
path raw class: real_or_degenerate
kappa flips selected forward J: True
kappa preserves birth order: False
```

## Key findings

### 1. Sequential dynamic birth produces nonzero local circulation

At final level `L = 7`, the mean log-circulation is positive in all modes:

```text
linear:      0.665658
log:         0.510948
saturating:  1.912531
```

Thus the response graph is not forward/reverse balanced. This is stronger than a static conductance imbalance because it compares directed products around the local sibling triangle.

### 2. The selected forward Z3 transport has a complex 2D sector in all completed triples

For all modes and all completed sibling triples to level 7:

```text
frac_forward_cycle_complex = 1.0
```

This means: if the directed closure `3 -> 1` is accepted as derived from the backreaction edge of the newest sibling to the oldest sibling, the selected local Z3 transport carries the usual complex pair.

### 3. The full local directed Markov transport also has complex eigenvalue pairs

This is the strongest positive result of the test:

```text
frac_full_markov_complex = 1.0
```

So the complex sector is not only present in the hand-selected forward cycle. It is also visible in the full local directed sibling influence matrix containing all six directed pair weights.

For the root in `linear` mode, for example:

```text
full Markov eigenvalues = 1, -0.5 ± 0.132389657 i
```

The imaginary part is smaller than the ideal `±sqrt(3)/2` of a pure Z3 permutation because reverse and symmetric influence channels are still present. But it is nonzero.

### 4. The controls behave correctly

At all levels and in all modes:

```text
frac_sym_raw_complex  = 0.0
frac_path_raw_complex = 0.0
```

So the complex pair disappears if either:

```text
1. the local matrix is symmetrized, or
2. the closure edge is removed and only the birth path remains.
```

This is the central control result:

```text
path alone      -> real/degenerate, no J
symmetrized     -> real, no J
directed closure / full directed response -> complex sector
```

### 5. κ flips the selected forward J, but does not preserve birth order

For all completed triples:

```text
frac_kappa_flips_forward_J = 1.0
frac_kappa_preserves_birth_order = 0.0
```

So κ sends the selected forward `J` to `-J`, but κ is not a symmetry of the irreversible birth-time history.

## Interpretation

The test gives a genuine positive diagnostic signal:

```text
sequential birth + backreaction
-> directed local sibling response
-> nonzero log-circulation
-> full local directed Markov operator has complex eigenvalues
```

This is stronger than the previous neutral-current test. It shows not only imbalance, but a rotational local response sector in the directed influence dynamics.

## What is still not shown

The test does not yet prove:

```text
1. a Lean theorem
2. a derived simplicial-complex closure rule
3. a non-coboundary class in the extracted effective geometry
4. J^2 = -I for the full network operator
5. Type III / vN / nuclearity
```

The strongest currently justified claim is:

```text
A dynamically updated directed sibling-response model produces complex local monodromy,
whereas the path-only and symmetrized controls do not.
```

## Next test

The next step should connect this local directed-response monodromy to the effective simplicial complex:

```text
test_dynamic_birth_simplicial_closure.py
```

It should check whether the local response cycles correspond to actual closure cycles / non-coboundary classes in the extracted NGF-like simplicial geometry, or whether the complex sector remains only a response-layer phenomenon.
