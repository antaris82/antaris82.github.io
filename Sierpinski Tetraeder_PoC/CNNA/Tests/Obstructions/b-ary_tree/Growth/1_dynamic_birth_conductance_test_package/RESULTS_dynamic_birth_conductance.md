# Dynamic birth-conductance test — RESULTS

## Status

This is a Python diagnostic surrogate, not a Lean theorem and not a physical claim.  
It tests the proposed CNNA/NGF provenance-growth intuition:

- a newborn cell has no own UV-tail;
- the newborn immediately acts as UV-tail/backreaction for its parent line;
- siblings are born sequentially and therefore do not share the same birth environment;
- response/conductance data changes after every birth;
- geometry lengths are not conductances.

The simulation was run to level `L = 7` for three update modes:

1. `linear`
2. `log`
3. `saturating`

Each mode has `3280` nodes and `3279` birth events at final level.

## Model assumptions

At each parent, children are born in order `1,2,3`.

For child `k` under parent `p`:

```text
birth_environment(child_k)
  = parent_line(p) + older_siblings(p)
```

Then:

```text
child_birth_g = response_function(birth_environment)
```

After birth:

```text
child acts as UV-tail/backreaction for:
  parent line up to root
  already-born older siblings
```

Directed influence edges distinguish:

```text
environment influence:
  old -> newborn

backreaction:
  newborn -> old
```

This is crucial: the support is not treated as a static symmetric graph.

## Root-level event trace, linear mode

After child 1:

```text
g_child1_birth = 1.220000
partial children = [1.220000]
```

After child 2:

```text
g_child2_birth = 1.500478
partial current children = [1.272517, 1.500478]
```

After child 3:

```text
g_child3_birth = 1.856992
partial/current children = [1.337511, 1.565473, 1.856992]
neutral |Z| = 0.451004
normalized |Z| = 0.284248
cycle log-bias forward/reverse = 1.379981
```

After full growth to level 7, the root children continue to be updated by descendants:

```text
root current conductances:
  [7.104291, 7.454674, 7.890918]
root current |Z| = 0.682590
root current phase ≈ -146.394°
```

So the conductance system is not static after a sibling triple is completed.

## Level summary

```text
      mode  level  nodes  undirected_H1_rank  mean_neutral_norm_current  mean_cycle_log_bias  min_g   max_g
    linear      0      1                   0                     0.0000               0.0000 1.0000  1.0000
    linear      1      4                   3                     0.2842               1.3800 1.2060  1.8570
    linear      2     13                  21                     0.2783               1.1415 1.3965  2.3942
    linear      3     40                 102                     0.2760               0.9754 1.6947  2.9185
    linear      4    121                 426                     0.2751               0.8590 1.9735  3.4369
    linear      5    364                1641                     0.2749               0.7758 2.1387  3.9604
    linear      6   1093                6015                     0.2748               0.7140 2.2915  6.3136
    linear      7   3280               21324                     0.2749               0.6657 2.4390 13.1182
       log      0      1                   0                     0.0000               0.0000 1.0000  1.0000
       log      1      4                   3                     0.0618               1.2385 1.1684  1.3339
       log      2     13                  21                     0.0379               0.9818 1.3014  1.5117
       log      3     40                 102                     0.0240               0.8054 1.3721  1.6484
       log      4    121                 426                     0.0158               0.6871 1.3981  1.8334
       log      5    364                1641                     0.0108               0.6078 1.4144  2.3996
       log      6   1093                6015                     0.0077               0.5526 1.4254  3.6673
       log      7   3280               21324                     0.0056               0.5109 1.4337  6.4766
saturating      0      1                   0                     0.0000               0.0000 1.0000  1.0000
saturating      1      4                   3                     0.0904               2.7829 1.2171  1.7299
saturating      2     13                  21                     0.0311               2.4566 1.3901  1.9612
saturating      3     40                 102                     0.0191               2.2425 1.6254  2.1378
saturating      4    121                 426                     0.0206               2.1045 1.7939  2.3755
saturating      5    364                1641                     0.0237               2.0155 1.7990  2.7997
saturating      6   1093                6015                     0.0263               1.9560 1.8022  4.4163
saturating      7   3280               21324                     0.0284               1.9125 1.8045  7.9882
```

## Key findings

### 1. Sequential birth generically breaks sibling equality

In all three update modes, the root birth conductances satisfy:

```text
linear:      [1.220000, 1.500478, 1.856992]
log:         [1.152492, 1.256192, 1.333861]
saturating:  [1.450000, 1.643973, 1.729891]
```

Thus `g1 = g2 = g3` is not generic under sequential birth.

### 2. Already-born siblings and ancestors keep changing

For linear mode:

```text
root birth conductances:
  [1.220000, 1.500478, 1.856992]

root current conductances after L=7:
  [7.104291, 7.454674, 7.890918]
```

This supports the corrected principle:

```text
newborn has no own UV-tail,
but newborn immediately becomes UV-tail/backreaction for its parent line.
```

### 3. Neutral-current / neutral-phasor imbalance appears immediately

For the ideal initial root state:

```text
Z_0 = 0
```

After the root has three sequentially born children, in linear mode:

```text
|Z| = 0.451004
phase ≈ -145.96°
```

At later levels the normalized neutral imbalance remains nonzero in `linear`,
shrinks in `log`, and remains small but nonzero in `saturating`.

This shows that imbalance is update-rule dependent, but not an artifact of a static `[1,2,1]` input.

### 4. Directed local cycle bias appears in the influence graph

For each completed sibling triple, the test compares:

```text
forward product:
  1 -> 2, 2 -> 3, 3 -> 1

reverse product:
  1 -> 3, 3 -> 2, 2 -> 1
```

The log-ratio remains positive in all modes to level 7:

```text
linear final mean log-bias:      0.665658
log final mean log-bias:         0.510948
saturating final mean log-bias:  1.912531
```

So the directed influence/backreaction system is not forward/reverse balanced.

### 5. Global support cycles grow rapidly, but this is not yet topological H¹

The reported `undirected_H1_rank` is computed from the undirected support of the influence graph:

```text
H1_support = E - V + components
```

At final level:

```text
H1_support = 21324
```

This means many support cycles exist in the influence graph.  
It does **not** yet prove non-coboundary holonomy of the effective simplicial complex.

## What is not shown

This test does **not** prove:

- a derived Z3 closure in the effective simplicial complex;
- non-coboundary monodromy;
- `J² = -I` from the full network;
- Type III behavior;
- physical conductance laws;
- operator-algebraic modular structure.

It shows only that the dynamic birth/backreaction model generically produces:

```text
sibling conductance asymmetry
neutral-phasor imbalance
directed local cycle bias
rapid growth of support cycles
```

## Next test

The next diagnostic should separate two questions:

1. Are the support cycles merely graph artifacts of directed response bookkeeping?
2. Does the extracted effective simplicial complex contain non-coboundary monodromy?

Recommended next file:

```text
test_dynamic_birth_monodromy.py
```

It should use the same dynamic birth-response model, extract closure-enabled local/global cycles, and then classify monodromy:

```text
real eigenvalues     -> flow / arrow, no J
complex eigenvalues  -> rotation candidate
kappa flip           -> orientation test
```
