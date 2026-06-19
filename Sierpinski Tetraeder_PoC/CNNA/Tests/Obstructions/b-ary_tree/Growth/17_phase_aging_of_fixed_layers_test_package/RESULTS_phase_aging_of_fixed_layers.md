# RESULTS — Phase Aging of Fixed Layers

## Status

Python diagnostic completed successfully to **level 10** for the shell-normalized kernel:

```text
K(d) = 1 / (3^(d-1) d²)
```

Final size:

```text
88573 nodes
29524 completed local response operators
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

The previous locality test showed that residual curvature is larger in old interior layers than near the active frontier.

This test asks:

```text
If a fixed parent level is observed while the tower continues to grow,
does its residual relax/change, or is it frozen after local completion?
```

## Crucial semantic split

The test separates two notions.

### 1. Stored local response operator

This is the operator used in the previous response/J-plane tests:

```text
local_w[parent][i -> j]
```

It is built during the birth of the three children of a parent. In the current model, once the parent has all three children, later descendant births do not modify this stored local response block.

### 2. Refreshed conductance snapshot

This is a current-state diagnostic from the present conductances of the three children:

```text
(g1(t), g2(t), g3(t))
```

Those conductances can still change under later descendant backreaction along parent lines.

This is not the same object as the stored response operator.

## Raw summary

```text
PHASE AGING OF FIXED LAYERS
  max_level=10

  L=1: nodes=4, completed=1, stored_drift=0.000000000e+00, neutral_phase_drift=0.000000000, neutral_norm_delta=0.000000000, gspread_delta=0.000000000, stored_loop=nan, neutral_loop=nan, elapsed=0.00s
  L=2: nodes=13, completed=4, stored_drift=0.000000000e+00, neutral_phase_drift=0.003020522, neutral_norm_delta=-0.001734413, gspread_delta=-0.001969422, stored_loop=0.136892345, neutral_loop=0.078733215, elapsed=0.00s
  L=3: nodes=40, completed=13, stored_drift=0.000000000e+00, neutral_phase_drift=0.007382486, neutral_norm_delta=-0.001474000, gspread_delta=-0.001667035, stored_loop=0.385962622, neutral_loop=0.392688932, elapsed=0.01s
  L=4: nodes=121, completed=40, stored_drift=0.000000000e+00, neutral_phase_drift=0.014029241, neutral_norm_delta=-0.001051858, gspread_delta=-0.001179882, stored_loop=0.355452909, neutral_loop=0.609928326, elapsed=0.02s
  L=5: nodes=364, completed=121, stored_drift=0.000000000e+00, neutral_phase_drift=0.023428389, neutral_norm_delta=-0.000744202, gspread_delta=-0.000823093, stored_loop=0.280956939, neutral_loop=0.748012816, elapsed=0.06s
  L=6: nodes=1093, completed=364, stored_drift=0.000000000e+00, neutral_phase_drift=0.034760726, neutral_norm_delta=-0.000553045, gspread_delta=-0.000599436, stored_loop=0.217305724, neutral_loop=0.823571771, elapsed=0.16s
  L=7: nodes=3280, completed=1093, stored_drift=0.000000000e+00, neutral_phase_drift=0.046300906, neutral_norm_delta=-0.000440719, gspread_delta=-0.000466417, stored_loop=0.171757124, neutral_loop=0.848569476, elapsed=0.52s
  L=8: nodes=9841, completed=3280, stored_drift=0.000000000e+00, neutral_phase_drift=0.056453832, neutral_norm_delta=-0.000375970, gspread_delta=-0.000388659, stored_loop=0.142420786, neutral_loop=0.849947920, elapsed=1.59s
  L=9: nodes=29524, completed=9841, stored_drift=0.000000000e+00, neutral_phase_drift=0.064420034, neutral_norm_delta=-0.000338763, gspread_delta=-0.000343341, stored_loop=0.124682857, neutral_loop=65.081360537, elapsed=4.56s
  L=10: nodes=88573, completed=29524, stored_drift=0.000000000e+00, neutral_phase_drift=0.070163353, neutral_norm_delta=-0.000317300, gspread_delta=-0.000317319, stored_loop=0.114290571, neutral_loop=52.354270420, elapsed=13.73s

FINAL LEVEL AGE SUMMARY
  parent_level=0, age=9, count=1, stored_drift=0.000e+00, neutral_phase_drift=0.013052121, neutral_norm_delta=-0.010319386, gspread_delta=-0.011718780
  parent_level=1, age=8, count=3, stored_drift=0.000e+00, neutral_phase_drift=0.030080540, neutral_norm_delta=-0.005219365, gspread_delta=-0.005885237
  parent_level=2, age=7, count=9, stored_drift=0.000e+00, neutral_phase_drift=0.055517012, neutral_norm_delta=-0.003206767, gspread_delta=-0.003566427
  parent_level=3, age=6, count=27, stored_drift=0.000e+00, neutral_phase_drift=0.090320194, neutral_norm_delta=-0.002243186, gspread_delta=-0.002444975
  parent_level=4, age=5, count=81, stored_drift=0.000e+00, neutral_phase_drift=0.129731385, neutral_norm_delta=-0.001731310, gspread_delta=-0.001842117
  parent_level=5, age=4, count=243, stored_drift=0.000e+00, neutral_phase_drift=0.166683635, neutral_norm_delta=-0.001435223, gspread_delta=-0.001490883
  parent_level=6, age=3, count=729, stored_drift=0.000e+00, neutral_phase_drift=0.196052203, neutral_norm_delta=-0.001240671, gspread_delta=-0.001262499
  parent_level=7, age=2, count=2187, stored_drift=0.000e+00, neutral_phase_drift=0.215132541, neutral_norm_delta=-0.001074597, gspread_delta=-0.001075910
  parent_level=8, age=1, count=6561, stored_drift=0.000e+00, neutral_phase_drift=0.213996639, neutral_norm_delta=-0.000839654, gspread_delta=-0.000831607
  parent_level=9, age=0, count=19683, stored_drift=0.000e+00, neutral_phase_drift=0.000000000, neutral_norm_delta=0.000000000, gspread_delta=0.000000000

FINAL LEVEL LOOP SUMMARY
  stored_response / parent_child_ring: count=9841, mean_abs_res=0.137512566, p95=0.299574848
  stored_response / parent_fan_triangle: count=29523, mean_abs_res=0.118213050, p95=0.270215921
  stored_response / sibling_cycle: count=9841, mean_abs_res=0.079301136, p95=0.175869757
  refreshed_neutral / parent_child_ring: count=9841, mean_abs_res=85.165026394, p95=124.132204097
  refreshed_neutral / parent_fan_triangle: count=29523, mean_abs_res=29.311610830, p95=41.729835728
  refreshed_neutral / sibling_cycle: count=9841, mean_abs_res=88.671493215, p95=128.689419433
```

## Final level age table

|   parent_level |   age |   count |   stored_phase_drift_abs_mean |   neutral_phase_drift_abs_mean |   neutral_norm_delta_mean |   g_rel_std_delta_mean |   g_spread_delta_mean |
|---------------:|------:|--------:|------------------------------:|-------------------------------:|--------------------------:|-----------------------:|----------------------:|
|              0 |     9 |       1 |                             0 |                      0.0130521 |              -0.0103194   |           -0.00486461  |          -0.0117188   |
|              1 |     8 |       3 |                             0 |                      0.0300805 |              -0.00521937  |           -0.00246043  |          -0.00588524  |
|              2 |     7 |       9 |                             0 |                      0.055517  |              -0.00320677  |           -0.00151168  |          -0.00356643  |
|              3 |     6 |      27 |                             0 |                      0.0903202 |              -0.00224319  |           -0.00105745  |          -0.00244497  |
|              4 |     5 |      81 |                             0 |                      0.129731  |              -0.00173131  |           -0.000816147 |          -0.00184212  |
|              5 |     4 |     243 |                             0 |                      0.166684  |              -0.00143522  |           -0.000676571 |          -0.00149088  |
|              6 |     3 |     729 |                             0 |                      0.196052  |              -0.00124067  |           -0.000584858 |          -0.0012625   |
|              7 |     2 |    2187 |                             0 |                      0.215133  |              -0.0010746   |           -0.00050657  |          -0.00107591  |
|              8 |     1 |    6561 |                             0 |                      0.213997  |              -0.000839654 |           -0.000395817 |          -0.000831607 |
|              9 |     0 |   19683 |                             0 |                      0         |               0           |            0           |           0           |

## Final level loop table

| phase_kind        | loop_mode           |   count |   mean_abs_residual |   p95_abs_residual |
|:------------------|:--------------------|--------:|--------------------:|-------------------:|
| stored_response   | parent_child_ring   |    9841 |           0.137513  |           0.299575 |
| stored_response   | parent_fan_triangle |   29523 |           0.118213  |           0.270216 |
| stored_response   | sibling_cycle       |    9841 |           0.0793011 |           0.17587  |
| refreshed_neutral | parent_child_ring   |    9841 |          85.165     |         124.132    |
| refreshed_neutral | parent_fan_triangle |   29523 |          29.3116    |          41.7298   |
| refreshed_neutral | sibling_cycle       |    9841 |          88.6715    |         128.689    |

## Main findings

### 1. Stored response phase freezes exactly

At every level:

```text
stored_drift = 0
```

Final level table:

```text
stored_phase_drift_abs_mean = 0
```

for all parent levels.

This is not surprising; it follows from the current operator definition. The stored local response block records the local birth event and is not refreshed by later subtree growth.

Interpretation:

```text
The previous local J-plane / response-phase operator is a frozen birth-history record.
```

### 2. Current conductance snapshots continue to age

The refreshed child-conductance neutral phase does drift after local completion.

At final level 10:

```text
parent_level 0 age 9: neutral phase drift ≈ 0.0131°
parent_level 4 age 5: neutral phase drift ≈ 0.1297°
parent_level 7 age 2: neutral phase drift ≈ 0.2151°
parent_level 8 age 1: neutral phase drift ≈ 0.2140°
parent_level 9 age 0: neutral phase drift = 0
```

So later descendant growth does affect the current child conductance snapshot.

### 3. Conductance imbalance slightly relaxes

For all aged layers, the neutral norm / spread deltas are negative:

```text
neutral_norm_delta < 0
g_spread_delta     < 0
```

Example at final level 10:

```text
parent_level 0: gspread_delta ≈ -0.01172
parent_level 4: gspread_delta ≈ -0.00184
parent_level 8: gspread_delta ≈ -0.00083
```

Interpretation:

```text
Current conductance imbalance relaxes slightly as descendant subtrees grow.
```

This is a real current-state aging effect, but it is not the same as modifying the stored response operator.

### 4. Stored loop residual remains the relevant operator diagnostic

At final level 10, stored-response loop residuals are small:

```text
sibling_cycle:        0.0793°
parent_fan_triangle:  0.1182°
parent_child_ring:    0.1375°
```

These match the earlier residual-curvature scale.

### 5. Refreshed-neutral loop residual is not a valid holonomy diagnostic

The refreshed-neutral loop residuals become huge:

```text
sibling_cycle:        ~88.7°
parent_child_ring:    ~85.2°
parent_fan_triangle:  ~29.3°
```

This should **not** be interpreted as physical curvature.

Reason:

```text
The neutral phasor angle of current child conductances is not a connection phase.
It is an imbalance direction in the sibling plane.
Using it additively around loops is not justified.
```

So the refreshed-neutral part is useful only as an aging/current-state diagnostic, not as a loop-holonomy diagnostic.

## Interpretation

The current model contains two layers:

```text
1. Frozen event record:
   stored local response operator
   -> J-plane / response phase / loop residual tests

2. Current conductance state:
   child conductances keep aging under descendant growth
   -> imbalance relaxes slightly
```

This is an important clarification.

The old-interior residuals seen before are not relaxing in the stored operator. They are frozen finite-depth birth-history defects. The global residual decreases because newer, more homogeneous layers dominate the tower count, not because old stored response blocks are later smoothed.

## Current CNNA meaning

The shell-normalized model now suggests:

```text
Birth creates a local response record.
That record carries the local J-plane and response phase.
Later growth changes current conductance loads,
but does not rewrite the historical response record.
```

If CNNA wants true relaxation of old operator blocks, then the model needs an explicit derived rule for refreshing/reintegrating old response operators from later backreaction. That rule is not present yet.

## What this does not prove

Still not proven:

```text
physical time
physical i
modular flow
global Hilbert/vN structure
unique metric G
Lean theorem
```

## Next decision

There are now two paths.

### Path A — keep stored-response semantics

Then old response records are intentionally frozen. The next test should study the tower as a growing archive of local response events:

```text
test_response_record_tower_spectrum.py
```

Goal:

```text
build block/tower spectrum from frozen local response records
track phase-density distribution
test spectral convergence / type-like direction
```

### Path B — introduce a refresh/backreaction-update semantics

Then we need a new rule:

```text
descendant backreaction updates old local response operators
```

and the next test should be:

```text
test_response_operator_refresh_rule.py
```

But this would be a new model assumption unless we derive the refresh rule from provenance.
