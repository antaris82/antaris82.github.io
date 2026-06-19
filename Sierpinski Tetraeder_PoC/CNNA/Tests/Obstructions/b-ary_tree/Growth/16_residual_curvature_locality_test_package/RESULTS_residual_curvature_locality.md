# RESULTS — Residual Curvature Locality Test

## Status

Python diagnostic completed successfully to **level 10** for the shell-normalized kernel:

```text
K(d) = 1 / (3^(d-1) d²)
```

Final size:

```text
88573 nodes
29524 completed local response operators
49205 polar loop records at level 10
```

This is a numerical/model diagnostic, not a Lean theorem.

## Purpose

The previous clock-renormalization test showed:

```text
theta_i = theta_inf + finite_level_gap(L) + small_local_curvature_i
```

This test asks where the small local/residual curvature sits:

```text
near the active growth frontier
or distributed through the old interior of the tower?
```

## Diagnostics

Primary curvature diagnostic:

```text
level-centered residual =
sum(theta_i) - sum(mean theta at each source parent level)
```

This removes the finite-level/background phase. The clock residual using fixed `theta_inf` is reported separately.

Distance convention:

```text
distance_to_frontier = max_completed_parent_level - max_source_level_in_loop
```

So:

```text
distance 0 = closest available loop layer to the active completed frontier
larger distance = deeper/older interior
```

## Raw summary

```text
RESIDUAL CURVATURE LOCALITY TEST
  max_level=10

  L=1: nodes=4, completed=1, max_parent_level=0, local_dev=0.000000000, loop_level_res=nan, clock_res=nan, elapsed=0.00s
  L=2: nodes=13, completed=4, max_parent_level=1, local_dev=0.171115431, loop_level_res=0.136892345, clock_res=30.007854896, elapsed=0.01s
  L=3: nodes=40, completed=13, max_parent_level=2, local_dev=0.165398882, loop_level_res=0.385962622, clock_res=20.317291864, elapsed=0.01s
  L=4: nodes=121, completed=40, max_parent_level=3, local_dev=0.133719939, loop_level_res=0.355452909, clock_res=13.100305185, elapsed=0.03s
  L=5: nodes=364, completed=121, max_parent_level=4, local_dev=0.104483377, loop_level_res=0.280956939, clock_res=8.171229962, elapsed=0.10s
  L=6: nodes=1093, completed=364, max_parent_level=5, local_dev=0.083938065, loop_level_res=0.217305724, clock_res=4.997816092, elapsed=0.31s
  L=7: nodes=3280, completed=1093, max_parent_level=6, local_dev=0.070810478, loop_level_res=0.171757124, clock_res=3.028072216, elapsed=0.95s
  L=8: nodes=9841, completed=3280, max_parent_level=7, local_dev=0.063032657, loop_level_res=0.142420786, clock_res=1.830803448, elapsed=2.41s
  L=9: nodes=29524, completed=9841, max_parent_level=8, local_dev=0.058517171, loop_level_res=0.124682857, clock_res=1.111206852, elapsed=7.04s
  L=10: nodes=88573, completed=29524, max_parent_level=9, local_dev=0.055924444, loop_level_res=0.114290571, clock_res=0.681150781, elapsed=20.60s

FINAL LEVEL POLAR BY DISTANCE TO FRONTIER
  dist=0: count=32805, level_res=0.109095220, p95=0.253056244, clock_res=0.466155519
  dist=1: count=10935, level_res=0.115817947, p95=0.269681967, clock_res=0.751573071
  dist=2: count=3645, level_res=0.127772738, p95=0.299619607, clock_res=1.232990235
  dist=3: count=1215, level_res=0.149076545, p95=0.353096789, clock_res=2.047253249
  dist=4: count=405, level_res=0.185873026, p95=0.459639949, clock_res=3.430698131
  dist=5: count=135, level_res=0.245088509, p95=0.543297939, clock_res=5.797971522
  dist=6: count=45, level_res=0.341893037, p95=0.790215475, clock_res=9.892755550
  dist=7: count=15, level_res=0.468986047, p95=0.863073960, clock_res=17.087104187
  dist=8: count=5, level_res=0.136892345, p95=0.330805012, clock_res=30.007854896

FINAL LEVEL POLAR LOCAL DEVIATION BY DISTANCE
  dist=0: count=19683, local_dev=0.054628146, p95=0.111506411, clock_gap=-0.126637025
  dist=1: count=6561, local_dev=0.056259773, p95=0.116306076, clock_gap=-0.202783324
  dist=2: count=2187, local_dev=0.059145525, p95=0.124755686, clock_gap=-0.331116367
  dist=3: count=729, local_dev=0.064255688, p95=0.139399568, clock_gap=-0.547888695
  dist=4: count=243, local_dev=0.073707684, p95=0.163465900, clock_gap=-0.915400477
  dist=5: count=81, local_dev=0.090045569, p95=0.205278192, clock_gap=-1.542171233
  dist=6: count=27, local_dev=0.118467115, p95=0.247505371, clock_gap=-2.620950703
  dist=7: count=9, local_dev=0.162858193, p95=0.346200272, clock_gap=-4.503092328
  dist=8: count=3, local_dev=0.228153908, p95=0.336517937, clock_gap=-7.849603252
  dist=9: count=1, local_dev=0.000000000, p95=0.000000000, clock_gap=-13.961008865

FINAL LEVEL POLAR BY MODE × DISTANCE
  parent_child_ring dist=0: count=6561, level_res=0.131171484, p95=0.278724345
  parent_child_ring dist=1: count=2187, level_res=0.139296035, p95=0.301217531
  parent_child_ring dist=2: count=729, level_res=0.153853488, p95=0.342952324
  parent_child_ring dist=3: count=243, level_res=0.180615038, p95=0.404447832
  parent_child_ring dist=4: count=81, level_res=0.226436587, p95=0.523369644
  parent_child_ring dist=5: count=27, level_res=0.300311708, p95=0.629218443
  parent_child_ring dist=6: count=9, level_res=0.418627128, p95=0.886961813
  parent_child_ring dist=7: count=3, level_res=0.579519025, p95=0.854170511
  parent_child_ring dist=8: count=1, level_res=0.000000000, p95=0.000000000
  parent_fan_triangle dist=0: count=19683, level_res=0.113042136, p95=0.257922626
  parent_fan_triangle dist=1: count=6561, level_res=0.119767496, p95=0.274303578
  parent_fan_triangle dist=2: count=2187, level_res=0.131630553, p95=0.300411829
  parent_fan_triangle dist=3: count=729, level_res=0.152487830, p95=0.351246029
  parent_fan_triangle dist=4: count=243, level_res=0.188684706, p95=0.445834647
  parent_fan_triangle dist=5: count=81, level_res=0.247762081, p95=0.524430231
  parent_fan_triangle dist=6: count=27, level_res=0.345023040, p95=0.779734156
  parent_fan_triangle dist=7: count=9, level_res=0.471348698, p95=0.821201294
  parent_fan_triangle dist=8: count=3, level_res=0.228153908, p95=0.336517937
  sibling_cycle dist=0: count=6561, level_res=0.075178207, p95=0.162156384
  sibling_cycle dist=1: count=2187, level_res=0.080491212, p95=0.177038455
  sibling_cycle dist=2: count=729, level_res=0.090118542, p95=0.203943751
  sibling_cycle dist=3: count=243, level_res=0.107304197, p95=0.241057525
  sibling_cycle dist=4: count=81, level_res=0.136874422, p95=0.318091453
  sibling_cycle dist=5: count=27, level_res=0.181844593, p95=0.381922675
  sibling_cycle dist=6: count=9, level_res=0.255768934, p95=0.540761541
  sibling_cycle dist=7: count=3, level_res=0.351365117, p95=0.517652573
  sibling_cycle dist=8: count=1, level_res=0.000000000, p95=0.000000000
```

## Polar level trend

|   global_level |   nodes |   completed |   local_mean_abs_dev |   loop_mean_abs_level_residual |   loop_mean_abs_clock_residual |
|---------------:|--------:|------------:|---------------------:|-------------------------------:|-------------------------------:|
|              1 |       4 |           1 |            0         |                     nan        |                     nan        |
|              2 |      13 |           4 |            0.171115  |                       0.136892 |                      30.0079   |
|              3 |      40 |          13 |            0.165399  |                       0.385963 |                      20.3173   |
|              4 |     121 |          40 |            0.13372   |                       0.355453 |                      13.1003   |
|              5 |     364 |         121 |            0.104483  |                       0.280957 |                       8.17123  |
|              6 |    1093 |         364 |            0.0839381 |                       0.217306 |                       4.99782  |
|              7 |    3280 |        1093 |            0.0708105 |                       0.171757 |                       3.02807  |
|              8 |    9841 |        3280 |            0.0630327 |                       0.142421 |                       1.8308   |
|              9 |   29524 |        9841 |            0.0585172 |                       0.124683 |                       1.11121  |
|             10 |   88573 |       29524 |            0.0559244 |                       0.114291 |                       0.681151 |

## Final level polar residual by distance to frontier

|   distance_to_frontier |   count |   mean_abs_level_residual |   p95_abs_level_residual |   mean_abs_clock_residual |
|-----------------------:|--------:|--------------------------:|-------------------------:|--------------------------:|
|                      0 |   32805 |                  0.109095 |                 0.253056 |                  0.466156 |
|                      1 |   10935 |                  0.115818 |                 0.269682 |                  0.751573 |
|                      2 |    3645 |                  0.127773 |                 0.29962  |                  1.23299  |
|                      3 |    1215 |                  0.149077 |                 0.353097 |                  2.04725  |
|                      4 |     405 |                  0.185873 |                 0.45964  |                  3.4307   |
|                      5 |     135 |                  0.245089 |                 0.543298 |                  5.79797  |
|                      6 |      45 |                  0.341893 |                 0.790215 |                  9.89276  |
|                      7 |      15 |                  0.468986 |                 0.863074 |                 17.0871   |
|                      8 |       5 |                  0.136892 |                 0.330805 |                 30.0079   |

## Final level polar local phase deviation by distance

|   distance_to_frontier |   count |   mean_abs_local_deviation |   p95_abs_local_deviation |   mean_clock_gap |
|-----------------------:|--------:|---------------------------:|--------------------------:|-----------------:|
|                      0 |   19683 |                  0.0546281 |                  0.111506 |        -0.126637 |
|                      1 |    6561 |                  0.0562598 |                  0.116306 |        -0.202783 |
|                      2 |    2187 |                  0.0591455 |                  0.124756 |        -0.331116 |
|                      3 |     729 |                  0.0642557 |                  0.1394   |        -0.547889 |
|                      4 |     243 |                  0.0737077 |                  0.163466 |        -0.9154   |
|                      5 |      81 |                  0.0900456 |                  0.205278 |        -1.54217  |
|                      6 |      27 |                  0.118467  |                  0.247505 |        -2.62095  |
|                      7 |       9 |                  0.162858  |                  0.3462   |        -4.50309  |
|                      8 |       3 |                  0.228154  |                  0.336518 |        -7.8496   |
|                      9 |       1 |                  0         |                  0        |       -13.961    |

## Main findings

### 1. Residual curvature is **not concentrated at the active frontier**

At level 10, polar level-centered loop residuals are smallest near the frontier:

```text
distance 0: 0.1091°
distance 1: 0.1158°
distance 2: 0.1278°
distance 3: 0.1491°
```

They increase into the older interior:

```text
distance 5: 0.2451°
distance 6: 0.3419°
distance 7: 0.4690°
```

The tiny count at distance 8 is an early-root transient/control edge case and should not be overinterpreted.

Interpretation:

```text
Residual curvature is mainly an old-interior/transient memory effect,
not a fresh frontier-localized effect.
```

### 2. Local phase deviations show the same pattern

At level 10, polar local deviations:

```text
distance 0: 0.0546°
distance 1: 0.0563°
distance 2: 0.0591°
distance 3: 0.0643°
distance 4: 0.0737°
distance 5: 0.0900°
distance 6: 0.1185°
distance 7: 0.1629°
distance 8: 0.2282°
```

Again, the newest completed layer is the most homogeneous. The older interior carries larger residual deviations.

### 3. Clock residual is dominated by old finite-level gap

Clock residual grows strongly with distance:

```text
distance 0: 0.4662°
distance 1: 0.7516°
distance 2: 1.2330°
distance 3: 2.0473°
distance 4: 3.4307°
distance 5: 5.7980°
distance 6: 9.8928°
distance 7: 17.0871°
```

That is exactly the finite-level `theta_L - theta_inf` memory of older levels. It confirms that fixed `theta_inf` subtraction is asymptotic, not the best finite-level curvature diagnostic.

### 4. Sibling cycles are cleaner than parent-fan and parent-child rings

At distance 0:

```text
sibling_cycle:       0.0752°
parent_fan_triangle: 0.1130°
parent_child_ring:   0.1312°
```

This ordering persists across distances.

Interpretation:

```text
Pure child-subtower loops are the cleanest.
Loops involving the parent carry more vertical/transient mismatch.
```

### 5. Global trend still decays

Polar level-centered loop residual across global level:

```text
L8:  0.1424°
L9:  0.1247°
L10: 0.1143°
```

So although older interior layers carry larger residuals, their relative contribution is diluted as the frontier grows.

## Interpretation

The residual curvature is best understood as:

```text
finite-depth memory in old interior layers
+ small mismatch from vertical parent involvement
```

not as:

```text
active-frontier curvature burst
```

The newest completed layers are actually the most homogeneous and lowest-residual part of the tower.

## Current CNNA reading

The shell-normalized model now looks like:

```text
frontier growth generates a coherent phase-density layer;
as the tower grows, newer layers are more homogeneous;
old layers retain finite-depth transient memory;
global residual decreases because the tower mass shifts to newer homogeneous layers.
```

This supports the idea of an asymptotic phase/clock background with decaying finite-depth defects.

## What this does not prove

Still not proven:

```text
physical time
physical i
modular flow
Type III
Lean theorem
unique metric G
```

## Next recommended test

Now the natural next diagnostic is not another global average. It is a **memory-flow / aging test**:

```text
test_phase_aging_of_fixed_layers.py
```

Question:

```text
If a fixed parent level is observed while the tower grows from L to L+k,
does its residual continue to change/relax, or is it frozen after local completion?
```

This will tell us whether old-interior residuals are:

```text
frozen birth-history defects
or
slowly relaxing under later ancestor backreaction.
```
