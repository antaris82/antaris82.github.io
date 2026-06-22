# RESULTS — candidate sharp-closure obstruction gate

## Focus table: real_growth_live_pair_plus_response with signed R

| family | gen | init dim | init prod resid | init # resid | prod alg dim | full M? | prod alg # resid | forced dim | extra dim | forced closed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diag_core | 4 | 4 | 0.499154 | 0.0224152 | 36 | True | 1.53496e-15 | 36 | 0 | True |
| cap_only | 6 | 6 | 0.433013 | 0.9398 | 7 | False | 0.959983 | 17 | 10 | True |
| pair_only | 4 | 4 | 0.707107 | 8.61732e-16 | 5 | False | 1.57532e-15 | 5 | 0 | True |
| all_seed | 10 | 10 | 0.893809 | 0.918548 | 36 | True | 2.2629e-15 | 36 | 0 | True |

## Highest # obstruction rows

| case | R | family | prod alg dim | # residual | forced dim | extra |
|---|---:|---|---:|---:|---:|---:|
| real_growth_live_pair_only:live:pair_only | unsigned | cap_only | 7 | 0.960567 | 17 | 10 |
| real_growth_live_pair_only:live:pair_only | signed | cap_only | 7 | 0.960567 | 17 | 10 |
| real_growth_record_plus_live_pair_plus_response:full:pair_plus_response | signed | cap_only | 7 | 0.959983 | 17 | 10 |
| real_growth_record_plus_live_pair_plus_response:full:pair_plus_response | unsigned | cap_only | 7 | 0.959983 | 17 | 10 |
| real_growth_record_only_pair_plus_response:record:pair_plus_response | signed | cap_only | 7 | 0.959983 | 17 | 10 |
| no_backreaction_record_pair_plus_response:record:pair_plus_response | signed | cap_only | 7 | 0.959983 | 17 | 10 |
| real_growth_live_pair_plus_response:live:pair_plus_response | signed | cap_only | 7 | 0.959983 | 17 | 10 |
| real_growth_live_pair_plus_response:live:pair_plus_response | unsigned | cap_only | 7 | 0.959983 | 17 | 10 |
| real_growth_record_only_pair_plus_response:record:pair_plus_response | unsigned | cap_only | 7 | 0.959983 | 17 | 10 |
| no_backreaction_record_pair_plus_response:record:pair_plus_response | unsigned | cap_only | 7 | 0.959983 | 17 | 10 |
| real_growth_record_only_pair_plus_response:record:pair_plus_response | unsigned | all_seed | 36 | 2.86468e-15 | 36 | 0 |
| no_backreaction_record_pair_plus_response:record:pair_plus_response | unsigned | all_seed | 36 | 2.86468e-15 | 36 | 0 |
| real_growth_live_pair_only:live:pair_only | signed | pair_only | 5 | 2.65892e-15 | 5 | 0 |
| real_growth_record_only_pair_plus_response:record:pair_plus_response | signed | diag_core | 36 | 2.4436e-15 | 36 | 0 |
| no_backreaction_record_pair_plus_response:record:pair_plus_response | signed | diag_core | 36 | 2.4436e-15 | 36 | 0 |
| real_growth_record_only_pair_plus_response:record:pair_plus_response | unsigned | diag_core | 36 | 2.27657e-15 | 36 | 0 |
| no_backreaction_record_pair_plus_response:record:pair_plus_response | unsigned | diag_core | 36 | 2.27657e-15 | 36 | 0 |
| real_growth_live_pair_plus_response:live:pair_plus_response | signed | all_seed | 36 | 2.2629e-15 | 36 | 0 |

## Main interpretation

The test separates the obstruction into families rather than treating the previous product algebra as one black box.

Read the result in three layers:

```text
1. initial # residual:
   whether the raw generator span is already # stable.

2. product algebra # residual:
   whether the product-closed algebra from the raw generators is # stable.

3. forced #-product dimension:
   how much extra finite real operator material is needed if # stability is imposed.
```

A derived real `*`-candidate would require product closure and # closure to appear without a large forced enlargement, without full-matrix saturation as the only repair, and without importing positivity, norm, Hodge duality, or complex scalars.

## Negative-control condition

`strict_symmetrized_control` remains trivial when β₂ and the real C² carrier vanish.  `decision_used_delta_beta_any` remains false in all generated rows.

## Next test

```text
test_native_reversal_search_gate.py
```

Goal: do not force `A# = R A^T R`.  Instead, search within the actually generated real product algebra for an internally defined involutive anti-automorphism candidate built from available real operators only: `R`, pair maps, cap maps, and closed/harmonic diagonal actions.  The test should reject candidates that require full ambient matrix algebra or an imported transpose/inner-product interpretation.
