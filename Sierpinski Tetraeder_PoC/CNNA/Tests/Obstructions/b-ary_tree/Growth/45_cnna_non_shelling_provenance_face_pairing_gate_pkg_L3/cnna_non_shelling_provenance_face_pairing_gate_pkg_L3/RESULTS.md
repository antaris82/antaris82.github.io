# RESULTS

## Core diagnosis

The growth error is now localized more precisely:

```text
Current growth = local outward stacking / shelling-like ball growth.
Needed growth  = local outward birth + complement boundary-pairing/gluing move.
```

The current complex after real-growth L=3 has:

```text
beta = (1, 0, 0, 0)
boundary_fraction = 0.678571
K_mean = 0.909526
harmonic_ratio = 0
```

So the local operator sector is present, but the base topological carrier is still trivial.

## Move audit result

For the real-growth run the enumerated candidate classes are:

```text
shelling_disk_move : 88
cap_move           : 30
handle_candidate   : 188
quotient_candidate : 218
illegal            : 123
```

The best handle candidate has response rank `3` and changes beta2 by `1` in the audited candidate application. The best quotient candidate has response rank `1` and reduces boundary faces by `-8`.

## Interpretation

This is a useful result because it distinguishes three possibilities:

1. No non-shelling candidates exist. Then the outward geometry never even creates complementary boundary patches.
2. Non-shelling candidates exist but response ranks them poorly. Then the score is wrong or not topology-aware.
3. Non-shelling candidates exist and response ranks them high, but the growth law does not apply them. Then the missing object is a permitted complement-pairing operation.

The real-growth run falls into case 3.

## Control warning

The controls fall into a similar case:

```text
symmetrized_birth : top_handle_rank=1, top_quotient_rank=2
no_backreaction   : top_handle_rank=1, top_quotient_rank=2
```

Therefore the response score is not yet sufficiently derived/discriminating. We should not claim that the handle/quotient move is forced by real backreaction. We can only claim that the current growth law is missing the move class.

## Next test

`test_apply_top_ranked_non_shelling_move_and_reaudit.py`

Apply the top-ranked legal handle and quotient candidates one at a time, recompute Betti numbers and K harmonic projection, and compare against equally high local shelling/cap moves. Then add ablation gates that require a real-growth candidate to outperform symmetrized/no-backreaction controls before calling it provenance-forced.
