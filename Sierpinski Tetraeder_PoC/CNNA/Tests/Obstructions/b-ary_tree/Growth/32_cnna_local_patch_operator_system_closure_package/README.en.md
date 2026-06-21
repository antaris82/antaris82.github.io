# CNNA Local Patch Operator-System Closure Diagnostic

This package contains `test_local_patch_operator_system_closure.py`.

It is the next stage after the pair-level Record/Live L/T operator tests.
Instead of testing a single Double-History pair, it constructs a local patch

```text
X_P = W_p1 ⊕ ... ⊕ W_pn,
W_p = B_p^record ⊕ B_p^live, B_p = R^3.
```

The primary patch groups together completed cells with the same `(level, suffix)` and
different root sectors. These are precisely the multi-history cells that are
identified under suffix-forgetting Double-History gluing.

No `i`, no `J`, no complex phase, no `J²=-I`, and no C*-norm are inserted.
The test asks only whether a real local operator-system closure becomes visible
at the patch level.

## Default run

```bash
python3 test_local_patch_operator_system_closure.py --out local_patch_operator_system_out_L45
```

The default run covers L4/L5 and the controls:

- real growth,
- symmetrized birth,
- no backreaction,
- identical-history patch,
- random same-level patch.

## Main outputs

- `patch_rows_all.csv`
- `summary_table_all.csv`
- `RESULTS_local_patch_operator_system_closure.md`
- `SUMMARY.txt`
