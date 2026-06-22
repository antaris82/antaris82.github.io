# CNNA non-shelling provenance face-pairing gate

Run one case:

```bash
python3 test_cqnm_non_shelling_provenance_face_pairing_gate.py --max-level 3 --out out_L3
```

The included package already contains three runs:

- `out_L3_real`
- `out_L3_symmetrized`
- `out_L3_no_backreaction`

Key files:

- `move_candidates.csv`: all enumerated candidate moves.
- `move_class_summary.csv`: per-class best candidates.
- `summary.json`: machine-readable run summary.
- `comparative_move_audit.csv`: compact comparison of the three runs.
- `comparative_summary.json`: combined JSON summary.
