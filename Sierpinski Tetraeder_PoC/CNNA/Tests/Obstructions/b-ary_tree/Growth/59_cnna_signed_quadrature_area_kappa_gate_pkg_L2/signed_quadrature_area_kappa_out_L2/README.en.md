# Signed Quadrature Area Kappa Gate

Run:

```bash
python3 test_signed_quadrature_area_kappa_gate.py
```

Outputs:

- comparative_summary.json
- comparative_signed_quadrature_area_summary.csv
- phase_flip_comparison.csv
- RESULTS.md
- SUMMARY.md
- SOURCE_AUDIT.md
- per-variant signed pair/face logs

The next gate depends on this result:

- if the signed area is small or does not flip: treat Q/P as magnitude-only and search for a
  native operator involution or quadrature pairing instead of spatial orientation;
- if the signed area is nontrivial and flips: construct a strict derived real symplectic-form
  closure test.
