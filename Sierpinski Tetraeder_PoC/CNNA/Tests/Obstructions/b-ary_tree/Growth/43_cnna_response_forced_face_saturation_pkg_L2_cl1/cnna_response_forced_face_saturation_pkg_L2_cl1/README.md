# Response-forced CQNM face saturation test

Run:

```bash
python3 test_cqnm_response_forced_face_saturation.py --max-level 4 --mode linear --closure-passes 2 --outdir out --package pkg
```

Outputs:

- `growth_summary.csv`
- `geometry_operator_summary.csv`
- `face_saturation_decisions.csv`
- `operator_kill_controls.csv`
- `summary.json`
- `SUMMARY.md`
- `RESULTS.md`
- `SOURCE_AUDIT_1_40.md`

The test is a diagnostic surrogate. It does not claim a derived CNNA theorem.
