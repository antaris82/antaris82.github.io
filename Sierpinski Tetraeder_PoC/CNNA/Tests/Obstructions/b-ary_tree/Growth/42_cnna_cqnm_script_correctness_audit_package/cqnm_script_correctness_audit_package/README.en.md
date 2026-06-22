# CQNM Script Correctness Audit Package

This package audits the uploaded `test_cqnm_sminus1_saturation_link_curvature_gate.py` against the CNNA requirements:

- Script-1 dynamic birth/backreaction growth
- CQNM/s=-1 saturated geometry
- frustration/noncommutativity via real DtN plaquette commutators
- topology/cohomology gates

Run:

```bash
python3 audit_cqnm_script_correctness.py
```

Outputs:

- `audit_report.json`
- `audit_report.stdout`
- `RESULTS.md`
- `SUMMARY.md`
