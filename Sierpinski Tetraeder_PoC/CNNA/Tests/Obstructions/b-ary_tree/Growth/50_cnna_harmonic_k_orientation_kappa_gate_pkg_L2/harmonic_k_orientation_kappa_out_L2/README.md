# Harmonic K orientation / kappa gate

Default:

```bash
python3 test_harmonic_k_orientation_kappa_gate.py
```

L3 local run:

```bash
python3 test_harmonic_k_orientation_kappa_gate.py \
  --max-level 3 \
  --max-cascade-per-birth 3 \
  --max-auto-pairings 2 \
  --out harmonic_kappa_out_L3 \
  --zip cnna_harmonic_k_orientation_kappa_gate_pkg_L3.zip
```

This test is diagnostic.  A kappa sign bias depends on the current face-orientation convention and must later be replaced by a Lean-level/provenance-level orientation definition.
