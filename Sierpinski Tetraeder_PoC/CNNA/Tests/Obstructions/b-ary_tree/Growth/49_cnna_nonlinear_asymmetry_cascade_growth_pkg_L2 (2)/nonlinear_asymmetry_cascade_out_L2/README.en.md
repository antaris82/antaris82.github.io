# Nonlinear asymmetry-cascade growth package

Default run:

```bash
python3 test_nonlinear_asymmetry_cascade_growth.py --max-level 2 --response-mode power_saturating --max-auto-pairings 2 --max-cascade-per-birth 2
```

For a faster audit, reduce `--max-pair-candidates` and `--max-rows`. For scaling, use local indexing before increasing the level too aggressively.
