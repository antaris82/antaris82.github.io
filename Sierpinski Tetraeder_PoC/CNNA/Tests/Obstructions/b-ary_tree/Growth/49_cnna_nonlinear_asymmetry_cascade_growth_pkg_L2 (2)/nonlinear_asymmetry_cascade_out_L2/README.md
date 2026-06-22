# Nonlinear asymmetry-cascade growth package

Run default:

```bash
python3 test_nonlinear_asymmetry_cascade_growth.py --max-level 2 --response-mode power_saturating --max-auto-pairings 2 --max-cascade-per-birth 2
```

For a faster audit, reduce `--max-pair-candidates` and `--max-rows`. For scaling, use local-indexing before increasing level too aggressively.
