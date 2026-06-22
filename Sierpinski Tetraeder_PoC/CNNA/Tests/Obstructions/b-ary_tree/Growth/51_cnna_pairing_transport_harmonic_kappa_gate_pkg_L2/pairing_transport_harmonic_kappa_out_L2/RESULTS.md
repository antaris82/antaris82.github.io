# RESULTS — pairing-transport harmonic kappa gate

## Comparative table

| variant | beta auto | pairings | scalar K harm | pair axial harm | pair scalar harm | Hdim | pair kappa | pair birth kappa | delta beta2 sum | used Δβ? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_growth | (1,0,2,0) | 2 | 0.121617 | 0.220125 | 0.208125 | 2 | 0.0339795 | 0.0339795 | 2 | False |
| strict_symmetrized_control | (1,0,0,0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| no_backreaction | (1,0,2,0) | 2 | 0.128934 | 0.216054 | 0.20887 | 2 | 0.0296715 | 0.0296715 | 2 | False |

## Interpretation

The test distinguishes three levels:

```text
1. beta2 carrier opens.
2. scalar |K| has a harmonic residual.
3. transported axial K_pair has a harmonic and kappa-biased component.
```

A strong candidate requires all three in real_growth and a strict kill in strict_symmetrized_control.

Important: `decision_used_delta_beta_any` must remain false.  If true, the topology would have entered the selection rule.  In this package, topology and harmonic projection are measured after the fact.

## Current status

See `comparative_pairing_transport_kappa_summary.csv` and each variant directory for pair-level logs and top harmonic-face support.
