# SUMMARY

This package tests whether the missing layer is not local sibling asymmetry, but inter-parent-fan transverse transport.

The base growth remains Script-1-like: a newborn detects its parent line and older siblings, then feeds back to ancestors and older siblings. The new diagnostic branch adds an explicit transport of the local Z3/transverse orientation into neighboring parent fans through the common grandparent context.

This is a controlled hypothesis test, not a derived CNNA law.

## Main numerical comparison

| variant | growth rule | interfan strength | interfan edges | beta | base harmonic | top handle rank | top quotient rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| real_local_only | real_growth | 0 | 0 | (1,0,0,0) | 0 | 1 | 16 |
| real_interfan_transport | real_growth | 0.16 | 36 | (1,0,0,0) | 0 | 1 | 7 |
| sym_interfan_transport | symmetrized_birth | 0.16 | 36 | (1,0,0,0) | 0 | 1 | 9 |
| no_backreaction_interfan_transport | no_backreaction | 0.16 | 36 | (1,0,0,0) | 0 | 1 | 6 |

## Core reading

The local sequential sibling offset is not the missing ingredient; it already exists in real_growth. The missing layer is whether that local transverse orientation is transported between neighboring parent-fans strongly enough to prioritize topology-effective non-shelling moves over local shelling/cap moves.

If interfan transport improves the response rank of beta-positive handle/quotient moves only in real_growth, this supports the inter-fan diagnosis. If symmetrized/no-backreaction controls improve similarly, the move remains externally introduced rather than CNNA-derived.