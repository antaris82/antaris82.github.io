# Notes

## Key distinction

The diagnostic distinguishes between two concepts of change:

- **Last-shell marginal update:** compares the same provenance node before and after the most recent growth shell. This is the definitive test of the locality of current growth. It confirms that frontier/active-parent nodes change the most and the root the least.
- **Cumulative aging since completion:** compares a completed local cell’s current live DtN data to its completion snapshot. This value can be large in older interior cells because memory and backreaction accumulate over many subsequent births. This is not the same as instantaneous local growth activity.

## Interpretation

The results support a two-scale interpretation:

- the current update is dominated by local and frontier processes;
- old interior cells store accumulated memory;
- a *-operator candidate should not be sought in a root-only or isolated old-pair subsystem, but rather in growth-defined local patches and, ultimately, in a local-net/limit construction.

## Current status

Positive:

- the last-shell marginal update gradient is clean and monotonic toward the active parent frontier;
- same-suffix growth patches remain structured and non-random.

Negative/open:

- finite degree-2 patch closure is still not a stable *-algebra;
- the patch-size/level trend is not yet sufficient as a limit signal;
- no J-candidate is extracted here.

## Next test

`test_active_shell_patch_net_refinement.py`

Goal: Restrict patch construction to active and near-active shells, then compare closure residuals across local refinements while treating old interior cells as accumulated memory/background rather than as the primary carrier.
