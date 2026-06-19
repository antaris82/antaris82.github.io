# Generator56a support-module policy

This package keeps the active entry point clean without physically quarantining
all useful microstep modules.

Active entry point:

- `GeneratorSpine.lean`
- `SchurDtnDerivedSeedRebase.lean`
- `SchurDtnDerivedAddressSeed.lean`

The remaining Lean modules stay in `CNNA.*` as support/comparison/proof-witness
modules. They are not exported one-by-one from `GeneratorSpine.lean`, but they
remain available for inspection, comparison, and transitive proof support.

This corrects the overly broad Generator56 quarantine: useful microsteps are not
moved into `CNNA.Quarantine.*`; only the active spine is kept small.
