# TWO-LAYER MODEL SETTING — provisional CNNA working decision

## Status

We adopt the two-layer response semantics as a **provisional model setting** for the current CNNA growing real complement-network path.

This is not yet a theorem of the final theory. It is a disciplined working structure supported by the Gen66 Python stress tests and by a structural single-layer obstruction described below.

```text
Record layer:
  immutable birth/provenance handoff record

Live layer:
  current conductance/response state under later descendant/ancestor backreaction
```

In the current code/test vocabulary:

```text
record_only:
  uses W_birth / local_w as the frozen completion-time record

live_only:
  replays the local response law from current conductances and current ancestor environment

record_plus_live:
  keeps W_birth and adds an explicitly audited live increment channel
```

The current preferred working semantics is therefore not `live_only`, but a two-layer semantics:

```text
Response = Record ⊕ Live
```

where `⊕` is not yet claimed to be the final algebraic direct sum of the physical theory. It currently means: do not collapse historical provenance and current backreaction state into one untyped local operator.

## Why this is not treated as accidental

The numerical result from `test_response_operator_refresh_rule.py` is structurally suggestive:

```text
record_only:
  strongest vertical/tower gluing
  but larger old-interior residuals

live_only:
  smallest level-centered residual curvature
  but weaker vertical/tower gluing

record_plus_live:
  intermediate residual reduction
  while nearly preserving record-like gluing
```

This is exactly the pattern expected if two different roles are being mixed:

```text
Record role:
  preserves birth history, handoff identity, and vertical provenance coherence.

Live role:
  tracks current state, aging, descendant loading, and backreaction response.
```

A one-layer operator is therefore suspected to be too coarse: it tries to carry two incompatible invariance/covariance requirements.

## Structural single-layer obstruction

Let `N_t` be a growing complement network at construction stage `t`, and let a completed local sibling triple/parent event be denoted by `p`.

Define two role requirements for a local response object attached to `p`.

### R. Record invariance

A birth/provenance record must be invariant under future extensions of the network.

For every later extension `N_t -> N_s` with `s >= t_complete(p)`:

```text
Record_p(N_s) = Record_p(N_t_complete)
```

The record is allowed to be read later, but not rewritten by later descendants. Otherwise it no longer certifies the original birth handoff.

### L. Live-state covariance

A live response state must depend on the current conductance/backreaction state.

For every later extension `N_t -> N_s`:

```text
Live_p(N_s) = ReplayResponse(current child conductances,
                             current ancestor environment,
                             descendant/ancestor backreaction,
                             shell-normalized kernel)
```

Thus, if later descendants change the effective conductance environment of `p`, the live object is allowed, and in fact expected, to change.

### A. Nontrivial aging/backreaction

The Gen66 tests already exhibit nontrivial aging:

```text
there exist p and s > t_complete(p) such that
Live_p(N_s) != Live_p(N_t_complete)
```

The observed live drift in old interior layers is the numerical witness of this condition.

### Consequence

Assume a single-layer object `W_p` is supposed to satisfy both roles:

```text
W_p = Record_p = Live_p
```

Then Record invariance gives:

```text
W_p(N_s) = W_p(N_t_complete)
```

but Live-state covariance plus nontrivial aging gives for some later extension:

```text
W_p(N_s) != W_p(N_t_complete)
```

Contradiction.

Therefore, under these three requirements:

```text
record invariance
+ live covariance
+ nontrivial aging/backreaction
```

a single untyped response operator is impossible unless one of the following is sacrificed:

```text
1. backreaction is trivialized,
2. birth provenance is allowed to be rewritten,
3. live state is ignored,
4. the two roles are separated.
```

The current CNNA path chooses option 4.

## Relation to AQFT-net provenance intuition

This also fits the emerging AQFT-net provenance split.

There may be different effective net layers with different provenance status, for example:

```text
poor / lean net:
  minimal provenance-carrying net
  candidate carrier for additivity-type constraints

rich net:
  enlarged state/response/context net
  may encode additional backreaction, memory, or live effective data
```

The current working caution is:

```text
Do not assume additivity for every enriched/live object.
```

A plausible future target is:

```text
additivity belongs first to the lean/provenance-controlled layer,
while richer live layers must earn their additivity by a derived handoff theorem.
```

This is only a target/analogy at the current stage. It is not yet an AQFT theorem.

## Formalization target

A future Lean/Python bridge should express the single-layer obstruction explicitly.

Minimal abstract objects:

```text
NetworkStage
Extension N_t N_s
ParentEvent p
RecordResponse p N
LiveResponse p N
NontrivialAging p N_t N_s
```

Desired no-go shape:

```text
record_invariant:
  Extension N_t N_s -> RecordResponse p N_s = RecordResponse p N_t

live_changes:
  NontrivialAging p N_t N_s -> LiveResponse p N_s != LiveResponse p N_t

single_layer_identification:
  forall N, RecordResponse p N = LiveResponse p N

NoGo:
  record_invariant + live_changes + single_layer_identification -> False
```

This would not prove the final physical two-layer theory. It would prove that, once the three role constraints are accepted, the two-layer split is not optional bookkeeping.

## Current methodological guardrails

Still not proved:

```text
physical i
physical time
modular flow
Type III
AQFT handoff
unique final two-layer algebra
```

Currently justified:

```text
The growing real complement network numerically distinguishes
birth provenance records from live backreaction states.

The provisional two-layer semantics is the least destructive way to keep both roles.

A canonical forcing route exists as a role-incompatibility/no-go theorem,
provided record invariance, live covariance, and nontrivial aging are accepted
as derived or certified constraints.
```
