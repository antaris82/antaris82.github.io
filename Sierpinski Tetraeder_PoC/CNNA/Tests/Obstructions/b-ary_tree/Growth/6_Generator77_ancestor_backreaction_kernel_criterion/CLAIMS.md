# Generator72 claims ledger

## Content theorems / tests

- `ConcreteSymmetricCutG1Test.symmetric_concrete_status`: static symmetric 2-node Schur/DtN cut leaves the sector unresolved (`kernel = 2`).
- `ConcreteGrowthIrreversibilityG1Test.growth_irreversibility_closure_status`: directed boundary-frontier growth creates a computed Schur/DtN trace residual and conditionally closes the local sign gate.
- `GrowthEdgeProvenance.derived_growth_status`: the local growth-edge provenance feeds the computed residual into the existing Schur/Cut/Handoff/J-sign closure path.
- `CanonicalFillOrderCriterion.current_fill_rank_nodeOf_eq_role_order_nodeOf`: fill-endpoint rank reproduces the current role-order node map pointwise.
- `CanonicalFillOrderCriterion.reversal_changes_fill_frontier_rank`: reversed role order sends the frontier role to rank 0 while the fill rank sends it to rank 2.
- `FillEndpointDerivationCriterion.current_derived_endpoint_eq_fill_endpoint`: the active local endpoint map is reconstructed from cut/fill data rather than by directly assigning endpoints to roles.
- `FillEndpointDerivationCriterion.current_derived_fill_nodeOf_eq_fill_order_nodeOf`: the reconstructed cut/fill endpoint ranks reproduce the Gen66 fill-order node map pointwise.
- `FillEndpointDerivationCriterion.derived_fill_reversal_rejects_frontier_zero`: the reconstructed fill rank keeps the frontier at rank 2 while the reversed role order sends it to rank 0.
- `FillDynamicsRankCriterion.current_dynamic_endpoint_eq_cut_fill_endpoint`: the active local endpoint map is reproduced from a three-phase dynamic rank profile (`beforeRank`, `afterRank`, `afterRank + 1`) rather than directly from endpoint labels.
- `FillDynamicsRankCriterion.current_dynamic_fill_nodeOf_eq_fill_order_nodeOf`: the dynamic-rank node map reproduces the Gen66 fill-order node map pointwise.
- `FillDynamicsRankCriterion.dynamic_fill_rank_reversal_rejects_frontier_zero`: the dynamic-rank frontier remains at rank 2 while the reversed role order sends it to rank 0.
- `FrontierRankProfileCriterion.rank_from_frontier_profile_eq_dynamic_rank`: the Gen68 dynamic-rank formula is reproduced from explicit phase predicates (`sourcePhase`, `activeFrontierPhase`) and a `FillPhase` classifier.
- `FrontierRankProfileCriterion.current_frontier_profile_nodeOf_eq_dynamic_nodeOf`: the current phase-profile node map reproduces the Gen68 dynamic node map pointwise.
- `FrontierRankProfileCriterion.current_phase_cut_root_source`, `current_phase_selected_transit`, and `current_phase_frontier_active`: the active local cut root, selected address, and frontier address are classified as source, transit, and active frontier respectively.
- `FrontierRankProfileCriterion.frontier_profile_reversal_rejects_frontier_zero`: the phase-profile frontier remains at rank 2 while the reversed role order sends it to rank 0.

- `FrontierPhaseCertificateCriterion.certificate_cut_root_phase_source`, `certificate_selected_phase_transit`, and `certificate_frontier_phase_active`: a local certificate record turns the source/transit/active-frontier phase decisions into explicit witnessed obligations rather than leaving the active phase classifier as an unstructured computation.
- `FrontierPhaseCertificateCriterion.current_certified_nodeOf_eq_frontier_profile_nodeOf`: the certified phase record reproduces the Gen69 frontier-profile node map pointwise.
- `FrontierPhaseCertificateCriterion.frontier_phase_certificate_reversal_rejects_frontier_zero`: the certified phase record keeps the frontier at rank 2 while the reversed role order sends it to rank 0.
- `FrontierEventPhaseBridgeCriterion.residual_event_matches_current_active_frontier_phase`: the active residual frontier event from the earlier Schur/DtN event-activation provenance agrees with the current `activeFrontierPhase` classification for the frontier address.
- `FrontierEventPhaseBridgeCriterion.inside_event_matches_current_selected_non_frontier_phase`: the inside-target control event agrees with the selected address being non-frontier in the current phase profile.
- `FrontierEventPhaseBridgeCriterion.inactive_event_shows_edge_present_not_in_phase`: the inactive-edge control exposes the current limitation: the existing phase predicate assumes the residual frontier event context and does not itself carry `edgePresent` as an input.
- `FrontierEventPhaseBridgeCriterion.current_event_backed_nodeOf_eq_fill_order_nodeOf`: the event-backed certificate still reproduces the fill-order node map pointwise.

## Transport / comparison lemmas

Most Spine lemmas after Generator55 are intentionally transport or comparison lemmas. They should not be counted as independent physical claims. They witness that the same local closure path can be viewed through address, abstract address, port numbering, role order, fill-rank, local cut/fill endpoint-derivation, dynamic three-phase fill-rank interfaces, an explicit frontier phase-profile interface, a witnessed frontier phase-certificate interface, and now a current event-activation compatibility bridge, and a local fill-orbit basin diagnostic.

## Remaining open point

`FrontierPhaseCertificateCriterion.frontierPhaseCertificateStillLocalWitnessRecord` remains true-by-marker. Generator70 does not prove a global theorem that every CNNA cut/interface canonically induces the source/transit/active-frontier phase certificate. It proves only that the active Gen69 phase classifier can be carried by explicit witness obligations. The next open core is to derive those witness obligations from a genuine fill law or cut/interface evolution principle, rather than supplying them as local certificate fields. Generator71 also shows a sharper local gap: the current `activeFrontierPhase` predicate is compatible with the active residual event, but it is not yet itself an event predicate because `edgePresent` is not one of its inputs.


## Generator72 fill-orbit diagnostic

- `FillOrbitBasinCriterion.local_fill_map_two_two_flows_to_one`: the formerly decorative local fill map is now used in the proof graph for the active local frontier value; at `b = 2`, the fill value `2` maps to the selected fixed value `1`.
- `FillOrbitBasinCriterion.frontier_flows_to_selected_but_endpoint_code_is_two`: the active frontier dynamically flows to the selected value under `localFillMap`, while its endpoint-code rank remains `2`. This intentionally prevents the false claim that the numeric rank `2` itself is the attracting fixed point.
- `FillOrbitBasinCriterion.residual_event_has_orbit_witness`: the active residual event is paired with the local fill-orbit witness.

Generator72 is a diagnostic attachment, not a global closure theorem. It shows that `localFillMap` can witness the active frontier flow `2 -> 1`, but the endpoint code `attractor -> 2` is still a coding layer over the orbit classification. The remaining open core is to derive the endpoint-code convention and the local active frontier state from a general cut/interface/fill evolution, not only from the current three-node local witness.

## Generator73 — Fill orbit endpoint-code semantics bridge

### New file

- `FillOrbitEndpointCodeCriterion.lean`

### Substantive claim

Generator73 separates the local orbit-flow semantics from the legacy
`FillEndpoint.attractor` code:

```text
fixed zero value          -> fixedZeroBase
fixed one value           -> fixedOneSelected
frontier value 2 flows to 1 -> frontierInflowToSelected
```

The frontier role is therefore not claimed to be a fixed-point attractor.
It is recorded as an inflow-to-selected orbit state whose legacy endpoint
code remains `FillEndpoint.attractor`, and hence rank/node code `2`, in the
current local path.

### Important negative clarification

Generator73 does not prove a global CNNA fill-dynamics theorem.  It also does
not prove that arbitrary cut/interface evolutions force this orbit-flow code.
It only proves, for the current local address seed, that the semantic split

```text
frontier value 2 -> localFillMap 2 2 = 1
frontier endpoint code -> FillEndpoint.attractor -> node/rank 2
```

is explicit and preserved through the existing local G1 closure path.

### Transport lemmas

The equalities to `FillOrbitBasinCriterion.currentOrbitNodeOf` and
`CanonicalFillOrderCriterion.currentFillRankNodeOf` are transport lemmas, not
new physical claims.

## Generator74 — Fill orbit cycle bridge diagnostic

### New file

- `FillOrbitCycleBridgeCriterion.lean`

### Substantive claim

Generator74 stops trying to read a local orbit-flow line as an orientation.
The current local orbit path is

```text
cutRoot/node 0 -> frontier/node 2 -> selected/node 1
```

and its circulation under the explicit three-node comparison cochain is zero:

```text
csum orbitCmp [(0,2),(2,1)] = 0
```

A nonzero circulation appears only after adding a closure edge:

```text
[(0,2),(2,1),(1,0)] -> -1
[(0,1),(1,2),(2,0)] -> +1
```

This is the first explicit bridge from the Gen73 local fill-orbit semantics to
the older `Circulation.lean` layer: the local fill orbit supplies a directed
line/flow segment, while a sign-carrying circulation requires a closed cycle
and therefore a gluing/closure datum.

### Important negative clarification

Generator74 does not prove a global H¹-orientation theorem and does not derive
the closure edge.  The marker

```lean
cycleClosureStillGlobalGluingInput
```

remains intentionally true-by-marker.  The content is diagnostic: local
frontier inflow plus the local orbit path has zero circulation; nonzero
circulation requires an additional cycle-closure/gluing input.

### Transport lemmas

The Spine lemmas for Generator74 are comparison/diagnostic lemmas. They should
not be counted as new G1 closure claims.

## Generator75 — Cycle circulation gate / height-coboundary no-go

### New file

- `CycleCirculationGate.lean`

### Substantive claim

Generator75 takes the Gen74 cycle diagnostic and separates two notions of
circulation.

1. A sign-circulation on the concrete growth triangle exists:

```text
csum cmpH [(0,3),(3,4),(4,0)] = 1
```

2. The genuine height-gradient circulation on the same closed walk vanishes:

```text
csum (gradOf height) [(0,3),(3,4),(4,0)] = 0
```

The general theorem is `grad_telescopes_closed`: for any node type, any
height function, and any closed walk constructed by `closedWalkEdges`, the
height-gradient circulation telescopes to zero.  Therefore any circulation
that is only a coboundary of a height/total-order datum is not an intrinsic
rotation datum.

### Important negative clarification

Generator75 does not prove a global CNNA orientation, a complex structure on
H¹, or a von-Neumann-algebraic limit. It proves the opposite diagnostic:
height/rank-derived cycle data can give a nonzero sign-circulation, but the
true gradient part is cohomologically trivial. This supports the conclusion
that local rank/fill/height data supplies a directed axis or arrow, not a
canonical plane orientation or `i`.

### Kappa / axis-bound test

`orientation_axis_bound` shows that the concrete sign-circulation changes
under the kappa-style height-axis swap. This is a real sign sensitivity, but
it is still axis-bound; it does not by itself produce an intrinsic rotation.

### Transport lemmas

The GeneratorSpine entries for Generator75 only expose the new diagnostic
claims. They should not be counted as new G1 closure claims.

## Generator76 — Schur cut spectral tower register

### New file

- `SchurCutSpectralTowerCriterion.lean`

### Substantive claim

Generator76 records the new Schur/DtN cut-spectrum finding as a finite
spectral-register gate.  For the full symmetric binary cut with equal
conductances, the observed and now registered denominator/condition profile is

```text
L = 1..8: 1, 3, 7, 15, 31, 63, 127, 255
```

Equivalently, the smallest registered nonzero cut-response scale behaves like

```text
lambda_min(L) = 1 / (2^L - 1)
cond(L)      = 2^L - 1
```

The threshold register for eigenvalues below `0.1` is

```text
0, 0, 0, 1, 3, 7, 15, 31
```

so the soft sector is not a single isolated global mode in the finite
register; it grows hierarchically with the full-frontier refinement.

### Important negative clarification

Generator76 does **not** claim a Type-III factor, a missing trace, nuclearity,
a modular operator, or a derived `i`.  The module records a finite Schur/DtN
spectral diagnostic: full-frontier cut response has no uniform gap in the
registered binary tower.  The operator-algebraic target still requires local
algebras, embeddings, compatible states/weights, a GNS or vN completion, and
separate modular analysis.

### Kappa/orientation status

The same symmetric equal-conductance register keeps the soft-shell/projector
surrogate kappa-invariant.  Hence the positive UV/cut-correlation signal is
paired with a negative orientation result:

```text
Schur/DtN soft-sector formation: yes.
Orientation from the symmetric soft sector alone: no.
```

This preserves the post-Gen75 distinction: a rich gapless cut spectrum can be a
necessary diagnostic for later Type-III/AQFT investigation, but it does not by
itself produce a handed `J`-orientation.

### Open core

The next positive search target is a derived conductance/growth-history
asymmetry, not another symmetric-spectrum test.  The key question is whether a
CNNA-derived growth rule can produce a non-kappa-invariant Schur/DtN soft
sector while remaining provenance-controlled and not importing orientation by
hand.


## Generator77 — Ancestor backreaction kernel criterion

### New file

- `AncestorBackreactionKernelCriterion.lean`

### Substantive claim

Generator77 records the conductance-scaling finding from the dynamic birth
Python tests as a small finite shell-control criterion.  For ternary branching
`b = 3`, the old remote ancestor kernel `K(d) = 1/d²` attenuates a single
remote birth, but it does not control the exponential shell count.  In the
registered shell-control profile it first fails at level 4:

```text
inverse_square profile, levels 1..8:
true, true, true, false, false, false, false, false
```

The shell-normalized kernel

```text
K(d) = 1 / (3^(d-1) d²)
```

controls all registered levels and carries the extra polynomial damping profile

```text
1, 4, 9, 16, 25, 36, 49, 64
```

This registers the new default candidate for high-level extrapolation: keep the
local birth/response dynamics, but damp remote ancestor backreaction by a
branching-shell factor plus polynomial depth damping.

### Important negative clarification

Generator77 does not claim a derived physical metric, a global `J`, a full
operator tower, or a Type-III/von-Neumann result.  It is still a response-layer
criterion.  Its role is to prevent the unnormalized `1/d²` ancestor kernel from
being used as if it were suitable for the infinite growth limit.

### Open core

The next positive target is an effective operator-sector closure test using a
shell-controlled kernel.  The remaining questions are whether the local
projected rotational sector becomes an invariant local `J`-plane, whether a
derived positive metric/weight exists, and whether adjacent local sectors glue
compatibly in a tower.
