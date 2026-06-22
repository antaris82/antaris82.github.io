# CQNM/s=-1 saturation-link-curvature script audit — RESULTS

## Verdict

The uploaded script is useful as a **topology carrier / geometry-gate prototype**, but it is **not yet a correct implementation of the full CNNA growth + frustration + noncommutativity stack**.

It correctly separates the conceptual layers P/G/R/J in its docstring, but the implementation currently uses:

- a toy `ProvenanceRecord` layer,
- a synthetic vector-valued face cochain `K`,
- a closed periodic topology control for the CQNM case,
- and no imported/reused dynamic birth/backreaction model from Script 1.

Therefore the result should be interpreted as:

> A closed saturated topology can provide a possible nontrivial carrier for a synthetic face cochain, unlike SG-like or naive outward controls.

It should **not** be interpreted as:

> The original dynamic CNNA growth with DtN/Record-Live noncommutative plaquette operators has produced a CQNM-derived J-locking mechanism.

## Audit check 1 — Does the new script implement growth as in Script 1?

No.

Script 1 contains the actual event-resolved dynamic growth mechanism:

- `DynamicBirthConductanceModel`
- `parent_line`
- `birth_environment_load`
- `directed_edges`
- newborn senses parent line + older siblings
- newborn backreacts onto parent line and older siblings
- current conductances evolve after birth

The uploaded CQNM script does not contain those mechanisms. It only creates `ProvenanceRecord` values with deterministic or synthetic `record_weight` / `live_weight` fields.

Audit flags:

```json
{
  "new_script_has_DynamicBirthConductanceModel": false,
  "new_script_has_parent_line_update": false,
  "new_script_has_directed_edges": false,
  "new_script_has_birth_environment_load": false,
  "script1_has_DynamicBirthConductanceModel": true,
  "script1_has_parent_line_update": true,
  "script1_has_directed_edges": true,
  "script1_has_birth_environment_load": true
}
```

So the script changes the **geometric carrier** but not with the real Script-1 growth engine.

## Audit check 2 — Is frustration / noncommutativity really implemented?

Only as a placeholder.

The script defines:

```python
def record_vector(record, mode): ...
def k_cochain_on_faces(model, faces, tets, mode): ...
```

This builds a vector-valued diagnostic face cochain from synthetic provenance fields. It is **not** the previous operatorial plaquette construction:

```text
A_ab = S_b - S_a
K_abc = [A_ab, A_bc]
```

So the current `K` tests the **topological possibility of a non-exact face cochain**, not the actual DtN / Record-Live / parent-fan noncommutativity found in the previous plaquette tests.

Correct interpretation:

```text
Current script:
  Does CQNM-like saturated topology provide a nontrivial carrier for K?

Not yet implemented:
  Does the real DtN plaquette commutator K=[A_ab,A_bc]
  survive on that CQNM carrier?
```

## Audit check 3 — Is model C really a valid T^3 CQNM/s=-1 control at the default setting?

No for the default `--periodic-n 2`.

The default summary says C has:

```json
"betti_z2": {"beta0": 1, "beta1": 0, "beta2": 4, "beta3": 1}
```

and Euler characteristic:

```text
χ = V - E + F - T = 8 - 28 + 48 - 24 = 4
```

This is not the topology of a 3-torus. A 3-torus over Z2 should have:

```text
β0=1, β1=3, β2=3, β3=1, χ=0
```

Audit result:

```json
"periodic_n_2": {
  "betti_z2": {"beta0": 1, "beta1": 0, "beta2": 4, "beta3": 1},
  "euler_characteristic": 4,
  "is_consistent_with_T3_betti_over_Z2": false,
  "is_consistent_with_closed_3_manifold_euler_0": false
}
```

For `--periodic-n 3`, the topology check is consistent with T^3:

```json
"periodic_n_3": {
  "betti_z2": {"beta0": 1, "beta1": 3, "beta2": 3, "beta3": 1},
  "euler_characteristic": 0,
  "face_occupancy_counts": {"2": 324},
  "boundary_face_count": 0,
  "is_consistent_with_T3_betti_over_Z2": true,
  "is_consistent_with_closed_3_manifold_euler_0": true
}
```

So the script should not use `periodic_n=2` as a valid T^3 claim. Use `periodic_n >= 3`, and preferably add a hard validation gate.

## Audit check 4 — Is CQNM/s=-1 growth implemented as actual growth?

Not yet.

Model C is built as:

```text
periodic Freudenthal triangulation of T^3
+ dual BFS birth ordering
```

This is a **closed saturated control geometry with an imposed birth ordering**, not a dynamic CQNM process in which growth itself produces closure by s=-1 face-saturation.

That distinction matters:

```text
Implemented now:
  closed saturated target/control geometry
  then assign provenance order by BFS

Needed next:
  dynamic growth on active faces
  occupancy n_alpha tracked during birth
  s=-1 admissibility / saturation during the growth process
  Script-1 birth/backreaction data attached to the actual birth events
```

## Interpretation of uploaded summary result

The uploaded summary supports only this limited statement:

```text
A SG-like control and B naive outward NGF remain topologically trivial for stage-4 purposes.
C, the closed saturated periodic control, has nonzero harmonic projection of synthetic K.
D random saturation does not reproduce C's clean topological carrier.
```

But because C at `periodic_n=2` is topologically not a valid T^3 triangulation, and because K is synthetic, the C result is **not yet a CNNA stage-4 result**.

The right reading is:

```text
Positive topology hint:
  closed saturated geometry can host a nontrivial cohomological carrier.

Still missing:
  true Script-1 dynamic growth
  true dynamic s=-1 saturation
  real DtN plaquette commutator K
  noncommutative frustration on links/cycles
  J-locking diagnostic after those are in place
```

## Required next implementation

The next script should be called for example:

```text
test_cqnm_dynamic_growth_with_dtn_plaquette_frustration.py
```

It must combine:

1. Script-1 dynamic birth/backreaction engine.
2. A primal CQNM/NGF geometry layer.
3. Active face list and face occupancy `n_alpha`.
4. s=-1 admissibility/saturation rule.
5. Real DtN/Record-Live data from previous response packages.
6. Parent-fan / link-level plaquette commutator:

```text
A_ab = S_b - S_a
K_abc = [A_ab, A_bc]
```

7. Link/Regge/defect holonomy on the saturated complex.
8. Controls:
   - identical history
   - symmetrized birth
   - no backreaction
   - diagonal/trace kill
   - random saturation
   - SG-like subdivision
   - naive outward NGF

## Final verdict

```text
The new script is directionally correct as a topology-gate prototype.
It is not yet correct as the full CNNA/CQNM growth-frustration-noncommutativity test.
```
