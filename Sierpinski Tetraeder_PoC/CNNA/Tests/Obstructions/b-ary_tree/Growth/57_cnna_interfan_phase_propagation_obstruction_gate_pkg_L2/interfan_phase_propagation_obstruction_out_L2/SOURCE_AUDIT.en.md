# SOURCE AUDIT

Hard constraints:

- no final `sym(M)` in the tested directed vertex operator;
- the antisymmetric term is derived from birth order: `q⊗h - h⊗q`;
- phase propagation uses only birth labels, local real K vectors, the shared-edge face graph, and the actual pairing graph;
- H2/kappa metrics are evaluated after phase labels are assigned;
- beta changes are logged for diagnostic purposes only and are not used for move or phase decisions.

Limits:

- `pair_graph_*` and `face_graph_*` are diagnostic synchronization tests; they are not yet a derived CNNA law;
- rotations are finite real 3D coordinate transformations used to compare local charts; they are not an imported complex scalar or J;
- NGF/CQNM remains only a comparison framework.
