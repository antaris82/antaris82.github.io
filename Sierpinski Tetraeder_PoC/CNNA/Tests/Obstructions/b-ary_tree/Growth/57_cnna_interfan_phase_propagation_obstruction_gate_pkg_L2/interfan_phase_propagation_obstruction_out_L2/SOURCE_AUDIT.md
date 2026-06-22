# SOURCE AUDIT

Hard constraints:

- no final `sym(M)` in the tested directed vertex operator;
- antisymmetric term is birth-order-derived: `q⊗h - h⊗q`;
- phase propagation uses only birth labels, local real K vectors, shared-edge face graph, and actual pairing graph;
- H2/kappa metrics are evaluated after phase labels are assigned;
- beta changes are logged/diagnostic only, not used for move or phase decisions.

Limits:

- `pair_graph_*` and `face_graph_*` are diagnostic synchronization tests, not yet a derived CNNA law;
- rotations are finite real 3D coordinate transports used to compare local charts, not an imported complex scalar or J;
- NGF/CQNM remains only a comparison framework.
