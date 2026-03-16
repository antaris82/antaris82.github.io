import Mathlib
import REALOQS.PillarA.Ideal.Adapter.TreeOfCliquesDtNSemantics

set_option autoImplicit false

/-!
# Tree-of-Cliques: end-to-end wiring (build-only)

This file is intentionally **minimal** and serves as a compilation gate.

It checks that:

* the IDEAL adapter `TreeOfCliquesApprox.mk` produces a `TwoStageApprox` carrier;
* the DtN semantics adapter `TreeOfCliquesDtNSemantics.mk` produces a
  `TailElimDtNSemantics` instance over that carrier;
* basic projections from the carrier state and semantics typecheck.

No numerical claims are made here; we do not construct concrete states.
-/

namespace REALOQS
namespace Examples
namespace TreeOfCliques_EndToEnd

noncomputable section

namespace ToC

open _root_.PillarA.LayerA

variable {b : Nat} (p : _root_.PillarA.LayerA.Policy b)

/-- The Phase-4 Tree-of-Cliques carrier (IDEAL adapter). -/
abbrev A : _root_.PillarA.Update.TwoStageApprox.{0} b :=
  _root_.PillarA.Update.Instances.TreeOfCliquesApprox.mk (b := b) p

/-- DtN semantics for the Tree-of-Cliques carrier. -/
abbrev Sem :
    _root_.PillarA.Update.TailElimDtNSemantics (A := A (p := p)) :=
  _root_.PillarA.Update.Instances.TreeOfCliquesDtNSemantics.mk (b := b) p

/-!
## Sanity checks (type-level)

We only show that the key objects and their main projections exist.
-/

-- The carrier state type exists.
abbrev State : Type := (A (p := p)).S

-- The semantics split boundary type exists.
abbrev Boundary : Type := (Sem (p := p)).split.B

-- The semantics provides an operator-valued boundary observable.
abbrev BoundaryOp (st : State (p := p)) : Matrix (Boundary (p := p)) (Boundary (p := p)) ℝ :=
  (Sem (p := p)).boundaryOp st

-- `boundaryOp` is tied to the BB-block of the full operator
-- (as provided by the semantics contract).
theorem boundaryOp_eq_blockBB (st : State (p := p)) :
    (Sem (p := p)).boundaryOp st = (Sem (p := p)).split.blockBB ((Sem (p := p)).L st) := by
  simpa using (Sem (p := p)).boundaryOp_def st

-- The Stage-1 boundary operator is the DtN reduction (as stated by the semantics contract).
theorem stage1_boundaryOp_eq_DtN (st : State (p := p)) :
    (Sem (p := p)).boundaryOp ((A (p := p)).stage1 st) =
      (Sem (p := p)).split.DtN_of ((Sem (p := p)).L st) ((Sem (p := p)).inv st) := by
  -- This is exactly the field provided by the semantics instance.
  simpa using (Sem (p := p)).stage1_boundaryOp_eq_DtN st

end ToC

end
end TreeOfCliques_EndToEnd
end Examples
end REALOQS
