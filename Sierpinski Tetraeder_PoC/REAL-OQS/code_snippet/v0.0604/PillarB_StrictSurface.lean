import REALOQS.Meta.PillarB_ActivePath
import REALOQS.Meta.PillarB_NoConfigHarnessInCriticalPath

set_option autoImplicit false

namespace Meta
namespace PillarB_StrictSurface

noncomputable section

open _root_.PillarB.AQFT.Derived.TreeOfCliquesFromPillarA.ToC
open _root_.PillarB.AQFT.Derived.TreeOfCliquesExtendedFromPillarA.ToC

variable {b : Nat}
variable (p : _root_.PillarA.LayerA.Policy b)

theorem aqft_surface_exposes_quasilocal_closure [Fact (0 < b)] :
    _root_.PillarB.AQFT.HaagKastler.StarAlgIso
      (SA := _root_.PillarB.AQFT.Derived.boundaryMatrixQuasiLocalSA
        (Ω := Ω (b := b) (p := p)))
      (SB := _root_.PillarB.AQFT.Derived.matrixStarAlgebraCStar
        (Ω (b := b) (p := p))) :=
  quasiLocalClosureIsoMatrix_FromA (b := b) (p := p)

theorem aqft_surface_exposes_locality [Fact (0 < b)] :
    _root_.PillarB.AQFT.Gates.HaagKastler.Locality
      (N := N_ToC (b := b) (p := p))
      (D := D_ToC (b := b) (p := p)) :=
  locality_ToC (b := b) (p := p)

theorem aqft_surface_exposes_isotony [Fact (0 < b)] :
    _root_.PillarB.AQFT.Gates.HaagKastler.Isotony
      (N := N_ToC (b := b) (p := p)) :=
  isotony_ToC (b := b) (p := p)

theorem aqft_surface_exposes_split [Fact (0 < b)] :
    _root_.PillarB.AQFT.Gates.SplitProperty
      (N := N_ToC (b := b) (p := p)) :=
  splitProperty_ToC (b := b) (p := p)

theorem aqft_surface_exposes_modular [Fact (0 < b)] :
    (modularDatum_top_ToC (b := b) (p := p)).topKMS.state =
      ρTop (b := b) (p := p) := by
  rfl

end
end PillarB_StrictSurface
end Meta
