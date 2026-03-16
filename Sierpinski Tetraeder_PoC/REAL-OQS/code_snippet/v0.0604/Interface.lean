/-
REALOQS / Pillar A: canonical interface

This is the single recommended entry point for using Pillar A as a
substrate-agnostic kernel:

  (i)   combinatorics / nesting contracts (substrate-agnostic)
  (ii)  approximant / tail interfaces
  (iii) symmetry / equivariance hooks
  (iv)  update pipeline hooks

Design intent:
- No IDEAL-specific imports.
- Keep gauge/matter/physics/geometry scaffolds out of the default surface.

For the heavy developer umbrella (incl. IDEAL adapters and scaffolds) use
`REALOQS.PillarA.All`.
-/

-- Core-only umbrella (contracts + operators + B/C handoff interfaces)
import REALOQS.PillarA.Core.Exports

-- Update layer (mismatch + pipeline + step)
import REALOQS.PillarA.Update.Seed
import REALOQS.PillarA.Update.Step
import REALOQS.PillarA.Update.MismatchFunctional
import REALOQS.PillarA.Update.MismatchDriver
import REALOQS.PillarA.Update.ApproximationPipeline
import REALOQS.PillarA.Update.DeterministicExport

-- OQS blocks (substrate-agnostic hooks)
import REALOQS.PillarA.OQS.SysEnv
import REALOQS.PillarA.OQS.DtN
import REALOQS.PillarA.OQS.DtN_TailHook
import REALOQS.PillarA.OQS.LiebRobinsonHook
import REALOQS.PillarA.OQS.SplitInvariants

-- RG scaffolds (scale windows / error budgets) for approximate symmetries
import REALOQS.PillarA.RG.ScaleWindow

-- Symmetry scaffolds (approximate isometries / approximate Poincaré)
import REALOQS.PillarA.Symmetry.ApproxIsometry
import REALOQS.PillarA.Symmetry.ApproxPoincare
import REALOQS.PillarA.Symmetry.PoincareLorentzSplit

-- SR kinematics blocks (hooks, worldlines, proper time, discrete spacetime)
import REALOQS.PillarA.Kinematics.FrameChange
import REALOQS.PillarA.Kinematics.KinematicsHooks
import REALOQS.PillarA.Kinematics.Worldline
import REALOQS.PillarA.Kinematics.ProperTime
import REALOQS.PillarA.Kinematics.Clock
import REALOQS.PillarA.Kinematics.DiscreteSpacetime
import REALOQS.PillarA.Kinematics.FromUpdate
import REALOQS.PillarA.Kinematics.SpatialMetric
import REALOQS.PillarA.Kinematics.Causality
import REALOQS.PillarA.Kinematics.ProperTimeFromMetric
import REALOQS.PillarA.Kinematics.SpacetimeRegions
