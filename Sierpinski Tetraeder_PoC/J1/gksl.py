# app_gksl_measurement.py
# Vollständige Streamlit-App:
#  - Mikro (Repeated Interactions, partieller Austausch) + partielle Spur
#  - GKSL (Amplitude Damping bei endlicher T) mit wählbaren Integratoren
#  - Verification Suite A–E (CPTP/Choi, GADC-Fit, Semigroup, Spohn, MCWF≈GKSL)
#  - Export (Markdown + auswählbare Plots)
#
# Wichtige Änderungen ggü. Basisversion:
#  • Farbstile so gesetzt, dass keine "weißen" Kurven verwendet werden (CI-Bänder mit sichtbaren Linien + Füllung).
#  • Integrator-Auswahl: "Strang (CPTP, 2. Ordnung)", "RK4 (mit PSD-Projektion)", "RK4 (ohne Projektion)".
#  • Verification-Supplement: dynamische Zielwerte-Box + Quellen (mit URLs).
#  • Stabilität bei mehrfachen Läufen: Dirty-Flag + Run-ID verhindern Inkonsistenzen/Abstürze bei Parameterwechseln in Tab 1.
#
# Hinweise:
#  • Die PSD-Projektion bei RK4 (EVD-Clipping) ist NUMERIK, kein physikalischer Mechanismus.
#  • Der "Strang (CPTP)"-Integrator erhält CP/TP für jeden Zeitschritt (Unitary + exakte GADC über dt + Unitary).
#  • Für harte Fehlerschranken: kleine Zeitschritte (Substeps↑), bevorzugt Strang (CPTP).

from __future__ import annotations

import streamlit as st
import numpy as np
import math, io, zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
import plotly.graph_objects as go
import numpy.linalg as npl

# ======================== Session-Init & Helpers ============================
if "traceout_data" not in st.session_state: st.session_state["traceout_data"] = None
if "tab2_data" not in st.session_state: st.session_state["tab2_data"] = None
if "figs" not in st.session_state: st.session_state["figs"] = {}
if "micro_dirty" not in st.session_state: st.session_state["micro_dirty"] = False
if "run_id" not in st.session_state: st.session_state["run_id"] = 0
if "integrator_choice" not in st.session_state: st.session_state["integrator_choice"] = "Strang (CPTP, 2. Ordnung)"
if "integrator_sub" not in st.session_state: st.session_state["integrator_sub"] = 5

def mark_micro_dirty():
    """Bei jeder Slider-Änderung in Tab 1 aufrufen."""
    st.session_state["micro_dirty"] = True

# ======================== Mathematische Utilities ===========================
def dagger(X: np.ndarray) -> np.ndarray:
    return X.conj().T

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def anticom(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A

def is_psd(rho: np.ndarray, tol: float = 1e-10) -> bool:
    w = np.linalg.eigvalsh((rho + dagger(rho)) / 2)
    return np.all(w >= -tol)

def partial_trace_last(rho_AB: np.ndarray, dimA: int, dimB: int) -> np.ndarray:
    """Tr_B rho_AB, B ist die letzte Komponente."""
    rho = rho_AB.reshape(dimA, dimB, dimA, dimB).transpose(0, 2, 1, 3)
    return np.trace(rho, axis1=2, axis2=3)

def trace_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """T(ρ1,ρ2) = 0.5 ||ρ1 - ρ2||_1 für Hermitesches delta via EVD."""
    delta = (r1 - r2 + dagger(r1 - r2)) / 2
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))

def logm_herm(A: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    """Matrixlog für pos.-def. hermitesche A via EVD (kleine Eigenwerte geklemmt)."""
    w, V = np.linalg.eigh((A + dagger(A)) / 2)
    w = np.clip(w.real, tol, None)
    return V @ np.diag(np.log(w)) @ dagger(V)

# =============================== Qubit-Basis ================================
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma_+
sm = np.array([[0, 0], [1, 0]], dtype=complex)  # sigma_-

# =================== Mikro-Modell (Repeated Interactions) ===================
@dataclass
class MicroParams:
    omega: float = 1.0        # H_S = (omega/2) σ_z
    beta: float  = 2.0        # inverse Temperatur (KMS)
    g: float     = 0.5        # Kopplungsstärke (Austausch)
    tau_int: float = 0.2      # Pulsdauer einer Kollision
    dt_coll: float = 0.2      # Kollisionsintervall
    n_steps: int = 60         # Anzahl Kollisionen
    theta0: float = math.pi/3 # Startzustand: θ
    phi0: float   = 0.0       # Startzustand: φ

@dataclass
class GKSLRates:
    gamma_down: float
    gamma_up: float

def thermal_qubit(beta: float, omega: float) -> np.ndarray:
    """τ_B ∝ e^{-β (ω/2) σ_z} = diag(e^{-βω/2}, e^{+βω/2}) / (2 cosh βω/2)."""
    x = beta*omega/2.0
    Z = 2.0*math.cosh(x)
    p_exc = math.exp(-x)/Z  # |0>, E=+ω/2
    p_gnd = 1.0 - p_exc     # |1>, E=-ω/2
    return np.array([[p_exc,0],[0,p_gnd]], dtype=complex)

def unitary_exchange(theta: float) -> np.ndarray:
    """U = exp(-i θ (σ_+⊗σ_- + σ_-⊗σ_+)); rotiert im |01>,|10|-Block."""
    c, s = math.cos(theta), math.sin(theta)
    U = np.eye(4, dtype=complex)
    # Basis: |00>,|01>,|10>,|11>
    U[1,1] = c; U[1,2] = -1j*s
    U[2,1] = -1j*s; U[2,2] = c
    return U

def one_collision_map(rhoS: np.ndarray, tauB: np.ndarray, theta: float) -> np.ndarray:
    U = unitary_exchange(theta)
    rhoAB = np.kron(rhoS, tauB)
    rhoABp = U @ rhoAB @ dagger(U)
    return partial_trace_last(rhoABp, 2, 2)

def kraus_from_micro(theta: float, tauB: np.ndarray) -> List[np.ndarray]:
    """Kraus aus U und gemischtem Badzustand τ = Σ_i p_i |i⟩⟨i| (Umgebungsbasis |0>,|1>).
       K_{ji} = √p_i ⟨j|U|i⟩."""
    U = unitary_exchange(theta)
    p0 = float(tauB[0,0].real); p1 = float(tauB[1,1].real)
    def block(j,i):
        B = np.zeros((2,2), dtype=complex)
        for sp_idx in range(2):
            for s_idx in range(2):
                row = sp_idx*2 + j
                col = s_idx*2 + i
                B[sp_idx, s_idx] = U[row, col]
        return B
    K00 = math.sqrt(p0) * block(0,0)
    K10 = math.sqrt(p0) * block(1,0)
    K01 = math.sqrt(p1) * block(0,1)
    K11 = math.sqrt(p1) * block(1,1)
    return [K00, K10, K01, K11]

def choi_from_kraus(ks: List[np.ndarray]) -> np.ndarray:
    """Choi(Φ) = Σ_i vec(K_i) vec(K_i)† (Qubit → 4x4)."""
    V = [K.reshape(-1,1,order='F') for K in ks]
    V = np.concatenate(V, axis=1)
    return V @ dagger(V)

def tp_defect(ks: List[np.ndarray]) -> float:
    S = np.zeros((2,2), dtype=complex)
    for K in ks:
        S += dagger(K) @ K
    return float(np.linalg.norm(S - I2))

def rates_from_micro(p: MicroParams) -> GKSLRates:
    """
    Kleiner Winkel θ = g τ_int, Kollisionsrate r = 1/Δt:
      γ↓ = r sin^2θ p_gnd,   γ↑ = r sin^2θ p_exc,   p_exc/p_gnd = e^{-βω}.
    """
    r = 1.0 / max(p.dt_coll, 1e-12)
    sin2 = math.sin(p.g * p.tau_int)**2
    tauB = thermal_qubit(p.beta, p.omega)
    p_exc = float(tauB[0,0].real); p_gnd = 1.0 - p_exc
    return GKSLRates(gamma_down=r*sin2*p_gnd, gamma_up=r*sin2*p_exc)

# =================== GKSL (Amplitude-Damping bei endlicher T) ==============
def gksl_rhs_amp_damp(rho: np.ndarray, omega: float, rates: GKSLRates) -> np.ndarray:
    """
    dot{ρ} = -i[H_S,ρ] + γ↓ D[σ_-](ρ) + γ↑ D[σ_+](ρ),  H_S=(ω/2)σ_z,  D[L](ρ)=LρL† - 0.5{L†L,ρ}.
    """
    H = 0.5*omega*sz
    gd, gu = rates.gamma_down, rates.gamma_up
    def D(L): return L @ rho @ dagger(L) - 0.5*anticom(dagger(L) @ L, rho)
    return -1j*comm(H, rho) + gd*D(sm) + gu*D(sp)

def rk4_step(rho: np.ndarray, dt: float, deriv, *args) -> np.ndarray:
    """RK4 mit PSD-Projektion (EVD-Clipping) → numerisch stabil (ρ≽0), aber nicht physikalischer Mechanismus."""
    k1 = deriv(rho, *args)
    k2 = deriv(rho + 0.5*dt*k1, *args)
    k3 = deriv(rho + 0.5*dt*k2, *args)
    k4 = deriv(rho + dt*k3, *args)
    rho_next = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    rho_next = 0.5*(rho_next + dagger(rho_next))
    tr = np.trace(rho_next)
    if abs(tr) > 1e-14: rho_next = rho_next / tr
    w,V = np.linalg.eigh(rho_next); w = np.clip(w.real, 0.0, None)
    return V @ np.diag(w) @ dagger(V)

def gibbs_populations(beta: float, omega: float) -> tuple[float,float]:
    x = beta*omega/2.0; Z = 2.0*math.cosh(x)
    return math.exp(-x)/Z, math.exp(+x)/Z  # (p0*, p1*)

# ============================= Helferfunktionen =============================
def pop00(rho): return float(rho[0,0].real)
def pop11(rho): return float(rho[1,1].real)
def coh_abs(rho): return float(abs(rho[0,1]))
def coh_phase(rho): return float(np.angle(rho[0,1]+0j))

def linreg_with_errors(x: np.ndarray, y: np.ndarray) -> tuple[float,float,float,float]:
    """Lineare Regression y = m x + b + ε; liefert m,b und Std.-Fehler se_m,se_b."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return 0.0, float(y[0] if len(y) else 0.0), 0.0, 0.0
    X = np.column_stack([x, np.ones_like(x)])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    m, b = beta_hat
    resid = y - (m*x + b)
    dof = max(len(x) - 2, 1)
    sigma2 = np.dot(resid, resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_m = math.sqrt(max(cov[0,0], 0.0)); se_b = math.sqrt(max(cov[1,1], 0.0))
    return float(m), float(b), float(se_m), float(se_b)

# ============================= GADC (Kraus) =================================
def gadc_kraus(eta: float, q_exc: float) -> List[np.ndarray]:
    """
    Generalized Amplitude Damping Channel (GADC):
      - η ∈ [0,1] (ein-Schritt-Transfer; für Partial-Exchange η≈sin²θ)
      - q_exc ∈ [0,1] Bad-Anregungswahrscheinlichkeit (thermisch: q_exc=p_exc)
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    q   = float(np.clip(q_exc, 0.0, 1.0))
    k0 = math.sqrt(q) * np.array([[1, 0],[0, math.sqrt(1-eta)]], dtype=complex)
    k1 = math.sqrt(q) * np.array([[0, math.sqrt(eta)],[0, 0]], dtype=complex)
    k2 = math.sqrt(1-q) * np.array([[math.sqrt(1-eta), 0],[0, 1]], dtype=complex)
    k3 = math.sqrt(1-q) * np.array([[0, 0],[math.sqrt(eta), 0]], dtype=complex)
    return [k0,k1,k2,k3]

def choi_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Frobenius-Norm der Choi-Differenz (Heuristik)."""
    return float(np.linalg.norm(C1 - C2))

# ============================= SU(2) & PTM ==================================
def Rz(alpha: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j*alpha), 0],[0, np.exp(0.5j*alpha)]], dtype=complex)

def Ry(beta: float) -> np.ndarray:
    c = math.cos(beta/2.0); s = math.sin(beta/2.0)
    return np.array([[c, -s],[s, c]], dtype=complex)

def su2_zyz(a: float, b: float, c: float) -> np.ndarray:
    return Rz(a) @ Ry(b) @ Rz(c)

def choi_conjugate(C: np.ndarray, Uout: np.ndarray, Uin: np.ndarray) -> np.ndarray:
    """Φ' = U_out ∘ Φ ∘ U_in ⇒ vec(U_out K U_in) = (U_in.T ⊗ U_out) vec(K) ⇒
       C' = (U_in.T ⊗ U_out) C (U_in.conj() ⊗ U_out.conj().T)."""
    A = np.kron(Uin.T, Uout)
    B = np.kron(Uin.conj(), Uout.conj().T)
    return A @ C @ dagger(B)

def pauli_transfer_affine(evolve_state, total_time: float) -> np.ndarray:
    """4x4-affine Pauli-Transfer-Matrix T(t) mit [1; r'] = T(t) [1; r]."""
    I_over_2 = 0.5*I2
    rho0_t = evolve_state(I_over_2, total_time)
    c = np.array([np.trace(rho0_t @ sx).real,
                  np.trace(rho0_t @ sy).real,
                  np.trace(rho0_t @ sz).real])
    def r_from_state(rho):
        return np.array([np.trace(rho @ sx).real, np.trace(rho @ sy).real, np.trace(rho @ sz).real])
    basis_states = [0.5*(I2 + sx), 0.5*(I2 + sy), 0.5*(I2 + sz)]
    Acols = []
    for S in basis_states:
        r_t = r_from_state(evolve_state(S, total_time))
        Acols.append(r_t - c)
    A = np.column_stack(Acols)
    T = np.zeros((4,4), float)
    T[0,0] = 1.0
    T[1:,0] = c
    T[1:,1:] = A
    return T

# ===== Basis-Alignment + adaptive SU(2)-Verfeinerung (für GADC-Fit) ========
X = sx  # Pauli-X

def basis_alignment_unitary(omega: float) -> np.ndarray:
    """
    Standard-GADC nimmt |1> als 'excited' an.
    Für H=(ω/2)σ_z ist bei ω>0 der 'excited'-Eigenzustand |0>.
    Wir alignen daher systematisch mit X (Swap |0><->|1|), sonst Identität.
    """
    return X if omega > 0 else I2

def su2_from_vector(v: np.ndarray) -> np.ndarray:
    """Kleine SU(2)-Rotation aus R^3: U = exp(-i/2 * (vx σx + vy σy + vz σz))."""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    A = vx*sx + vy*sy + vz*sz
    theta = np.linalg.norm([vx,vy,vz])
    if theta < 1e-12:
        return I2
    c = math.cos(theta/2.0); s = math.sin(theta/2.0)
    return c*I2 - 1j*s*(A/theta)

def refine_unitaries_adaptive(C_target: np.ndarray,
                              C_seed: np.ndarray,
                              U_align: np.ndarray,
                              max_iter: int = 60,
                              step0: float = 0.3,
                              step_min: float = 1e-3,
                              patience: int = 10) -> tuple[np.ndarray,np.ndarray,float]:
    """
    Minimiert || C_target - (U_in^T ⊗ U_out) C_seed (U_in* ⊗ U_out^†) ||_F
    mit U_in = U_align @ δU_in, U_out = δU_out @ U_align.
    δU_in/out werden per koordinatenweiser Schrittweiten-Reduktion gesucht.
    """
    Uin_delta = I2.copy()
    Uout_delta = I2.copy()

    def apply(Uin_d, Uout_d):
        Uin = U_align @ Uin_d
        Uout = Uout_d @ U_align
        return choi_conjugate(C_seed, Uout, Uin)

    C_cur = apply(Uin_delta, Uout_delta)
    best = np.linalg.norm(C_target - C_cur)
    no_improve = 0
    step = step0

    axes = [np.array([1.0,0,0]), np.array([0,1.0,0]), np.array([0,0,1.0])]

    for _ in range(max_iter):
        improved = False
        for which in ("in","out"):
            for ax in axes:
                for sgn in (+1.0, -1.0):
                    v = sgn*step*ax
                    U_try = su2_from_vector(v)
                    if which == "in":
                        C_try = apply(U_try @ Uin_delta, Uout_delta)
                        cand = np.linalg.norm(C_target - C_try)
                        if cand + 1e-12 < best:
                            Uin_delta = U_try @ Uin_delta
                            best = cand
                            C_cur = C_try
                            improved = True
                    else:
                        C_try = apply(Uin_delta, U_try @ Uout_delta)
                        cand = np.linalg.norm(C_target - C_try)
                        if cand + 1e-12 < best:
                            Uout_delta = U_try @ Uout_delta
                            best = cand
                            C_cur = C_try
                            improved = True
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step *= 0.5
            if step < step_min or no_improve >= patience:
                break

    Uin_total  = U_align @ Uin_delta
    Uout_total = Uout_delta @ U_align
    return Uin_total, Uout_total, float(best)

# =================== Numerik: weitere GKSL-Step-Varianten ===================
def rk4_step_raw(rho: np.ndarray, dt: float, deriv, *args) -> np.ndarray:
    """RK4 ohne PSD-Clipping (nur Hermitisieren + Tracenorm)."""
    k1 = deriv(rho, *args)
    k2 = deriv(rho + 0.5*dt*k1, *args)
    k3 = deriv(rho + 0.5*dt*k2, *args)
    k4 = deriv(rho + dt*k3, *args)
    rho_next = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    rho_next = 0.5*(rho_next + dagger(rho_next))
    tr = np.trace(rho_next)
    if abs(tr) > 1e-14:
        rho_next = rho_next / tr
    return rho_next

def gksl_step_rk4(rho: np.ndarray, dt: float, omega: float, rates: GKSLRates, project_psd: bool=True) -> np.ndarray:
    """GKSL-Schritt mit RK4; optional PSD-Projektion (wie 'rk4_step')."""
    if project_psd:
        return rk4_step(rho, dt, gksl_rhs_amp_damp, omega, rates)  # PSD-Projektion
    else:
        return rk4_step_raw(rho, dt, gksl_rhs_amp_damp, omega, rates)

def unitary_half_step_z(rho: np.ndarray, omega: float, dt: float) -> np.ndarray:
    """Halber unitärer Schritt für H=(ω/2)σ_z: ρ → U ρ U† mit U=exp(-i H dt/2)."""
    phase = 0.5*omega*(dt/2.0)
    U = np.array([[np.exp(-1j*phase), 0],[0, np.exp(+1j*phase)]], dtype=complex)
    return U @ rho @ dagger(U)

def apply_gadc_exact_dt(rho: np.ndarray, dt: float, rates: GKSLRates) -> np.ndarray:
    """
    Exakter dissipativer Schritt (CPTP) für finite dt:
      - Gesamt-Relaxationsrate K = γ↓ + γ↑
      - η(dt) = 1 - exp(-K·dt)
      - q = γ↑/K  (für K>0, sonst q=0, η=0)
    Entspricht der 'generalized amplitude damping'-Map über Zeit dt.
    """
    K = rates.gamma_down + rates.gamma_up
    if K <= 0:
        return rho.copy()
    eta = 1.0 - math.exp(-K*dt)
    q   = rates.gamma_up / K
    Ks = gadc_kraus(eta, q)
    acc = np.zeros_like(rho, dtype=complex)
    for Kk in Ks:
        acc += Kk @ rho @ dagger(Kk)
    acc = 0.5*(acc + dagger(acc))
    tr = np.trace(acc)
    if abs(tr) > 1e-14:
        acc = acc / tr
    return acc

def gksl_step_strang_cp(rho: np.ndarray, dt: float, omega: float, rates: GKSLRates) -> np.ndarray:
    """
    Strang-Splitting (2. Ordnung), CP/TP:
       1) U(dt/2)  2) dissipativ (exakte GADC über dt)  3) U(dt/2)
    Bewahrt CPTP ohne PSD-Clipping, ist physikalisch interpretierbar.
    """
    r = unitary_half_step_z(rho, omega, dt)
    r = apply_gadc_exact_dt(r, dt, rates)
    r = unitary_half_step_z(r, omega, dt)
    return r

# ============================== Streamlit UI ================================
st.set_page_config(page_title="GKSL-Verifikation (Micro → GKSL)", layout="wide")
st.title("GKSL aus Repeated Interactions + VERIFIKATION")

# Minimal-CSS für Badges (OK/FAIL)
st.markdown("""
<style>
.badge-ok{background:#10b98122;color:#065f46;border:1px solid #10b981;border-radius:999px;padding:2px 8px;font-weight:600;font-size:0.85rem;}
.badge-fail{background:#ef444422;color:#7f1d1d;border:1px solid #ef4444;border-radius:999px;padding:2px 8px;font-weight:600;font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

def badge_html(ok: bool) -> str:
    return "<span class='badge-ok'>OK</span>" if ok else "<span class='badge-fail'>FAIL</span>"

TAB1, TAB2, TAB3, TAB4 = st.tabs(["1) Trace-out & Raten", "2) GKSL & Fits", "3) Verification Suite", "4) Export"])

# ================================== Tab 1 ===================================
with TAB1:
    st.subheader("Repeated Interactions: unitäre Kollisionen + partielle Spur")
    colL, colR = st.columns(2)
    with colL:
        omega = st.slider("ω (System)", 0.1, 5.0, 1.0, 0.1, key="omega_slider", on_change=mark_micro_dirty)
        beta  = st.slider("β (Bad)", 0.0, 10.0, 2.0, 0.1, key="beta_slider", on_change=mark_micro_dirty)
        g     = st.slider("Kopplung g", 0.0, 2.0, 0.5, 0.05, key="g_slider", on_change=mark_micro_dirty)
        tau_int = st.slider("Pulsdauer τ_int", 0.02, 1.0, 0.2, 0.02, key="tau_slider", on_change=mark_micro_dirty)
        dt_coll = st.slider("Kollisionsintervall Δt", 0.02, 1.0, 0.2, 0.02, key="dt_slider", on_change=mark_micro_dirty)
        n_steps = st.slider("Anzahl Kollisionen", 5, 400, 60, 5, key="steps_slider", on_change=mark_micro_dirty)
        theta0  = st.slider("θ (Startzustand)", 0.0, float(math.pi), float(math.pi/3), 0.01, key="theta0_slider", on_change=mark_micro_dirty)
        phi0    = st.slider("φ (Startzustand)", 0.0, 2*float(math.pi), 0.0, 0.01, key="phi0_slider", on_change=mark_micro_dirty)
    with colR:
        run = st.button("Run micro trace-out", type="primary")
        psi = np.array([math.cos(theta0/2.0), math.sin(theta0/2.0)*np.exp(1j*phi0)], dtype=complex)
        rhoS0 = np.outer(psi, psi.conj())
        tauB = thermal_qubit(beta, omega)
        p_exc = float(tauB[0,0].real); p_gnd = 1.0 - p_exc
        P = MicroParams(omega=omega, beta=beta, g=g, tau_int=tau_int, dt_coll=dt_coll,
                        n_steps=n_steps, theta0=theta0, phi0=phi0)
        rates = rates_from_micro(P)
        kms_gap = rates.gamma_up/(rates.gamma_down+1e-30) - math.exp(-beta*omega)
        st.caption(f"Ancilla thermal: p_exc≈{p_exc:.4f}, p_gnd≈{p_gnd:.4f}  (KMS: p_exc/p_gnd=e^(-βω))")
        st.write(f"Raten (aus Mikro): γ↓≈{rates.gamma_down:.5f}, γ↑≈{rates.gamma_up:.5f}")
        st.write(f"KMS-Check: γ↑/γ↓ − e^(-βω) ≈ {kms_gap:.3e}")

    if run:
        ts = [0.0]
        rhos_coll = [rhoS0.copy()]
        theta = g * tau_int
        for k in range(1, n_steps+1):
            rhos_coll.append(one_collision_map(rhos_coll[-1], tauB, theta))
            ts.append(k*dt_coll)
        ts = np.array(ts)

        # Plots Tab1
        fig1 = go.Figure()
        fig1.update_layout(template="plotly_white")
        fig1.add_scatter(x=ts, y=[pop00(r) for r in rhos_coll], name="ρ00 (trace-out)",
                         mode="lines", line=dict(width=1.8, color="rgba(23,94,165,1)"))
        fig1.add_scatter(x=ts, y=[pop11(r) for r in rhos_coll], name="ρ11 (trace-out)",
                         mode="lines", line=dict(width=1.8, color="rgba(200,90,0,1)"))
        p0_star, p1_star = gibbs_populations(beta, omega)
        fig1.add_hline(y=p0_star, line_dash="dash", annotation_text="p0* (Gibbs)")
        fig1.add_hline(y=p1_star, line_dash="dash", annotation_text="p1* (Gibbs)")
        fig1.update_layout(xaxis_title="t", yaxis_title="population")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.update_layout(template="plotly_white")
        fig2.add_scatter(x=ts, y=[coh_abs(r) for r in rhos_coll], name="|ρ01| (trace-out)",
                         mode="lines", line=dict(width=1.8, color="rgba(31,119,180,1)"))
        fig2.update_layout(xaxis_title="t", yaxis_title="coherence magnitude")
        st.plotly_chart(fig2, use_container_width=True)
        st.success(f"PSD(last)={is_psd(rhos_coll[-1])}, Tr≈{np.trace(rhos_coll[-1]).real:.6f}")

        # Session-State sauber setzen
        st.session_state["traceout_data"] = {
            "params": P, "ts": ts, "rhos": rhos_coll,
            "gibbs": (p0_star, p1_star), "rates": rates,
            "kms_gap": kms_gap, "p_exc": p_exc, "p_gnd": p_gnd,
            "theta": theta, "tauB": tauB, "run_id": st.session_state["run_id"] + 1
        }
        st.session_state["figs"] = {}
        st.session_state["tab2_data"] = None
        st.session_state["verif"] = None
        st.session_state["run_id"] += 1
        st.session_state["micro_dirty"] = False

    with st.expander("📚 Supplement (Tab 1: Micro → Raten, KMS)"):
        st.markdown("**Collision-Map & partielle Spur**")
        st.latex(r"\rho_S' = \mathrm{Tr}_B\!\left[\,U\,(\rho_S\!\otimes\!\tau_B)\,U^\dagger\,\right]")
        st.latex(r"U = e^{-i\theta(\sigma_+\otimes\sigma_- + \sigma_-\otimes\sigma_+)}")
        st.markdown("**Thermischer Ancilla-Zustand (KMS)**")
        st.latex(r"\tau_B = \frac{e^{-\beta(\omega/2)\sigma_z}}{\mathrm{Tr}\,e^{-\beta(\omega/2)\sigma_z}} = \mathrm{diag}(p_{\rm exc}, p_{\rm gnd})")
        st.latex(r"\frac{p_{\rm exc}}{p_{\rm gnd}}=e^{-\beta\omega}")
        st.markdown("**Raten im kleinen Winkel**")
        st.latex(r"\gamma_\downarrow = \frac{\sin^2\theta}{\Delta t}\,p_{\rm gnd},\qquad \gamma_\uparrow = \frac{\sin^2\theta}{\Delta t}\,p_{\rm exc}.")

# ================================== Tab 2 ===================================
with TAB2:
    st.subheader("GKSL-Integration & Fits")
    data = st.session_state.get("traceout_data")

    if data is None:
        st.info("Bitte zuerst in Tab 1 rechnen.")
        st.stop()

    if st.session_state.get("micro_dirty", False):
        st.warning("Parameter in Tab 1 wurden geändert. Bitte erneut **Run micro trace-out** starten.")
        st.stop()

    P: MicroParams = data["params"]
    rates: GKSLRates = data["rates"]
    ts_coll = data["ts"]
    rhos_coll = data["rhos"]
    p0_star, p1_star = data["gibbs"]

    # ---- Integrator-Auswahl ----
    colInt1, colInt2 = st.columns([2,1])
    with colInt1:
        integrator = st.radio(
            "Integrator für GKSL",
            ["Strang (CPTP, 2. Ordnung)", "RK4 (mit PSD-Projektion)", "RK4 (ohne Projektion)"],
            index=["Strang (CPTP, 2. Ordnung)", "RK4 (mit PSD-Projektion)", "RK4 (ohne Projektion)"].index(st.session_state.get("integrator_choice","Strang (CPTP, 2. Ordnung)")),
            horizontal=True
        )
    with colInt2:
        sub = st.slider("Substeps pro Δt", 1, 20, st.session_state.get("integrator_sub",5), 1,
                        help="Feinere Substeps → geringerer Diskretisierungsfehler")

    st.session_state["integrator_choice"] = integrator
    st.session_state["integrator_sub"]    = sub

    # ---- GKSL-Integration mit gewähltem Integrator ----
    rho = rhos_coll[0].copy()
    rhos_gksl = [rho.copy()]
    dt_sub = P.dt_coll / max(1, sub)

    def do_step(rho, dt):
        if "Strang" in integrator:
            return gksl_step_strang_cp(rho, dt, P.omega, rates)
        elif "ohne Projektion" in integrator:
            return gksl_step_rk4(rho, dt, P.omega, rates, project_psd=False)
        else:
            return gksl_step_rk4(rho, dt, P.omega, rates, project_psd=True)

    for _ in range(1, len(ts_coll)):
        for __ in range(sub):
            rho = do_step(rho, dt_sub)
        rhos_gksl.append(rho)

    # ---------- Plots (sichtbare, nicht-weiße Farben) ----------
    # ρ00
    fig21 = go.Figure(); fig21.update_layout(template="plotly_white")
    fig21.add_scatter(x=ts_coll, y=[pop00(r) for r in rhos_coll], name="ρ00 (trace-out)",
                      mode="lines", line=dict(width=1.8, color="rgba(23,94,165,1)"))
    fig21.add_scatter(x=ts_coll, y=[pop00(r) for r in rhos_gksl], name="ρ00 (GKSL)",
                      mode="lines", line=dict(width=1.8, dash="dash", color="rgba(31,119,180,1)"))
    fig21.add_hline(y=p0_star, line_dash="dot", annotation_text="p0* (Gibbs)")
    fig21.update_layout(xaxis_title="t", yaxis_title="population ρ00")
    st.plotly_chart(fig21, use_container_width=True)

    # |ρ01|
    fig22 = go.Figure(); fig22.update_layout(template="plotly_white")
    fig22.add_scatter(x=ts_coll, y=[coh_abs(r) for r in rhos_coll], name="|ρ01| (trace-out)",
                      mode="lines", line=dict(width=1.8, color="rgba(200,90,0,1)"))
    fig22.add_scatter(x=ts_coll, y=[coh_abs(r) for r in rhos_gksl], name="|ρ01| (GKSL)",
                      mode="lines", line=dict(width=1.8, dash="dash", color="rgba(255,127,14,1)"))
    fig22.update_layout(xaxis_title="t", yaxis_title="coherence magnitude")
    st.plotly_chart(fig22, use_container_width=True)

    # arg(ρ01)
    fig23 = go.Figure(); fig23.update_layout(template="plotly_white")
    fig23.add_scatter(x=ts_coll, y=[coh_phase(r) for r in rhos_coll], name="arg ρ01 (trace-out)",
                      mode="lines", line=dict(width=1.8, color="rgba(31,119,180,1)"))
    fig23.add_scatter(x=ts_coll, y=[coh_phase(r) for r in rhos_gksl], name="arg ρ01 (GKSL)",
                      mode="lines", line=dict(width=1.8, dash="dash", color="rgba(23,94,165,1)"))
    fig23.update_layout(xaxis_title="t", yaxis_title="phase arg ρ01")
    st.plotly_chart(fig23, use_container_width=True)

    # Trace distance
    fig24 = go.Figure(); fig24.update_layout(template="plotly_white")
    fig24.add_scatter(x=ts_coll, y=[trace_distance(a,b) for a,b in zip(rhos_coll, rhos_gksl)],
                      name="Trace distance", mode="lines",
                      line=dict(width=1.8, color="rgba(128,0,128,1)"))
    fig24.update_layout(xaxis_title="t", yaxis_title="T(ρ_trace, ρ_GKSL)")
    st.plotly_chart(fig24, use_container_width=True)

    # ---------- Fits (T1/T2) + CI-Bänder ----------
    st.subheader("Exponentielle Fits (T1/T2) + 95%-CI")
    colL, colR = st.columns(2)
    with colL:
        fit_source = st.radio("Fit-Quelle", ["Trace-out", "GKSL"], index=0, horizontal=True)
        show_ci = st.checkbox("95%-CI anzeigen", value=True)
    with colR:
        T1_exp = 1.0 / (rates.gamma_down + rates.gamma_up + 1e-30)
        T2_exp = 2.0 * T1_exp
        st.write(f"Erwartet (ohne reine Dephasierung):  T1≈{T1_exp:.3f},  T2≈{T2_exp:.3f}  ⇒  T2≈2·T1")

    rhos_fit = rhos_coll if "Trace-out" in fit_source else rhos_gksl
    y_pop = np.array([r[1,1].real for r in rhos_fit])   # p11(t)
    y_coh = np.array([abs(r[0,1]) for r in rhos_fit])   # |p01|(t)

    eps = 1e-12

    # --- Fit T1 ---
    mask1 = np.abs(y_pop - p1_star) > 1e-10
    if np.count_nonzero(mask1) >= 2:
        t1 = ts_coll[mask1]; ln1 = np.log(np.abs(y_pop[mask1] - p1_star) + eps)
        m1, b1, se_m1, se_b1 = linreg_with_errors(t1, ln1)
        T1_fit = float('inf') if m1 == 0 else -1.0/m1
        sgn = 1.0 if (y_pop[0] - p1_star) >= 0 else -1.0
        amp_fit = np.exp(b1 + m1*ts_coll)
        y_pop_fit = p1_star + sgn*amp_fit
        if show_ci:
            z = 1.96
            amp_up  = np.exp((b1 + z*se_b1) + (m1 + z*se_m1)*ts_coll)
            amp_low = np.exp((b1 - z*se_b1) + (m1 - z*se_m1)*ts_coll)
            y_pop_low_band = np.minimum(p1_star + sgn*amp_low, p1_star + sgn*amp_up)
            y_pop_up_band  = np.maximum(p1_star + sgn*amp_low, p1_star + sgn*amp_up)
    else:
        T1_fit = float('nan')
        y_pop_fit = y_pop.copy()
        show_ci = False  # keine CI berechenbar

    # --- Fit T2 ---
    mask2 = y_coh > 1e-12
    if np.count_nonzero(mask2) >= 2:
        t2 = ts_coll[mask2]; ln2 = np.log(y_coh[mask2] + eps)
        m2, b2, se_m2, se_b2 = linreg_with_errors(t2, ln2)
        T2_fit = float('inf') if m2 == 0 else -1.0/m2
        y_coh_fit = np.exp(b2 + m2*ts_coll)
        if show_ci:
            z = 1.96
            y_coh_low = np.exp((b2 - z*se_b2) + (m2 - z*se_m2)*ts_coll)
            y_coh_up  = np.exp((b2 + z*se_b2) + (m2 + z*se_m2)*ts_coll)
    else:
        T2_fit = float('nan')
        y_coh_fit = y_coh.copy()
        show_ci = False

    st.write(f"Fit:  T1≈{T1_fit:.3f},  T2≈{T2_fit:.3f},  Verhältnis T2/(2·T1)≈{(T2_fit/(2*T1_fit)) if T1_fit>0 else float('nan'):.3f}")

    # Farben für Fits/CI (keine weißen Kurven)
    data_color_T1   = "rgba(23, 94, 165, 1.00)"
    fit_color_T1    = "rgba(31, 119, 180, 1.00)"
    ci_line_T1      = "rgba(31, 119, 180, 0.90)"
    ci_fill_T1      = "rgba(31, 119, 180, 0.18)"

    data_color_T2   = "rgba(200, 90, 0, 1.00)"
    fit_color_T2    = "rgba(255, 127, 14, 1.00)"
    ci_line_T2      = "rgba(255, 127, 14, 0.90)"
    ci_fill_T2      = "rgba(255, 127, 14, 0.18)"

    # ---------- fig25 (T1) ----------
    fig25 = go.Figure(); fig25.update_layout(template="plotly_white")
    fig25.add_scatter(
        x=ts_coll, y=y_pop, name="p11 (Daten)",
        mode="lines", line=dict(width=1.8, color=data_color_T1)
    )
    if show_ci:
        fig25.add_scatter(
            x=ts_coll, y=y_pop_low_band,
            name="fit −95% CI",
            mode="lines",
            line=dict(color=ci_line_T1, width=1.2, dash="dot"),
            showlegend=True, hoverinfo="skip",
            legendgroup="T1_CI"
        )
        fig25.add_scatter(
            x=ts_coll, y=y_pop_up_band,
            name="fit +95% CI",
            mode="lines",
            line=dict(color=ci_line_T1, width=1.2, dash="dot"),
            fill="tonexty", fillcolor=ci_fill_T1,
            showlegend=True, hoverinfo="skip",
            legendgroup="T1_CI"
        )
    fig25.add_scatter(
        x=ts_coll, y=y_pop_fit, name="exp-fit T1",
        mode="lines", line=dict(color=fit_color_T1, width=2.2, dash="dash")
    )
    fig25.add_hline(y=p1_star, line_dash="dot", annotation_text="p1* (Gibbs)")
    fig25.update_layout(xaxis_title="t", yaxis_title="population p11")
    st.plotly_chart(fig25, use_container_width=True)

    # ---------- fig26 (T2) ----------
    fig26 = go.Figure(); fig26.update_layout(template="plotly_white")
    fig26.add_scatter(
        x=ts_coll, y=y_coh, name="|p01| (Daten)",
        mode="lines", line=dict(width=1.8, color=data_color_T2)
    )
    if show_ci:
        fig26.add_scatter(
            x=ts_coll, y=y_coh_low,
            name="fit −95% CI",
            mode="lines",
            line=dict(color=ci_line_T2, width=1.2, dash="dot"),
            showlegend=True, hoverinfo="skip",
            legendgroup="T2_CI"
        )
        fig26.add_scatter(
            x=ts_coll, y=y_coh_up,
            name="fit +95% CI",
            mode="lines",
            line=dict(color=ci_line_T2, width=1.2, dash="dot"),
            fill="tonexty", fillcolor=ci_fill_T2,
            showlegend=True, hoverinfo="skip",
            legendgroup="T2_CI"
        )
    fig26.add_scatter(
        x=ts_coll, y=y_coh_fit, name="exp-fit T2",
        mode="lines", line=dict(color=fit_color_T2, width=2.2, dash="dash")
    )
    fig26.update_layout(xaxis_title="t", yaxis_title="coherence magnitude")
    st.plotly_chart(fig26, use_container_width=True)

    # ---------- Δt-Konvergenz ----------
    st.subheader("Δt-Konvergenz: max Trace-Distance vs Δt (Log-Log)")
    T_end = ts_coll[-1]
    K = rates.gamma_down + rates.gamma_up
    base = P.dt_coll
    dts = [base/(2**k) for k in range(0,5)]
    TDmax, DTused = [], []
    tauB = thermal_qubit(P.beta, P.omega)
    rho0 = rhos_coll[0].copy()

    for dt_prime in dts:
        r_prime = 1.0/dt_prime
        sin2 = K / r_prime
        if not (0.0 <= sin2 <= 1.0): continue
        theta_prime = math.asin(math.sqrt(max(min(sin2,1.0),0.0)))
        n_steps_prime = int(round(T_end / dt_prime))
        if n_steps_prime <= 0: continue
        rhos_to = [rho0.copy()]
        for _ in range(n_steps_prime):
            rhos_to.append(one_collision_map(rhos_to[-1], tauB, theta_prime))

        # GKSL-Referenz mit gewähltem Integrator
        subp = max(1, int(round((P.dt_coll/dt_prime)*sub)))
        dtsub = dt_prime/subp
        rhoG = rho0.copy(); rhos_g = [rhoG.copy()]
        def do_step_local(rho, dt):
            if "Strang" in integrator:
                return gksl_step_strang_cp(rho, dt, P.omega, rates)
            elif "ohne Projektion" in integrator:
                return gksl_step_rk4(rho, dt, P.omega, rates, project_psd=False)
            else:
                return gksl_step_rk4(rho, dt, P.omega, rates, project_psd=True)
        for _ in range(n_steps_prime):
            for __ in range(subp):
                rhoG = do_step_local(rhoG, dtsub)
            rhos_g.append(rhoG)

        TDarr = [trace_distance(a,b) for a,b in zip(rhos_to, rhos_g)]
        TDmax.append(max(TDarr)); DTused.append(dt_prime)

    if DTused:
        fig27 = go.Figure(); fig27.update_layout(template="plotly_white")
        fig27.add_scatter(x=DTused, y=TDmax, mode="lines+markers", name="max TD",
                          line=dict(width=1.8, color="rgba(128,0,128,1)"))
        c = TDmax[0]/(DTused[0]**2) if DTused[0] > 0 else 1.0
        dt_ref = np.linspace(min(DTused), max(DTused), 50)
        fig27.add_scatter(x=dt_ref, y=c*(dt_ref**2), name="~ Δt² (ref)",
                          mode="lines", line=dict(width=1.5, dash="dash", color="rgba(0,128,0,1)"))
        fig27.update_xaxes(type="log"); fig27.update_yaxes(type="log")
        fig27.update_layout(xaxis_title="Δt", yaxis_title="max Trace distance")
        st.plotly_chart(fig27, use_container_width=True)
        st.session_state["figs"]["tab2_dt_convergence"] = fig27
    else:
        st.info("Δt zu klein für gegebene Raten (sin²θ ≤ 1). Δt erhöhen oder Raten reduzieren.")

    ratio_gap = rates.gamma_up/(rates.gamma_down+1e-30) - math.exp(-P.beta*P.omega)
    st.write(f"Raten: γ↓≈{rates.gamma_down:.5f}, γ↑≈{rates.gamma_up:.5f}  ⇒  T1≈{T1_exp:.3f}, T2≈{T2_exp:.3f}")
    st.write(f"KMS-Check: γ↑/γ↓ - e^(-βω) ≈ {ratio_gap:.3e}")

    st.session_state["figs"] |= {
        "tab2_pop00": fig21, "tab2_coh_abs": fig22, "tab2_phase": fig23,
        "tab2_trace_distance": fig24, "tab2_fit_T1": fig25, "tab2_fit_T2": fig26
    }
    st.session_state["tab2_data"] = {"T1_fit": float(T1_fit), "T2_fit": float(T2_fit),
                                     "T1_theory": float(T1_exp), "T2_theory": float(T2_exp)}

    with st.expander("📚 Supplement (Tab 2): GKSL, T1/T2, Δt-Konvergenz", expanded=False):
        st.markdown("**GKSL**")
        st.latex(r"\dot\rho=-i[H,\rho]+\sum_k \big(L_k\rho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\rho\}\big)")
        st.markdown("Thermische Relaxation (ohne reine Dephasierung):")
        st.latex(r"L_1=\sqrt{\gamma_\downarrow}\,\sigma_- ,\quad L_2=\sqrt{\gamma_\uparrow}\,\sigma_+")
        st.markdown("**T1/T2** (ohne reine Dephasierung)")
        st.latex(r"T_1=1/(\gamma_\downarrow+\gamma_\uparrow),\quad T_2=2T_1")
        st.markdown("**Δt-Scaling (Collision→GKSL)**")
        st.latex(r"\text{halte } K=\sin^2\theta/\Delta t \ \text{konstant, Fehler } \propto \Delta t^2")
    with st.expander("🧮 Numerik-Hinweis", expanded=False):
        st.markdown(
            "- **RK4 + PSD-Projektion** (EVD-Clipping) hält ρ≽0 numerisch stabil, ist **kein physikalischer Mechanismus**.\n"
            "- **RK4 (ohne Projektion)** zeigt echten Integrationsfehler; bei zu grobem dt können kleine Negativ-Eigenwerte auftreten.\n"
            "- **Strang-Splitting (CPTP)**: U(dt/2) → exakte **GADC** über dt → U(dt/2). Erhält **CP/TP** für jeden Schritt (2. Ordnung).\n"
            "- Für **harte Fehlergrenzen**: dt fein wählen (Substeps↑) und bevorzugt **Strang (CPTP)** nutzen."
        )

# ============================ Tab 3: Verification ===========================
with TAB3:
    st.subheader("Verification Suite (CPTP/Choi, GADC-Fit, Semigroup, Spohn, MCWF≈GKSL)")
    data = st.session_state.get("traceout_data")

    if data is None:
        st.info("Bitte zuerst in Tab 1 rechnen.")
        st.stop()

    if st.session_state.get("micro_dirty", False):
        st.warning("Parameter in Tab 1 wurden geändert. Bitte erneut **Run micro trace-out** starten.")
        st.stop()

    P: MicroParams = data["params"]
    rates: GKSLRates = data["rates"]
    tauB = data["tauB"]
    theta = data["theta"]
    rhos_coll = data["rhos"]
    rho0 = rhos_coll[0].copy()

    # ---- A) CPTP/Choi ----
    st.markdown("### A) Choi + CPTP (ein Kollisionsschritt)")
    ks_micro = kraus_from_micro(theta, tauB)
    C_micro = choi_from_kraus(ks_micro)
    tp_err = tp_defect(ks_micro)
    min_eig = float(np.min(np.linalg.eigvalsh((C_micro + dagger(C_micro))/2)).real)
    colA1, colA2 = st.columns(2)
    with colA1: st.metric("TP-Defekt ‖ΣK†K−I‖", f"{tp_err:.3e}")
    with colA2: st.metric("Choi min-Eig (≥0 erwartet)", f"{min_eig:.3e}")
    A_ok = (tp_err <= 1e-12) and (min_eig >= -1e-12)
    st.markdown(f"{badge_html(A_ok)} Ziel: TP-Defekt = 0, minEig(Choi) ≥ 0", unsafe_allow_html=True)

    # ---- B) GADC-Fit (basis-aligned + adaptive) ----
    st.markdown("### B) GADC-Fit (ein Schritt) — basis-aligned + adaptive")
    eta_target = math.sin(theta)**2
    q_target   = float(tauB[0,0].real)  # p_exc(β,ω)
    U_align = basis_alignment_unitary(P.omega)

    with st.expander("Fit-Optionen"):
        search_width = st.slider("η/q-Suchbreite um Ziel", 0.00, 0.30, 0.10, 0.01)
        coarse_pts   = st.slider("Koarse η/q-Punkte (je Achse)", 3, 11, 5, 1)
        step0        = st.slider("Startschritt (rad) für δU", 0.05, 1.0, 0.3, 0.05)
        max_iter     = st.slider("max_iter (adaptive SU(2))", 20, 120, 60, 10)

    colZ1, colZ2 = st.columns(2)
    with colZ1: st.metric("Ziel η = sin²θ", f"{eta_target:.6f}")
    with colZ2: st.metric("Ziel q = p_exc(β,ω)", f"{q_target:.6f}")

    etas = np.clip(np.linspace(eta_target - search_width, eta_target + search_width, int(coarse_pts)), 0, 1)
    qs   = np.clip(np.linspace(q_target   - search_width, q_target   + search_width, int(coarse_pts)), 0, 1)

    # Start mit exakt passenden Parametern (robuste Konvergenz)
    Cgad_base = choi_from_kraus(gadc_kraus(eta_target, q_target))
    CgadA_base = choi_conjugate(Cgad_base, U_align, U_align)
    Uin_opt, Uout_opt, d0 = refine_unitaries_adaptive(C_micro, CgadA_base, U_align,
                                                      max_iter=int(max_iter), step0=float(step0))
    best = {"dist": d0, "eta": eta_target, "q": q_target, "Uin": Uin_opt, "Uout": Uout_opt}

    for e in etas:
        for q in qs:
            Cgad0 = choi_from_kraus(gadc_kraus(e, q))            # Standard-GADC (excited=|1>)
            CgadA = choi_conjugate(Cgad0, U_align, U_align)      # Basis-Ausrichtung
            Uin_o, Uout_o, d = refine_unitaries_adaptive(
                C_micro, CgadA, U_align, max_iter=int(max_iter), step0=float(step0)
            )
            if d < best["dist"]:
                best = {"dist": d, "eta": e, "q": q, "Uin": Uin_o, "Uout": Uout_o}

    d_best, eta_best, q_best = best["dist"], best["eta"], best["q"]
    colB1, colB2, colB3 = st.columns(3)
    with colB1: st.metric("min ‖ΔChoi‖_F", f"{d_best:.3e}")
    with colB2: st.metric("η*", f"{eta_best:.6f}")
    with colB3: st.metric("q*", f"{q_best:.6f}")
    st.write(f"Δη = {abs(eta_best-eta_target):.3e},   Δq = {abs(q_best-q_target):.3e}")

    B_ok = (abs(eta_best-eta_target) <= 1e-3) and (abs(q_best-q_target) <= 1e-3) and (d_best <= 5e-3)
    st.markdown(f"{badge_html(B_ok)} Ziel: η≈sin²θ, q≈p_exc(β,ω) und ΔChoi minimal", unsafe_allow_html=True)

    # ---- C) Semigroup (affine PTM) ----
    st.markdown("### C) Semigroup-Test (GKSL):  Φ_{t+s} ≈ Φ_t ∘ Φ_s (affine PTM)")
    # Integrator-Einstellung aus Tab 2 (Fallbacks)
    integ_choice = st.session_state.get("integrator_choice", "Strang (CPTP, 2. Ordnung)")
    sub_tab3   = st.session_state.get("integrator_sub", 5)

    def do_step_v(rho, dt):
        if "Strang" in integ_choice:
            return gksl_step_strang_cp(rho, dt, P.omega, rates)
        elif "ohne Projektion" in integ_choice:
            return gksl_step_rk4(rho, dt, P.omega, rates, project_psd=False)
        else:
            return gksl_step_rk4(rho, dt, P.omega, rates, project_psd=True)

    def evolve_state_to_t(rho, t):
        """Propagiere bis Zeit t mit gewähltem Integrator (sub-Steps für Stabilität)."""
        if t <= 0:
            return rho.copy()
        steps = max(1, int(round(t / (P.dt_coll/max(1,sub_tab3)))))
        dt = t / steps
        r = rho.copy()
        for _ in range(steps):
            r = do_step_v(r, dt)
        return r

    t = st.slider("Zeit t", 0.0, max(0.2, float(P.n_steps*P.dt_coll)), float(P.dt_coll*5), 0.01)
    s = st.slider("Zeit s", 0.0, max(0.2, float(P.n_steps*P.dt_coll)), float(P.dt_coll*3), 0.01)
    T_t  = pauli_transfer_affine(lambda R,tt: evolve_state_to_t(R, tt), t)
    T_s  = pauli_transfer_affine(lambda R,tt: evolve_state_to_t(R, tt), s)
    T_ts = pauli_transfer_affine(lambda R,tt: evolve_state_to_t(R, tt), t+s)
    defect = float(np.linalg.norm(T_ts - T_t @ T_s))
    st.metric("Semigroup-Defekt ‖T(t+s)−T(t)T(s)‖_F", f"{defect:.3e}")
    C_ok = (defect <= 1e-2)
    st.markdown(f"{badge_html(C_ok)} Ziel: Semigroup-Defekt = 0", unsafe_allow_html=True)

    # ---- D) Gibbs-Fixpunkt & Spohn-Monotonie ----
    st.markdown("### D) Gibbs-Fixpunkt & Spohn-Monotonie")
    p0s, p1s = gibbs_populations(P.beta, P.omega)
    rho_beta = np.array([[p0s,0],[0,p1s]], dtype=complex)
    T_end = float(P.n_steps*P.dt_coll)
    nT = 60
    ts = np.linspace(0, T_end, nT)
    path = []
    r = rho0.copy()
    last_t = 0.0
    for tcur in ts:
        dt_loc = tcur - last_t
        steps_loc = max(1, int(round(dt_loc/(P.dt_coll/max(1,sub_tab3))))) if dt_loc>0 else 1
        dt_sub = (dt_loc/steps_loc) if steps_loc>0 else 0.0
        for _ in range(steps_loc):
            r = do_step_v(r, dt_sub)
        path.append(r.copy())
        last_t = tcur
    Dvals = []
    Lb = logm_herm(rho_beta)
    for R in path:
        Lr = logm_herm(R)
        D = float(np.trace(R@(Lr - Lb)).real)
        Dvals.append(D)
    figD = go.Figure(); figD.update_layout(template="plotly_white")
    figD.add_scatter(x=ts, y=Dvals, name="D(ρ_t || ρ_β)",
                     mode="lines", line=dict(width=1.8, color="rgba(31,119,180,1)"))
    figD.update_layout(xaxis_title="t", yaxis_title="relative entropy")
    st.plotly_chart(figD, use_container_width=True)
    st.session_state["figs"]["tab3_D_relent"] = figD
    colD1, colD2 = st.columns(2)
    with colD1: st.metric("min D(ρ||ρβ)", f"{min(Dvals):.6f}")
    with colD2: st.metric("max D(ρ||ρβ)", f"{max(Dvals):.6f}")
    dD = np.diff(Dvals)/np.diff(ts)
    max_pos = float(np.max(dD)) if len(dD)>0 else 0.0
    st.metric("max d/dt D(ρ_t||ρ_β)", f"{max_pos:.3e}")
    D_ok = (max_pos <= 1e-6)
    st.markdown(f"{badge_html(D_ok)} Ziel: max d/dt D ≤ 0 (Spohn)", unsafe_allow_html=True)

    # ---- E) MCWF vs GKSL ----
    st.markdown("### E) Trajektorien (MCWF) vs. GKSL")
    with st.expander("Parameter & Start"):
        ntraj = st.slider("Anzahl Trajektorien", 50, 5000, 400, 50)
        dt_tr = st.slider("dt (Trajektorien)", 1e-3, 5e-2, 1e-2, 1e-3, format="%.3f")
        tmax  = st.slider("t_max (Trajektorien)", 0.1, max(1.0, float(P.n_steps*P.dt_coll)), float(min(2.0, P.n_steps*P.dt_coll)), 0.1)
        seed  = st.number_input("Seed", value=7, step=1)
    np.random.seed(int(seed))
    H = 0.5*P.omega*sz
    Ls = [(math.sqrt(max(rates.gamma_down,0.0)), sm),
          (math.sqrt(max(rates.gamma_up,0.0)),   sp)]

    def step_mcwf(psi: np.ndarray, dt: float) -> np.ndarray:
        Heff = H - 0.5j*sum((c**2)*(dagger(L)@L) for c,L in Ls)
        psi_t = (np.eye(2)+(-1j*Heff*dt)) @ psi
        norm2 = float(np.vdot(psi_t, psi_t).real)
        ps = [ (c**2)*float(np.vdot(psi, (dagger(L)@L)@psi).real)*dt for c,L in Ls ]
        p_total = sum(ps)
        if np.random.rand() > p_total:
            psi_t = psi_t / math.sqrt(max(norm2,1e-30))
            return psi_t
        rrand = np.random.rand()*p_total
        acc = 0.0
        for j,(c,L) in enumerate(Ls):
            acc += ps[j]
            if rrand <= acc:
                psi_t = (L @ psi) / math.sqrt(max(float(np.vdot(psi, dagger(L)@L@psi).real),1e-30))
                break
        return psi_t

    w,V = np.linalg.eigh(rho0); idx = int(np.argmax(w.real))
    psi0 = V[:,idx]
    nsteps = int(round(tmax/dt_tr))

    # Referenz (GKSL) mit gewähltem Integrator
    rho_ref = rho0.copy(); ref_path = [rho_ref.copy()]
    for _ in range(nsteps):
        rho_ref = do_step_v(rho_ref, dt_tr)
        ref_path.append(rho_ref.copy())

    # MCWF-Mittel
    acc_rho = np.zeros((2,2), dtype=complex)
    for _ in range(ntraj):
        psi = psi0.copy()
        for __ in range(nsteps):
            psi = step_mcwf(psi, dt_tr)
        acc_rho += psi[:,None]@psi[None,:].conj()
    rho_avg = acc_rho / ntraj
    TD_mc = trace_distance(rho_avg, ref_path[-1])
    st.metric("MCWF vs GKSL (Endzeit) — Trace distance", f"{TD_mc:.3e}")
    E_ok = (TD_mc <= 1e-2)
    st.markdown(f"{badge_html(E_ok)} Ziel: Trace distance → 0 (ntraj↑, dt↓)", unsafe_allow_html=True)
    st.caption("Konvergiert gegen 0 für ntraj↑ und dt↓.")

    # -- Kennzahlen für Export speichern
    st.session_state["verif"] = {
        "tp_defect": float(tp_err),
        "choi_min_eig": float(min_eig),
        "gadc_d": float(d_best),
        "gadc_eta": float(eta_best),
        "gadc_q": float(q_best),
        "semigroup_defect": float(defect),
        "relent_min": float(min(Dvals)),
        "relent_max": float(max(Dvals)),
        "relent_max_dD": float(max_pos),
        "mcwf_td": float(TD_mc),
        "A_ok": bool(A_ok), "B_ok": bool(B_ok), "C_ok": bool(C_ok),
        "D_ok": bool(D_ok), "E_ok": bool(E_ok)
    }

    # ---- Supplementary (Zielwerte + Primärquellen) ----
    with st.expander("📚 Supplementary: Methodik, Zielwerte & Primärquellen (A–E)", expanded=False):
        # Methodik / kurze Formeln
        st.markdown("**A) CPTP & Choi (Kraus)** — Ziel: TP-Defekt = 0; minEig(Choi) ≥ 0.")
        st.latex(r"\Phi(\rho)=\sum_i K_i\rho K_i^\dagger,\quad \sum_i K_i^\dagger K_i=\mathbb{1},\quad C_\Phi=\sum_i \mathrm{vec}(K_i)\mathrm{vec}(K_i)^\dagger\succeq 0.")
        st.markdown("**B) Generalized Amplitude Damping (GADC)** — Ziel: "
                    r"$\eta\approx\sin^2\theta,\ q\approx p_{\rm exc}(\beta,\omega)$; "
                    "Basis-Ausrichtung + adaptive Vor/Nach-Unitäre.")
        st.markdown("**C) GKSL-Semigruppe (affine PTM)** — Ziel: "
                    r"$\Phi_{t+s}=\Phi_t\circ\Phi_s \Rightarrow \|T(t+s)-T(t)T(s)\|_F=0$.")
        st.markdown("**D) Gibbs-Fixpunkt & Spohn** — Ziel: "
                    r"$\max_t \frac{d}{dt}D(\rho_t\Vert\rho_\beta)\le 0$ (numerisch bis auf Toleranz).")
        st.markdown("**E) MCWF ≡ GKSL im Mittel** — Ziel: "
                    "Trace-Distance(⟨Traj⟩, GKSL) → 0 für $n_{\\rm traj}\\uparrow$, $dt\\downarrow$.")

        # Dynamische Zielwerte dieses Runs
        eta_tgt = math.sin(theta)**2
        q_tgt   = float(tauB[0,0].real)       # p_exc(β,ω)
        kms_tgt = math.exp(-P.beta*P.omega)

        # Toleranzen / OK-Schwellen (wie in den Tests verwendet)
        thr_A_tp, thr_A_mineig  = 1e-12, -1e-12
        thr_B_eta, thr_B_q, thr_B_dchoi = 1e-3, 1e-3, 5e-3
        thr_C_defect = 1e-2
        thr_D_pos    = 1e-6
        thr_E_TD     = 1e-2

        st.markdown("### 🎯 Zielwerte dieses Runs")
        colZ1, colZ2, colZ3 = st.columns(3)
        with colZ1:
            st.metric("θ = g·τ_int", f"{theta:.6f}")
            st.metric("η_target = sin²θ", f"{eta_tgt:.6f}")
        with colZ2:
            st.metric("q_target = p_exc(β,ω)", f"{q_tgt:.6f}")
            st.metric("KMS-Ziel e^{−βω}", f"{kms_tgt:.6f}")
        with colZ3:
            st.metric("γ↑/γ↓ − e^{−βω} (Ziel)", "0")

        st.markdown("### ✅ OK-Schwellen (Verification)")
        st.markdown(
            f"""
- **A)** TP-Defekt ≤ {thr_A_tp:.0e}; minEig(Choi) ≥ {thr_A_mineig:.0e}  
- **B)** |η*−η_target| ≤ {thr_B_eta:.0e}; |q*−q_target| ≤ {thr_B_q:.0e}; min‖ΔChoi‖_F ≤ {thr_B_dchoi:.0e}  
- **C)** Semigroup-Defekt ≤ {thr_C_defect:.0e}  
- **D)** max d/dt D(ρ_t||ρ_β) ≤ {thr_D_pos:.0e}  
- **E)** Trace-Distance(⟨Traj⟩, GKSL) ≤ {thr_E_TD:.0e}
"""
        )

        st.markdown("### 🔗 Primärquellen & Referenzen (mit URLs)")
        st.markdown("- **Choi (1975)**: *Completely positive linear maps on complex matrices*, **LAA**. DOI: 10.1016/0024-3795(75)90075-0.  \n"
                    "  https://www.sciencedirect.com/science/article/pii/0024379575900750")
        st.markdown("- **Lindblad (1976)**: *On the generators of quantum dynamical semigroups*, **Commun. Math. Phys.** 48, 119–130.  \n"
                    "  https://link.springer.com/article/10.1007/BF01608499")
        st.markdown("- **Gorini–Kossakowski–Sudarshan (1976)**: *Completely positive dynamical semigroups of N-level systems*, **J. Math. Phys.** 17, 821.  \n"
                    "  https://pubs.aip.org/aip/jmp/article/17/5/821/225427/Completely-positive-dynamical-semigroups-of-N")
        st.markdown("- **Davies (1974)**: *Markovian master equations*, **Commun. Math. Phys.** 39, 91–110.  \n"
                    "  https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-39/issue-2/Markovian-master-equations/cmp/1103860160.full")
        st.markdown("- **Spohn (1978)**: *Entropy production for quantum dynamical semigroups*, **J. Math. Phys.** 19, 1227–1230.  \n"
                    "  https://pubs.aip.org/aip/jmp/article/19/5/1227/460049/Entropy-production-for-quantum-dynamical")
        st.markdown("- **Dalibard–Castin–Mølmer (1992)**: *Wave-function approach to dissipative processes in quantum optics*, **Phys. Rev. Lett.** 68, 580.  \n"
                    "  https://link.aps.org/doi/10.1103/PhysRevLett.68.580")
        st.markdown("- **Plenio–Knight (1998)**: *The quantum-jump approach to dissipative dynamics in quantum optics*, **Rev. Mod. Phys.** 70, 101.  \n"
                    "  https://link.aps.org/doi/10.1103/RevModPhys.70.101")
        st.markdown("- **GADC (Kraus-Form & Parametrisierung)**: Pennylane-Doku »GeneralizedAmplitudeDamping«.  \n"
                    "  https://docs.pennylane.ai/en/stable/code/api/pennylane.GeneralizedAmplitudeDamping.html")

# ================================== Tab 4 ===================================
with TAB4:
    st.subheader("Export als ZIP")
    with st.expander("📚 Supplement (Tab 4): Export & Reproduzierbarkeit", expanded=False):
        st.markdown("**ZIP** enthält wählbare Plots + Markdown-Dateien (settings/results/verification).")

    data = st.session_state.get("traceout_data")
    if data is None:
        st.info("Bitte zuerst in Tab 1 (und optional Tab 2/3) rechnen, dann exportieren.")
    else:
        st.markdown("**Was soll exportiert werden?**")
        colA, colB = st.columns(2)
        with colA:
            exp_settings = st.checkbox("Einstellungen (Parameter) als .md", value=True)
            exp_values   = st.checkbox("Werte/Ergebnisse (Raten, KMS, T1/T2, Verifikation) als .md", value=True)
        with colB:
            st.write("**Grafiken aus Tab 1:**")
            g_t1_pop = st.checkbox("Tab 1 – Populations", value=True)
            g_t1_coh = st.checkbox("Tab 1 – |ρ01|", value=True)
            st.write("**Grafiken aus Tab 2:**")
            g_t2_p00  = st.checkbox("Tab 2 – ρ00 (trace-out vs GKSL)", value=True)
            g_t2_coh  = st.checkbox("Tab 2 – |ρ01| (trace-out vs GKSL)", value=True)
            g_t2_phase= st.checkbox("Tab 2 – arg  ρ01", value=True)
            g_t2_td   = st.checkbox("Tab 2 – Trace distance", value=True)
            g_t2_fit1 = st.checkbox("Tab 2 – Fit T1", value=True)
            g_t2_fit2 = st.checkbox("Tab 2 – Fit T2", value=True)
            g_t2_conv = st.checkbox("Tab 2 – Δt-Konvergenz", value=True)
            st.write("**Grafiken aus Tab 3 (Verification Suite):**")
            g_t3_D = st.checkbox("Tab 3 – D) Relative entropy", value=True)

        import plotly.io as pio
        def figure_bytes(fig, name: str) -> tuple[str, bytes, str]:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{name}_{stamp}"
            try:
                b = fig.to_image(format="png", scale=2)
                return f"{fname}.png", b, "image/png"
            except Exception:
                b = pio.to_html(fig).encode("utf-8")
                return f"{fname}.html", b, "text/html"

        if st.button("ZIP erstellen", type="primary"):
            to_zip = io.BytesIO()
            zf = zipfile.ZipFile(to_zip, mode="w", compression=zipfile.ZIP_DEFLATED)

            # 1) Markdown: Einstellungen
            if exp_settings:
                P: MicroParams = data["params"]
                p0_star, p1_star = data["gibbs"]
                rates: GKSLRates = data["rates"]
                kms_gap = data["kms_gap"]
                md = []
                md.append("# Einstellungen\n")
                md.append(f"- ω = {P.omega}\n- β = {P.beta}\n- g = {P.g}\n- τ_int = {P.tau_int}\n"
                          f"- Δt = {P.dt_coll}\n- steps = {P.n_steps}\n- θ0 = {P.theta0}\n- φ0 = {P.phi0}\n")
                md.append(f"\n**Gibbs-Belegungen:** p0* ≈ {p0_star:.6f}, p1* ≈ {p1_star:.6f}\n")
                md.append(f"**Raten aus Mikro:** γ↓ ≈ {rates.gamma_down:.6f}, γ↑ ≈ {rates.gamma_up:.6f}\n")
                md.append(f"**KMS-Check:** γ↑/γ↓ − e^(-βω) ≈ {kms_gap:.3e}\n")
                zf.writestr("settings.md", "".join(md))

            # 2) Markdown: Werte/Ergebnisse inkl. Verifikation
            if exp_values:
                t2 = st.session_state.get("tab2_data", {})
                v = st.session_state.get("verif", {})
                mdv = []
                mdv.append("# Ergebnisse\n")
                mdv.append("## Trace-out (S⊗Bad → partielle Spur)\n")
                mdv.append(f"- letzte Tr(ρ) ≈ {np.trace(data['rhos'][-1]).real:.6f}\n")
                mdv.append(f"- PSD(last)   ≈ {is_psd(data['rhos'][-1])}\n")
                mdv.append(f"- Raten: γ↓≈{data['rates'].gamma_down:.6f}, γ↑≈{data['rates'].gamma_up:.6f}\n")
                mdv.append(f"- KMS-Delta: γ↑/γ↓ − e^(-βω) ≈ {data['kms_gap']:.3e}\n")
                if t2:
                    mdv.append("\n## GKSL-Fits\n")
                    mdv.append(f"- T1_fit≈{t2.get('T1_fit', float('nan')):.6f}, T1_theory≈{t2.get('T1_theory', float('nan')):.6f}\n")
                    mdv.append(f"- T2_fit≈{t2.get('T2_fit', float('nan')):.6f}, T2_theory≈{t2.get('T2_theory', float('nan')):.6f}\n")
                    if t2.get('T1_fit', 0) and t2.get('T1_fit', 0) > 0:
                        mdv.append(f"- Verhältnis T2_fit/(2·T1_fit)≈{t2.get('T2_fit', float('nan'))/(2*t2.get('T1_fit', 1.0)):.6f}\n")
                if v:
                    mdv.append("\n## Verification (Tab 3)\n")
                    def yesno(b): return "OK" if b else "FAIL"
                    mdv.append(f"- A (CPTP/Choi): TP-defect≈{v.get('tp_defect', float('nan')):.3e}, "
                               f"min-eig≈{v.get('choi_min_eig', float('nan')):.3e} [{yesno(v.get('A_ok', False))}]\n")
                    mdv.append(f"- B (GADC): ΔChoi≈{v.get('gadc_d', float('nan')):.3e}, η*≈{v.get('gadc_eta', float('nan')):.6f}, "
                               f"q*≈{v.get('gadc_q', float('nan')):.6f} [{yesno(v.get('B_ok', False))}]\n")
                    mdv.append(f"- C (Semigroup): ‖T(t+s)-T(t)T(s)‖_F≈{v.get('semigroup_defect', float('nan')):.3e} "
                               f"[{yesno(v.get('C_ok', False))}]\n")
                    mdv.append(f"- D (RelEnt): min≈{v.get('relent_min', float('nan')):.6f}, max≈{v.get('relent_max', float('nan')):.6f}, "
                               f"max dD≈{v.get('relent_max_dD', float('nan')):.3e} [{yesno(v.get('D_ok', False))}]\n")
                    mdv.append(f"- E (MCWF vs GKSL): TD≈{v.get('mcwf_td', float('nan')):.3e} [{yesno(v.get('E_ok', False))}]\n")
                zf.writestr("results.md", "".join(mdv))

            # 3a) Figuren
            figs = st.session_state["figs"]
            def add_fig(key, name):
                if key in figs:
                    fn, b, mt = figure_bytes(figs[key], name)
                    zf.writestr(f"figures/{fn}", b)

            if st.checkbox("Alle verfügbaren Grafiken exportieren", value=False):
                for key in list(figs.keys()):
                    add_fig(key, key)
            else:
                if st.checkbox("Tab 1 – Populations exportieren", value=True) and "tab1_populations" in figs:
                    add_fig("tab1_populations", "tab1_populations")
                if st.checkbox("Tab 1 – |ρ01| exportieren", value=True) and "tab1_coherence" in figs:
                    add_fig("tab1_coherence", "tab1_coherence")
                if st.checkbox("Tab 2 – ρ00 exportieren", value=True) and "tab2_pop00" in figs:
                    add_fig("tab2_pop00", "tab2_pop00")
                if st.checkbox("Tab 2 – |ρ01| exportieren", value=True) and "tab2_coh_abs" in figs:
                    add_fig("tab2_coh_abs", "tab2_coh_abs")
                if st.checkbox("Tab 2 – phase exportieren", value=True) and "tab2_phase" in figs:
                    add_fig("tab2_phase", "tab2_phase")
                if st.checkbox("Tab 2 – Trace distance exportieren", value=True) and "tab2_trace_distance" in figs:
                    add_fig("tab2_trace_distance", "tab2_trace_distance")
                if st.checkbox("Tab 2 – Fit T1 exportieren", value=True) and "tab2_fit_T1" in figs:
                    add_fig("tab2_fit_T1", "tab2_fit_T1")
                if st.checkbox("Tab 2 – Fit T2 exportieren", value=True) and "tab2_fit_T2" in figs:
                    add_fig("tab2_fit_T2", "tab2_fit_T2")
                if st.checkbox("Tab 2 – Δt-Konvergenz exportieren", value=True) and "tab2_dt_convergence" in figs:
                    add_fig("tab2_dt_convergence", "tab2_dt_convergence")
                if st.checkbox("Tab 3 – RelEnt exportieren", value=True) and "tab3_D_relent" in figs:
                    add_fig("tab3_D_relent", "tab3_D_relent")

            # 4) Verification Markdown (kompakt)
            try:
                v = st.session_state.get("verif", {})
                md = "# Verification Suite (Tab 3)\n"
                if v:
                    def yesno(b): return "OK" if b else "FAIL"
                    md += f"- A: TP-defect≈{v.get('tp_defect', float('nan')):.3e}, min-eig≈{v.get('choi_min_eig', float('nan')):.3e} [{yesno(v.get('A_ok', False))}]\n"
                    md += f"- B: ΔChoi≈{v.get('gadc_d', float('nan')):.3e}, η*≈{v.get('gadc_eta', float('nan')):.6f}, q*≈{v.get('gadc_q', float('nan')):.6f} [{yesno(v.get('B_ok', False))}]\n"
                    md += f"- C: ‖T(t+s)-T(t)T(s)‖_F≈{v.get('semigroup_defect', float('nan')):.3e} [{yesno(v.get('C_ok', False))}]\n"
                    md += f"- D: min≈{v.get('relent_min', float('nan')):.6f}, max≈{v.get('relent_max', float('nan')):.6f}, max dD≈{v.get('relent_max_dD', float('nan')):.3e} [{yesno(v.get('D_ok', False))}]\n"
                    md += f"- E: TD≈{v.get('mcwf_td', float('nan')):.3e} [{yesno(v.get('E_ok', False))}]\n"
                zf.writestr("verification.md", md)
            except Exception:
                pass

            readme = [
                "# Export\n",
                "Dieses Archiv enthält die gewählten Grafiken und Markdown-Dateien (Einstellungen/Ergebnisse/Verification).\n",
                "Verifikation umfasst: CPTP/Choi, GADC-Fit, Semigroup-Test, Gibbs/Spohn, MCWF≈GKSL.\n",
                "Hinweis: Falls keine PNGs enthalten sind, wurden die Plots als HTML exportiert (kaleido nicht verfügbar).\n"
            ]
            zf.writestr("README.md", "".join(readme))
            zf.close()
            to_zip.seek(0)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="ZIP herunterladen",
                data=to_zip.getvalue(),
                file_name=f"export_{stamp}.zip",
                mime="application/zip",
                type="primary"
            )
