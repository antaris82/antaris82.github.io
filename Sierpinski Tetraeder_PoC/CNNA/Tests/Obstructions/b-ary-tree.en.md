# CNNA-ToC / (J)-sign / Non-commutativity — complete inventory of tests and obstructions

Status: based on chat history, not as a lean theorem. Most findings are numerical, conceptual, or derived from Python diagnostics. The central result is now more precise than at the beginning:

$$
\boxed{
\text{Der flache, reellwertige, reziproke ToC-/Schur-/DtN-Sektor erzeugt keine ausgezeichnete }J\text{-Orientierung.}
}
$$

It delivers multiple times:

$$
{+J,-J},\qquad {+\tau,-\tau},\qquad \text{radiale Ordnung},\qquad \text{DtN-/Spektralstruktur}.
$$

It has not yet yielded:

$$
\boxed{
J\text{ statt }-J.
}
$$

The uniform obstruction is now no longer just “symmetry,” but more precisely:

$$
\boxed{
\text{Eine abgeleitete Achse }F1\text{ genügt nicht. Nichtkommutativität/Chiralität braucht eine zweite abgeleitete Achse.}
}
$$

---

# 0. Global status of the test series

## 0.1 What A/ToC has delivered positively so far

The ToC/DtN sector delivers robust precursors:

$$
\text{radiale Provenienzordnung},
$$

$$
\text{UV/Env-Coorientierung},
$$

$$
\text{Cauchy-Dopplung},
$$

$$
{+J,-J},
$$

$$
{+\tau,-\tau},
$$

$$
\text{reelle DtN-/Schur-Handoff-Matrizen}.
$$

This is not trivial. It means:

$$
\boxed{
\text{Der ToC ist als lokale Provenienzfaser und flacher Referenzsektor wertvoll.}
}
$$

## 0.2 What A/ToC has not yet delivered

What has not been delivered is an absolute orientation:

$$
\boxed{
J \neq \text{derived uniquely from flat ToC data}.
}
$$

Also not yet delivered are:

$$
\text{Chirotopie},
$$

$$
\text{nichttriviale Holonomie},
$$

$$
\text{Krümmung},
$$

$$
\text{nichtkommutative Handoff-Algebra},
$$

$$
\text{echte partielle Spur/OQS-Struktur}.
$$

## 0.3 New interpretive status

The tests do not concern “CNNA in general,” but rather the specific sector:

$$
\boxed{
\text{flacher, homogener, reell-reziproker ToC-/DtN-Sektor.}
}
$$

It follows that:

$$
\boxed{
\text{Der globale Einzel-ToC als ontischer Weltbaum ist für }J\text{ falsifiziert.}
}
$$

But not:

$$
\boxed{
\text{lokale ToC-Fasern, DtN-Geometrie oder CNNA als Gesamtprogramm sind falsifiziert.}
}
$$

---

# 1. Didactic and Proxy Tests

## 1.1 Gradio ToC Concept Explorer

**Script / File**

```text
app.py
```

**Initial Situation**

Visualization of a (b)inary ToC with parameters:

$$
b,\qquad L_{\max},\qquad \text{Approximant root},\qquad L.
$$

Levels shown:

$$
\text{ToC}
\to
\text{proper subsystem}
\to
\text{UV-tail}
\to
\text{Environment}
\to
\text{Cauchy-/}J\text{-Kandidat}
\to
\text{Complex-plane overlay}.
$$

**Findings**

Highly effective for teaching. It visually distinguishes:

$$
\text{Approximant},
\qquad
\text{UV-tail},
\qquad
\text{Environment},
\qquad
\text{Interface}.
$$

**Obstruction Location**

Visualization is not proof. Early tilt/angle values were partly chart/rendering proxies, not DtN invariants.

**Status**

Didactically valuable, mathematically secondary.

---

## 1.2 Stage-6 Chart Proxy / Tilt Test

**Script / File**

Part of the interactive `app.py`.

**Initial Situation**

Deep embedding of approximants, e.g.

$$
0.1,\qquad 0.1.1,\qquad 0.1.1.0,\ldots
$$

with fixed parameters such as:

$$
b=3,\qquad L_{\max}=4.
$$

**Findings**

Visual tilt decreased with deeper embedding:

$$
|\mathrm{tilt}|\downarrow.
$$

**Interpretation**

Deeper-embedded approximants appeared more balanced between UV and Env.

**Obstruction location**

No true Schur/DtN value:

$$
\text{Proxy} \neq \text{Invariante}.
$$

**Status**

Heuristic motivation; later replaced by true DtN/Schur tests.

---

# 2. Single-Approximant Schur/DtN Tests

## 2.1 Projected-tail (J)/Rotation Test

**Script / File**

Implemented in Chat; functionality was later incorporated into (\alpha_{\mathrm{orth}})- and DtN diagnostics.

**Initial situation**

Finite approximant with effective operator:

$$
M=L_\Omega+\text{projected UV/Env loads}.
$$

Two channel responses:

$$
u_{\mathrm{Env}},\qquad u_{\mathrm{UV}}.
$$

Measured variable:

$$
\rho_M
§§X118§§

\frac{\langle u_{\mathrm{Env}},u_{\mathrm{UV}}\rangle_M}
{|u_{\mathrm{Env}}|*M,|u*{\mathrm{UV}}|_M}.
$$

**Findings**

Near orthogonality:

$$
|\rho_M|\ll 1,
$$

partially numerically close (90^\circ).

**Obstruction location**

Orthogonality of a real 2-plane yields at most:

$$
{+J,-J}.
$$

The plane exists; the direction of rotation does not.

**Status**

Positive precursor of a pre-complex plane. No sign proof.

---

## 2.2 Real finite-network Schur/DtN test

**Script / File**

Implemented in the chat as a finite unit-edge graph test; no uniquely isolated final filename.

**Initial situation**

Finite tree graph with Laplace matrix:

$$
L_{\mathrm{graph}}.
$$

Edge (B), internal node (I), Schur complement:

$$
\Lambda_B
§§X120§§

L_{BB}-L_{BI}L_{II}^{-1}L_{IB}.
$$

**Result**

Numerically practically orthogonal for deterministic centered single modes, for example:

$$
|\rho_M|\approx 10^{-18}.
$$

**Obstruction location**

A single mode can be orthogonal while the full boundary response space still retains structure. Furthermore, the DtN operator remains real-symmetric.

**Status**

Strong indication of true Schur/DtN orthogonality in certain modes; no (J) sign.

---

## 2.3 Dirichlet/Cut Regularization Test

**Script / File**

Part of the Schur/DtN tests.

**Initial situation**

Question:

$$
\text{Braucht man eine externe Regularisierung oder Pseudoinverse?}
$$

**Result**

No, provided that a genuine Dirichlet/Boundary-Cut is set. Then the inner block:

$$
L_{II}
$$

is invertible.

**Obstruction point**

The DtN operator is cut-relative:

$$
\Lambda_{\partial A}.
$$

There is no cut-free universal DtN operator for the entire infinite ToC.

**Status**

Important positive result: no Ridge/pseudoinverse setting required.

---

# 3. $\alpha_{\mathrm{orth}}$- and invariant tests

## 3.1 $\Xi$- / $\alpha_{\mathrm{orth}}$-diagnostics

**Script / File**

```text
alpha_orth_invariant.py
```

Packages:

```text
cnna_alpha_orth_invariant_v2.zip
cnna_alpha_orth_invariant_v3.zip
cnna_alpha_orth_invariant_v4.zip
cnna_alpha_orth_invariant_v5.zip
cnna_alpha_orth_invariant_v6.zip
cnna_alpha_orth_invariant_v7.zip
```

**Initial Situation**

Control variable:

$$
\Xi=(1+\lambda_{\mathrm{UV}})(1+\lambda_{\mathrm{Env}}),
$$

with

$$
\lambda_{\mathrm{UV}}
§§X123§§

\frac{b^k\alpha_{\mathrm{UV}}}{C_k},
\qquad
\lambda_{\mathrm{Env}}
§§X124§§

\frac{\alpha_{\mathrm{Env}}}{C_k}.
$$

Typical orthogonality diagnosis:

$$
|\rho|\sim \Xi^{-1/2}.
$$

**Findings**

The UV term dominates strongly for increasing depth:

$$
|\rho|\sim b^{-k/2}.
$$

Thus:

$$
\boxed{
\text{UV-Auflösung treibt Orthogonalität.}
}
$$

**Obstruction location**

(\alpha_{\mathrm{Env}}) was model-dependent in early versions:

```text
none
constant
power
exponential
ladder
```

Therefore, the exact numerical value was not a fully derived physical value.

**Status**

Good diagnostic parameter. No fine-structure constant claim. No (J) sign.

---

## 3.2 Environment-sensitive models

**Script / File**

```text
alpha_orth_invariant.py
```

**Initial Situation**

Comparison of various (\alpha_{\mathrm{Env}}) models.

**Findings**

For large (k), the UV term often dominates so strongly that the choice of environment model becomes subdominant.

**Location of obstruction**

In regimes where Environment is not subdominant, a true complement family/DtN derivation of (\alpha_{\mathrm{Env}}) is required.

**Status**

Good methodological findings:

$$
\text{definierbar}\neq\text{erzwungen}.
$$

---

# 4. Parent–Child and Handoff Tests

## 4.1 Two-Approximant / Flow-Sign Test

**Script / File**

```text
two_approximant_flow_sign.py
```

**Initial Situation**

Parent–Child Handoff:

$$
A_{\mathrm{parent}}\to A_{\mathrm{child}}.
$$

Objective: to check whether the transition yields a (J) sign.

**Findings**

Radial transition signatures may arise.

**Obstruction Location**

Radiality is not chirality:

$$
\text{Parent}\to\text{Child}
$$

Yields depth direction, but no direction of rotation.

Furthermore, flow can easily introduce a sign via the excitation direction.

**Status**

Radial handoff structure: yes. (J) sign: no.

---

## 4.2 Schur-before-Flow criterion

**Script / File**

Methodologically derived from parent–child tests.

**Initial Situation**

Possible handoff types:

1. Restriction
2. Aggregation
3. Schur handoff
4. Flow handoff

**Findings**

Restriction/Aggregation/Schur are more canonical than Flow.

**Location of obstruction**

Flow may contain a directed stimulus. In that case, the sign would not be derived but set.

**Status**

Methodological rule:

$$
\boxed{
\text{Schur zuerst, Flow nur als Konsistenztest.}
}
$$

---

# 5. Two-boundary/shell chirality tests

## 5.1 V4 — Two-boundary shell chirality

**Script / File**

```text
two_boundary_shell_chirality.py
```

Package:

```text
cnna_alpha_orth_invariant_v4.zip
```

**Initial conditions**

Parent–child difference shell, two boundary ports, real DtN matrix:

$$
\Lambda_\Delta.
$$

Cauchy pairing:

$$
\omega((q,p),(q',p'))=q^Tp'-p^Tq'.
$$

The following holds for a DtN graph:

$$
p=\Lambda q.
$$

**Result**

For self-adjoint DtN graphs:

$$
\omega((q,\Lambda q),(r,\Lambda r)) = q^T\Lambda r-r^T\Lambda q = 0.
$$

**Obstruction Location**

A single passive symmetric DtN graph is Lagrangian.

**Status**

Clean negative result. Too restrictive for family/handoff tests, but correct for a single graph.

---

## 5.2 V5 — Family handoff chirality

**Script / File**

```text
family_handoff_chirality.py
```

Package:

```text
cnna_alpha_orth_invariant_v5.zip
```

**Initial situation**

Family of DtN matrices:

$$
{\Lambda_i}.
$$

Cross-graph Cauchy pairing:

$$
\omega_{ij}(q,r)=q^T\Lambda_jr-r^T\Lambda_iq.
$$

Additionally, Handoff Square:

$$
A\to B_i\to C,
\qquad
A\to B_j\to C.
$$

**Findings**

Cross-graph signals may occur:

$$
\omega_{ij}\neq 0.
$$

But:

```text
sibling_flip_detected = false
handoff_holonomy_detected = false
```

**Obstruction Location**

Signal is family/metric difference, not chirality. No sibling sign reversal, no true handoff holonomy.

**Status**

Important test: “Not just a graph” was checked. Result remains achiral.

---

# 6. Triadic tests

## 6.1 V6 — Triadic interface chirality

**Script / File**

```text
triadic_interface_chirality.py
```

Package:

```text
cnna_alpha_orth_invariant_v6.zip
```

**Initial situation**

Triad:

$$
\text{UV-channel},
\qquad
\text{Environment-channel},
\qquad
\text{Handoff/Regulator-channel}.
$$

Regulator candidate:

$$
r_i=(\Lambda_{\mathrm{child},i}-\Lambda_{\mathrm{parent}})a.
$$

Triadic surface:

$$
\tau_i
§§X131§§

\det(e_{\mathrm{UV}}-e_{\mathrm{Env}},,r_i-e_{\mathrm{Env}}).
$$

**Findings**

For canonical modes:

```text
tau_signs = 1,1,1
nonzero_tau_count = 3
sibling_flip_detected = false
```

**Obstruction site**

The triad is radially or sibling-invariant.

**Status**

Triadic signal yes. Chiral sibling asymmetry no.

---

## 6.2 Non-canonical positive controls

**Script / File**

```text
triadic_interface_chirality.py
family_handoff_chirality.py
```

**Initial Situation**

Control modes:

```text
sibling_index
cyclic_order
```

**Findings**

They predictably generate sign/flip effects.

**Obstruction Location**

They break symmetry via label or external order.

**Status**

Detector check only. No CNNA-derived proof.

---

# 7. V7 — Oriented UV/Environment Cauchy shell

## 7.1 Oppositely oriented UV/Env boundary sides

**Script / File**

```text
oriented_cauchy_shell_gate.py
```

Package:

```text
cnna_alpha_orth_invariant_v7.zip
```

**Initial condition**

UV-tail and Environment-tail are interpreted as oppositely oriented boundary faces of a shell.

Cauchy data space:

$$
(q_{\mathrm{Env}},q_{\mathrm{UV}},p_{\mathrm{Env}},p_{\mathrm{UV}}).
$$

Oriented boundary form:

$$
\omega_\partial=\omega_{\mathrm{Env}}-\omega_{\mathrm{UV}}.
$$

Metric:

$$
g=\operatorname{diag}(k_{\mathrm{Env}},k_{\mathrm{UV}},1/k_{\mathrm{Env}},1/k_{\mathrm{UV}}).
$$

Construction:

$$
J=-g^{-1}\omega_\partial.
$$

**Findings**

Tested:

```text
J_square_error = 0.0
metric_compat_error = 0.0
omega_compat_error = 0.0
swap_to_minus_J_error = 0.0
```

Thus:

$$
J^2=-I,
\qquad
J^TgJ=g,
\qquad
J^T\omega J=\omega.
$$

**Obstruction-Location**

The co-orientation is chosen. With the opposite choice, the following also arises consistently:

$$
J\mapsto -J.
$$

**Status**

Very important positive result:

$$
\text{UV/Env-Coorientierung}\Rightarrow {J,-J}\text{-Cauchy-Struktur}.
$$

No absolute sign.

---

# 8. Root, Co-root, and Depth-Reading Tests

## 8.1 Root as the outer model boundary

**Initial Situation**

The ToC does not grow ontically; it is infinitely given.

$$
\ell(\mathrm{root})=0,
\qquad
\ell\to\infty
$$

Inward.

**Findings**

Depth order provides relative oppositeness:

$$
\text{Env-Seite}: \ell\downarrow,
\qquad
\text{UV-Seite}: \ell\uparrow.
$$

**Site of obstruction**

Depth order is polar, not chiral:

$$
\text{innen/außen}\neq\text{Drehsinn}.
$$

**Status**

Semantically supports V7. Not absolute (J).

---

## 8.2 Negative-root / Co-root Hypothesis

**Initial Situation**

Hypothesis:

$$
\text{formale Root ist Interface;}
\qquad
\text{dahinter liegt negative Wurzelfamilie}.
$$

**Findings**

Could support Cauchy doubling and (\alpha_{\mathrm{Env}})-derivation.

**Obstruction point**

A negative root family does not automatically remain chiral under real passive symmetry.

**Status**

Possible candidate for environment derivation; no sign proof.

---

# 9. Sibling, $S_b$, and address symmetry tests

## 9.1 $S_b$-sibling obstruction

**Initial position**

In an unordered binary tree, siblings are

$$
S_b
$$

interchangeable.

**Result**

Canonical sizes lie in the trivial $S_b$ component.

**Obstruction Point**

The signum representation is not chosen canonically:

$$
S_b\text{-Äquivarianz}
\Rightarrow
\text{keine kanonische sibling-chirality}.
$$

**Status**

Robust negative line.

---

## 9.2 Hamming weight classes

**Initial situation**

Pages such as:

$$
000,001,010,011,100,101,110,111.
$$

Classes:

$$
|x|_1=1,
\qquad
|x|_1=2.
$$

**Findings**

Address-intrinsic relation transverse to the prefix structure.

**Obstruction location**

Hamming weight is magnitude, not orientation. Bit reversal remains possible.

**Status**

Structural finding, but achiral.

---

## 9.3 Cyclic bit shift

**Initial state**

Approximately:

$$
{001,010,100}
$$

there is a cyclic shift:

$$
001\to010\to100\to001.
$$

**Findings**

Address-intrinsic loop without geometric embedding.

**Obstruction location**

Bit reversal conjugates left shift into right shift:

$$
\mathrm{reverse}\circ\rho=\rho^{-1}\circ\mathrm{reverse}.
$$

Thus:

$$
\text{Schleife ja, Drehsinn nein.}
$$

**Status**

Important candidate for multi-ToC/frustration structures. No local (J) sign.

---

# 10. SG/ST, chirotopy, and sign-line tests

## 10.1 SG/ST as IFS/quotient structures

**Initial situation**

The Sierpinski Gasket (SG) and Sierpinski Tetrahedron/Tetrix (ST) were considered as ToC-related quotient/IFS structures.

**Findings**

They introduce loops and co-cycles:

$$
H^1\neq0.
$$

Exemplary dimensions:

$$
d_s(SG)=\frac{2\log 3}{\log 5},
\qquad
d_s(ST)=\frac{2\log 4}{\log 6}.
$$

**Obstruction site**

SG/ST are not the bare ToC. They are IFS/address quotients. Their additional relations are not automatically derived from the ToC.

**Status**

Useful as a comparison and structure test; no direct (J) breakthrough.

---

## 10.2 Chirotopy / Sign-Line (S_b/A_b)

**Initial situation**

Chirality on siblings lies in the sign information:

$$
S_b/A_b\simeq \mathbb Z_2.
$$

**Findings**

If the local isotropy group (H) does not lie in (A_b), there is no canonical non-vanishing chiral topology.

For the symmetric ToC:

$$
H=S_b.
$$

**Obstruction point**

$$
S_b\not\subset A_b.
$$

Therefore, a sign line is not canonically distinguished.

**Status**

A very central "no-go" statement.

---

## 10.3 $Z_b$-cyclicity is not enough

**Starting point**

Test whether the cyclic order $Z_b$ replaces the missing chiropathy.

**Result**

No. For (b=4), a 4-cycle can be odd as a label permutation; geometric orientation and permutation parity do not automatically coincide.

**Obstruction point**

Cyclic order is not yet a sign line.

**Status**

Important correction against hasty “cycle = orientation” conclusions.

---

# 11. Hodge, Dirac, and Dual Complex Tests

## 11.1 Cellular Dirac $K=d-d^*$

**Initial Situation**

Cellular operator:

$$
K=d-d^*
$$

on

$$
C^0\oplus C^1\oplus C^2.
$$

**Result**

(K) is real skew. On $\operatorname{im}K$, a formal polar structure can provide a J-like component.

**Obstruction point**

The operator mixes degrees. On a pure $C^1$ space, the relevant block is not automatically a local (J).

**Status**

Formal (J)-like structure possible, but not derived as a local handoff (J).

---

## 11.2 Hodge Star / Dual Complex

**Initial Situation**

Test whether dual cells or Hodge (\star) provide the orientation.

**Result**

A genuine Hodge $\star$ requires an orientation or a metric/volume structure.

With full $S_b$-symmetry, there is no canonical skew-equivariant operator.

With chirotopy, the symmetry is reduced and a (J)-block may appear.

**Obstruction Point**

Direction is:

$$
\text{Chirotopie}\Rightarrow J\text{-Modus},
$$

not:

$$
J\text{-Modus}\Rightarrow\text{Chirotopie}.
$$

**Status**

Confirms that orientation does not arise from Hodge alone.

---

# 12. Recursive SG/ST and Schur/DtN Tests

## 12.1 Recursive SG/ST-DtN Matrices

**Initial Situation**

Boundary-DtN matrices for recursive SG/ST approximations.

**Result**

Boundary-DtN remains fully symmetric:

$$
\Lambda_n=a_n(bI-\mathbf 1\mathbf 1^T).
$$

Typically:

$$
a_n(SG)=\left(\frac35\right)^n,
\qquad
a_n(ST)=\left(\frac23\right)^n.
$$

**Obstruction point**

Full (S_b)-invariance is preserved. No reduction:

$$
S_b\to A_b.
$$

**Status**

SG/ST-Schur/DtN provides scale structure, no chirotopy.

---

## 12.2 IFS Generation Process Test

**Initial Situation**

Test whether the IFS growth process itself generates an order.

**Findings**

Unordered contractions:

$$
{\phi_i}
$$

remain (S_b)-equivariant.

**Obstruction Point**

An ordered/chiral IFS family could carry chirotopy, but only if the order itself is derived.

**Status**

IFS growth alone does not solve the sign problem.

---

# 13. Multicell Holonomy

## 13.1 Permutation Holonomy between Local ToC Fibers

**Initial situation**

Gluing edges with:

$$
\varphi_{\alpha\beta}\in S_b.
$$

Loop holonomy:

$$
h_\gamma
§§X147§§

\varphi_{\alpha_{k-1}\alpha_k}\cdots\varphi_{\alpha_0\alpha_1}.
$$

**Result**

If the centralizer

$$
C_{S_b}(h_\gamma)
$$

lies in (A_b), local odd permutations can be ruled out.

Example:

$$
b=3,\quad h=(012),
\qquad
C_{S_3}(h)=A_3.
$$

**Obstruction point**

The direction

$$
h \text{ vs. } h^{-1}
$$

leaves exactly the chiral choice. Unoriented class:

$$
{h,h^{-1}}
$$

localizes only one pair.

**Status**

Strong multi-ToC candidate, but without derived directed holonomy, no (J) sign.

---

# 14. F1 Holonomy and F1-only No-Go

## 14.1 F1-only Port Rules

**Initial Situation**

F1 is the radial provenance/filling arrow. Test: Can an F1-only rule permute ports in a non-trivial way?

**Result**

An F1-only port rule that is relabeling-natural must commute with all

$$
\sigma\in S_b
$$

. Therefore, it lies at the center:

$$
Z(S_b)={e}
\qquad (b\ge3).
$$

**Obstruction Point**

F1 alone has no transverse port order.

**Status**

Strong no-go: Nonlinearity in depth does not help as long as relabeling naturalness applies.

---

## 14.2 Screw rule as an import

**Initial situation**

Rule such as:

$$
(n,i)\mapsto(n+1,\sigma(i)),
\qquad
\sigma=(012).
$$

**Findings**

Seems to generate rotation.

**Obstruction location**

Under odd relabeling:

$$
\tau\sigma\tau^{-1}=\sigma^{-1}.
$$

The rule imports a port order.

**Status**

Control import, not a CNNA-derived mechanism.

---

# 15. Value-based F1-Coupling

## 15.1 Depth-dependent value coupling

**Script / File**

Documented in the process as `toc_paper_v10_f1_value_coupling`; Python test in chat.

**Initial situation**

Not port permutation, but value-based coupling:

$$
w_{\alpha\beta}=f(d_\alpha,d_\beta,\ldots).
$$

**Findings**

Skew components can arise when binding is depth-dependent and not symmetric.

**Obstruction location**

“Deep feeds strongly” and “shallow feeds strongly” are two sign choices:

$$
K^+=-K^-.
$$

The rule selects a sign if it is not derived.

**Status**

Shows how non-reciprocity could arise. But without derived selection, ({+K,-K}) remains.

---

# 16. Block RG and Shell Coupling

## 16.1 Collective Shell Coupling

**Initial Situation**

Not node-to-node, but relabeling-natural level-shell-to-level-shell:

$$
S_k(A)\leftrightarrow S_k(B),
$$

with mean mode:

$$
u_{A,k}
§§X152§§

\frac{1}{\sqrt{|S_k|}}\mathbf 1_{S_k(A)}.
$$

Coupling:

$$
C_{AB}
§§X153§§

\sum_k\gamma_k u_{A,k}u_{B,k}^T.
$$

**Findings**

Reciprocal shell coupling generates spectral structure and, if applicable, cycles.

**Obstruction Site**

The coupling remains symmetric:

$$
C_{AB}=C_{BA}^T.
$$

Therefore, A/B mirroring survives.

**Status**

Structure yes, chirality no.

---

## 16.2 Four-case test: Address-fixed vs. Role-fixed

**Initial situation**

Distinction:

$$
\text{Adressort}
\neq
\text{Skalenrolle}.
$$

Four cases:

| Case | Scale reading | Bonding location |
| ---- | ------------ | ------------------------------ |
| A    | Root coarse  | Root |
| B    | Root fine  | Root |
| C    | Root coarse  | Coarse end = Root |
| D    | Root fine  | Coarse end = Level-(L)-shell |

**Findings**

Case D is structurally new.

Reported finding:

$$
\beta_1: 0\to 6560,
$$

$$
d_s: 1.385\to 3.647.
$$

**Obstruction location**

Despite significant structural changes, A/B mirroring persists in all cases.

Reason:

$$
\text{Gate hängt an Reziprozität der transversalen Kopplung, nicht am Verklebungsort.}
$$

**Status**

Very important finding: inverse scale reading is a true structural parameter, but not a (J) mechanism.

---

# 17. Inverse UV/Env-Cut

## 17.1 UV-cut under inverse scale reading

**Initial situation**

Standard:

$$
\text{UV an Blättern},
\qquad
\text{Env an Wurzel}.
$$

Inverse reading:

$$
\text{UV an Wurzel},
\qquad
\text{Env an Blättern}.
$$

**Findings**

Identified as a genuine additional test; not fully concluded as a separate final positive finding.

**Obstruction Site**

Would introduce scale roles directly into the operator structure. However, as long as the resulting operators remain real-symmetric and relabeling-natural, chirality is not to be expected.

**Status**

Open or marked as the next precise test, but partially classified by subsequent DtN/flatness diagnosis.

---

# 18. DtN Handoff Operator Tests

## 18.1 Two DtN Matrices on a Shared Handoff Space

**Initial situation**

After correction: Handoff no longer sees ToC nodes, but operators:

$$
(H_\partial,\Lambda).
$$

Target:

$$
K=$$\Lambda_A,\Lambda_B$$.
$$

**Findings**

Only meaningful if both operators reside on the same handoff space.

**Obstruction-Location**

Spectral order alone does not identify eigenspaces. Diagonalized in their respective eigenspaces, both commute trivially.

**Status**

Important category correction.

---

## 18.2 DtN-RG commutator

**Initial Situation**

Successive RG/Schur steps of the same sequence:

$$
\Lambda_n,
\qquad
\widetilde{\Lambda}_{n+1}.
$$

Commutator:

$$
K_n=$$\Lambda_n,\widetilde{\Lambda}_{n+1}$$.
$$

**Result**

Reported:

$$
K_n=0
$$

for canonical RG projection.

**Obstruction point**

Both operators lie on the same radial F1 axis and share the same symmetry-adapted shell basis.

**Status**

Very important mechanism:

$$
\text{abgeleitete Reihenfolge durch F1}
\Rightarrow
\text{gleiche Achse}
\Rightarrow
\text{Kommutativität}.
$$

---

# 19. Superimposed DtN Matrix Algebra Towers

## 19.1 Matrix Tower Idea

**Initial Situation**

Proposal:

$$
M_2\to M_4\to M_8\to\cdots
$$

or multiple ToC-DtN matrices on growing handoff spaces.

**Findings**

Non-commutativity could arise if several symmetric operators on the same space do not share a common eigenbasis.

**Obstruction Point**

Examples involving spin chains introduce tensor product order and neighborhood:

$$
A_{12},\qquad A_{23}.
$$

This left-right structure is not derived from the bare ToC.

**Status**

Interesting as a possible A→B algebra path, but only allowed with derived embeddings.

---

## 19.2 Child Partition/ToC-Derived Embedding Test

**Initial Situation**

Derived embeddings via child subtrees or (S_b)-symmetric partitions.

**Result**

Child-restricted DtN operators commute:

* disjoint supports → trivial commutators,
* full DtN vs. block-diagonal part → commutes numerically.

**Obstruction location**

All decompositions respect the same (S_b)/radial symmetry and share the symmetry-adapted eigenbasis.

**Status**

Matrix Tower route negative in the flat derived ToC sector.

---

# 20. Connes/Noncommutativity route

## 20.1 Fundamental Question: Where Does Noncommutativity Come From in Connes?

**Starting Point**

Connes replaces space with algebra:

$$
(\mathcal A,\mathcal H,D).
$$

Noncommutativity lies in:

$$
ab\neq ba.
$$

**Finding**

In Connes, the noncommutative algebra is typically the input structure, not derived from a flat ToC.

**Obstruction point for CNNA**

CNNA would first have to provide a handoff algebra:

$$
\mathcal A_{\mathrm{eff}} = \operatorname{Alg}\{\Lambda_i\}
$$

with

$$
$$\Lambda_i,\Lambda_j$$\neq0.
$$

**Status**

Connes is a target/comparison structure, not a generator.

---

## 20.2 Two reduction regimes

**Script / File**

```text
two_reduction_regimes.py
```

**Initial Situation**

Comparison:

$$
\Lambda_{\mathrm{UV}}
$$

versus

$$
\Lambda_{\mathrm{Env}}
$$

on the same leaf-boundary space.

**Findings**

Reported:

$$
|$$\Lambda_{\mathrm{UV}},\Lambda_{\mathrm{Env}}$$|\sim 10^{-16}.
$$

**Obstruction location**

Root self-energy shifts eigenvalues but does not rotate eigenspaces. Radial remains radial.

**Status**

Negative for exact derived regimes.

---

## 20.3 Spectrally truncated reduction

**Script / File**

```text
two_reduction_regimes.py
truncation_sign_test.py
```

**Initial situation**

Comparison:

$$
\Lambda_{\mathrm{full}}
$$

versus spectrally truncated reduction:

$$
\Lambda_{\mathrm{trunc}}.
$$

**Result**

For any (m):

$$
|$$\Lambda_{\mathrm{full}},\Lambda_{\mathrm{trunc}}$$|\approx 0.017
$$

for average (m) values; (K) is skewed.

**Obstruction location**

Initially misinterpreted: (\pm i\lambda) pairs were read as “both chiralities.” Correction:

$$
\pm i\lambda
$$

is the normal spectrum of a real (J) block.

The true sign test is:

$$
K\text{ oder }-K\text{ ausgezeichnet?}
$$

**Status**

Only an apparently positive candidate; had to be retested to ensure it was free of degeneracy.

---

## 20.4 Degeneracy-proof cluster truncation

**Script / File**

Retest of `truncation_sign_test.py`; described in chat.

**Initial situation**

Truncation not by arbitrary (m), but only by integer eigenvalue clusters:

$$
P_{\le \lambda} = \sum_{\mu\le\lambda}P_\mu.
$$

**Findings**

For all canonical cluster boundaries:

$$
|K|\approx 10^{-16}.
$$

Non-commutativity occurred only when (m) intersected degenerate eigenspaces exactly in the middle.

**Obstruction point**

A cut through degenerate eigenspaces selects a non-canonical `numpy` basis. This is not a ToC-derived mechanism.

**Status**

Strong negative result:

$$
\boxed{
\text{relabeling-natürliche exakte und cluster-sichere DtN-Reduktionen kommutieren.}
}
$$

---

# 21. Node Elimination vs. Partial Trail

## 21.1 Incorrect “Trail Elimination” Test

**Initial Situation**

System/environment nodes were separated:

$$
\mathbb R^N=\mathbb R^S\oplus\mathbb R^E.
$$

Then diffusion (e^{-tL}) was calculated and the environment was treated as a steady state.

**Findings**

Skew could have arisen.

**Obstruction Location**

This was not a partial trace. A partial trace requires:

$$
\mathcal H=\mathcal H_S\otimes\mathcal H_E.
$$

However, the node space provides a direct sum, not a tensor product.

The skew resulted from asymmetric input/restriction:

$$
\text{Umgebung speist ein, System-Abfluss wird verworfen}.
$$

**Status**

Invalid as an OQS/partial-trace test. At most, a test of an asymmetric boundary condition.

---

## 21.2 Correct Node Reduction

**Initial Situation**

For node splitting:

$$
L=
\begin{pmatrix}
L_{SS} & L_{SE}\
L_{ES} & L_{EE}
\end{pmatrix}.
$$

Correct elimination:

$$
L_{\mathrm{eff}} = L_{SS}-L_{SE}L_{EE}^{-1}L_{ES}
$$

**Findings**

For real symmetric (L):

$$
L_{\mathrm{eff}}^T=L_{\mathrm{eff}}.
$$

**Obstruction location**

Node elimination does not generate OQS irreversibility or an antisymmetric Hamiltonian part.

**Status**

Central method correction:

$$
\boxed{
\text{Auf Knoten wird eliminiert, nicht ausgespurt.}
}
$$

---

# 22. Flat Sector and Curvature

## 22.1 Flat Real-Reciprocal ToC/DtN Sector

**Initial Situation**

Ideal ToC or ToC fibers without curvature, holonomy, or regulator backreaction.

**Findings**

All natural operators remain jointly diagonalizable.

**Obstruction point**

There is no connection:

$$
\nabla,
$$

no holonomy:

$$
U_\gamma\neq I,
$$

and no curvature:

$$
$$\nabla_\mu,\nabla_\nu$$\neq0.
$$

**Status**

Change of interpretation:

$$
\boxed{
\text{Die No-Gos betreffen den flachen ToC-/DtN-Sektor.}
}
$$

Not CNNA as a whole.

---

## 22.2 Curvature as a possible later origin of non-commutativity

**Initial situation**

In geometry/gauge theory:

$$
$$\nabla_\mu,\nabla_\nu$$=R_{\mu\nu}
$$

or

$$
$$D_\mu,D_\nu$$=F_{\mu\nu}.
$$

**Findings**

In the CNNA context, non-commutativity could rather be an emergent curvature/holonomy phenomenon.

**Obstruction location**

Curvature must not be imported as a savior. It would have to arise from handoff/regulator/backreaction data.

**Status**

Open curved-sector target:

$$
\text{Block-RG/DtN}\to\text{Connection}\to\text{Holonomie/Krümmung}.
$$

---

# 23. IDEAL ToC Fiber Lattice

## 23.1 Double-infinite IDEAL Sector

**Initial configuration**

Instead of a universal single ToC:

$$
T_b^\infty
$$

one defines a ToC fiber lattice:

$$
\mathcal I_{\mathrm{ToCGrid}} = \Gamma_\infty\times T_b^\infty
$$

With:

$$
x\in\Gamma_\infty,
\qquad
w\in T_b^\infty.
$$

Two infinities:

$$
\Gamma_\infty
$$

transversal and

$$
T_b^\infty
$$

internal per fiber.

**Findings**

Fully ideal sector:

$$
\text{flach, homogen, reziprok, intern ToC-skaleninvariant}.
$$

Transverse isotropy only discrete or dependent on (\Gamma_\infty).

**Obstruction point**

The lattice introduces transverse neighborhood as a new IDEAL reference datum. It is not derived from a single ToC.

**Status**

A very useful final ToC-related test before substrate change.

---

## 23.2 Finite Double Section

**Initial Situation**

Calculable sector:

$$
\Omega_{R,L} = W_R\times T_{\le L}​
$$

With:

$$
W_R\subset\Gamma_\infty,
\qquad
T_{\le L}\subset T_b^\infty.
$$

**Findings**

Being a subsystem necessarily breaks the IDEAL symmetry:

$$
\operatorname{Aut}(\mathcal I_{\mathrm{ToCGrid}})
\to
\operatorname{Aut}(\Omega_{R,L}).
$$

The following arise:

$$
\text{äußeres Gitter-Komplement},
$$

$$
\text{interner UV-tail},
$$

$$
\text{Rand/Ecken/Mischkomplemente}.
$$

**Obstruction point**

Being a subsystem generates effective boundary/spectral/DtN geometry, but not automatically chiropathy.

**Status**

Positive geometry/DtN test, negative (J) test in the flat reciprocal case.

---

## 23.3 DtN on the ToC fiber lattice

**Initial situation**

Operator on (\Omega_{R,L}):

$$
L_{R,L}.
$$

Schur/DtN:

$$
\Lambda_{R,L} = L_{\partial\partial} - L_{\partial I}L_{II}^{-1}L_{I\partial}
$$

**Findings**

This is closer to A→B than raw node gluing. B would not see ToC nodes, but rather handoff matrices.

**Obstruction Location**

As long as the lattice is homogeneous, reciprocal, and flat, a spectrum and effective geometry arise, but no excellent chiropathy.

**Status**

Important final reference test:

$$
\boxed{
\text{ToC-Faser-Gitter kann Geometrie testen, nicht }J\text{ erzwingen.}
}
$$

---

# 24. Holonomy/Connection Test in the Fiber Lattice

## 24.1 Effective Intertwiners between Local Handoff Spaces

**Initial Situation**

For local handoff spaces:

$$
H_x,\qquad H_y
$$

one would need derived intertwiner:

$$
U_{xy}:H_x\to H_y.
$$

Loop holonomy:

$$
U_\gamma = U_{wx}U_{zw}U_{yz}U_{xy}
$$

**Findings**

Expected in the homogeneous flat case:

$$
U_\gamma=I
$$

or gauge-trivial.

**Obstruction point**

A non-trivial rotational component would have to come from inhomogeneity, the regulator, backreaction, or frustration.

**Status**

Open curved-sector test. Not yet proven positive.

---

# 25. Lorentz/time-structure tests

## 25.1 Lorentz signature

**Initial situation**

Signature:

$$
\eta=\operatorname{diag}(-1,+1,\ldots,+1).
$$

**Findings**

Separates time-like and space-like.

**Obstruction location**

Time reversal remains a symmetry:

$$
T\eta T=\eta.
$$

Light cone remains a double cone:

$$
C^+\cup C^-.
$$

**Status**

Reduces problem to temporal orientation, does not solve it.

---

## 25.2 Real-time flow precursor

**Initial Situation**

Real-symmetric generator (H), flow pair:

$$
{e^{+tH},e^{-tH}}.
$$

**Findings**

Delivers:

$$
{+\tau,-\tau}.
$$

**Obstruction point**

For real-symmetric (H), every spectral function remains symmetric. A (J) is antisymmetric:

$$
J\neq f(H).
$$

**Status**

Time pair yes. Locking with (J) no.

---

# 26. Pillar C / OQS / Entropy

## 26.1 Lindblad/OQS Time Arrow

**Initial Situation**

Open quantum dynamics / Lindblad generator.

**Findings**

Dissipation can choose the direction of time:

$$
+\tau.
$$

**Obstruction location**

Hamiltonian term already contains:

$$
-i$$H,\rho$$.
$$

Thus, OQS presupposes (i) or (J).

**Status**

Pillar C can choose (\tau), but cannot generate (J) on its own.

---

# 27. AQFT / Type-I / Type-III / Handoff Structure

## 27.1 A as a Type-I/Type-III precursor layer

**Initial situation**

Pillar A is not intended to directly prove Type III, but rather to provide precursors:

$$
\mathcal C_{d,k} = (Q_{d,k}\oplus P_{d,k},g_{d,k},\omega_{d,k},\{J,-J\})
$$

Finite:

$$
k<\infty
\Rightarrow
\text{Type-I-artige Vorläufer}.
$$

Infinite:

$$
k\to\infty
\Rightarrow
\text{Type-III-fähige Komplementfamilien-Vorläufer}.
$$

**Findings**

Architecturally sound.

**Location of obstruction**

Dimension/infinity provides no orientation:

$$
\text{finite/infinite}\neq J\text{-sign}.
$$

**Status**

Important architectural shift.

---

## 27.2 Triadic Handoff (B|B'|C)

**Initial Situation**

Handoffs are not passive arrows, but separate interface objects.

Triad:

$$
C\text{-Regulator}
\triangleright
H_{B|B'}(B,B')
\to
\text{stable record}.
$$

**Findings**

Best location for:

$$
\omega_{\mathrm{lock}}.
$$

**Obstruction location**

Not yet formalized. Type I/Type III asymmetry is initially algebraic/dimensional asymmetry, not orientation.

**Status**

Still the most important open (J)-locking candidate.

---

# 28. Multi-ToC / Detector / Multi-particle structure

## 28.1 Mini-ToCs as detector elements

**Initial situation**

A detector consists of many local ToC fibers:

$$
T_1,T_2,\ldots,T_N.
$$

Each carries locally:

$$
{J_i,-J_i}.
$$

**Findings**

Local sign can be gauge:

$$
J_i\mapsto -J_i.
$$

Relevant physical data would include relative or cyclic data:

$$
\sigma_{ij},
\qquad
\Phi_\gamma=\prod_{(ij)\in\gamma}\sigma_{ij}.
$$

**Obstruction Location**

The mechanism for (\sigma_{ij}) has not yet been derived.

**Status**

Strong candidate for the next non-local test.

---

## 28.2 Frustration / Spin-net-like structure

**Initial situation**

Many local ToC fibers are coupled. Possible cycle product:

$$
\Phi_\gamma=-1.
$$

**Findings**

If (\Phi_\gamma) is invariant under local gauge flips

$$
J_i\mapsto -J_i
$$

, true global frustration arises.

**Obstruction location**

(\sigma_{ij}) must not be set.

**Status**

Most important open multi-ToC test path.

---

# 29. Motor/Multiphase Analogy

## 29.1 Two-phase three-phase motor

**Initial Situation**

In two-phase operation, a three-phase motor does not generate a stably directed rotating field, but rather a superposition:

$$
\text{Vorwärtsdrehfeld}+\text{Rückwärtsdrehfeld}.
$$

**Findings**

Good analogy to:

$$
{+J,-J}.
$$

**Location of obstruction**

Without a third-phase configuration or connection arrangement, there is no stable direction of rotation.

**Status**

Highly effective for teaching.

---

## 29.2 Three Phases / Connection Order

**Initial Situation**

Balanced system:

$$
(1,a,a^2),
\qquad
a=e^{2\pi i/3}.
$$

Swapping:

$$
(1,a,a^2)
\leftrightarrow
(1,a^2,a).
$$

**Findings**

Direction of rotation is determined by the connection order.

**CNNA Translation**

Not the local (J_i) sign, but rather the handoff sequence or cycle order could be decisive.

**Obstruction Location**

Connection order must be derived.

**Status**

Good candidate for Multi-ToC-Handoff-Sequence-Gate.

---

# 30. Cayley-Dickson / higher division algebras

## 30.1 CD-/Hurwitz candidate

**Initial situation**

Route:

$$
\mathbb R\to\mathbb C\to\mathbb H\to\mathbb O.
$$

**Findings**

Negative for the first (J)-sign problem. Higher algebra does not resolve the origin of the first complex orientation.

**Obstruction Location**

Dimensional doubling and norm multiplicativity are not enforced from intersection data.

Open Objects:

```text
positiveDefiniteNormSq
divisionFromNormSq
alternativeLaw
```

**Status**

Not a current path for (J)-sign. Not ruled out as a later target structure.

---

# 31. Substrate-change candidates

## 31.1 ToC remains local provenance fiber

**Initial situation**

The single ToC as a global world tree fails at the (J)-gate.

**Findings**

As a local fiber, ToC remains valuable:

$$
\text{Provenienz}
\to
\text{Approximant}
\to
\text{Schur/DtN}
\to
\text{lokaler Handoff-Operator}.
$$

**Site of obstruction**

Global ontology as a single tree is too limited for a second axis, chirotopy, curvature.

**Status**

No total rejection of the ToC; a change of roles.

---

## 31.2 Event Structures as Candidates

**Initial Situation**

Event structures possess two relations:

$$
\leq \qquad\text{und}\qquad \#
$$

**Findings**

They could represent two axes, namely causality and conflict.

**Obstruction Location**

Both relations would initially be primitive input data as long as they are not CNNA-derived.

**Status**

Strong candidate within (i)-free substrate classes, but not yet a derived result.

---

# 32. Essential Scripts by Chat Status

## 32.1 Reliably Named Older Scripts

```text
app.py
```

ToC Concept Explorer.

```text
alpha_orth_invariant.py
```

(\Xi)-, (\rho)-, (\alpha_{\mathrm{orth}})-diagnostics.

```text
two_approximant_flow_sign.py
```

Parent–Child/Flow-Sign diagnostics.

```text
two_boundary_shell_chirality.py
```

V4: Single-graph Cauchy pairing.

```text
family_handoff_chirality.py
```

V5: Cross-graph Cauchy and Handoff Square.

```text
triadic_interface_chirality.py
```

V6: UV/Env/Regulator triad.

```text
oriented_cauchy_shell_gate.py
```

V7: Opposite-direction UV/Env Cauchy shell.

## 32.2 Recent test scripts mentioned in the chat

```text
two_reduction_regimes.py
```

DtN regimes: bare UV-DtN vs. root-env-DtN; additionally, truncated reduction.

```text
truncation_sign_test.py
```

Sign/commutator test for spectral truncation.

## 32.3 Scripts that make sense to implement next

```text
cluster_safe_truncation_test.py
```

Explicitly encapsulate degeneracy-safe cluster truncation.

```text
toc_fiber_grid_dtn_test.py
```

IDEAL ToC fiber lattice, double cut (\Omega_{R,L}), DtN spectral tests.

```text
fiber_grid_connection_holonomy_test.py
```

Derived Intertwiner (U_{xy}) and loop holonomy in the fiber lattice.

```text
multi_toc_frustration_gate.py
```

Gauge-invariant cycle products:

$$
\Phi_\gamma=\prod\sigma_{ij}.
$$

```text
handoff_phase_sequence_gate.py
```

Handoff sequence/multi-phase gate.

---

# 33. Obstruction Locations by Type

## 33.1 Reciprocity

$$
\Lambda=\Lambda^T.
$$

Passive Schur/DtN reduction remains symmetric. No antisymmetric (J) generator.

## 33.2 Real conjugation symmetry

$$
J\mapsto -J.
$$

Real structures do not choose a complex orientation.

## 33.3 (S_b)-Equivariance

Sibling permutations preserve canonical quantities in the trivial sector. No sign selection.

## 33.4 Radial Uniaxial Structure (F1)

F1 provides order:

$$
n\to n+1.
$$

But only along one axis. Non-commutativity requires two independent axes.

## 33.5 Degeneracy

Degenerate eigenspaces must not be cut by arbitrary numerical bases. Only entire clusters are relabeling-natural.

## 33.6 No partial trace on nodes

$$
\mathbb R^N=\mathbb R^S\oplus\mathbb R^E
$$

is a direct sum, not a tensor product.

## 33.7 Bit reversal

Address cycles can mirror the direction of rotation:

$$
\rho\leftrightarrow\rho^{-1}.
$$

## 33.8 Boundary reversal

UV/Env co-orientation yields:

$$
{J,-J}.
$$

## 33.9 Handoff reversal

$$
A_{\gamma^{-1}}=-A_\gamma.
$$

Without a directed handoff sequence, there is no absolute direction of rotation.

## 33.10 OQS dependence on (i)

Lindblad/OQS can provide time direction, but requires Hamilton-(i).

## 33.11 Flatness

Missing in the flat ToC/DtN sector:

$$
\text{Connection},
\qquad
\text{Holonomie},
\qquad
\text{Krümmung}.
$$

---

# 34. Current overall formula

$$
\boxed{
\text{Alle Einzelbaum-, Einzelapproximant-, passiven Schur-/DtN- und lokalen Triadentests enden bei }{J,-J}.
}
$$

$$
\boxed{
\text{Exakte und cluster-sichere Handoff-Operatoren im flachen ToC-/DtN-Sektor kommutieren.}
}
$$

$$
\boxed{
\text{Nichtkommutativität entsteht bisher nur durch gesetzte Ordnung, nicht-kanonische Trunkierung oder asymmetrische Randvorschrift.}
}
$$

$$
\boxed{
\text{Der nächste echte positive Suchraum ist nicht ein weiterer flacher Einzel-ToC-Test, sondern Curved-sector, Multi-ToC-Frustration oder triadisches Handoff-Locking.}
}
$$

The most important upcoming ToC-related test before the substrate change remains:

$$
\boxed{
\mathcal I_{\mathrm{ToCGrid}}=\Gamma_\infty\times T_b^\infty,
\qquad
\Omega_{R,L}=W_R\times T_{\le L},
\qquad
\Lambda_{R,L}.
}
$$

Goal:

$$
\text{effektive Geometrie aus Subsystem-Sein testen},
$$

but separate from that:

$$
\text{(J)-/Chirotopie-/Nichtkommutativitäts-Gate weiter offen halten}.
$$
