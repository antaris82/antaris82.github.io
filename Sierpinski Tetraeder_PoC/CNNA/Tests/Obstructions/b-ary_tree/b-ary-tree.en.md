# CNNA-ToC / J-sign / Non-commutativity — complete inventory of tests and obstructions

Status: based on chat discussions, not as a lean theorem. Most findings are numerical, conceptual, or derived from the diagnostic artifacts listed here. The central result is now more precise than at the outset:

This version additionally includes the substrate gate from the revised paper version: Event structures are no longer treated as permissible foundation candidates because they would already contain a causal or temporal order as primitive data via $\leq$. They remain only comparison or target structures.

> **Key statement.** The flat, real-valued, reciprocal ToC/Schur/DtN sector does not generate a distinguished J-orientation.

It delivers multiple times:

$$
\{+J,-J\},\qquad \{+\tau,-\tau\},\qquad \text{radiale Ordnung},\qquad \text{DtN-/Spektralstruktur}.
$$

It has not yet delivered:

> **Key statement.** J instead of -J.

The uniform obstruction is now no longer just “symmetry,” but more precisely:

> **Key statement.** A derived axis F1 is not sufficient. Non-commutativity requires at least two derived operator axes that cannot be diagonalized together; chirality additionally requires a derived orientation or sign-line selection.

---

# 0. Global status of the test series

## 0.0 Why the b-ary tree was chosen as the ToC reference substrate

The b-ary tree was not introduced as an arbitrary world tree. The historical motivation was the tamability, symmetry, and scale invariance of the Sierpinski Gasket (SG) and the Sierpinski Tetrahedron/Tetrix (ST). These objects were attractive because they form a controlled, self-similar, and highly symmetric test class. It was precisely this tameness that was methodologically important: if even the most symmetric and controlled candidate does not enforce the desired direction of $J$, then the obstruction lies not in numerical wildness, but in the structure of the flat real-reciprocal sector.

The corresponding provenance page of SG and ST is a binary address tree. For the Sierpinski gasket, the natural address tree is ternary; for the Sierpinski tetrahedron, it is quaternary:

$$
SG:\quad b=3,\qquad ST:\quad b=4.
$$

Underlying every geometric embedding, every quotient relation, and every orientation is the pure address/provenance structure

$$
A_b^{§§X199§§ **Kernaussage.** Der ToC ist als lokale Provenienzfaser und flacher Referenzsektor wertvoll.

Dabei ist ein ToC-Knoten kein physikalischer Freiheitsgrad. Seine Rolle im flachen Referenzsektor ist zunächst die eines Provenienzindex. Der zulässige Lesepfad ist:

$$
\text{ToC node} \to \text{provenance index} \to \text{approximant} \to \text{Schur/DtN} \to \text{effective handoff operator} \to \text{possible physical degree of freedom}.
$$

Nicht zulässig ist die Kurzidentifikation:

$$
\text{ToC node}=\text{physical degree of freedom}.
$$

§§X4§§0.2 Was A/ToC bisher nicht liefert

Nicht geliefert wird eine absolute Orientierung:

> **Kernaussage.** J ≠ derived uniquely from flat ToC data.

Auch nicht geliefert werden bisher:

$$
\text{chirotopy},
$$

$$
\text{non-trivial holonomy},
$$

$$
\text{Curvature},
$$

$$
\text{non-commutative handoff algebra},
$$

$$
\text{genuine partial trace/OQS structure}.
$$

§§X5§§0.3 Neuer Interpretationsstatus

Die Tests betreffen nicht „CNNA überhaupt“, sondern den spezifischen Sektor:

> **Kernaussage.** flacher, homogener, reell-reziproker ToC-/DtN-Sektor.

Daraus folgt:

> **Kernaussage.** Der b-äre Einzelbaum ist unter den flach-reziproken Derived-only-Prämissen als globaler J-Generator falsifiziert.

Aber nicht:

> **Kernaussage.** lokale ToC-Fasern, DtN-Geometrie oder CNNA als Gesamtprogramm sind falsifiziert.

Ebenso ist nicht das ToC-Konzept als solches obstruiert. Obstruiert ist bisher nur die Lesart

> **Kernaussage.** b-ärer Einzelbaum = globaler Träger des Universums und einer ausgezeichnet gerichteten komplexen Struktur.

Das positive Gegenfinding ist sogar stärker: Komplement-, Handoff- und lokale-Algebra-Strukturen bleiben nicht nur zulässig, sondern wirken für den Anschluss an AQFT-artige lokale Algebren weiterhin notwendig. Diese Aussage ist hier kein bewiesener AQFT-Rekonstruktionssatz, sondern ein Architektur- und Anschlussbefund:

$$
\text{local algebras} \Longleftrightarrow \text{complement/intersection/handoff structures remain central.}
$$

Der Rollenwechsel ist daher selbst ein Finding:

> **Kernaussage.** Der b-äre Einzelbaum ist nicht Weltbaum, sondern lokale Provenienzfaser.

Ein endlicher Approximant ist entsprechend nicht automatisch ein Vielteilchensystem. Er ist zunächst ein effektiver lokaler Handoff-/Objektkandidat:

$$
\Omega(a,L) \Rightarrow \text{effective local handoff/object candidate}.
$$

Viele Objekte, Detektoren oder Vakuum-Gluing-Strukturen entstehen erst aus einer Familie lokaler Fasern und deren Verklebungen:

$$
\{T_i\}_{i\in I} \Rightarrow \text{multi-ToC/gluing structure}.
$$

§§X6§§0.4 Zusätzlicher Substrat-Gate: keine primitive Kausalität

Der nächste Substratkandidat darf keine der Strukturen enthalten, die CNNA erst rekonstruieren soll:

> **Kernaussage.** kein primitives i, · kein primitives J, · keine primitive Orientierung, · keine primitive Tensorstruktur, · keine primitive Kausalität.

Insbesondere sind Ereignisstrukturen mit einer gegebenen Relation $\leq$ nicht als Fundament zulässig, sofern $\leq$ kausal oder zeitartig gelesen wird. Eine solche Relation würde bereits eine Zeit-/Kausalordnung einführen. Zulässig ist nur die umgekehrte Richtung:

$$
\text{non-causal CNNA pre-structure} \longrightarrow \text{emergent events} \longrightarrow \text{emergent causal order}.
$$

Damit wird der Substratwechsel-Gate verschärft: Gesucht ist nicht einfach ein reichhaltigeres Substrat, sondern ein reichhaltigeres Substrat ohne importierte Kausalität.

§§X130§§

§§X7§§0.5 Grunddefinitionen des flachen ToC-Sektors

Dieser Abschnitt fixiert die Minimalnotation, auf die alle folgenden Tests bezogen sind. Er ist keine zusätzliche physikalische Annahme, sondern eine Konventionsschicht für den flachen, homogenen, reell-reziproken ToC-/DtN-Referenzsektor. Dieser Referenzsektor ist die b-äre Provenienzseite der SG/ST-Motivation und nicht das vollständige CNNA-ToC-Konzept mit möglichen Clique-, Gluing-, Regulator- oder lokalen-Algebra-Anreicherungen.

§§X8§§0.5.1 Adressalphabet, Wortbaum und Konkatenation

Fixiere

$$
b\ge 2
$$

und das Adressalphabet

$$
A_b=\{0,\ldots,b-1\}.
$$

Die Elemente von $A_b$ sind zunächst nur Adresssymbole. Sie tragen keine physikalische Ordnung, keine zyklische Ordnung und keine Orientierung. Jede spätere Ordnung auf Geschwistern wäre daher eine zusätzliche, zu begründende Struktur.

Der unendliche b-äre ToC ist der Wortbaum

$$
T_b^\infty=A_b^{<\omega} = \bigcup_{n\ge 0} A_b^n.
$$

Die Wurzel ist das leere Wort

$$
\varnothing\in A_b^0.
$$

Für Wörter $u,v\in T_b^\infty$ bezeichnet

$$
uv
$$

die Wortverkettung. Für $i\in A_b$ ist also $wi$ das Wort, das aus $w$ durch Anhängen des Symbols $i$ entsteht.

Die Tiefe eines Knotens $w$ ist die Wortlänge

$$
|w|.
$$

Für $w\ne\varnothing$ ist der Parent-Knoten

$$
\pi(w)
$$

das Wort, das durch Entfernen des letzten Symbols von $w$ entsteht. Die Kindermenge von $w$ wird mit $C_b(w)$ bezeichnet:

$$
C_b(w)=\{wi:i\in A_b\}.
$$

§§X9§§0.5.2 Präfixordnung, Kantenrelation und Unit-edge-Graph

Die natürliche Provenienzordnung des ToC ist die Präfixordnung

$$
u\preceq v \quad\Longleftrightarrow\quad \exists r\in A_b^{<\omega}: v=ur.
$$

Dabei bedeutet $u\preceq v$, dass $u$ ein Vorfahr von $v$ ist. Die strikte Präfixordnung ist

$$
u\prec v \quad\Longleftrightarrow\quad u\preceq v\ \text{und}\ u\ne v.
$$

Die ungerichtete Baumkante ist

$$
x\sim y \quad\Longleftrightarrow\quad x=\pi(y)\ \text{oder}\ y=\pi(x).
$$

Der bare ToC-Graph ist damit

$$
G_b^\infty=(T_b^\infty,E_b^\infty),
$$

mit

$$
E_b^\infty = \bigl\{\{w,wi\}:w\in T_b^\infty,\ i\in A_b\bigr\}.
$$

Jede bare Kante hat Gewicht $1$. Es gibt im baren Sektor keine eingebettete Geometrie, keine Winkel, keine Längen außer graph distance, keine Orientierung, keine komplexe Struktur, keine Zeit und keine Kausalordnung. Die einzige bare Abstandsgröße ist die graph distance

$$
d_G(x,y),
$$

insbesondere

$$
d_G(\varnothing,w)=|w|.
$$

Diese graph-distance/depth-Lesart ist eine Provenienz- und Skalenordnung, aber keine Raumzeitmetrik.

Wichtig ist die Rollenbegrenzung: Knoten von $T_b^\infty$ sind im flachen ToC-Sektor keine physikalischen Freiheitsgrade. Sie sind Adress- und Provenienzindizes. Physikalisch relevante Freiheitsgrade dürfen erst nach einem Schnitt, einer Schur-/DtN-Eliminierung und einem Handoff entstehen. Damit ist der direkte Schluss

$$
\text{Knoten im ToC}\Rightarrow\text{physikalisches Teilchen oder Feld-DOF}
$$

nicht zulässig.

§§X10§§0.5.3 Relabeling-Gauge und Kanonizitätsbedingung

Solange keine Geschwisterordnung abgeleitet wurde, sind die $b$ Kinder eines Knotens nur bis auf Relabeling unterscheidbar. Lokal wirkt daher auf jeder Geschwisterfamilie eine Permutationsgruppe

$$
S_b.
$$

Ein lokales Relabeling am Knoten $w$ ersetzt

$$
wi\mapsto w\sigma(i), \qquad \sigma\in S_b,
$$

und setzt sich auf den darunterliegenden Teilbaum fort. Globale oder lokale Relabelings dieser Art sind Gauge-artige Adresswechsel, solange keine zusätzliche Struktur sie bricht.

Eine Größe, die im flachen ToC-Sektor als kanonisch gelten soll, muss daher relabeling-natürlich bzw. invariant formuliert sein. Insbesondere ist eine Aussage, die eine konkrete Reihenfolge

$$
0<1<\cdots<b-1
$$

oder eine zyklische Ordnung

$$
0\to1\to\cdots\to b-1\to0
$$

benutzt, nicht derived-only, solange diese Ordnung nicht zuvor aus ToC-/Handoff-Daten abgeleitet wurde.

Diese Relabeling-Bedingung ist der technische Grund, warum reine Adresszyklen, Hamming-Klassen oder Screw-Regeln noch keine CNNA-abgeleitete Chirotopie liefern.

§§X11§§0.5.4 Endliche Approximanten als induzierte Teilgraphen

Für einen Anchor

$$
a\in T_b^\infty
$$

mit Einbettungstiefe

$$
k=|a|
$$

und eine innere Approximantentiefe

$$
L\ge 0
$$

ist die Knotenmenge des endlichen Approximanten

$$
\Omega(a,L)=\{av:v\in A_b^{\le L}\}.
$$

Hier ist

$$
A_b^{\le L}=\bigcup_{0\le n\le L}A_b^n.
$$

Der zugehörige Approximantengraph ist der induzierte Teilgraph

$$
G_\Omega=(\Omega(a,L),E_\Omega),
$$

mit

$$
E_\Omega = \bigl\{\{x,y\}\in E_b^\infty:x,y\in\Omega(a,L)\bigr\}.
$$

Seine Knotenanzahl ist

$$
|\Omega(a,L)| =1+b+\dots+b^L =\frac{b^{L+1}-1}{b-1}.
$$

Die inneren relativen Level des Approximanten sind

$$
\Omega_\ell(a,L)=\{av:v\in A_b^\ell\}, \qquad 0\le \ell\le L.
$$

Der Approximantenroot ist

$$
a\in\Omega_0(a,L).
$$

Die UV-Boundary bzw. Blattmenge ist

$$
\partial_{\mathrm{UV}}\Omega = \Omega_L(a,L) = \{av:|v|=L\}.
$$

Der Bright-Sektor ist

$$
\Omega(a,L).
$$

Der Dark-Sektor ist schnittrelativ

$$
T_b^\infty\setminus\Omega(a,L).
$$

Er zerfällt in den UV-tail an den Blättern von $\Omega(a,L)$ und, falls $k>0$, in the environment portion on the parent/root side. For $k=0$, the noOuterEnvironment interpretation applies.

The Environment port, if $k>0$, is the root-side interface port at the approximant root $a$. It is not an additional Bright node, but rather the interface to the outer complement side.

The approximant itself also initially has a role limitation: $\Omega(a,L)$ is not an automatically interpreted many-particle system. In the flat ToC sector, it is a section-relative local handoff/object candidate. Only the Schur/DtN data generated from it and subsequent gluing/regime formations can carry physical degrees of freedom or multi-object structures.

### 0.5.5 Bright-Laplace Operator and Complement Loads

The Bright-Laplace operator $L_\Omega$ is the Laplace operator of the induced Bright graph $G_\Omega$:

$$
(L_\Omega)_{xy} = \begin{cases} d_\Omega(x), & x=y,\\ -1, & x\sim y\text{ innerhalb von }\Omega,\\ 0, & \text{sonst}. \end{cases}
$$

Here,

$$
d_\Omega(x)=|\{y\in\Omega:x\sim y\}|
$$

and counts only neighbors within $\Omega$. Complementary branches are not included in $\deg_\Omega$. Their effect is supplemented exclusively via Schur/DtN/Load terms. This avoids double counting of out-edges.

A UV-cut or environment-cut is already a Dirichlet-type boundary condition. Schur/DtN elimination is therefore not stabilized by an external numerical regularization, but by the cut-relative boundary status itself. The regularization is internal to the cut:

$$
\text{UV-cut oder Environment-cut} \Rightarrow \text{Dirichlet-Boundary} \Rightarrow L_{II}^{-1}\text{ wohldefiniert},
$$

provided that the inner block under consideration is actually coupled to the set boundary. External auxiliary conditions such as ridge terms, pseudo-inverses, or artificial mass terms do not belong to the flat derived-only ToC/DtN core.

The effective operator has the form

$$
M_\Omega=L_\Omega+\Sigma_{\mathrm{Env}}+\Sigma_{\mathrm{UV}}.
$$

In the simplest load-based proxy, one can write

$$
\Sigma_{\mathrm{Env}} = \sigma_{\mathrm{Env}}\,P_{\mathrm{root}}, \qquad \Sigma_{\mathrm{UV}} = \sigma_{\mathrm{UV}}\,P_{\partial_{\mathrm{UV}}\Omega},
$$

where this is considered derived only if the values originate from an explicit Schur/DtN elimination of the respective complement families. Early constant or ladder models for $\sigma_{\mathrm{Env}}$ and $\alpha_{\mathrm{Env}}$ are diagnostic models, not ontic CNNA inputs.

The two loads act on opposite sides of the approximant:

$$
\Sigma_{\mathrm{UV}}\text{ wirkt leafseitig an den feinsten/cut-Knoten}, \qquad \Sigma_{\mathrm{Env}}\text{ wirkt rootseitig am Parent-/Environment-Port}.
$$

Thus, the intersection creates a true internal scale break in the approximant. However, this scale break is initially radial or longitudinal:

$$
\text{UV/Env-Skalenbruch}\neq\text{Chiralität}.
$$

For channel sources $f_{\mathrm{Env}}$ and $f_{\mathrm{UV}}$, the responses are

$$
u_{\mathrm{Env}}=M_\Omega^{-1}f_{\mathrm{Env}}, \qquad u_{\mathrm{UV}}=M_\Omega^{-1}f_{\mathrm{UV}}.
$$

By default, $f_{\mathrm{Env}}$ is a root-side source at the environment port and $f_{\mathrm{UV}}$ is a symmetric or normalized leaf source on $\partial_{\mathrm{UV}}\Omega$. Any deviating normalization must be explicitly documented in the respective artifact.

The energy inner product is

$$
\langle x,y\rangle_M=x^TM_\Omega y.
$$

The orthogonality diagnosis is

$$
\rho_M = \frac{\langle u_{\mathrm{Env}},u_{\mathrm{UV}}\rangle_M} {\|u_{\mathrm{Env}}\|_M\|u_{\mathrm{UV}}\|_M}.
$$

The quantities $\alpha_{\mathrm{UV}}$, $\alpha_{\mathrm{Env}}$, $C_k$, and $\Xi$ are diagnostic quantities as long as they are not derived from the complete complement families. In the early tests, $C_k$ denotes a cut- or depth-dependent normalization/capacity measure of the approximator; its exact value depends on the artifact or diagnostic and should therefore not be interpreted as a universal CNNA constant.

### 0.5.6 J-Problem, F1/F2, and Locking Object

A complex structure on a real handoff space is an endomorphism $J$ with

$$
J^2=-I.
$$

The $J$-sign problem is not merely the existence of such a block, but the derived-only selection of $J$ over $-J$. A real, symmetric, relabeling-natural structure therefore yields at most

$$
\{+J,-J\},
$$

as long as there is no additional derived orientation or locking structure.

$F1$ denotes the radial provenance/depth axis

$$
|w|\mapsto |w|+1.
$$

A second axis $F2$ is not an input, but an open target object: an independently derived transverse structure that is not trivialized by full $S_b$-symmetry.

$\omega_{\mathrm{lock}}$ denotes the still open handoff form, which would have to couple a $J$-orientation with a flow/time/handoff orientation $\tau$. It is not identical to a mere Cauchy boundary form as long as the latter only

$$
\{+J,-J\}
$$

. The Cauchy shell can thus be positive without solving the actual locking problem:

$$
\omega_\partial\Rightarrow\{+J,-J\}, \qquad \omega_{\mathrm{lock}}:(J,\tau)\mapsto\text{stabiler orientierter Record}.
$$

---

# 1. Didactic and Proxy Tests

## 1.1 Hugging Face ToC Concept Explorer

**Artifact Reference**

```text
Hugging-Face-Space: https://huggingface.co/spaces/antaris/b-ary_tree
app.py
```

`app.py` is the visualization script for the Hugging Face Space. It is intended for illustrative purposes only and does not serve as proof or primary diagnostics.

**Initial Situation**

Visualization of a (b)inary ToC with parameters:

$$
b,\qquad L_{\max},\qquad \text{Approximant root},\qquad L.
$$

Levels shown:

$$
\text{ToC} \to \text{proper subsystem} \to \text{UV-tail} \to \text{Environment} \to \text{Cauchy-/}J\text{-Kandidat} \to \text{Complex-plane overlay}.
$$

**Findings**

Highly educational. It clearly distinguishes:

$$
\text{Approximant}, \qquad \text{UV-tail}, \qquad \text{Environment}, \qquad \text{Interface}.
$$

**Location of Obstruction**

Visualization is not proof. Early tilt/angle values were partly chart/rendering proxies, not DtN invariants.

**Status**

Didactically valuable, mathematically secondary.

---

## 1.2 Stage-6 Chart Proxy / Tilt Test

**Artifact Reference**

Part of the Hugging Face visualization `app.py`; only illustrative and proxy level.

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

**Obstruction Location**

No true Schur/DtN value:

$$
\text{Proxy} \neq \text{Invariante}.
$$

**Status**

Heuristic motivation; later replaced by true DtN/Schur tests.

---

# 2. Single-Approximant Schur/DtN Tests

## 2.1 Projected-tail J/Rotation Test

**Artifact Reference**

No standalone appended artifact in this version; the section remains as a consolidated finding from the later $\alpha_{\mathrm{orth}}$ and DtN diagnostics.

**Initial situation**

Finite approximant with effective operator:

$$
M=L_\Omega+\text{projected UV/Env loads}.
$$

Two channel responses:

$$
u_{\mathrm{Env}},\qquad u_{\mathrm{UV}}.
$$

Measured quantity:

$$
\rho_M = \frac{\langle u_{\mathrm{Env}},u_{\mathrm{UV}}\rangle_M} {\|u_{\mathrm{Env}}\|_M\,\|u_{\mathrm{UV}}\|_M}.
$$

**Findings**

Near orthogonality:

$$
|\rho_M|\ll 1,
$$

partially numerically close to $90^\circ$.

**Obstruction location**

Orthogonality of a real 2-plane yields at most:

$$
\{+J,-J\}.
$$

The plane is there; the direction of rotation is not.

**Status**

Positive precursor of a pre-complex plane. No sign proof.

---

## 2.2 Real finite-network Schur/DtN test

**Artifact reference**

No standalone appended artifact in this version; the section remains as a consolidated methodological finding.

**Initial situation**

Finite tree graph with Laplace matrix:

$$
L_{\mathrm{graph}}.
$$

Edge (B), internal node (I), Schur complement:

$$
\Lambda_B = L_{BB}-L_{BI}L_{II}^{-1}L_{IB}.
$$

**Findings**

Numerically practically orthogonal for deterministic centered single modes, for example:

$$
|\rho_M|\approx 10^{-18}.
$$

**Obstruction location**

A single mode can be orthogonal while the full boundary response space still carries structure. Furthermore, the DtN operator remains real symmetric.

**Status**

Strong indication of true shear/DtN orthogonality in certain modes; no $J$ sign.

---

## 2.3 Dirichlet/Cut regularization test

**Artifact Reference**

No standalone appended artifact in this version; the section establishes the methodological findings.

**Initial Situation**

Question:

$$
\text{Braucht man eine externe Regularisierung oder Pseudoinverse?}
$$

More specifically: Does the tree or the Dirichlet network need to be artificially regularized, or does a set UV or environment cut already have a regularizing effect in itself?

**Findings**

No, the Dirichlet network does not need to be artificially regularized. A true UV cut or environment cut already has a regularizing effect in itself, because the removed complementary portion is treated as a Dirichlet/boundary side. This makes the interior block

$$
L_{II}
$$

invertible, provided that the interior region under consideration is actually coupled to the specified boundary.

The regularization is therefore internal to the cut:

$$
\text{UV-cut oder Environment-cut} \Rightarrow \text{Dirichlet-Boundary} \Rightarrow L_{II}^{-1}\text{ wohldefiniert}.
$$

It is not an external numerical auxiliary condition:

$$
\text{kein Ridge},\qquad \text{keine Pseudoinverse},\qquad \text{kein künstlicher Massenterm}.
$$

**Obstruction Point**

The DtN operator remains cut-relative:

$$
\Lambda_{\partial A}.
$$

The cut-internal regularization thus provides a well-defined DtN/Schur operator for the respective cut, but not a cut-free universal DtN operator for the entire infinite ToC.

**Status**

Important positive result: UV and environment cuts provide the necessary Dirichlet regularization themselves. No ridge/pseudoinverse/mass term setting is necessary.

---

## 2.4 Hard UV/Env scale break in the approximation

**Artifact reference**

Conceptually derived from the Schur/DtN and $\alpha_{\mathrm{orth}}$ tests; traceable in the attached diagnostic artifacts via $M_\Omega$, $\Sigma_{\mathrm{UV}}$, and $\Sigma_{\mathrm{Env}}$.

**Initial Situation**

A proper subsystem has two distinct complement sides:

$$
\text{UV-tail an den feinsten/cut-Knoten}, \qquad \text{Environment am Root-/Parent-Port}.
$$

**Findings**

The two complement projections load the approximant not uniformly, but oppositely in the inner scale direction:

$$
\text{UV-tail} \Rightarrow \text{Load an feinsten/cut-Knoten},
$$

$$
\text{Environment} \Rightarrow \text{Load am Root-/Parent-Port}.
$$

Thus:

$$
\Sigma_{\mathrm{UV}}\text{ wirkt leafseitig}, \qquad \Sigma_{\mathrm{Env}}\text{ wirkt rootseitig}.
$$

This is a genuine, sharp scale break in the approximant. It is not merely a visualization or a chart artifact.

**Obstruction Point**

The break is radial or longitudinal. It distinguishes between inside/outside, fine/coarse, UV/Environment, but it does not yet generate transverse handedness:

$$
\text{Skalenbruch}\neq\text{Chiralität}.
$$

**Status**

Positive finding for the physics of approximants and for $F1$. No proof of $J$-sign.

---

## 2.5 Passive Dirichlet/resistance networks do not generate phase

**Artifact reference**

Cross-sectional finding from the real Schur/DtN, Cauchy-Shell, and motor analogy tests; no independent additional artifact reference.

**Initial Situation**

The flat ToC/DtN sector is real, passive, and reciprocal. Mathematically, it behaves like a Dirichlet/resistance network with energy form, diffusion, and symmetric boundary response.

**Findings**

A purely resistive/passive sector provides imbalance, axes, loads, Dirichlet energy, diffusion, and DtN responses:

$$
\text{passive resistance/load} \Rightarrow \text{imbalance/axis}.
$$

However, it does not provide an independent $90^\circ$ phase shift or a stably directed rotating field:

$$
\text{passive resistance/load} \not\Rightarrow \text{rotating phase}.
$$

**Obstruction Location**

For oscillation, phase, or Hamiltonian-like rotation, a second storage structure, a derived skew sector, or a handoff locking mechanism would be required—one that is not already imported as a complex phase.

**Status**

Technical form of the motor/capacitor analogy: The real resistance sector can provide an axis and pulsation, but not the missing phase itself.

---

# 3. alpha_orth and invariant tests

## 3.1 Xi / alpha_orth diagnostics

**Artifact Reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/alpha_orth_invariant.py
```

**Initial Situation**

Control variable:

$$
\Xi=(1+\lambda_{\mathrm{UV}})(1+\lambda_{\mathrm{Env}}),
$$

with

$$
\lambda_{\mathrm{UV}} = \frac{b^k\alpha_{\mathrm{UV}}}{C_k}, \qquad \lambda_{\mathrm{Env}} = \frac{\alpha_{\mathrm{Env}}}{C_k}.
$$

Typical orthogonality diagnosis:

$$
|\rho|\sim \Xi^{-1/2}.
$$

**Findings**

The UV term dominates strongly as depth increases:

$$
|\rho|\sim b^{-k/2}.
$$

Therefore:

> **Key message.** UV resolution drives orthogonality.

**Obstruction location**

$\alpha_{\mathrm{Env}}$ was model-dependent in early versions:

```text
none
constant
power
exponential
ladder
```

Therefore, the exact numerical value was not a fully derived physical value.

**Status**

Good diagnostic measure. No fine-structure constant claim. No $J$ sign.

---

## 3.2 Environment-Sensitivity Models

**Artifact Reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/alpha_orth_invariant.py
```

**Initial Situation**

Comparison of various $\alpha_{\mathrm{Env}}$ models.

**Findings**

For large $k$, the UV term often dominates so strongly that the choice of environment model becomes subdominant.

**Location of obstruction**

In regimes where the environment is not subdominant, a genuine complement family/DtN derivation of $\alpha_{\mathrm{Env}}$ is required.

**Status**

Good methodological finding:

$$
\text{definierbar}\neq\text{erzwungen}.
$$

---

# 4. Parent–Child and Handoff Tests

## 4.1 Two-Approximant / Flow-Sign Test

**Artifact Reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/two_approximant_flow_sign.py
```

**Initial Situation**

Parent–Child Handoff:

$$
A_{\mathrm{parent}}\to A_{\mathrm{child}}.
$$

Objective: to check whether the transition yields a $J$ sign.

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

Radial handoff structure: yes. $J$-sign: no.

---

## 4.2 Schur-before-Flow criterion

**Artifact reference**

Methodologically derived from the parent–child tests; no additional independent artifact.

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

> **Key statement.** Schur first, Flow only as a consistency check.

---

# 5. Two-boundary/shell chirality tests

## 5.1 V4 — Two-boundary shell chirality

**Artifact reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/two_boundary_shell_chirality.py
```

**Initial situation**

Parent–child difference shell, two boundary ports, real DtN matrix:

$$
\Lambda_\Delta.
$$

Cauchy pairing:

$$
\omega((q,p),(q',p'))=q^Tp'-p^Tq'.
$$

On a DtN graph, the following holds:

$$
p=\Lambda q.
$$

**Findings**

For self-adjoint DtN graphs:

$$
\omega((q,\Lambda q),(r,\Lambda r)) = q^T\Lambda r-r^T\Lambda q = 0.
$$

**Obstruction location**

A single passive symmetric DtN graph is Lagrangian.

**Status**

Clean negative result. Too restrictive for family/handoff tests, but correct for a single graph.

---

## 5.2 V5 — Family handoff chirality

**Artifact reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/family_handoff_chirality.py
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
A\to B_i\to C, \qquad A\to B_j\to C.
$$

**Findings**

Cross-graph signals may occur:

$$
\omega_{ij}\neq 0.
$$

However:

```text
sibling_flip_detected = false
handoff_holonomy_detected = false
```

**Obstruction Location**

Signal is family/metric difference, not chirality. No sibling sign reversal, no true handoff holonomy.

**Status**

Important test: “Not just a graph” has been verified. Result remains achiral.

---

# 6. Triadic Tests

## 6.1 V6 — Triadic interface chirality

**Artifact reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/triadic_interface_chirality.py
```

**Initial situation**

Triad:

$$
\text{UV-channel}, \qquad \text{Environment-channel}, \qquad \text{Handoff/Regulator-channel}.
$$

Regulator candidate:

$$
r_i=(\Lambda_{\mathrm{child},i}-\Lambda_{\mathrm{parent}})a.
$$

Triadic surface:

$$
\tau_i = \det(e_{\mathrm{UV}}-e_{\mathrm{Env}},\,r_i-e_{\mathrm{Env}}).
$$

**Findings**

For canonical modes:

```text
tau_signs = 1,1,1
nonzero_tau_count = 3
sibling_flip_detected = false
```

**Site of obstruction**

The triad is radially or sibling-invariant.

**Status**

Triadic signal yes. Chiral sibling asymmetry no.

---

## 6.2 Non-canonical positive controls

**Artifact Reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Dateien darin:
- cnna_alpha_orth_invariant_v7/triadic_interface_chirality.py
- cnna_alpha_orth_invariant_v7/family_handoff_chirality.py
```

**Initial Situation**

Control Modes:

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

## 7.1 Oppositely oriented UV/Env boundary faces

**Artifact reference**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/oriented_cauchy_shell_gate.py
```

**Initial situation**

UV-tail and Environment-tail are interpreted as opposite-facing boundary sides of a shell.

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
g= \begin{pmatrix} k_{\mathrm{Env}} & 0 & 0 & 0\\ 0 & k_{\mathrm{UV}} & 0 & 0\\ 0 & 0 & k_{\mathrm{Env}}^{-1} & 0\\ 0 & 0 & 0 & k_{\mathrm{UV}}^{-1} \end{pmatrix}.
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
J^2=-I, \qquad J^TgJ=g, \qquad J^T\omega J=\omega.
$$

**Obstruction Point**

The co-orientation is chosen. With the opposite choice, the following is also consistent:

$$
J\mapsto -J.
$$

The Cauchy boundary form is therefore not identical to the sought-after locking object. It provides a symplectically compatible Cauchy structure, but not yet a locking of $J$ with a flow/time/handoff orientation $\tau$:

$$
\omega_\partial\Rightarrow\{+J,-J\}, \qquad \omega_{\mathrm{lock}}:(J,\tau)\mapsto\text{stabiler orientierter Record}.
$$

**Status**

Very important positive result:

$$
\text{UV/Env-Ko-Orientierung}\Rightarrow \{+J,-J\}\text{-Cauchy-Struktur}.
$$

Cauchy-Shell positive, but locking is missing. No absolute sign.

---

# 8. Root, Co-root, and Depth-First Search Tests

## 8.1 Root as Outer Model Boundary

**Initial Situation**

The ToC does not grow ontically; it is given as infinite.

$$
\ell(\mathrm{root})=0, \qquad \ell\to\infty
$$

inward.

**Findings**

Depth order provides relative oppositeness:

$$
\text{Env-Seite}: \ell\downarrow, \qquad \text{UV-Seite}: \ell\uparrow.
$$

**Site of obstruction**

Depth order is polar, not chiral:

$$
\text{innen/außen}\neq\text{Drehsinn}.
$$

**Status**

Semantically supports V7. No absolute $J$.

---

## 8.2 Negative-root / Co-root Hypothesis

**Initial Situation**

Hypothesis:

$$
\text{formale Root ist Interface;} \qquad \text{dahinter liegt negative Wurzelfamilie}.
$$

**Findings**

Could support Cauchy doubling and $\alpha_{\mathrm{Env}}$ derivation.

**Obstruction point**

A negative root family does not automatically remain chiral under real passive symmetry.

**Status**

Possible candidate for environment derivation; no sign proof.

---

# 9. Sibling, S_b, and address symmetry tests

## 9.1 S_b Sibling Obstruction

**Initial Situation**

In an unordered binary tree, siblings are

$$
S_b
$$

interchangeable.

**Findings**

Canonical sizes lie in the trivial $S_b$ component.

**Obstruction location**

The signum representation is not chosen canonically:

$$
S_b\text{-Äquivarianz} \Rightarrow \text{keine kanonische sibling-chirality}.
$$

**Status**

Robust negative line.

---

## 9.2 Hamming weight classes

**Initial situation**

See also:

$$
000,001,010,011,100,101,110,111.
$$

Classes:

$$
|x|_1=1, \qquad |x|_1=2.
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

Approximately at:

$$
\{001,010,100\}
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

Important candidate for multi-ToC/frustration structures. No local $J$ sign.

---

# 10. SG/ST, chirotopy, and sign-line tests

## 10.1 SG/ST as IFS/quotient structures

**Initial Situation**

The Sierpinski Gasket (SG) and the Sierpinski Tetrahedron/Tetrix (ST) were regarded as ToC-like quotient/IFS structures. Their role in the history of testing was no accident: SG and ST were the first natural fractal stress objects due to their tamability, high symmetry, p.c.f. controllability, and scale invariance.

The b-ary tree is the provenance or address side of these structures:

$$
SG:\quad A_3^{§§X201§§ **Kernaussage.** relabeling-natürliche exakte und cluster-sichere DtN-Reduktionen kommutieren.

§§X175§§

§§X78§§21. Knoten-Elimination vs. partielle Spur

§§X79§§21.1 Falscher „Ausspuren“-Test

**Ausgangslage**

System/Umwelt-Knoten wurden getrennt:

$$
\mathbb R^N=\mathbb R^S\oplus\mathbb R^E.
$$

Dann wurde Diffusion $e^{-tL}$ gerechnet und Umgebung mit festem Zustand behandelt.

**Befund**

Skew konnte entstehen.

**Obstruktions-Ort**

Das war keine partielle Spur. Eine partielle Spur braucht:

$$
\mathcal H=\mathcal H_S\otimes\mathcal H_E.
$$

Der Knotenraum liefert aber direkte Summe, kein Tensorprodukt.

Der Skew kam aus asymmetrischer Einspeisung/Restriktion:

$$
\text{Environment feeds in, system outflow is discarded}.
$$

**Status**

Ungültig als OQS-/Partial-trace-Test. Höchstens Test einer asymmetrischen Randbedingung.

§§X176§§

§§X80§§21.2 Korrekte Knotenreduktion

**Ausgangslage**

Für Knotenaufteilung:

$$
L= \begin{pmatrix} L_{SS} & L_{SE}\\ L_{ES} & L_{EE} \end{pmatrix}.
$$

Korrekte Eliminierung:

$$
L_{\mathrm{eff}} = L_{SS}-L_{SE}L_{EE}^{-1}L_{ES}
$$

**Befund**

Für reell symmetrisches $L$:

$$
L_{\mathrm{eff}}^T=L_{\mathrm{eff}}.
$$

**Obstruktions-Ort**

Knoten-Elimination erzeugt keine OQS-Irreversibilität und keinen antisymmetrischen Hamilton-Teil.

**Status**

Zentrale Methodenkorrektur:

> **Kernaussage.** Auf Knoten wird eliminiert, nicht ausgespurt.

§§X177§§

§§X81§§22. Flacher Sektor und Krümmung

§§X82§§22.1 Flacher reell-reziproker ToC-/DtN-Sektor

**Ausgangslage**

Idealer ToC bzw. ToC-Fasern ohne Krümmung, Holonomie, Regulator-Backreaction.

**Befund**

Alle natürlichen Operatoren bleiben gemeinsam diagonalisierbar.

**Obstruktions-Ort**

Es gibt keine Connection:

$$
\nabla,
$$

keine Holonomie:

$$
U_\gamma \neq I,
$$

und keine Krümmung:

$$
[\nabla_\mu,\nabla_\nu]\neq0.
$$

**Status**

Interpretationswechsel:

> **Kernaussage.** Die No-Gos betreffen den flachen ToC-/DtN-Sektor.

Nicht CNNA insgesamt.

§§X178§§

§§X83§§22.2 Krümmung als möglicher späterer Ursprung von Nichtkommutativität

**Ausgangslage**

In Geometrie/Eichtheorie:

$$
[\nabla_\mu,\nabla_\nu]=R_{\mu\nu}.
$$

bzw.

$$
[D_\mu,D_\nu]=F_{\mu\nu}.
$$

**Befund**

Nichtkommutativität könnte im CNNA-Kontext eher ein emergentes Krümmungs-/Holonomiephänomen sein.

**Obstruktions-Ort**

Krümmung darf nicht als Retter importiert werden. Sie müsste aus Handoff-/Regulator-/Backreaction-Daten entstehen.

**Status**

Offener Curved-sector target:

$$
\text{Block-RG/DtN}\to\text{Connection}\to\text{Holonomy/Curvature}.
$$

§§X179§§

§§X84§§23. IDEAL-ToC-Faser-Gitter

§§X85§§23.1 Doppelt unendlicher IDEAL-Sektor

**Ausgangslage**

Statt eines universalen Einzel-ToC:

$$
T_b^\infty
$$

definiert man ein ToC-Faser-Gitter:

$$
\mathcal I_{\mathrm{ToCGrid}} = \Gamma_\infty\times T_b^\infty
$$

Mit:

$$
x\in\Gamma_\infty, \qquad w\in T_b^\infty.
$$

Zwei Unendlichkeiten:

$$
\Gamma_\infty
$$

transversal und

$$
T_b^\infty
$$

intern pro Faser.

**Befund**

Vollidealer Sektor:

$$
\text{flat, homogeneous, reciprocal, internally ToC-scale-invariant}.
$$

Transversale Isotropie nur diskret bzw. abhängig von $\Gamma_\infty$.

**Obstruktions-Ort**

Das Gitter bringt transversale Nachbarschaft als neues IDEAL-Vergleichsdatum mit. Sie ist nicht aus einem einzelnen ToC abgeleitet.

**Status**

Sehr sinnvoller letzter ToC-naher Test vor Substratwechsel.

§§X180§§

§§X86§§23.2 Endlicher Doppelschnitt

**Ausgangslage**

Berechenbarer Sektor:

$$
\Omega_{R,L} = W_R\times T_{\le L}
$$

Mit:

$$
W_R\subset\Gamma_\infty, \qquad T_{\le L}\subset T_b^\infty.
$$

**Befund**

Subsystem-Sein bricht zwingend die IDEAL-Symmetrie:

$$
\mathrm{Aut}(\mathcal I_{\mathrm{ToCGrid}}) \to \mathrm{Aut}(\Omega_{R,L}).
$$

Es entstehen:

$$
\text{outer grid complement},
$$

$$
\text{internal UV-tail},
$$

$$
\text{edge/corner/mixed complements}.
$$

**Obstruktions-Ort**

Subsystem-Sein erzeugt effektive Rand-/Spektral-/DtN-Geometrie, aber nicht automatisch Chirotopie.

**Status**

Positiver Geometrie-/DtN-Test, negativer $J$-Test im flachen reziproken Fall.

§§X181§§

§§X87§§23.3 DtN auf dem ToC-Faser-Gitter

**Ausgangslage**

Operator auf $\Omega_{R,L}$:

$$
L_{R,L}.
$$

Schur/DtN:

$$
\Lambda_{R,L} = L_{\partial\partial} - L_{\partial I}L_{II}^{-1}L_{I\partial}
$$

**Befund**

Dies ist A→B-näher als rohe Knotenverklebung. B würde nicht ToC-Knoten sehen, sondern Handoff-Matrizen.

**Obstruktions-Ort**

Solange das Gitter homogen, reziprok und flach ist, entstehen zwar Spektrum und effektive Geometrie, aber keine ausgezeichnete Chirotopie.

**Status**

Wichtiger letzter Referenztest:

> **Kernaussage.** ToC-Faser-Gitter kann Geometrie testen, nicht J erzwingen.

§§X182§§

§§X88§§24. Holonomie-/Connection-Test im Faser-Gitter

§§X89§§24.1 Effektive Intertwiner zwischen lokalen Handoff-Räumen

**Ausgangslage**

Für lokale Handoff-Räume:

$$
H_x,\qquad H_y
$$

bräuchte man derived Intertwiner:

$$
U_{xy}:H_x\to H_y.
$$

Loop-Holonomie:

$$
U_\gamma = U_{wx}U_{zw}U_{yz}U_{xy}
$$

**Befund**

Im homogenen flachen Fall erwartbar:

$$
U_\gamma=I
$$

oder gauge-trivial.

**Obstruktions-Ort**

Ein nichttrivialer Rotationsanteil müsste aus Inhomogenität, Regulator, Backreaction oder Frustration kommen.

**Status**

Offener Curved-sector-Test. Noch nicht positiv gezeigt.

§§X183§§

§§X90§§25. Lorentz-/Zeitstruktur-Tests

§§X91§§25.1 Lorentz-Signatur

**Ausgangslage**

Signatur:

$$
\eta=\mathrm{diag}(-1,+1,\ldots,+1).
$$

**Befund**

Trennt zeitartig und raumartig.

**Obstruktions-Ort**

Zeitumkehr bleibt Symmetrie:

$$
T\eta T=\eta.
$$

Lichtkegel bleibt Doppelkegel:

$$
C^+\cup C^-.
$$

**Status**

Reduziert Problem auf Zeitorientierung, löst sie nicht.

§§X184§§

§§X92§§25.2 Reeller Zeitfluss-Vorläufer

**Ausgangslage**

Reell-symmetrischer Generator $H$, Flusspaar:

$$
{e^{+tH},e^{-tH}}.
$$

**Befund**

Liefert:

$$
\{+\tau,-\tau\}.
$$

**Obstruktions-Ort**

Für reell-symmetrisches $H$ bleibt jede spektrale Funktion symmetrisch. Ein $J$ ist antisymmetrisch:

$$
J\neq f(H).
$$

**Status**

Zeitpaar ja. Verriegelung mit $J$ nein.

§§X185§§

§§X93§§26. Pillar C / OQS / Entropie

§§X94§§26.1 Lindblad-/OQS-Zeitpfeil

**Ausgangslage**

Offene Quantendynamik / Lindblad-Generator.

**Befund**

Dissipation kann Zeitrichtung wählen:

$$
+\tau.
$$

**Obstruktions-Ort**

Hamiltonischer Term enthält bereits:

$$
-i[H,\rho].
$$

Also setzt OQS $i$ bzw. $J$ voraus.

**Status**

Pillar C kann $\tau$ wählen, aber $J$ nicht allein erzeugen.

§§X186§§

§§X95§§27. AQFT / Type-I / Type-III / Handoff-Struktur

§§X96§§27.1 A als Type-I-/Type-III-Vorläuferschicht

**Ausgangslage**

Pillar A soll nicht direkt Type III beweisen, sondern Vorläufer liefern:

$$
\mathcal C_{d,k} = (Q_{d,k}\oplus P_{d,k},g_{d,k},\omega_{d,k},\{J,-J\})
$$

Endlich:

$$
k<\infty \Rightarrow \text{Type-I-artige Vorläufer}.
$$

Unendlich:

$$
k\to\infty \Rightarrow \text{Type-III-fähige Komplementfamilien-Vorläufer}.
$$

**Befund**

Architektonisch sinnvoll.

**Obstruktions-Ort**

Dimension/Unendlichkeit liefert keine Orientierung:

$$
\text{finite/infinite}\neq J\text{-sign}.
$$

**Status**

Wichtiger Architekturshift.

§§X187§§

§§X97§§27.2 Triadischer Handoff (B|B'|C)

**Ausgangslage**

Handoffs sind nicht passive Pfeile, sondern eigene Interface-Objekte.

Triade:

$$
C\text{-Regulator} \triangleright H_{B|B'}(B,B') \to \text{stable record}.
$$

**Befund**

Bester Ort für:

$$
\omega_{\mathrm{lock}}.
$$

**Obstruktions-Ort**

Noch nicht formalisiert. Type-I/Type-III-Asymmetrie ist zunächst Algebra-/Dimensionsasymmetrie, nicht Orientierung.

**Status**

Weiterhin wichtigster offener $J$-Locking-Kandidat.

§§X188§§

§§X98§§28. Multi-ToC / Detektor / Vielobjektstruktur

Dieser Abschnitt darf nicht als Rückfall in die Lesart „ToC-Knoten sind Teilchen“ verstanden werden. Viele Objekte entstehen nicht durch viele Knoten innerhalb eines einzelnen ToC, sondern durch viele lokale ToC-Fasern, deren Approximanten und Handoff-Daten relativ zueinander verklebt werden.

$$
\{T_i\}_{i\in I} \Rightarrow \text{Multi-ToC-/Gluing-Struktur}, \qquad T_i\text{-Knoten}\neq\text{Teilchen}.
$$

§§X99§§28.1 Mini-ToCs als Detektorelemente

**Ausgangslage**

Ein Detektor besteht aus vielen lokalen ToC-Fasern:

$$
T_1,T_2,\ldots,T_N.
$$

Jede trägt lokal:

$$
{J_i,-J_i}.
$$

**Befund**

Lokales Vorzeichen kann Gauge sein:

$$
J_i\mapsto -J_i.
$$

Physikalisch relevant wären relative oder zyklische Daten:

$$
\sigma_{ij}, \qquad \Phi_\gamma=\prod_{(ij)\in\gamma}\sigma_{ij}.
$$

**Obstruktions-Ort**

Mechanismus für $\sigma_{ij}$ ist noch nicht derived. Außerdem wäre ein Zyklusprodukt zunächst eine relative, gauge-invariante Struktur, nicht automatisch ein absolutes $J$-Vorzeichen:

$$
\Phi_\gamma=\prod_{(ij)\in\gamma}\sigma_{ij} \quad\Rightarrow\quad \text{relative Orientierung},
$$

aber nicht unmittelbar

$$
\Rightarrow\text{absolute Orientierung}.
$$

**Status**

Starker Kandidat für nächsten nichtlokalen Test. Methodisch gilt:

$$
\text{relative Orientierung}\neq\text{absolute Orientierung}.
$$

§§X189§§

§§X100§§28.2 Frustration / Spin-netz-artige Struktur

**Ausgangslage**

Viele lokale ToC-Fasern werden gekoppelt. Mögliches Zyklusprodukt:

$$
\Phi_\gamma=-1.
$$

**Befund**

Falls $\Phi_\gamma$ invariant unter lokalen Gauge-Flips

$$
J_i\mapsto -J_i
$$

ist, entsteht echte globale Frustration.

**Obstruktions-Ort**

$\sigma_{ij}$ darf nicht gesetzt werden. Auch ein nichttriviales $\Phi_\gamma$ wäre zunächst eine globale Sektor-/Frustrationsstruktur. Es müsste zusätzlich gezeigt werden, dass daraus ein orientierter Record oder ein $\omega_{\mathrm{lock}}$ folgt, nicht nur eine relative Holonomieklasse.

**Status**

Wichtigster offener Multi-ToC-Testpfad. Positiv wäre hier zuerst eine gauge-invariante relative Struktur; das absolute $J$-Vorzeichen bliebe danach separat zu prüfen.

§§X190§§

§§X101§§29. Motor-/Mehrphasen-Analogie

§§X102§§29.1 Zweiphasiger Dreiphasenmotor

**Ausgangslage**

Zweiphasig erzeugt ein Dreiphasenmotor kein stabil gerichtetes Drehfeld, sondern Überlagerung:

$$
\text{Vorwärtsdrehfeld}+\text{Rückwärtsdrehfeld}.
$$

**Befund**

Gute Analogie zu:

$$
\{+J,-J\}.
$$

**Obstruktions-Ort**

Ohne dritte Phasenordnung bzw. Anschlussordnung kein stabiler Drehsinn.

**Status**

Didaktisch stark. Technische Lesart: Der reelle passive Dirichlet-/Widerstandssektor kann Imbalance, Achse und Pulsation erzeugen, aber keine eigenständige Phase. Die fehlende Rolle ist die eines abgeleiteten kapazitiven/speichernden/skew-Hamilton-artigen Sektors oder eines äquivalenten Handoff-Lockings.

§§X191§§

§§X103§§29.2 Drei Phasen / Anschlussordnung

**Ausgangslage**

Balanciertes System:

$$
(1,a,a^2), \qquad a=e^{2\pi i/3}.
$$

Vertauschung:

$$
(1,a,a^2) \leftrightarrow (1,a^2,a).
$$

**Befund**

Drehrichtung liegt in der Anschlussordnung.

**CNNA-Übersetzung**

Nicht lokales (J_i)-Vorzeichen, sondern Handoff-Sequenz bzw. Zyklusordnung könnte entscheidend sein.

**Obstruktions-Ort**

Anschlussordnung muss derived sein.

**Status**

Guter Kandidat für Multi-ToC-Handoff-Sequence-Gate.

§§X192§§

§§X104§§30. Cayley-Dickson / höhere Divisionsalgebren

§§X105§§30.1 CD-/Hurwitz-Kandidat

**Ausgangslage**

Route:

$$
\mathbb R\to\mathbb C\to\mathbb H\to\mathbb O.
$$

**Befund**

Für das erste $J$-Vorzeichenproblem negativ. Höhere Algebra löst nicht die Herkunft der ersten komplexen Orientierung.

**Obstruktions-Ort**

Dimensionsverdopplung und Normmultiplikativität werden nicht aus Schnittdaten erzwungen.

Offene Objekte:

```text
positiveDefiniteNormSq
divisionFromNormSq
alternativeLaw
```

**Status**

Nicht aktueller Weg für $J$-Vorzeichen. Als spätere Zielstruktur nicht ausgeschlossen.

§§X193§§

§§X106§§31. Substratwechsel-Kandidaten

§§X107§§31.1 ToC bleibt lokale Provenienzfaser

**Ausgangslage**

Der b-äre Einzelbaum als flacher ToC-Referenzsektor scheitert unter den flach-reziproken Derived-only-Prämissen am $J$-Gate. Damit ist nicht das ToC-Konzept insgesamt obstruiert, sondern nur die spezielle Lesart, dass ein einzelner b-ärer Baum globaler Träger des Universums und zugleich Ursprung einer ausgezeichnet gerichteten komplexen Struktur sein kann.

**Befund**

Als lokale Faser bleibt ToC wertvoll. Der präzise Rollenpfad lautet:

$$
\text{ToC-Knoten} \to \text{Provenienzindex} \to \text{Approximant} \to \text{Schur/DtN} \to \text{lokaler Handoff-Operator} \to \text{möglicher physikalischer Freiheitsgrad}.
$$

Ein endlicher Approximant ist daher zunächst ein effektiver lokaler Handoff-/Objektkandidat, kein automatisch gegebenes Vielteilchensystem.

**Obstruktions-Ort**

Globale Ontologie als einzelner Baum ist zu arm für zweite Achse, Chirotopie, Krümmung. Umgekehrt wäre die direkte Deutung von ToC-Knoten als physikalische Freiheitsgrade ein Rollenfehler.

**Status**

Kein Totalverwerfen des ToC und keine Falsifikation der Complement Net Architecture; Rollenwechsel:

> **Key statement.** A b-ary single tree is not a world tree, but a local provenance fiber.

On the contrary, the complement side remains structurally necessary as soon as local handoff operators, local algebras, relative complements, and subsequent AQFT connection conditions are taken seriously.

---

## 31.2 Event structures as a comparative structure, not a foundation

**Initial situation**

Event structures typically possess two relations:

$$
\leq \qquad\text{und}\qquad \#.
$$

Here, $\leq$ is not neutral as soon as it is interpreted as a causal or temporal order. The relation $\#$ denotes conflict, incompatibility, or exclusion.

**Findings**

Event structures are of interest as subsequent target or comparison structures. They could describe how emergent events, conflicts, and a causal order arise from a CNNA-derived pre-structure.

The permissible direction is therefore:

$$
\text{CNNA-derived nicht-kausale Vorstruktur} \longrightarrow \text{emergente Ereignisse} \longrightarrow (E,\leq,\#).
$$

**Obstruction-Location**

Event structures would be too strong as a foundation. The relation $\leq$ would already introduce causality or temporal order as primitive data. This would establish precisely what CNNA would first have to reconstruct.

The impermissible direction would be:

$$
(E,\leq,\#) \longrightarrow \text{CNNA-Fundament}.
$$

Methodologically, this would be the same type of import as:

$$
\text{komplexe Zahlen setzen},\qquad \text{Orientierung setzen},\qquad \text{Tensorprodukt setzen},\qquad \text{Hodge-Star setzen}.
$$

Only here, the imported content would be:

> **Core statement.** Establish causality.

**Status**

Event structures must be downgraded as the next candidate for a foundation. They remain target/comparison structures, but are not a valid substrate core prior to a derived causality reconstruction.

> **Core statement.** Event structures: comparison structure yes, foundation no.

## 31.3 Non-causal substrate-change gate

**Initial situation**

The b-ary single tree is falsified as a global world tree for the $J$-sector under the flat-reciprocal derived-only premises. It does not follow from this that arbitrarily richer relational substrates are admissible. A new substrate must not simply contain the missing target structures as primitive relations.

**Findings**

A permissible next substrate candidate must satisfy at least the following exclusions:

> **Core Statement.** no primitive i, · no primitive J, · no primitive chiropathy, · no primitive orientation, · no primitive tensor factorization, · no primitive causal order.

It may carry a non-causal relational, combinatorial, or topological pre-structure, provided that its subsequent causal interpretation is enforced only through handoff, regime formation, spectral structure, regulators, or backreaction.

**Obstruction Site**

Any substrate that already contains a directed time, causal, orientation, or phase structure bypasses the actual CNNA test. In that case, the missing second axis would not be derived, but imported.

**Status**

The strictest currently permissible intermediate step therefore remains the non-causal IDEAL-ToC fiber lattice as a flat reference test:

$$
\mathcal I_{\mathrm{ToCGrid}}=\Gamma_\infty\times T_b^\infty,\qquad \Omega_{R,L}=W_R\times T_{\le L}.
$$

Here, $\Gamma_\infty$ is merely a homogeneous relational index carrier, not yet spacetime and not yet a causal order. Any metric, spatial, directed, or oriented interpretation of $\Gamma_\infty$ is a comparison/test structure and not ontological input.

---

## 31.4 Sierpinski carpet as a non-p.c.f. stress class

**Initial situation**

The Sierpinski carpet is more interesting than SG/ST as a non-p.c.f. stress class when testing multiscale boundary/trace structures. The Menges sponge is not pursued further in this version.

**Findings**

Non-p.c.f. structure means: wilder, multiscale intersection and boundary contacts are possible. This can be useful for handoff, trace, gluing, and frustration tests:

$$
\text{nicht-p.c.f.} \Rightarrow \text{wildere, mehrskalige Boundary/Trace-Struktur}.
$$

**Obstruction Location**

However, more holes or wilder boundary structures do not automatically result in a derived-only orientation:

$$
\text{mehr Löcher} \neq \text{derived }J\text{-Vorzeichen}.
$$

In particular, it remains to be checked whether every loop, area, trace, or Hodge-like structure used truly arises from the non-causal pre-structure or was imported via embedding/orientation.

**Status**

Meaningful substrate stress class, but no current foundation candidate and no solution to the $J$-sign problem.

---

# 32. Identified artifact locations in this version

This version only lists artifacts that have either been appended or are explicitly referenced as Hugging Face visualizations. Older package names, non-attached post-tests, and hypothetical future implementations are no longer listed as a reproducible artifact basis for this file.

## 32.1 Hugging Face visualization

```text
Hugging-Face-Space: https://huggingface.co/spaces/antaris/b-ary_tree
app.py
```

The visualization serves to illustrate the ToC/Approximant/UV/Environment concept. It is itself merely a proxy and representation layer; tilt, angle, or chart values derived from it should not be interpreted as Schur/DtN invariants.

## 32.2 Appendix `cnna_alpha_orth_invariant_v7(1).zip`

```text
cnna_alpha_orth_invariant_v7/alpha_orth_invariant.py
cnna_alpha_orth_invariant_v7/two_approximant_flow_sign.py
cnna_alpha_orth_invariant_v7/two_boundary_shell_chirality.py
cnna_alpha_orth_invariant_v7/family_handoff_chirality.py
cnna_alpha_orth_invariant_v7/triadic_interface_chirality.py
cnna_alpha_orth_invariant_v7/oriented_cauchy_shell_gate.py
```

The appendix also contains associated CSV, JSON, PNG, and Markdown reports. These artifacts form the documented reproducible basis for the $\alpha_{\mathrm{orth}}$-, Flow-Sign-, Cauchy-Shell-, Family-Handoff-, Triadic Interface-, and UV/Env-Cauchy-Shell findings of this version.

## 32.3 Appendix `files(1).zip`

```text
F9_H1_test_zusammenfassung.md
build_structures.py
build_gasket.py
generator_test.py
h1_tests.py
```

This appendix documents the Baum vs. Sierpinski Gasket control test: Baum as the $b_1=0$ control group, Gasket as a non-trivial $H_1$ stress case, generative $\kappa$-blindness test, and $H_1$-dynamic test.

---

# 33. Obstruction Locations by Type

## 33.1 Reciprocity

$$
\Lambda=\Lambda^T.
$$

Passive Schur/DtN reduction remains symmetric. No antisymmetric $J$-generator.

## 33.2 Real conjugation symmetry

$$
J\mapsto -J.
$$

Real structures do not choose a complex orientation.

## 33.3 S_b-equivariance

Sibling permutations preserve canonical quantities in the trivial sector. No sign choice.

## 33.4 Radial Uniaxial Structure (F1)

F1 provides order:

$$
n\to n+1.
$$

But only along one axis. Noncommutativity requires two independent axes.

## 33.5 Degeneracy

Degenerate eigenspaces must not be cut by an arbitrary numerical basis. Only entire clusters are relabeling-natural.

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

## 33.10 OQS Dependence on i

Lindblad/OQS can provide the direction of time, but requires Hamiltonian $i$.

## 33.11 Flatness

Missing in the flat ToC/DtN sector:

$$
\text{Connection}, \qquad \text{Holonomie}, \qquad \text{Krümmung}.
$$

## 33.12 Causality import

A primitive causal order $\leq$ is not a neutral structural carrier. It would already incorporate a temporal/causal structure and thus bypass the subsequent reconstruction step.

> **Key point.** (E,≤,\#) is the target structure, not the foundation.

The permissible test is therefore not whether a causal substrate can carry CNNA, but whether CNNA can generate a causal order from a non-causal pre-structure.

---

# 34. Current overall formula

> **Key statement.** All single-tree, single-approximant, passive Schur/DtN, and local triad tests end at {J,-J}.

> **Key statement.** Exact and cluster-safe handoff operators in the flat ToC/DtN sector commute.

> **Key statement.**Non-commutativity arises so far only through imposed order, non-canonical truncation, or asymmetric boundary conditions.

> **Key statement.** ToC nodes are provenance indices, not physical degrees of freedom.

> **Key statement.** The b-ary tree was chosen as the provenance side of SG/ST: SG↔ b=3, · ST↔ b=4.

> **Key statement.** What is obstructed is not CNNA and not ToC in general, but the b-ary single tree as a global carrier of directed complex structure.

> **Key statement.** Complementary, handoff, and local algebra structures remain positively relevant for the AQFT connection.

> **Key statement.** UV/Env generate a genuine radial scale break, but no chirality.

> **Key statement.** \omega_\partial⇒{+J,-J}, · \omega_{lock} remains the open locking object.

> **Key point.** Relative holonomy/frustration is not automatically absolute orientation.

> **Key statement.** The next genuine positive search space is not another flat single-ToC test, but rather curved-sector, multi-ToC frustration, or triadic handoff locking.

The most important next ToC-related test before substrate change remains:

$$
\mathcal I_{\mathrm{ToCGrid}}=\Gamma_\infty\times T_b^\infty, \qquad \Omega_{R,L}=W_R\times T_{\le L}, \qquad \Lambda_{R,L}.
$$

Goal:

$$
\text{effektive Geometrie aus Subsystem-Sein testen},
$$

but separate from that:

$$
\text{J-/Chirotopie-/Nichtkommutativitäts-Gate weiter offen halten}.
$$
