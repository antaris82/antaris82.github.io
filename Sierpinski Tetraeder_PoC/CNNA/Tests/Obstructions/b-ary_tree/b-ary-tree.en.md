# CNNA-ToC / binary tree / J-sign — renderer-safe version

Status: consolidated, renderer-safe Markdown version based on the current state of the discussion. Do not read as a lean theorem. The findings are numerical, conceptual, or derived from the reported diagnostic artifacts.

> **Key statement.** The flat, real-valued, reciprocal ToC/Schur/DtN sector does not generate a distinguished J-orientation. It repeatedly yields the pair {+J,-J}, but not J instead of -J.

---

# 0. Why the b-ary tree was chosen

The b-ary tree was not introduced as an arbitrary world tree. The historical motivation was the tractability, symmetry, and scale invariance of the Sierpinski Gasket (SG) and the Sierpinski Tetrahedron/Tetrix (ST). These objects form a controlled, self-similar, and highly symmetric test class.

The corresponding provenance page for SG and ST is a b-ary address tree. For the Sierpinski Gasket, the natural address tree is 3-ary; for the Sierpinski Tetrahedron, it is 4-ary:

$$
SG: b=3, \qquad ST: b=4.
$$

Prior to geometric embedding, quotient relation, orientation, or Hodge structure, there is the pure address and provenance structure

$$
A_b^{§§X32§§ **Kernaussage.** b-ärer Einzelbaum = globaler Träger des Universums und einer ausgezeichnet gerichteten komplexen Struktur.

Der positive Rollenwechsel lautet:

> **Kernaussage.** Der b-äre Einzelbaum ist nicht Weltbaum, sondern lokale Provenienzfaser.

§§X23§§

§§X2§§1. Grunddefinitionen des flachen ToC-Sektors

Fixiere

$$
b\ge 2
$$

und das Adressalphabet

$$
A_b=\{0,\ldots,b-1\}.
$$

Die Elemente von A_b sind zunächst nur Adresssymbole. Sie tragen keine physikalische Ordnung, keine zyklische Ordnung und keine Orientierung.

Der unendliche b-äre ToC ist der Wortbaum

$$
T_b^\infty=A_b^{<\omega}=\bigcup_{n\ge 0} A_b^n.
$$

Die Wurzel ist das leere Wort

$$
\varnothing\in A_b^0.
$$

Für Wörter u und v bezeichnet uv die Wortverkettung. Für i in A_b ist wi das Wort, das aus w durch Anhängen von i entsteht.

Die Tiefe eines Knotens w ist die Wortlänge

$$
|w|.
$$

Für w ungleich der Wurzel ist pi(w) der Parent-Knoten. Die Kindermenge von w wird mit C_b(w) bezeichnet:

$$
C_b(w)=\{wi:i\in A_b\}.
$$

Die natürliche Provenienzordnung ist die Präfixordnung

$$
u\preceq v \quad\Longleftrightarrow\quad \exists r\in A_b^{<\omega}: v=ur.
$$

Die ungerichtete Baumkante ist

$$
x\sim y \quad\Longleftrightarrow\quad x=\pi(y)\ \text{oder}\ y=\pi(x).
$$

Der bare ToC-Graph ist

$$
G_b^\infty=(T_b^\infty,E_b^\infty),
$$

mit

$$
E_b^\infty=\{\{w,wi\}:w\in T_b^\infty,\ i\in A_b\}.
$$

Jede bare Kante hat Gewicht 1. Es gibt im baren Sektor keine eingebettete Geometrie, keine Winkel, keine Längen außer graph distance, keine Orientierung, keine komplexe Struktur, keine Zeit und keine Kausalordnung. Die einzige bare Abstandsgröße ist die graph distance

$$
d_G(x,y).
$$

Insbesondere gilt

$$
d_G(\varnothing,w)=|w|.
$$

Knoten von T_b^infty sind keine physikalischen Freiheitsgrade. Sie sind Adress- und Provenienzindizes.

Der zulässige Lesepfad ist:

$$
\text{ToC-Knoten}\to\text{Provenienzindex}\to\text{Approximant}\to\text{Schur/DtN}\to\text{effektiver Handoff-Operator}.
$$

Nicht zulässig ist:

$$
\text{ToC-Knoten}=\text{physikalischer Freiheitsgrad}.
$$

§§X24§§

§§X3§§2. Approximanten, Bright, Dark und Interface

Für einen Anchor a in T_b^infty mit

$$
k=|a|
$$

und eine Tiefe L ist der endliche Approximant

$$
\Omega(a,L)=\{av:v\in A_b^{\le L}\}.
$$

Die Levelmengen sind

$$
\Omega_\ell(a,L)=\{av:v\in A_b^\ell\},\qquad 0\le \ell\le L.
$$

Die Knotenanzahl ist

$$
|\Omega(a,L)|=1+b+\cdots+b^L=\frac{b^{L+1}-1}{b-1}.
$$

Der Bright-Sektor ist Omega(a,L). Der Dark-Sektor ist das Komplement relativ zum unendlichen ToC. Er zerfällt schnittrelativ in den UV-tail an den Blättern und, falls k größer 0 ist, in den Environment-Anteil auf der Parent-Seite.

Die UV-Boundary ist

$$
\partial_{\mathrm{UV}}\Omega=\Omega_L(a,L).
$$

Für k=0 gilt noOuterEnvironment. Für k größer 0 gibt es einen rootseitigen Environment-Port am Approximantenroot a.

Ein endlicher Approximant ist nicht automatisch ein Vielteilchensystem. Er ist zunächst ein effektiver lokaler Handoff- oder Objektkandidat:

$$
\Omega(a,L)\Rightarrow\text{effektiver lokaler Handoff-/Objektkandidat}.
$$

Viele Objekte, Detektoren oder Vakuum-Gluing-Strukturen entstehen erst aus Familien lokaler Fasern und deren Verklebungen:

$$
\{T_i\}_{i\in I}\Rightarrow\text{Multi-ToC-/Gluing-Struktur}.
$$

§§X25§§

§§X4§§3. Laplace-, Schur- und DtN-Konvention

Der Bright-Laplaceoperator L_Omega ist der Laplaceoperator des induzierten Bright-Graphen. Komplementzweige werden nicht in der Bright-Degree mitgezählt.

$$
(L_\Omega)_{xy}=\begin{cases}
d_\Omega(x),&x=y,\\
-1,&x\sim y\ \text{innerhalb von }\Omega,\\
0,&\text{sonst}.
\end{cases}
$$

Alle entfernten Komplementanteile werden ausschließlich durch Schur-, DtN- oder Load-Terme ergänzt. Dadurch wird Doppelzählung vermieden.

Ein UV-cut oder Environment-cut wirkt bereits wie eine Dirichlet-artige Randsetzung. Die Regularisierung ist schnittintern:

$$
\text{UV-cut oder Environment-cut}\Rightarrow\text{Dirichlet-Boundary}\Rightarrow L_{II}^{-1}\text{ wohldefiniert}.
$$

Es wird kein externer Ridge-Term, keine Pseudoinverse und kein künstlicher Massenterm gesetzt.

Der effektive Operator hat die Form

$$
M_\Omega=L_\Omega+\Sigma_{\mathrm{Env}}+\Sigma_{\mathrm{UV}}.
$$

Im einfachsten load-basierten Proxy kann man schreiben

$$
\Sigma_{\mathrm{Env}}=\sigma_{\mathrm{Env}}P_{\mathrm{root}},\qquad
\Sigma_{\mathrm{UV}}=\sigma_{\mathrm{UV}}P_{\partial_{\mathrm{UV}}\Omega}.
$$

Diese Form gilt nur dann als derived, wenn die Werte aus einer expliziten Schur-/DtN-Eliminierung der Komplementfamilien stammen. Frühe Konstanten- oder Ladder-Modelle sind Diagnosemodelle, keine ontischen CNNA-Eingaben.

Die beiden Loads wirken an entgegengesetzten Seiten des Approximanten:

$$
\Sigma_{\mathrm{UV}}\text{ wirkt leafseitig},\qquad \Sigma_{\mathrm{Env}}\text{ wirkt rootseitig}.
$$

Das erzeugt einen echten inneren Skalenbruch des Approximanten. Dieser Skalenbruch ist aber zunächst radial bzw. longitudinal:

$$
\text{Skalenbruch}\ne\text{Chiralität}.
$$

§§X26§§

§§X5§§4. Kanalantworten und Orthogonalitätsdiagnostik

Für Kanalquellen f_Env und f_UV sind die Antworten

$$
u_{\mathrm{Env}}=M_\Omega^{-1}f_{\mathrm{Env}},\qquad
u_{\mathrm{UV}}=M_\Omega^{-1}f_{\mathrm{UV}}.
$$

Das Energie-Innenprodukt ist

$$
\langle x,y\rangle_M=x^T M_\Omega y.
$$

Die Orthogonalitätsdiagnose ist

$$
\rho_M=\frac{\langle u_{\mathrm{Env}},u_{\mathrm{UV}}\rangle_M}{\|u_{\mathrm{Env}}\|_M\|u_{\mathrm{UV}}\|_M}.
$$

Die Größen alpha_UV, alpha_Env, C_k und Xi sind Diagnosegrößen, solange sie nicht aus vollständigen Komplementfamilien abgeleitet sind. C_k ist in den frühen Tests eine schnitt- bzw. tiefenabhängige Normierungs-/Kapazitätsgröße des Approximanten und keine universale CNNA-Konstante.

§§X27§§

§§X6§§5. J-Vorzeichenproblem

Eine komplexe Struktur auf einem reellen Handoff-Raum ist ein Endomorphismus J mit

$$
J^2=-I.
$$

Das J-Vorzeichenproblem ist nicht die bloße Existenz eines solchen Blocks, sondern die derived-only-Auswahl von J gegenüber -J.

Eine reelle, symmetrische und relabeling-natürliche Struktur liefert höchstens

$$
\{+J,-J\}.
$$

Gesucht wäre dagegen ein Locking-Objekt, das J mit einer Fluss-, Zeit- oder Handoff-Orientierung tau koppelt:

$$
\omega_{\mathrm{lock}}:(J,\tau)\mapsto\text{stabiler orientierter Record}.
$$

Die Cauchy-Randform der UV/Env-Shell ist ein positives Vorläuferobjekt, liefert aber weiterhin nur das Paar {+J,-J}. Deshalb gilt:

$$
\omega_\partial\Rightarrow\{+J,-J\},\qquad \omega_{\mathrm{lock}}\text{ bleibt offen}.
$$

§§X28§§

§§X7§§6. Zentrale Obstruktionsorte

§§X8§§6.1 Reziprozität

Passive Schur-/DtN-Reduktion bleibt symmetrisch:

$$
\Lambda=\Lambda^T.
$$

Daraus entsteht kein antisymmetrischer J-Generator.

§§X9§§6.2 Reelle Konjugationssymmetrie

Reelle Strukturen wählen keine komplexe Orientierung:

$$
J\mapsto -J.
$$

§§X10§§6.3 S_b-Äquivarianz

Geschwisterpermutationen halten kanonische Größen im trivialen Sektor. Eine Signum-Auswahl wird nicht kanonisch erzeugt.

§§X11§§6.4 Radiale Einachsenstruktur

F1 ist die radiale Provenienz- und Tiefenachse. Sie liefert Ordnung entlang der Tiefe, aber keine transversale Händigkeit.

$$
\text{F1 allein}\Rightarrow\text{Achse, aber keine Chiralität}.
$$

Nichtkommutativität braucht mindestens zwei nicht gemeinsam diagonalisierbare abgeleitete Operatorachsen. Chiralität braucht zusätzlich eine abgeleitete Orientierungs- oder Sign-Line-Auswahl.

§§X12§§6.5 Degenerazien

Entartete Eigenräume dürfen nicht durch eine willkürliche numerische Basis geschnitten werden. Nur ganze Spektralcluster sind relabeling-natürlich. Kommutatorsignale aus Schnitten mitten durch entartete Räume sind Symmetriebruch durch numerische Basiswahl, kein ToC-derived Mechanismus.

§§X13§§6.6 Keine partielle Spur auf Knoten

Eine Knotenzerlegung ist eine direkte Summe:

$$
\mathbb R^N=\mathbb R^S\oplus\mathbb R^E.
$$

Sie ist kein Tensorprodukt. Daher ist Knoten-Elimination keine partielle Spur. Die korrekte Knotenreduktion ist Schur-Eliminierung.

§§X14§§6.7 OQS-Abhängigkeit von i

Lindblad-/OQS-Strukturen können eine Zeitrichtung stabilisieren, setzen aber im Hamilton-Term bereits i voraus:

$$
-i[H,\rho].
$$

Sie können daher J nicht ursprünglich erzeugen.

§§X15§§6.8 Flachheit

Im flachen ToC-/DtN-Sektor fehlen Connection, Holonomie und Krümmung:

$$
\text{Connection},\qquad \text{Holonomie},\qquad \text{Krümmung}.
$$

Nichtkommutativität könnte später eher aus Curved-sector-, Handoff-, Regulator- oder Backreaction-Daten entstehen. Sie darf aber nicht importiert werden.

§§X29§§

§§X16§§7. SG, ST und Sierpinski-Teppich

SG und ST sind nützliche Vergleichs- und Stressstrukturen, aber nicht identisch mit dem baren ToC. Sie enthalten IFS-, Quotient- oder Einbettungsstruktur, die nicht rückwirkend in den flachen Baum importiert werden darf.

Für SG/ST kann man Schleifen, Zellstruktur und mehr Randdaten sehen. Aber auch dort gilt:

$$
\text{mehr Struktur}\ne\text{kanonisches J-Vorzeichen}.
$$

Der Sierpinski-Teppich ist als nicht-p.c.f.-Stressklasse interessant, weil er wildere mehrskalige Boundary- und Trace-Strukturen besitzt. Der Mengerschwamm wird in dieser Fassung nicht weiterverfolgt.

Auch für den Teppich gilt:

$$
\text{mehr Löcher}\ne\text{derived J-Vorzeichen}.
$$

§§X30§§

§§X17§§8. Artefaktlage

Diese Fassung nennt nur Artefakte, die angehängt wurden oder als Hugging-Face-Visualisierung ausdrücklich referenziert sind.

§§X18§§8.1 Hugging-Face-Visualisierung

```text
https://huggingface.co/spaces/antaris/b-ary_tree
app.py
```

Die Visualisierung dient der Anschauung des ToC-/Approximanten-/UV-/Environment-Konzepts. Tilt-, Winkel- oder Chartwerte daraus sind keine Schur-/DtN-Invarianten.

§§X19§§8.2 Anhang cnna_alpha_orth_invariant_v7(1).zip

```text
alpha_orth_invariant.py
two_approximant_flow_sign.py
two_boundary_shell_chirality.py
family_handoff_chirality.py
triadic_interface_chirality.py
oriented_cauchy_shell_gate.py
```

Diese Artefakte bilden die ausgewiesene reproduzierbare Basis für die alpha_orth-, Flow-Sign-, Cauchy-Shell-, Familien-Handoff-, triadischen Interface- und UV/Env-Cauchy-Shell-Befunde.

§§X20§§8.3 Anhang files(1).zip

```text
F9_H1_test_zusammenfassung.md
build_structures.py
build_gasket.py
generator_test.py
h1_tests.py
```

Dieser Anhang dokumentiert den Baum-vs.-Sierpinski-Gasket-Kontrolltest: Baum als b1=0-Kontrollgruppe, Gasket als nichttrivialer H1-Stressfall, generatorischer kappa-Blindheitstest und H1-Dynamiktest.

§§X31§§

§§X21§§9. Aktuelle Gesamtformel

> **Key result.** All single-tree, single-approximant, passive Schur/DtN, and local triad tests terminate at {+J,-J}.

> **Key statement.** Exact and cluster-safe handoff operators in the flat ToC/DtN sector commute.

> **Key statement.** So far, non-commutativity arises only from imposed order, non-canonical truncation, or asymmetric boundary conditions.

> **Key statement.** ToC nodes are provenance indices, not physical degrees of freedom.

> **Key statement.** The b-ary tree was chosen as the provenance side of SG/ST: SG corresponds to b=3, ST corresponds to b=4.

> **Key statement.** What is obstructed is not CNNA and not ToC in general, but the b-ary single tree as a global carrier of directed complex structure.

> **Key statement.** Complement, handoff, and local algebra structures remain positively relevant for the AQFT connection.

> **Key statement.** UV/Env generate a genuine radial scale break, but no chirality.

The most important upcoming ToC-related test before the substrate change remains the non-causal IDEAL ToC fiber lattice:

$$
\mathcal I_{\mathrm{ToCGrid}}=\Gamma_\infty\times T_b^\infty.
$$

with finite double-slit

$$
\Omega_{R,L}=W_R\times T_{\le L}.
$$

The goal is to test effective geometry from the perspective of subsystem status, while leaving the J-gate, the chiral topology gate, and the noncommutativity gate open.
