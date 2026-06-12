# CNNA-ToC / $J$-Prefix / Non-commutativity — Complete Test and Obstruction Inventory



Jan Seeck, ChatGPT 5.5, Claude Opus 4.8

June 12, 2026




Most findings are numerical, conceptual, or derived from the diagnostic artifacts presented here. The central result is now more precise than at the outset:

This version additionally includes the substrate gate from the revised paper version: Event structures are no longer treated as permissible foundation candidates because they would already contain a causal or temporal order as primitive data via $\\leq$. They remain only as comparison or target structures.

$$
\\boxed{
\\text{Der flache, reellwertige, reziproke ToC-/Schur-/DtN-Sektor erzeugt keine ausgezeichnete }J\\text{-Orientierung.}
}
$$

It delivers multiple times:

$$
{+J,-J},\\qquad {+\\tau,-\\tau},\\qquad \\text{radiale Ordnung},\\qquad \\text{DtN-/Spektralstruktur}.
$$

It has not yet delivered:

$$
\\boxed{
J\\text{ statt }-J.
}
$$

The uniform obstruction is now no longer just “symmetry,” but more precisely:

$$
\\boxed{
\\text{Eine abgeleitete Achse }F1\\text{ genügt nicht. Nichtkommutativität braucht mindestens zwei nicht gemeinsam diagonalisierbare abgeleitete Operatorachsen; Chiralität braucht zusätzlich eine abgeleitete Orientierungs- bzw. Sign-Line-Auswahl.}
}
$$

\---

# 0\. Global status of the test series

## 0.0 Why the b-ary tree was chosen as the ToC reference substrate

The b-ary tree was not introduced as an arbitrary world tree. The historical motivation was the tamability, symmetry, and scale invariance of the Sierpinski Gasket (SG) and the Sierpinski Tetrahedron/Tetrix (ST). These objects were attractive because they form a controlled, self-similar, and highly symmetric test class. It was precisely this tameness that was methodologically important: if even the most symmetric and controlled candidate does not enforce the desired direction of $J$, then the obstruction lies not in numerical wildness, but in the structure of the flat real-reciprocal sector.

The corresponding provenance page of SG and ST is a binary address tree. For the Sierpinski gasket, the natural address tree is ternary; for the Sierpinski tetrahedron, it is quaternary:

$$
SG:\\quad b=3,\\qquad ST:\\quad b=4.
$$

Underlying every geometric embedding, every quotient relation, and every orientation is the pure address/provenance structure

$$
A\_b^{§§X146§§0$, in den Environment-Anteil auf der Parent-/Root-Seite. Für $k=0$ gilt die noOuterEnvironment-Lesart.

Der Environment-Port ist, falls $k>0$, der rootseitige Interface-Port am Approximantenroot $a$. Er ist kein zusätzlicher Bright-Knoten, sondern die Schnittstelle zur äußeren Komplementseite.

Auch der Approximant selbst hat zunächst eine Rollenbegrenzung: $\\Omega(a,L)$ ist kein automatisch interpretiertes Vielteilchensystem. Im flachen ToC-Sektor ist er ein schnittrelativer lokaler Handoff-/Objektkandidat. Erst die aus ihm erzeugten Schur-/DtN-Daten und spätere Gluing-/Regimebildungen können physikalische Freiheitsgrade oder Vielobjektstruktur tragen.

§§X13§§0.5.5 Bright-Laplaceoperator und Komplement-Loads

Der Bright-Laplaceoperator $L\_\\Omega$ ist der Laplaceoperator des induzierten Bright-Graphen $G\_\\Omega$:

$$
(L\_\\Omega)\_{xy}
===

\\begin{cases}
\\deg\_\\Omega(x), \& x=y,\\
-1, \& x\\sim y\\text{ within }\\Omega,\\
0, \& \\text{otherwise}.
\\end{cases}
$$

Dabei ist

$$
\\deg\_\\Omega(x)=|{y\\in\\Omega:x\\sim y}|
$$

und zählt nur Nachbarn innerhalb von $\\Omega$. Komplementzweige werden nicht in $\\deg\_\\Omega$ mitgezählt. Ihre Wirkung wird ausschließlich über Schur-/DtN-/Load-Terme ergänzt. Dadurch wird eine Doppelzählung von Außenkanten vermieden.

Ein UV-cut oder Environment-cut ist bereits eine Dirichlet-artige Randsetzung. Die Schur-/DtN-Eliminierung wird daher nicht durch eine externe numerische Regularisierung stabilisiert, sondern durch den schnittrelativen Boundary-Status selbst. Die Regularisierung ist schnittintern:

$$
\\text{UV-cut or environment-cut}
\\Rightarrow
\\text{Dirichlet boundary}
\\Rightarrow
L\_{II}^{-1}\\text{ is well-defined},
$$

sofern der betrachtete Innenblock tatsächlich an die gesetzte Boundary gekoppelt ist. Externe Hilfssetzungen wie Ridge-Terme, Pseudoinversen oder künstliche Massenterme gehören nicht zum flachen derived-only ToC-/DtN-Kern.

Der effektive Operator hat die Form

$$
M\_\\Omega=L\_\\Omega+\\Sigma\_{\\mathrm{Env}}+\\Sigma\_{\\mathrm{UV}}.
$$

Im einfachsten load-basierten Proxy kann man schreiben

$$
\\Sigma\_{\\mathrm{Env}}
===

\\sigma\_{\\mathrm{Env}},P\_{\\mathrm{root}},
\\qquad
\\Sigma\_{\\mathrm{UV}}
===

\\sigma\_{\\mathrm{UV}},P\_{\\partial\_{\\mathrm{UV}}\\Omega},
$$

wobei dies nur dann als derived gilt, wenn die Werte aus einer expliziten Schur-/DtN-Eliminierung der jeweiligen Komplementfamilien stammen. Frühe Konstanten- oder Ladder-Modelle für $\\sigma\_{\\mathrm{Env}}$ bzw. $\\alpha\_{\\mathrm{Env}}$ sind Diagnosemodelle, keine ontischen CNNA-Eingaben.

Die beiden Loads wirken an entgegengesetzten Seiten des Approximanten:

$$
\\Sigma\_{\\mathrm{UV}}\\text{ acts on the leaf side at the finest/cut nodes},
\\qquad
\\Sigma\_{\\mathrm{Env}}\\text{ acts on the root side at the parent/environment port}.
$$

Damit erzeugt der Schnitt einen echten inneren Skalenbruch des Approximanten. Dieser Skalenbruch ist jedoch zunächst radial bzw. longitudinal:

$$
\\text{UV/Env scale breaking}\\≠\\text{chirality}.
$$

Für Kanalquellen $f\_{\\mathrm{Env}}$ und $f\_{\\mathrm{UV}}$ sind die Antworten

$$
u\_{\\mathrm{Env}}=M\_\\Omega^{-1}f\_{\\mathrm{Env}},
\\qquad
u\_{\\mathrm{UV}}=M\_\\Omega^{-1}f\_{\\mathrm{UV}}.
$$

Standarddiagnostisch ist $f\_{\\mathrm{Env}}$ eine rootseitige Quelle am Environment-Port und $f\_{\\mathrm{UV}}$ eine symmetrische bzw. normierte Blattquelle auf $\\partial\_{\\mathrm{UV}}\\Omega$. Jede abweichende Normierung muss im jeweiligen Artefakt explizit dokumentiert werden.

Das Energie-Innenprodukt ist

$$
\\langle x,y\\rangle\_M=x^TM\_\\Omega y.
$$

Die Orthogonalitätsdiagnose ist

$$
\\rho\_M
===

\\frac{\\langle u\_{\\mathrm{Env}},u\_{\\mathrm{UV}}\\rangle\_M}
{|u\_{\\mathrm{Env}}|*M|u*{\\mathrm{UV}}|\_M}.
$$

Die Größen $\\alpha\_{\\mathrm{UV}}$, $\\alpha\_{\\mathrm{Env}}$, $C\_k$ und $\\Xi$ sind Diagnosegrößen, solange sie nicht aus den vollständigen Komplementfamilien abgeleitet sind. In den frühen Tests bedeutet $C\_k$ eine schnitt- bzw. tiefenabhängige Normierungs-/Kapazitätsgröße des Approximanten; ihr genauer Wert ist artefakt- bzw. diagnostikabhängig und daher nicht als universale CNNA-Konstante zu lesen.

§§X14§§0.5.6 $J$-Problem, F1/F2 und Locking-Objekt

Eine komplexe Struktur auf einem reellen Handoff-Raum ist ein Endomorphismus $J$ mit

$$
J^2=-I.
$$

Das $J$-Vorzeichenproblem ist nicht die bloße Existenz eines solchen Blocks, sondern die derived-only-Auswahl von $J$ gegenüber $-J$. Eine reelle, symmetrische, relabeling-natürliche Struktur liefert daher höchstens

$$
{+J,-J},
$$

solange keine zusätzliche abgeleitete Orientierungs- oder Locking-Struktur vorliegt.

$F1$ bezeichnet die radiale Provenienz-/Tiefenachse

$$
|w|\\mapsto |w|+1.
$$

Eine zweite Achse $F2$ ist kein Input, sondern ein offenes Zielobjekt: eine unabhängig abgeleitete transversale Struktur, die nicht durch volle $S\_b$-Symmetrie trivialisiert wird.

$\\omega\_{\\mathrm{lock}}$ bezeichnet die noch offene Handoff-Form, die eine $J$-Orientierung mit einer Fluss-/Zeit-/Handoff-Orientierung $\\tau$ koppeln müsste. Sie ist nicht identisch mit einer bloßen Cauchy-Randform, solange diese nur

$$
{+J,-J}
$$

liefert. Die Cauchy-Shell kann also positiv sein, ohne das eigentliche Locking-Problem zu lösen:

$$
\\omega\_\\partial\\Rightarrow{+J,-J},
\\qquad
\\omega\_{\\mathrm{lock}}:(J,\\tau)\\mapsto\\text{stable oriented record}.
$$

\---

§§X15§§1\. Didaktische und Proxy-Tests

§§X16§§1.1 Hugging-Face-ToC-Concept-Explorer

**Artefaktbezug**

§§X147§§

§§X166§§ ist das Visualisierungsskript des Hugging-Face-Spaces. Es dient nur der Anschauung und nicht als Beweis- oder Primärdiagnostik.

**Ausgangslage**

Visualisierung eines (b)-ären ToC mit Parametern:

$$
b,\\qquad L\_{\\max},\\qquad \\text{approximant root},\\qquad L.
$$

Dargestellte Stufen:

$$
\\text{ToC}
\\to
\\text{proper subsystem}
\\to
\\text{UV-tail}
\\to
\\text{Environment}
\\to
\\text{Cauchy–}J\\text{-candidate}
\\to
\\text{Complex-plane overlay}.
$$

**Befund**

Didaktisch stark. Es trennt sichtbar:

$$
\\text{Approximant},
\\qquad
\\text{UV-tail},
\\qquad
\\text{Environment},
\\qquad
\\text{Interface}.
$$

**Obstruktions-Ort**

Visualisierung ist kein Beweis. Frühe Tilt-/Winkelwerte waren teilweise Chart-/Rendering-Proxies, nicht DtN-Invarianten.

**Status**

Didaktisch wertvoll, mathematisch sekundär.

\---

§§X17§§1.2 Stage-6 Chart-Proxy / Tilt-Test

**Artefaktbezug**

Teil der Hugging-Face-Visualisierung §§X167§§; nur Anschauungs- und Proxyebene.

**Ausgangslage**

Tiefe Einbettung von Approximanten, z. B.

$$
0.1,\\qquad 0.1.1,\\qquad 0.1.1.0,\\ldots
$$

bei festen Parametern wie:

$$
b=3,\\qquad L\_{\\max}=4.
$$

**Befund**

Visueller Tilt wurde mit tieferer Einbettung kleiner:

$$
|\\mathrm{tilt}|\\downarrow.
$$

**Interpretation**

Tiefer eingebettete Approximanten wirkten balancierter zwischen UV und Env.

**Obstruktions-Ort**

Kein echter Schur-/DtN-Wert:

$$
\\text{Proxy} \\neq \\text{Invariance}.
$$

**Status**

Heuristische Motivation; später durch echte DtN-/Schur-Tests ersetzt.

\---

§§X18§§2\. Einzel-Approximant-Schur-/DtN-Tests

§§X19§§2.1 Projected-tail $J$-/Rotationstest

**Artefaktbezug**

Kein eigenständiger angehängter Artefakt in dieser Fassung; der Abschnitt bleibt als konsolidierter Befund aus der späteren $\\alpha\_{\\mathrm{orth}}$- und DtN-Diagnostik.

**Ausgangslage**

Endlicher Approximant mit effektivem Operator:

$$
M=L\_\\Omega+\\text{projected UV/Env loads}.
$$

Zwei Kanalantworten:

$$
u\_{\\mathrm{Env}},\\qquad u\_{\\mathrm{UV}}.
$$

Messgröße:

$$
\\rho\_M
===

\\frac{\\langle u\_{\\mathrm{Env}},u\_{\\mathrm{UV}}\\rangle\_M}
{|u\_{\\mathrm{Env}}|*M,|u*{\\mathrm{UV}}|\_M}.
$$

**Befund**

Nahe Orthogonalität:

$$
|\\rho\_M|\\ll 1,
$$

teilweise numerisch nahe $90^\\circ$.

**Obstruktions-Ort**

Orthogonalität einer reellen 2-Ebene liefert höchstens:

$$
{+J,-J}.
$$

Die Ebene ist da; der Drehsinn nicht.

**Status**

Positiver Vorläufer einer prä-komplexen Ebene. Kein Vorzeichenbeweis.

\---

§§X20§§2.2 Real finite-network Schur/DtN-Test

**Artefaktbezug**

Kein eigenständiger angehängter Artefakt in dieser Fassung; der Abschnitt bleibt als konsolidierter methodischer Befund.

**Ausgangslage**

Endlicher Baumgraph mit Laplace-Matrix:

$$
L\_{\\mathrm{graph}}.
$$

Rand (B), Innenknoten (I), Schur-Komplement:

$$
\\Lambda\_B
===

L\_{BB}-L\_{BI}L\_{II}^{-1}L\_{IB}.
$$

**Befund**

Für deterministische zentrierte Einzelmodi numerisch praktisch orthogonal, etwa:

$$
|\\rho\_M|\\approx 10^{-18}.
$$

**Obstruktions-Ort**

Ein Einzelmodus kann orthogonal sein, während der volle Randantwortsraum noch Struktur trägt. Außerdem bleibt der DtN-Operator reell symmetrisch.

**Status**

Starker Hinweis auf echte Schur-/DtN-Orthogonalität in bestimmten Modi; kein $J$-Vorzeichen.

\---

§§X21§§2.3 Dirichlet-/Cut-Regularisierungstest

**Artefaktbezug**

Kein eigenständiger angehängter Artefakt in dieser Fassung; der Abschnitt fixiert den methodischen Befund.

**Ausgangslage**

Frage:

$$
\\text{Is external regularization or the pseudoinverse needed?}
$$

Genauer: Muss der Baum bzw. das Dirichlet-Netzwerk künstlich regularisiert werden, oder wirkt ein gesetzter UV- bzw. Environment-cut bereits selbst regularisierend?

**Befund**

Nein, das Dirichlet-Netzwerk muss nicht künstlich regularisiert werden. Ein echter UV-cut oder Environment-cut wirkt selbst bereits regularisierend, weil der entfernte Komplementanteil als Dirichlet-/Boundary-Seite behandelt wird. Dadurch wird der Innenblock

$$
L\_{II}
$$

invertierbar, sofern der betrachtete Innenbereich tatsächlich an die gesetzte Boundary gekoppelt ist.

Die Regularisierung ist daher schnittintern:

$$
\\text{UV-cut or environment-cut}
\\Rightarrow
\\text{Dirichlet boundary}
\\Rightarrow
L\_{II}^{-1}\\text{ is well-defined}.
$$

Sie ist keine externe numerische Hilfssetzung:

$$
\\text{no ridge},\\qquad
\\text{no pseudoinverse},\\qquad
\\text{no artificial mass term}.
$$

**Obstruktions-Ort**

Der DtN-Operator bleibt cut-relativ:

$$
\\Lambda\_{\\partial A}.
$$

Die schnittinterne Regularisierung liefert also einen wohldefinierten DtN-/Schur-Operator für den jeweiligen Cut, aber keinen cut-freien universalen DtN-Operator des ganzen unendlichen ToC.

**Status**

Wichtiges positives Ergebnis: UV- und Environment-cuts liefern die nötige Dirichlet-Regularisierung selbst. Keine Ridge-/Pseudoinversen-/Massenterm-Setzung nötig.

\---

§§X22§§2.4 Harter UV/Env-Skalenbruch im Approximanten

**Artefaktbezug**

Konzeptionell aus den Schur-/DtN- und $\\alpha\_{\\mathrm{orth}}$-Tests; in den angehängten Diagnostikartefakten über $M\_\\Omega$, $\\Sigma\_{\\mathrm{UV}}$ und $\\Sigma\_{\\mathrm{Env}}$ nachvollziehbar.

**Ausgangslage**

Ein proper subsystem besitzt zwei verschiedene Komplementseiten:

$$
\\text{UV-tail at the finest/cut node},
\\qquad
\\text{Environment at the root/parent port}.
$$

**Befund**

Die beiden Komplementprojektionen laden den Approximanten nicht gleichartig, sondern entgegengesetzt in der inneren Skalenrichtung:

$$
\\text{UV-tail}
\\Rightarrow
\\text{Load at finest/cut nodes},
$$

$$
\\text{Environment}
\\Rightarrow
\\text{Load at root/parent port}.
$$

Also:

$$
\\Sigma\_{\\mathrm{UV}}\\text{ acts on the leaf side},
\\qquad
\\Sigma\_{\\mathrm{Env}}\\text{ acts on the root side}.
$$

Das ist ein echter harter Skalenbruch im Approximanten. Er ist nicht bloß Visualisierung oder Chart-Artefakt.

**Obstruktions-Ort**

Der Bruch ist radial bzw. longitudinal. Er unterscheidet innen/außen, fein/grob, UV/Environment, aber er erzeugt noch keine transversale Händigkeit:

$$
\\text{scale break}\\≠\\text{chiralitas}.
$$

**Status**

Positives Finding für die Approximantenphysik und für $F1$. Kein $J$-Vorzeichenbeweis.

\---

§§X23§§2.5 Passive Dirichlet-/Widerstandsnetzwerke erzeugen keine Phase

**Artefaktbezug**

Querschnittsbefund aus den realen Schur-/DtN-, Cauchy-Shell- und Motor-Analogie-Tests; kein eigenständiger zusätzlicher Artefaktbezug.

**Ausgangslage**

Der flache ToC-/DtN-Sektor ist reell, passiv und reziprok. Er verhält sich mathematisch wie ein Dirichlet-/Widerstandsnetzwerk mit Energieform, Diffusion und symmetrischer Randantwort.

**Befund**

Ein rein resistiver/passiver Sektor liefert Imbalance, Achsen, Loads, Dirichletenergie, Diffusion und DtN-Antworten:

$$
\\text{passive resistance/load}
\\Rightarrow
\\text{imbalance/axis}.
$$

Er liefert aber keine eigenständige $90^\\circ$-Phasenverschiebung und kein stabil gerichtetes Drehfeld:

$$
\\text{passive resistance/load}
\\not\\Rightarrow
\\text{rotating phase}.
$$

**Obstruktions-Ort**

Für Oszillation, Phase oder Hamilton-artige Rotation bräuchte man eine zweite Speicherstruktur, einen abgeleiteten skew-Sektor oder ein Handoff-Locking, das nicht bereits als komplexe Phase importiert wird.

**Status**

Technische Form der Motor-/Kondensator-Analogie: Reeller Widerstandssektor kann eine Achse und Pulsation liefern, aber nicht die fehlende Phase selbst.

\---

§§X24§§3\. $\\alpha\_{\\mathrm{orth}}$- und Invarianten-Tests

§§X25§§3.1 $\\Xi$- / $\\alpha\_{\\mathrm{orth}}$-Diagnostik

**Artefaktbezug**

§§X148§§

**Ausgangslage**

Kontrollgröße:

$$
\\Xi=(1+\\lambda\_{\\mathrm{UV}})(1+\\lambda\_{\\mathrm{Env}}),
$$

mit

$$
\\lambda\_{\\mathrm{UV}}
===

\\frac{b^k\\alpha\_{\\mathrm{UV}}}{C\_k},
\\qquad
\\lambda\_{\\mathrm{Env}}
===

\\frac{\\alpha\_{\\mathrm{Env}}}{C\_k}.
$$

Typische Orthogonalitätsdiagnose:

$$
|\\rho|\\sim \\Xi^{-1/2}.
$$

**Befund**

Der UV-Term dominiert für wachsende Tiefe stark:

$$
|\\rho|\\sim b^{-k/2}.
$$

Also:

$$
\\boxed{
\\text{UV resolution drives orthogonality.}
}
$$

**Obstruktions-Ort**

$\\alpha\_{\\mathrm{Env}}$ war in frühen Versionen modellabhängig:

§§X149§§

Daher war der exakte Zahlenwert kein vollständig abgeleiteter physikalischer Wert.

**Status**

Gute Diagnosegröße. Kein Feinstrukturkonstanten-Claim. Kein $J$-Vorzeichen.

\---

§§X26§§3.2 Environment-Sensitivitätsmodelle

**Artefaktbezug**

§§X150§§

**Ausgangslage**

Vergleich verschiedener $\\alpha\_{\\mathrm{Env}}$-Modelle.

**Befund**

Für große $k$ dominiert häufig der UV-Term so stark, dass die Environment-Modellwahl subdominant wird.

**Obstruktions-Ort**

In Regimen, in denen Environment nicht subdominant ist, braucht man eine echte Komplementfamilien-/DtN-Ableitung von $\\alpha\_{\\mathrm{Env}}$.

**Status**

Guter methodischer Befund:

$$
\\text{definable}\\≠\\text{enforced}.
$$

\---

§§X27§§4\. Parent–Child- und Handoff-Tests

§§X28§§4.1 Two-Approximant / Flow-Sign-Test

**Artefaktbezug**

§§X151§§

**Ausgangslage**

Parent–Child-Handoff:

$$
A\_{\\mathrm{parent}}\\to A\_{\\mathrm{child}}.
$$

Ziel: prüfen, ob der Übergang ein $J$-Vorzeichen liefert.

**Befund**

Radiale Übergangssignaturen können entstehen.

**Obstruktions-Ort**

Radialität ist nicht Chiralität:

$$
\\text{Parent}\\to\\text{Child}
$$

liefert Tieferichtung, aber keinen Drehsinn.

Außerdem kann Flow leicht durch Anregungsrichtung ein Vorzeichen einschmuggeln.

**Status**

Radiale Handoff-Struktur: ja. $J$-Vorzeichen: nein.

\---

§§X29§§4.2 Schur-vor-Flow-Kriterium

**Artefaktbezug**

Methodisch aus den Parent–Child-Tests abgeleitet; kein zusätzlicher eigenständiger Artefakt.

**Ausgangslage**

Mögliche Handoff-Typen:

1. Restriction
2. Aggregation
3. Schur-Handoff
4. Flow-Handoff

**Befund**

Restriction/Aggregation/Schur sind kanonischer als Flow.

**Obstruktions-Ort**

Flow kann eine gerichtete Anregung enthalten. Dann wäre das Vorzeichen nicht abgeleitet, sondern gesetzt.

**Status**

Methodische Regel:

$$
\\boxed{
\\text{Schur first, Flow only as a consistency check.}
}
$$

\---

§§X30§§5\. Zwei-Rand-/Shell-Chiralitätstests

§§X31§§5.1 V4 — Two-boundary shell chirality

**Artefaktbezug**

§§X152§§

**Ausgangslage**

Parent–Child-Differenzschale, zwei Boundary-Ports, reale DtN-Matrix:

$$
\\Lambda\_\\Delta.
$$

Cauchy-Paarung:

$$
\\omega((q,p),(q',p'))=q^Tp'-p^Tq'.
$$

Auf einem DtN-Graphen gilt:

$$
p=\\Lambda q.
$$

**Befund**

Für selbstadjungierten DtN-Graphen:

$$
\\omega((q,\\Lambda q),(r,\\Lambda r)) = q^T\\Lambda r-r^T\\Lambda q = 0.
$$

**Obstruktions-Ort**

Ein einzelner passiver symmetrischer DtN-Graph ist Lagrangesch.

**Status**

Sauberes Negativergebnis. Zu eng für Familien-/Handoff-Tests, aber korrekt für Einzelgraph.

\---

§§X32§§5.2 V5 — Family handoff chirality

**Artefaktbezug**

§§X153§§

**Ausgangslage**

Familie von DtN-Matrizen:

$$
{\\Lambda\_i}.
$$

Cross-Graph-Cauchy-Pairing:

$$
\\omega\_{ij}(q,r)=q^T\\Lambda\_jr-r^T\\Lambda\_iq.
$$

Zusätzlich Handoff-Square:

$$
A\\to B\_i\\to C,
\\qquad
A\\to B\_j\\to C.
$$

**Befund**

Cross-Graph-Signale können auftreten:

$$
\\omega\_{ij}\\neq 0.
$$

Aber:

§§X154§§

**Obstruktions-Ort**

Signal ist Familien-/Metrikdifferenz, nicht Chiralität. Keine Geschwister-Vorzeichenumkehr, keine echte Handoff-Holonomie.

**Status**

Wichtiger Test: „Nicht nur ein Graph“ wurde geprüft. Ergebnis bleibt achiral.

\---

§§X33§§6\. Triadische Tests

§§X34§§6.1 V6 — Triadic interface chirality

**Artefaktbezug**

§§X155§§

**Ausgangslage**

Triade:

$$
\\text{UV-channel},
\\qquad
\\text{Environment-channel},
\\qquad
\\text{Handoff/Regulator-channel}.
$$

Regulator-Kandidat:

$$
r\_i=(\\Lambda\_{\\mathrm{child},i}-\\Lambda\_{\\mathrm{parent}})a.
$$

Triadische Fläche:

$$
\\tau\_i
===

\\det(e\_{\\mathrm{UV}}-e\_{\\mathrm{Env}},,r\_i-e\_{\\mathrm{Env}}).
$$

**Befund**

Für kanonische Modi:

§§X156§§

**Obstruktions-Ort**

Die Triade ist radial bzw. sibling-invariant.

**Status**

Triadisches Signal ja. Chirale Geschwister-Asymmetrie nein.

\---

§§X35§§6.2 Nichtkanonische positive Controls

**Artefaktbezug**

§§X157§§

**Ausgangslage**

Kontrollmodi:

§§X158§§

**Befund**

Sie erzeugen erwartbar Vorzeichen-/Flip-Effekte.

**Obstruktions-Ort**

Sie brechen Symmetrie per Label oder externer Ordnung.

**Status**

Nur Detektorkontrolle. Kein CNNA-derived Beweis.

\---

§§X36§§7\. V7 — Oriented UV/Environment Cauchy shell

§§X37§§7.1 Gegengerichtete UV/Env-Randseiten

**Artefaktbezug**

§§X159§§

**Ausgangslage**

UV-tail und Environment-tail werden als gegengerichtete Randseiten einer Shell gelesen.

Cauchy-Datenraum:

$$
(q\_{\\mathrm{Env}},q\_{\\mathrm{UV}},p\_{\\mathrm{Env}},p\_{\\mathrm{UV}}).
$$

Orientierte Randform:

$$
\\omega\_\\partial=\\omega\_{\\mathrm{Env}}-\\omega\_{\\mathrm{UV}}.
$$

Metrik:

$$
g=\\operatorname{diag}(k\_{\\mathrm{Env}},k\_{\\mathrm{UV}},1/k\_{\\mathrm{Env}},1/k\_{\\mathrm{UV}}).
$$

Konstruktion:

$$
J=-g^{-1}\\ω\_\\partial.
$$

**Befund**

Getestet:

§§X160§§

Also:

$$
J^2=-I,
\\qquad
J^TgJ=g,
\\qquad
J^T\\omega J=\\omega.
$$

**Obstruktions-Ort**

Die Ko-Orientierung wird gewählt. Mit der Gegenwahl entsteht ebenso konsistent:

$$
J\\mapsto -J.
$$

Die Cauchy-Randform ist daher nicht identisch mit dem gesuchten Locking-Objekt. Sie liefert eine symplektisch-kompatible Cauchy-Struktur, aber noch keine Verriegelung von $J$ mit einer Fluss-/Zeit-/Handoff-Orientierung $\\tau$:

$$
\\omega\_\\partial\\Rightarrow{+J,-J},
\\qquad
\\omega\_{\\mathrm{lock}}:(J,\\tau)\\mapsto\\text{stable oriented record}.
$$

**Status**

Sehr wichtiges Positivergebnis:

$$
\\text{UV/Env co-orientation}\\Rightarrow {+J,-J}\\text{-Cauchy structure}.
$$

Cauchy-Shell positiv, aber Locking fehlt. Kein absolutes Vorzeichen.

\---

§§X38§§8\. Root-, Co-root- und Tiefenlesart-Tests

§§X39§§8.1 Root als äußerer Modellrand

**Ausgangslage**

Der ToC wächst nicht ontisch; er ist unendlich gegeben.

$$
\\ell(\\mathrm{root})=0,
\\qquad
\\ell\\to\\infty
$$

nach innen.

**Befund**

Tiefenordnung liefert relative Gegengerichtetheit:

$$
\\text{Env side}: \\ell\\downarrow,
\\qquad
\\text{UV side}: \\ell\\uparrow.
$$

**Obstruktions-Ort**

Tiefenordnung ist polar, nicht chiral:

$$
\\text{inside/outside}\\≠\\text{direction of rotation}.
$$

**Status**

Stützt V7 semantisch. Kein absolutes $J$.

\---

§§X40§§8.2 Negative-root / Co-root-Hypothese

**Ausgangslage**

Hypothese:

$$
\\text{formal root is interface;}
\\qquad
\\text{behind it lies a negative root family}.
$$

**Befund**

Könnte Cauchy-Doppelung und $\\alpha\_{\\mathrm{Env}}$-Ableitung unterstützen.

**Obstruktions-Ort**

Eine negative Wurzelfamilie bleibt bei reeller passiver Symmetrie nicht automatisch chiral.

**Status**

Möglicher Kandidat für Environment-Ableitung; kein Vorzeichenbeweis.

\---

§§X41§§9\. Geschwister-, $S\_b$- und Adresssymmetrie-Tests

§§X42§§9.1 $S\_b$-Sibling-Obstruktion

**Ausgangslage**

Im ungeordneten b-ären Baum sind Geschwister unter

$$
S\_b
$$

austauschbar.

**Befund**

Kanonische Größen liegen in der trivialen $S\_b$-Komponente.

**Obstruktions-Ort**

Die Signum-Darstellung wird nicht kanonisch ausgewählt:

$$
S\_b\\text{-equivariance}
\\Rightarrow
\\text{no canonical sibling chirality}.
$$

**Status**

Robuste Negativlinie.

\---

§§X43§§9.2 Hamming-Gewichtsklassen

**Ausgangslage**

Blätter wie:

$$
000,001,010,011,100,101,110,111.
$$

Klassen:

$$
|x|\_1=1,
\\qquad
|x|\_1=2.
$$

**Befund**

Adressintrinsische Relation quer zur Präfixstruktur.

**Obstruktions-Ort**

Hamming-Gewicht ist Betrag, keine Orientierung. Bit-Umkehr bleibt möglich.

**Status**

Strukturfund, aber achiral.

\---

§§X44§§9.3 Zyklische Bitverschiebung

**Ausgangslage**

Auf etwa:

$$
{001,010,100}
$$

gibt es zyklische Verschiebung:

$$
001\\to010\\to100\\to001.
$$

**Befund**

Adressintrinsische Schleife ohne geometrische Einbettung.

**Obstruktions-Ort**

Bit-Reversal konjugiert Links-Shift in Rechts-Shift:

$$
\\mathrm{reverse}\\circ\\rho=\\rho^{-1}\\circ\\mathrm{reverse}.
$$

Also:

$$
\\text{Loop yes, direction of rotation no.}
$$

**Status**

Wichtiger Kandidat für Multi-ToC-/Frustrationsstrukturen. Kein lokales $J$-Vorzeichen.

\---

§§X45§§10\. SG/ST-, Chirotopie- und Sign-Line-Tests

§§X46§§10.1 SG/ST als IFS-/Quotient-Strukturen

**Ausgangslage**

Sierpinski-Gasket (SG) und Sierpinski-Tetrahedron/Tetrix (ST) wurden als ToC-nahe Quotient-/IFS-Strukturen betrachtet. Ihre Rolle in der Testgeschichte war nicht zufällig: SG und ST waren wegen Zahmheit, hoher Symmetrie, p.c.f.-Kontrollierbarkeit und Skaleninvarianz die ersten natürlichen Fraktal-Stressobjekte.

Der b-äre Baum ist dabei die Provenienz- bzw. Adressseite dieser Strukturen:

$$
SG:\\quad A\_3^{<\\omega},\\qquad ST:\\quad A\_4^{<\\omega}.
$$

SG/ST selbst entstehen erst, wenn zur reinen Adressprovenienz zusätzliche IFS-/Quotient-/Randidentifikationen und meist eine geometrische Einbettung hinzukommen. Diese Zusatzrelationen dürfen im CNNA-derived-only-Test nicht unkontrolliert als Orientierung, Hodge-Struktur oder komplexe Phase zurückimportiert werden.

**Befund**

Sie bringen Schleifen und Kozyklen:

$$
H^1\\neq0.
$$

Beispielhafte Größen:

$$
d\_s(SG)=\\frac{2\\log 3}{\\log 5},
\\qquad
d\_s(ST)=\\frac{2\\log 4}{\\log 6}.
$$

**Obstruktions-Ort**

SG/ST sind nicht der bare ToC. Sie sind IFS-/Adressquotienten. Ihre zusätzlichen Relationen sind nicht automatisch aus dem ToC abgeleitet.

**Status**

Nützlich als Vergleichs- und Strukturtest; kein direkter $J$-Durchbruch. Der b-äre Baum bleibt die bewusst entkleidete Provenienzseite von SG/ST, nicht deren geometrisch orientierte Einbettung.

\---

§§X47§§10.2 Chirotopie / Sign-Line (S\_b/A\_b)

**Ausgangslage**

Chiralität auf Geschwistern liegt in der Signum-Information:

$$
S\_b/A\_b\\simeq \\mathbb Z\_2.
$$

**Befund**

Wenn die lokale Isotropiegruppe $H$ nicht in $A\_b$ liegt, gibt es keine kanonische nichtverschwindende Chirotopie.

Für den symmetrischen ToC:

$$
H=S\_b.
$$

**Obstruktions-Ort**

$$
S\_b\\not\\subset A\_b.
$$

Daher ist eine Sign-Line nicht kanonisch ausgezeichnet.

**Status**

Sehr zentrale No-Go-Formulierung.

\---

§§X48§§10.3 $Z\_b$-Zyklizität ist nicht genug

**Ausgangslage**

Test, ob zyklische Ordnung $Z\_b$ die fehlende Chirotopie ersetzt.

**Befund**

Nein. Bei $b=4$ kann ein 4-Zyklus als Labelpermutation ungerade sein; geometrische Orientierung und Permutationsparität fallen nicht automatisch zusammen.

**Obstruktions-Ort**

Zyklische Ordnung ist noch keine Sign-Line.

**Status**

Wichtige Korrektur gegen voreilige „Zyklus = Orientierung“-Schlüsse.

\---

§§X49§§10.4 Angehangener SG/H₁-Kontrolltest

**Artefaktbezug**

§§X161§§

**Ausgangslage**

Der b-äre Baum wurde als Kontrollgruppe gegen das Sierpinski-Gasket betrachtet: Der Baum hat keinen Zyklenraum, während das Gasket bereits auf Graphniveau viele Zyklen besitzt. Damit prüft der Test die Hypothese, ob das fehlende $i$ bzw. $J$ nicht im Baum, sondern in Schleifen bzw. $H\_1$ liegen könnte.

**Befund**

Der Baum hat erwartungsgemäß $b\_1=0$. Das Gasket besitzt nichttriviales $H\_1$, aber die reine Graphenform reicht nicht aus, um ein kanonisches $J$ zu erzwingen. Die planare Zyklenorientierung hängt an einer gewählten Ebenenorientierung und kippt unter Spiegelung. Der reine Down-Kanten-Laplace annihiliert den Zyklenraum, weil Zyklen im Graphen harmonisch sind.

**Obstruktions-Ort**

Schleifen allein liefern noch keine Dynamik und keine kanonische komplexe Orientierung. Für eine nichttriviale Dynamik auf $H\_1$ wären echte $2$-Zellen bzw. ein Kettenkomplex mit Up-Laplace erforderlich. Das wäre eine neue, separat zu prüfende Struktur und darf nicht aus der planaren Einbettung importiert werden.

**Status**

Der angehängte SG/H₁-Test stützt die Hauptlinie: mehr Topologie als im Baum ist hilfreich als Stressklasse, aber reine Graphenschleifen liefern noch kein derived-only $J$-Vorzeichen.

\---

§§X50§§11\. Hodge-, Dirac- und Dualkomplex-Tests

§§X51§§11.1 Cellular Dirac $K=d-d^\*$

**Ausgangslage**

Zellulärer Operator:

$$
K=d-d^\*
$$

auf

$$
C^0\\oplus C^1\\oplus C^2.
$$

**Befund**

$K$ ist reell schief. Auf $\\operatorname{im}K$ kann eine formale Polarstruktur einen J-artigen Anteil liefern.

**Obstruktions-Ort**

Der Operator mischt Grade. Auf reinem $C^1$-Raum ist der relevante Block nicht automatisch ein lokales $J$.

**Status**

Formale $J$-ähnliche Struktur möglich, aber nicht als lokaler Handoff-$J$ abgeleitet.

\---

§§X52§§11.2 Hodge-Star / Dualkomplex

**Ausgangslage**

Test, ob duale Zellen oder Hodge-$\\star$ die Orientierung liefern.

**Befund**

Ein echter Hodge-$\\star$ braucht Orientierung bzw. Metrik-/Volumenstruktur.

Bei voller $S\_b$-Symmetrie gibt es keinen kanonischen schiefen äquivarianten Operator.

Mit Chirotopie reduziert sich die Symmetrie und ein $J$-Block kann erscheinen.

**Obstruktions-Ort**

Richtung ist:

$$
\\text{Chirotopy}\\Rightarrow J\\text{-mode},
$$

nicht:

$$
J-mode\\Rightarrow\\text{chirotopy}.
$$

**Status**

Bestätigt, dass Orientierung nicht durch Hodge allein entsteht.

\---

§§X53§§12\. Recursive SG/ST- und Schur/DtN-Tests

§§X54§§12.1 Rekursive SG/ST-DtN-Matrizen

**Ausgangslage**

Boundary-DtN-Matrizen für rekursive SG/ST-Approximationen.

**Befund**

Boundary-DtN bleibt voll symmetrisch:

$$
\\Lambda\_n=a\_n(bI-\\mathbf 1\\mathbf 1^T).
$$

Typisch:

$$
a\_n(SG)=\\left(\\frac35\\right)^n,
\\qquad
a\_n(ST)=\\left(\\frac23\\right)^n.
$$

**Obstruktions-Ort**

Volle $S\_b$-Invarianz bleibt erhalten. Keine Reduktion:

$$
S\_b\\to A\_b.
$$

**Status**

SG/ST-Schur/DtN liefert Skalenstruktur, keine Chirotopie.

\---

§§X55§§12.2 IFS-Erzeugungsprozess-Test

**Ausgangslage**

Test, ob der IFS-Wachstumsprozess selbst eine Ordnung erzeugt.

**Befund**

Ungeordnete Kontraktionen:

$$
{\\phi\_i}
$$

bleiben $S\_b$-äquivariant.

**Obstruktions-Ort**

Eine geordnete/chirale IFS-Familie könnte Chirotopie tragen, aber nur, wenn die Ordnung selbst abgeleitet ist.

**Status**

IFS-Wachstum allein löst das Vorzeichenproblem nicht.

\---

§§X56§§13\. Mehrzellen-Holonomie

§§X57§§13.1 Permutations-Holonomie zwischen lokalen ToC-Fasern

**Ausgangslage**

Gluing-Kanten mit:

$$
\\varphi\_{\\alpha\\beta}\\in S\_b.
$$

Loop-Holonomie:

$$
h\_\\gamma
===

\\varphi\_{\\alpha\_{k-1}\\alpha\_k}\\cdots\\varphi\_{\\alpha\_0\\alpha\_1}.
$$

**Befund**

Wenn der Zentralisator

$$
C\_{S\_b}(h\_\\gamma)
$$

in $A\_b$ liegt, können lokale odd permutations ausgeschlossen werden.

Beispiel:

$$
b=3,\\quad h=(012),
\\qquad
C\_{S\_3}(h)=A\_3.
$$

**Obstruktions-Ort**

Die Richtung

$$
h \\text{ vs. } h^{-1}
$$

bleibt genau die chirale Wahl. Unorientierte Klasse:

$$
{h,h^{-1}}
$$

lokalisiert nur ein Paar.

**Status**

Starker Multi-ToC-Kandidat, aber ohne derived gerichtete Holonomie kein $J$-Vorzeichen.

\---

§§X58§§14\. F1-Holonomie und F1-only-No-Go

§§X59§§14.1 F1-only Port-Regeln

**Ausgangslage**

F1 ist der radiale Provenienz-/Auffüllpfeil. Test: Kann eine F1-only-Regel Ports nichttrivial permutieren?

**Befund**

Eine relabeling-natürliche F1-only-Portregel muss mit allen

$$
\\sigma\\in S\_b
$$

kommutieren. Daher liegt sie im Zentrum:

$$
Z(S_b)={e}
\\qquad (b\ge3).
$$

**Obstruktions-Ort**

F1 allein hat keine transversale Portordnung.

**Status**

Starker No-Go: Nichtlinearität in Tiefe hilft nicht, solange Relabeling-Natürlichkeit gilt.

\---

§§X60§§14.2 Screw-Regel als Import

**Ausgangslage**

Regel wie:

$$
(n,i)\\mapsto(n+1,\\sigma(i)),
\\qquad
\\sigma=(012).
$$

**Befund**

Erzeugt scheinbar Drehung.

**Obstruktions-Ort**

Unter odd relabeling:

$$
\\tau\\sigma\\tau^{-1}=\\sigma^{-1}.
$$

Die Regel importiert eine Portordnung.

**Status**

Kontrollimport, kein CNNA-derived Mechanismus.

\---

§§X61§§15\. Value-based F1-Coupling

§§X62§§15.1 Tiefenabhängige Wertkopplung

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Abschnitt bleibt als konzeptioneller Befund.

**Ausgangslage**

Nicht Portpermutation, sondern wertbasierte Kopplung:

$$
w\_{\\alpha\\beta}=f(d\_\\alpha,d\_\\beta,\\ldots).
$$

**Befund**

Skew-Komponenten können entstehen, wenn Kopplung tiefenabhängig und nicht symmetrisch ist.

**Obstruktions-Ort**

„Tief speist stark“ und „flach speist stark“ sind zwei Vorzeichenwahlen:

$$
K^+=-K^-.
$$

Die Regel wählt ein Vorzeichen, wenn sie nicht abgeleitet ist.

**Status**

Zeigt, wie Nichtreziprozität entstehen könnte. Aber ohne derived Auswahl bleibt ({+K,-K}).

\---

§§X63§§16\. Block-RG und Schalenkopplung

§§X64§§16.1 Kollektive Schalenkopplung

**Ausgangslage**

Nicht Knoten-an-Knoten, sondern relabeling-natürliche Level-Schale an Level-Schale:

$$
S\_k(A)\\leftrightarrow S\_k(B),
$$

mit Mean-Mode:

$$
u\_{A,k}
===

\\frac{1}{\\sqrt{|S\_k|}}\\mathbf 1\_{S\_k(A)}.
$$

Kopplung:

$$
C\_{AB}
===

\\sum\_k\\gamma\_k u\_{A,k}u\_{B,k}^T.
$$

**Befund**

Reziproke Schalenkopplung erzeugt Spektralstruktur und ggf. Zyklen.

**Obstruktions-Ort**

Die Kopplung bleibt symmetrisch:

$$
C\_{AB}=C\_{BA}^T.
$$

Daher überlebt A/B-Spiegelung.

**Status**

Struktur ja, Chiralität nein.

\---

§§X65§§16.2 Vier-Fälle-Test: Adress-fixiert vs. Rollen-fixiert

**Ausgangslage**

Unterscheidung:

$$
\\text{Address role}
\\≠
\\text{Scale role}.
$$

Vier Fälle:

|Fall|Skalenlesart|Verklebungsort|
|-|-|-|
|A|Wurzel grob|Wurzel|
|B|Wurzel fein|Wurzel|
|C|Wurzel grob|grobes Ende = Wurzel|
|D|Wurzel fein|grobes Ende = Level-$L$-Schale|

**Befund**

Fall D ist strukturell neu.

Gemeldeter Befund:

$$
\\beta\_1: 0\\to 6560,
$$

$$
d\_s: 1.385\\to 3.647.
$$

**Obstruktions-Ort**

Trotz starker Strukturänderung überlebt A/B-Spiegelung in allen Fällen.

Grund:

$$
\\text{Gate depends on the reciprocity of the transverse coupling, not on the bonding site.}
$$

**Status**

Sehr wichtiger Befund: inverse Skalenlesart ist echter Strukturparameter, aber kein $J$-Mechanismus.

\---

§§X66§§17\. Inverser UV/Env-Cut

§§X67§§17.1 UV-cut unter umgekehrter Skalenlesart

**Ausgangslage**

Standard:

$$
\\text{UV on leaves},
\\qquad
\\text{Env at root}.
$$

Inverse Lesart:

$$
\\text{UV at root},
\\qquad
\\text{Env on leaves}.
$$

**Befund**

Als echter weiterer Test identifiziert; nicht vollständig als eigener finaler positiver Befund abgeschlossen.

**Obstruktions-Ort**

Würde Skalenrollen direkt in die Operatorstruktur einbringen. Aber solange die resultierenden Operatoren reell symmetrisch und relabeling-natürlich bleiben, ist Chiralität nicht zu erwarten.

**Status**

Offen bzw. als nächster präziser Test markiert, aber durch spätere DtN-/Flachheitsdiagnose teilweise eingeordnet.

\---

§§X68§§18\. DtN-Handoff-Operator-Tests

§§X69§§18.1 Zwei DtN-Matrizen auf gemeinsamem Handoff-Raum

**Ausgangslage**

Nach Korrektur: Handoff sieht keine ToC-Knoten mehr, sondern Operatoren:

$$
(H\_\\partial,\\Lambda).
$$

Ziel:

$$
K=\[\\Lambda\_A,\\Lambda\_B].
$$

**Befund**

Nur sinnvoll, wenn beide Operatoren auf demselben Handoff-Raum leben.

**Obstruktions-Ort**

Spektralordnung allein identifiziert keine Eigenräume. In jeweiliger Eigenbasis diagonalisiert, kommutieren beide trivial.

**Status**

Wichtige Kategoriekorrektur.

\---

§§X70§§18.2 DtN-RG-Kommutator

**Ausgangslage**

Aufeinanderfolgende RG-/Schur-Stufen derselben Sequenz:

$$
\\Lambda\_n,
\\qquad
\\widetilde{\\Lambda}\_{n+1}.
$$

Kommutator:

$$
K\_n=\[\\Lambda\_n,\\widetilde{\\Lambda}\_{n+1}].
$$

**Befund**

Gemeldet:

$$
K\_n=0
$$

für kanonische RG-Projektion.

**Obstruktions-Ort**

Beide Operatoren liegen auf derselben radialen F1-Achse und teilen dieselbe symmetrieadaptierte Schalenbasis.

**Status**

Sehr wichtiger Mechanismus:

$$
\\text{order derived by F1}
\\Rightarrow
\\text{same axis}
\\Rightarrow
\\text{commutativity}.
$$

\---

§§X71§§19\. Überlagerte DtN-Matrixalgebra-Türme

§§X72§§19.1 Matrix-Tower-Idee

**Ausgangslage**

Vorschlag:

$$
M\_2\\to M\_4\\to M\_8\\to\\cdots
$$

bzw. mehrere ToC-DtN-Matrizen auf wachsenden Handoff-Räumen.

**Befund**

Nichtkommutativität könnte entstehen, wenn mehrere symmetrische Operatoren auf demselben Raum keine gemeinsame Eigenbasis haben.

**Obstruktions-Ort**

Beispiele mit Spin-Ketten importieren Tensorproduktordnung und Nachbarschaft:

$$
A\_{12},\\qquad A\_{23}.
$$

Diese Links-Rechts-Struktur ist nicht aus barem ToC abgeleitet.

**Status**

Als möglicher A→B-Algebraweg interessant, aber nur mit derived Einbettungen erlaubt.

\---

§§X73§§19.2 Kinderpartition-/ToC-derived-Einbettungstest

**Ausgangslage**

Abgeleitete Einbettungen über Kinder-Teilbäume bzw. $S\_b$-symmetrische Partitionen.

**Befund**

Kind-restringierte DtN-Operatoren kommutieren:

* disjunkte Supports → triviale Kommutatoren,
* volle DtN gegen blockdiagonalen Teil → kommutiert numerisch.

**Obstruktions-Ort**

Alle Zerlegungen respektieren dieselbe $S\_b$-/Radialsymmetrie und teilen die symmetrieadaptierte Eigenbasis.

**Status**

Matrix-Tower-Route negativ im flachen abgeleiteten ToC-Sektor.

\---

§§X74§§20\. Connes-/Nichtkommutativitätsroute

§§X75§§20.1 Grundfrage: Woher kommt Nichtkommutativität bei Connes?

**Ausgangslage**

Connes ersetzt Raum durch Algebra:

$$
(\\mathcal A,\\mathcal H,D).
$$

Nichtkommutativität liegt in:

$$
ab\\≠ ba.
$$

**Befund**

Bei Connes ist die nichtkommutative Algebra typischerweise Eingabestruktur, nicht aus einem flachen ToC abgeleitet.

**Obstruktions-Ort für CNNA**

CNNA müsste erst eine Handoff-Algebra liefern:

$$
\\mathcal A\_{\\mathrm{eff}} = \\operatorname{Alg}{\\Lambda\_i}
$$

mit

$$
\[\\Lambda\_i,\\Lambda\_j]\\neq0.
$$

**Status**

Connes ist Ziel-/Vergleichsstruktur, nicht Generator.

\---

§§X76§§20.2 Zwei Reduktionsregimes

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Befund bleibt als konsolidierter Diagnostikstand.

**Ausgangslage**

Vergleich:

$$
\\Lambda\_{\\mathrm{UV}}
$$

gegen

$$
\\Lambda\_{\\mathrm{Env}}
$$

auf demselben Leaf-Boundary-Raum.

**Befund**

Gemeldet:

$$
|\[\\Lambda\_{\\mathrm{UV}},\\Lambda\_{\\mathrm{Env}}]|\\sim 10^{-16}.
$$

**Obstruktions-Ort**

Root-Selbstenergie verschiebt Eigenwerte, aber dreht keine Eigenräume. Radial bleibt radial.

**Status**

Negativ für exakte derived Regime.

\---

§§X77§§20.3 Spektral trunkierte Reduktion

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Befund bleibt als konsolidierter Diagnostikstand.

**Ausgangslage**

Vergleich:

$$
\\Lambda\_{\\mathrm{full}}
$$

gegen spektral trunkierte Reduktion:

$$
\\Lambda\_{\\mathrm{trunc}}.
$$

**Befund**

Bei beliebigem $m$:

$$
|\[\\Lambda\_{\\mathrm{full}},\\Lambda\_{\\mathrm{trunc}}]|\\approx 0.017
$$

für mittlere $m$-Werte; $K$ ist schief.

**Obstruktions-Ort**

Zunächst falsch interpretiert: $\\pm i\\lambda$-Paare wurden als „beide Chiralitäten“ gelesen. Korrektur:

$$
\\pm i\\lambda
$$

ist normales Spektrum eines reellen $J$-Blocks.

Der echte Vorzeichentest ist:

$$
K\\text{ or }-K\\text{ distinguished?}
$$

**Status**

Nur scheinbar positiver Kandidat; musste degenerazien-sicher nachgetestet werden.

\---

§§X78§§20.4 Degenerazien-sichere Cluster-Trunkierung

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Abschnitt hält nur den methodischen Nachbefund fest.

**Ausgangslage**

Trunkierung nicht nach beliebigem $m$, sondern nur nach ganzen Eigenwert-Clustern:

$$
P\_{\\le \\lambda} = \\sum\_{\\mu\\le\\lambda}P\_\\mu.
$$

**Befund**

Bei allen kanonischen Cluster-Grenzen:

$$
|K| ≈ 10^(−16).
$$

Nichtkommutativität trat nur auf, wenn $m$ mitten durch degenerierte Eigenräume schnitt.

**Obstruktions-Ort**

Ein Schnitt durch entartete Eigenräume wählt eine nicht-kanonische §§X168§§-Basis. Das ist kein ToC-derived Mechanismus.

Warnung: Nicht-kanonische Trunkierung mitten durch entartete Eigenräume ist ein Symmetriebruch durch numerische Basiswahl und darf nicht als derived Nichtkommutativität gezählt werden.

**Status**

Starker Negativbefund:

$$
\\boxed{
\\text{Relabeling-natural exact and cluster-safe DtN reductions commute.}
}
$$

\---

§§X79§§21\. Knoten-Elimination vs. partielle Spur

§§X80§§21.1 Falscher „Ausspuren“-Test

**Ausgangslage**

System/Umwelt-Knoten wurden getrennt:

$$
\\mathbb R^N=\\mathbb R^S\\oplus\\mathbb R^E.
$$

Dann wurde Diffusion $e^{-tL}$ gerechnet und Umgebung mit festem Zustand behandelt.

**Befund**

Skew konnte entstehen.

**Obstruktions-Ort**

Das war keine partielle Spur. Eine partielle Spur braucht:

$$
\\mathcal H=\\mathcal H\_S\\otimes\\mathcal H\_E.
$$

Der Knotenraum liefert aber direkte Summe, kein Tensorprodukt.

Der Skew kam aus asymmetrischer Einspeisung/Restriktion:

$$
\\text{Environment feeds in, system outflow is discarded}.
$$

**Status**

Ungültig als OQS-/Partial-trace-Test. Höchstens Test einer asymmetrischen Randbedingung.

\---

§§X81§§21.2 Korrekte Knotenreduktion

**Ausgangslage**

Für Knotenaufteilung:

$$
L=
\\begin{pmatrix}
L\_{SS} \& L\_{SE}\\
L\_{ES} \& L\_{EE}
\\end{pmatrix}.
$$

Korrekte Eliminierung:

$$
L\_{\\mathrm{eff}} = L\_{SS}-L\_{SE}L\_{EE}^{-1}L\_{ES}
$$

**Befund**

Für reell symmetrisches $L$:

$$
L\_{\\mathrm{eff}}^T=L\_{\\mathrm{eff}}.
$$

**Obstruktions-Ort**

Knoten-Elimination erzeugt keine OQS-Irreversibilität und keinen antisymmetrischen Hamilton-Teil.

**Status**

Zentrale Methodenkorrektur:

$$
\\boxed{
\\text{At nodes, we eliminate, not trace.}
}
$$

\---

§§X82§§22\. Flacher Sektor und Krümmung

§§X83§§22.1 Flacher reell-reziproker ToC-/DtN-Sektor

**Ausgangslage**

Idealer ToC bzw. ToC-Fasern ohne Krümmung, Holonomie, Regulator-Backreaction.

**Befund**

Alle natürlichen Operatoren bleiben gemeinsam diagonalisierbar.

**Obstruktions-Ort**

Es gibt keine Connection:

$$
\\nabla,
$$

keine Holonomie:

$$
U\_\\gamma\\neq I,
$$

und keine Krümmung:

$$
\[\\nabla\_\\mu,\\nabla\_\\nu]\\neq0.
$$

**Status**

Interpretationswechsel:

$$
\\boxed{
\\text{The no-gos concern the flat ToC/DtN sector.}
}
$$

Nicht CNNA insgesamt.

\---

§§X84§§22.2 Krümmung als möglicher späterer Ursprung von Nichtkommutativität

**Ausgangslage**

In Geometrie/Eichtheorie:

$$
\[\\nabla\_\\mu,\\nabla\_\\nu]=R\_{\\mu\\nu}.
$$

bzw.

$$
\[D\_\\mu,D\_\\nu]=F\_{\\mu\\nu}.
$$

**Befund**

Nichtkommutativität könnte im CNNA-Kontext eher ein emergentes Krümmungs-/Holonomiephänomen sein.

**Obstruktions-Ort**

Krümmung darf nicht als Retter importiert werden. Sie müsste aus Handoff-/Regulator-/Backreaction-Daten entstehen.

**Status**

Offener Curved-sector target:

$$
\\text{Block-RG/DtN}\\to\\text{Connection}\\to\\text{Holonomy/Curvature}.
$$

\---

§§X85§§23\. IDEAL-ToC-Faser-Gitter

§§X86§§23.1 Doppelt unendlicher IDEAL-Sektor

**Ausgangslage**

Statt eines universalen Einzel-ToC:

$$
T\_b^\\infty
$$

definiert man ein ToC-Faser-Gitter:

$$
\\mathcal I\_{\\mathrm{ToCGrid}} = \\Gamma\_\\infty\\times T\_b^\\infty
$$

Mit:

$$
x\\in\\Gamma\_\\infty,
\\qquad
w\\in T\_b^\\infty.
$$

Zwei Unendlichkeiten:

$$
\\Gamma\_\\infty
$$

transversal und

$$
T\_b^\\infty
$$

intern pro Faser.

**Befund**

Vollidealer Sektor:

$$
\\text{flat, homogeneous, reciprocal, internally ToC-scale invariant}.
$$

Transversale Isotropie nur diskret bzw. abhängig von $\\Gamma\_\\infty$.

**Obstruktions-Ort**

Das Gitter bringt transversale Nachbarschaft als neues IDEAL-Vergleichsdatum mit. Sie ist nicht aus einem einzelnen ToC abgeleitet.

**Status**

Sehr sinnvoller letzter ToC-naher Test vor Substratwechsel.

\---

§§X87§§23.2 Endlicher Doppelschnitt

**Ausgangslage**

Berechenbarer Sektor:

$$
\\Omega\_{R,L} = W\_R\\times T\_{\\le L}
$$

Mit:

$$
W\_R\\subset\\Gamma\_\\infty,
\\qquad
T\_{\\le L}\\subset T\_b^\\infty.
$$

**Befund**

Subsystem-Sein bricht zwingend die IDEAL-Symmetrie:

$$
\\operatorname{Aut}(\\mathcal I\_{\\mathrm{ToCGrid}})
\\to
\\operatorname{Aut}(\\Omega\_{R,L}).
$$

Es entstehen:

$$
\\text{outer grid complement},
$$

$$
\\text{internal UV-tail},
$$

$$
\\text{edge/corner/mixed complements}.
$$

**Obstruktions-Ort**

Subsystem-Sein erzeugt effektive Rand-/Spektral-/DtN-Geometrie, aber nicht automatisch Chirotopie.

**Status**

Positiver Geometrie-/DtN-Test, negativer $J$-Test im flachen reziproken Fall.

\---

§§X88§§23.3 DtN auf dem ToC-Faser-Gitter

**Ausgangslage**

Operator auf $\\Omega\_{R,L}$:

$$
L\_{R,L}.
$$

Schur/DtN:

$$
\\Lambda\_{R,L} = L\_{\\partial\\partial} - L\_{\\partial I}L\_{II}^{-1}L\_{I\\partial}
$$

**Befund**

Dies ist A→B-näher als rohe Knotenverklebung. B würde nicht ToC-Knoten sehen, sondern Handoff-Matrizen.

**Obstruktions-Ort**

Solange das Gitter homogen, reziprok und flach ist, entstehen zwar Spektrum und effektive Geometrie, aber keine ausgezeichnete Chirotopie.

**Status**

Wichtiger letzter Referenztest:

$$
\\boxed{
\\text{ToC fiber lattices can test geometry, not }J\\text{ enforce it.}
}
$$

\---

§§X89§§24\. Holonomie-/Connection-Test im Faser-Gitter

§§X90§§24.1 Effektive Intertwiner zwischen lokalen Handoff-Räumen

**Ausgangslage**

Für lokale Handoff-Räume:

$$
H\_x,\\qquad H\_y
$$

bräuchte man derived Intertwiner:

$$
U\_{xy}:H\_x\\to H\_y.
$$

Loop-Holonomie:

$$
U\_\\gamma = U\_{wx}U\_{zw}U\_{yz}U\_{xy}
$$

**Befund**

Im homogenen flachen Fall erwartbar:

$$
U\_\\gamma=I
$$

oder gauge-trivial.

**Obstruktions-Ort**

Ein nichttrivialer Rotationsanteil müsste aus Inhomogenität, Regulator, Backreaction oder Frustration kommen.

**Status**

Offener Curved-sector-Test. Noch nicht positiv gezeigt.

\---

§§X91§§25\. Lorentz-/Zeitstruktur-Tests

§§X92§§25.1 Lorentz-Signatur

**Ausgangslage**

Signatur:

$$
\\eta=\\operatorname{diag}(-1,+1,\\ldots,+1).
$$

**Befund**

Trennt zeitartig und raumartig.

**Obstruktions-Ort**

Zeitumkehr bleibt Symmetrie:

$$
T\\eta T=\\eta.
$$

Lichtkegel bleibt Doppelkegel:

$$
C^+\\cup C^-.
$$

**Status**

Reduziert Problem auf Zeitorientierung, löst sie nicht.

\---

§§X93§§25.2 Reeller Zeitfluss-Vorläufer

**Ausgangslage**

Reell-symmetrischer Generator $H$, Flusspaar:

$$
{e^{+tH},e^{-tH}}.
$$

**Befund**

Liefert:

$$
{+\\tau,-\\tau}.
$$

**Obstruktions-Ort**

Für reell-symmetrisches $H$ bleibt jede spektrale Funktion symmetrisch. Ein $J$ ist antisymmetrisch:

$$
J≠ f(H).
$$

**Status**

Zeitpaar ja. Verriegelung mit $J$ nein.

\---

§§X94§§26\. Pillar C / OQS / Entropie

§§X95§§26.1 Lindblad-/OQS-Zeitpfeil

**Ausgangslage**

Offene Quantendynamik / Lindblad-Generator.

**Befund**

Dissipation kann Zeitrichtung wählen:

$$
+τ.
$$

**Obstruktions-Ort**

Hamiltonischer Term enthält bereits:

$$
-i[H, ρ].
$$

Also setzt OQS $i$ bzw. $J$ voraus.

**Status**

Pillar C kann $\\tau$ wählen, aber $J$ nicht allein erzeugen.

\---

§§X96§§27\. AQFT / Type-I / Type-III / Handoff-Struktur

§§X97§§27.1 A als Type-I-/Type-III-Vorläuferschicht

**Ausgangslage**

Pillar A soll nicht direkt Type III beweisen, sondern Vorläufer liefern:

$$
\\mathcal C\_{d,k} = (Q\_{d,k}\\oplus P\_{d,k},g\_{d,k},\\omega\_{d,k},{J,-J})
$$

Endlich:

$$
k<\\infty
\\Rightarrow
\\text{Type-I-like precursors}.
$$

Unendlich:

$$
k\\to\\infty
\\Rightarrow
\\text{Type-III-capable complement family precursors}.
$$

**Befund**

Architektonisch sinnvoll.

**Obstruktions-Ort**

Dimension/Unendlichkeit liefert keine Orientierung:

$$
\\text{finite/infinite}\\neq J\\text{-sign}.
$$

**Status**

Wichtiger Architekturshift.

\---

§§X98§§27.2 Triadischer Handoff (B|B'|C)

**Ausgangslage**

Handoffs sind nicht passive Pfeile, sondern eigene Interface-Objekte.

Triade:

$$
C\\text{-regulator}
\\triangleright
H\_{B|B'}(B,B')
\\to
\\text{stable record}.
$$

**Befund**

Bester Ort für:

$$
\\omega\_{\\mathrm{lock}}.
$$

**Obstruktions-Ort**

Noch nicht formalisiert. Type-I/Type-III-Asymmetrie ist zunächst Algebra-/Dimensionsasymmetrie, nicht Orientierung.

**Status**

Weiterhin wichtigster offener $J$-Locking-Kandidat.

\---

§§X99§§28\. Multi-ToC / Detektor / Vielobjektstruktur

Dieser Abschnitt darf nicht als Rückfall in die Lesart „ToC-Knoten sind Teilchen“ verstanden werden. Viele Objekte entstehen nicht durch viele Knoten innerhalb eines einzelnen ToC, sondern durch viele lokale ToC-Fasern, deren Approximanten und Handoff-Daten relativ zueinander verklebt werden.

$$
{T\_i}\_{i\\in I}
\\Rightarrow
\\text{Multi-ToC/gluing structure},
\\qquad
T\_i\\text{-nodes}\\≠\\text{particles}.
$$

§§X100§§28.1 Mini-ToCs als Detektorelemente

**Ausgangslage**

Ein Detektor besteht aus vielen lokalen ToC-Fasern:

$$
T\_1,T\_2,\\ldots,T\_N.
$$

Jede trägt lokal:

$$
{J\_i,-J\_i}.
$$

**Befund**

Lokales Vorzeichen kann Gauge sein:

$$
J\_i\\mapsto -J\_i.
$$

Physikalisch relevant wären relative oder zyklische Daten:

$$
\\sigma\_{ij},
\\qquad
\\Phi\_\\gamma=\\prod\_{(ij)\\in\\gamma}\\sigma\_{ij}.
$$

**Obstruktions-Ort**

Mechanismus für $\\sigma\_{ij}$ ist noch nicht derived. Außerdem wäre ein Zyklusprodukt zunächst eine relative, gauge-invariante Struktur, nicht automatisch ein absolutes $J$-Vorzeichen:

$$
\\Phi\_\\gamma=\\prod\_{(ij)\\in\\gamma}\\sigma\_{ij}
\\quad\\Rightarrow\\quad
\\text{relative orientation},
$$

aber nicht unmittelbar

$$
\\Rightarrow\\text{absolute orientation}.
$$

**Status**

Starker Kandidat für nächsten nichtlokalen Test. Methodisch gilt:

$$
\\text{relative orientation}\\≠\\text{absolute orientation}.
$$

\---

§§X101§§28.2 Frustration / Spin-netz-artige Struktur

**Ausgangslage**

Viele lokale ToC-Fasern werden gekoppelt. Mögliches Zyklusprodukt:

$$
\\Phi\_\\gamma=-1.
$$

**Befund**

Falls $\\Phi\_\\gamma$ invariant unter lokalen Gauge-Flips

$$
J\_i\\mapsto -J\_i
$$

ist, entsteht echte globale Frustration.

**Obstruktions-Ort**

$\\sigma\_{ij}$ darf nicht gesetzt werden. Auch ein nichttriviales $\\Phi\_\\gamma$ wäre zunächst eine globale Sektor-/Frustrationsstruktur. Es müsste zusätzlich gezeigt werden, dass daraus ein orientierter Record oder ein $\\omega\_{\\mathrm{lock}}$ folgt, nicht nur eine relative Holonomieklasse.

**Status**

Wichtigster offener Multi-ToC-Testpfad. Positiv wäre hier zuerst eine gauge-invariante relative Struktur; das absolute $J$-Vorzeichen bliebe danach separat zu prüfen.

\---

§§X102§§29\. Motor-/Mehrphasen-Analogie

§§X103§§29.1 Zweiphasiger Dreiphasenmotor

**Ausgangslage**

Zweiphasig erzeugt ein Dreiphasenmotor kein stabil gerichtetes Drehfeld, sondern Überlagerung:

$$
\\text{forward rotating field}+\\text{reverse rotating field}.
$$

**Befund**

Gute Analogie zu:

$$
{+J,-J}.
$$

**Obstruktions-Ort**

Ohne dritte Phasenordnung bzw. Anschlussordnung kein stabiler Drehsinn.

**Status**

Didaktisch stark. Technische Lesart: Der reelle passive Dirichlet-/Widerstandssektor kann Imbalance, Achse und Pulsation erzeugen, aber keine eigenständige Phase. Die fehlende Rolle ist die eines abgeleiteten kapazitiven/speichernden/skew-Hamilton-artigen Sektors oder eines äquivalenten Handoff-Lockings.

\---

§§X104§§29.2 Drei Phasen / Anschlussordnung

**Ausgangslage**

Balanciertes System:

$$
(1,a,a^2),
\\qquad
a=e^{2\\pi i/3}.
$$

Vertauschung:

$$
(1,a,a^2)
\\leftrightarrow
(1,a^2,a).
$$

**Befund**

Drehrichtung liegt in der Anschlussordnung.

**CNNA-Übersetzung**

Nicht lokales (J\_i)-Vorzeichen, sondern Handoff-Sequenz bzw. Zyklusordnung könnte entscheidend sein.

**Obstruktions-Ort**

Anschlussordnung muss derived sein.

**Status**

Guter Kandidat für Multi-ToC-Handoff-Sequence-Gate.

\---

§§X105§§30\. Cayley-Dickson / höhere Divisionsalgebren

§§X106§§30.1 CD-/Hurwitz-Kandidat

**Ausgangslage**

Route:

$$
\\mathbb R\\to\\mathbb C\\to\\mathbb H\\to\\mathbb O.
$$

**Befund**

Für das erste $J$-Vorzeichenproblem negativ. Höhere Algebra löst nicht die Herkunft der ersten komplexen Orientierung.

**Obstruktions-Ort**

Dimensionsverdopplung und Normmultiplikativität werden nicht aus Schnittdaten erzwungen.

Offene Objekte:

§§X162§§

**Status**

Nicht aktueller Weg für $J$-Vorzeichen. Als spätere Zielstruktur nicht ausgeschlossen.

\---

§§X107§§31\. Substratwechsel-Kandidaten

§§X108§§31.1 ToC bleibt lokale Provenienzfaser

**Ausgangslage**

Der b-äre Einzelbaum als flacher ToC-Referenzsektor scheitert unter den flach-reziproken Derived-only-Prämissen am $J$-Gate. Damit ist nicht das ToC-Konzept insgesamt obstruiert, sondern nur die spezielle Lesart, dass ein einzelner b-ärer Baum globaler Träger des Universums und zugleich Ursprung einer ausgezeichnet gerichteten komplexen Struktur sein kann.

**Befund**

Als lokale Faser bleibt ToC wertvoll. Der präzise Rollenpfad lautet:

$$
\\text{ToC node}
\\to
\\text{provenance index}
\\to
\\text{approximant}
\\to
\\text{Schur/DtN}
\\to
\\text{local handoff operator}
\\to
\\text{possible physical degree of freedom}.
$$

Ein endlicher Approximant ist daher zunächst ein effektiver lokaler Handoff-/Objektkandidat, kein automatisch gegebenes Vielteilchensystem.

**Obstruktions-Ort**

Globale Ontologie als einzelner Baum ist zu arm für zweite Achse, Chirotopie, Krümmung. Umgekehrt wäre die direkte Deutung von ToC-Knoten als physikalische Freiheitsgrade ein Rollenfehler.

**Status**

Kein Totalverwerfen des ToC und keine Falsifikation der Complement Net Architecture; Rollenwechsel:

$$
\\boxed{\\text{b-ary single tree is not a world tree, but a local provenance fiber.}}
$$

Die Komplementseite bleibt im Gegenteil strukturell notwendig, sobald lokale Handoff-Operatoren, lokale Algebren, relative Komplemente und spätere AQFT-Anschlussbedingungen ernst genommen werden.

\---

§§X109§§31.2 Ereignisstrukturen als Vergleichsstruktur, kein Fundament

**Ausgangslage**

Ereignisstrukturen besitzen typischerweise zwei Relationen:

$$
\\leq \\qquad\\text{and}\\qquad #.
$$

Dabei ist $\\leq$ nicht neutral, sobald es als kausale oder zeitartige Ordnung gelesen wird. Die Relation $#$ markiert Konflikt, Inkompatibilität oder Exklusion.

**Befund**

Als spätere Ziel- oder Vergleichsstruktur sind Ereignisstrukturen interessant. Sie könnten beschreiben, wie aus einer CNNA-derived Vorstruktur emergente Ereignisse, Konflikte und eine kausale Ordnung entstehen.

Die zulässige Richtung ist daher:

$$
\\text{CNNA-derived non-causal pre-structure}
\\longrightarrow
\\text{emergent events}
\\longrightarrow
(E,\\leq,#).
$$

**Obstruktions-Ort**

Als Fundament wären Ereignisstrukturen zu stark. Die Relation $\\leq$ würde Kausalität bzw. Zeitordnung bereits als primitives Datum einführen. Damit würde genau das gesetzt, was CNNA erst rekonstruieren müsste.

Die unzulässige Richtung wäre:

$$
(E,\\leq,#)
\\longrightarrow
\\text{CNNA foundation}.
$$

Das wäre methodisch derselbe Importtyp wie:

$$
\\text{set complex numbers},\\qquad
\\text{set orientation},\\qquad
\\text{set tensor product},\\qquad
\\text{set Hodge star}.
$$

Nur wäre der importierte Inhalt hier:

$$
\\boxed{
\\text{set causality.}
}
$$

**Status**

Ereignisstrukturen sind als nächster Fundament-Kandidat zurückzustufen. Sie bleiben Ziel-/Vergleichsstruktur, aber kein zulässiger Substratkern vor einer abgeleiteten Kausalitätsrekonstruktion.

$$
\\boxed{
\\text{Event structures: comparison structure yes, foundation no.}
}
$$

§§X110§§31.3 Nicht-kausaler Substratwechsel-Gate

**Ausgangslage**

Der b-äre Einzelbaum ist als globaler Weltbaum für den $J$-Sektor unter den flach-reziproken Derived-only-Prämissen falsifiziert. Daraus folgt nicht, dass beliebig reichere relationale Substrate zulässig sind. Ein neues Substrat darf nicht einfach die fehlenden Zielstrukturen als primitive Relationen enthalten.

**Befund**

Ein zulässiger nächster Substratkandidat muss mindestens folgende Ausschlüsse erfüllen:

$$
\\boxed{
\\text{no primitive }i,\\quad
\\text{no primitive }J,\\quad
\\text{no primitive chiropathy,}\\quad
\\text{no primitive orientation,}\\quad
\\text{no primitive tensor factorization,}\\quad
\\text{no primitive causal order.}
}
$$

Er darf eine nicht-kausale relationale, kombinatorische oder topologische Vorstruktur tragen, solange deren spätere kausale Lesart erst durch Handoff, Regimebildung, Spektralstruktur, Regulatoren oder Backreaction erzwungen wird.

**Obstruktions-Ort**

Jedes Substrat, das bereits eine gerichtete Zeit-, Kausal-, Orientierungs- oder Phasenstruktur enthält, umgeht den eigentlichen CNNA-Test. Dann wäre die fehlende zweite Achse nicht abgeleitet, sondern importiert.

**Status**

Der strengste derzeit zulässige Zwischenschritt bleibt daher das nicht-kausale IDEAL-ToC-Faser-Gitter als flacher Referenztest:

$$
\\mathcal I\_{\\mathrm{ToCGrid}}=\\Gamma\_\\infty\\times T\_b^\\infty,\\qquad
\\Omega\_{R,L}=W\_R\\times T\_{\\le L}.
$$

Hier ist $\\Gamma\_\\infty$ nur ein homogener relationaler Indexträger, nicht bereits Raumzeit und nicht bereits Kausalordnung. Jede metrische, räumliche, gerichtete oder orientierte Lesart von $\\Gamma\_\\infty$ ist Vergleichs-/Teststruktur und kein ontischer Input.

\---

§§X111§§31.4 Sierpinski-Teppich als nicht-p.c.f.-Stressklasse

**Ausgangslage**

Der Sierpinski-Teppich ist als nicht-p.c.f.-Stressklasse interessanter als SG/ST, wenn man mehrskalige Boundary-/Trace-Strukturen testen will. Der Mengerschwamm wird in dieser Fassung nicht weiterverfolgt.

**Befund**

Nicht-p.c.f.-Struktur bedeutet: wildere, mehrskalige Schnitt- und Randkontakte sind möglich. Das kann für Handoff-, Trace-, Gluing- und Frustrationstests nützlich sein:

$$
\\text{non-p.c.f.}
\\Rightarrow
\\text{more irregular, multiscale boundary/trace structure}.
$$

**Obstruktions-Ort**

Mehr Löcher oder wildere Randstruktur liefern aber nicht automatisch eine derived-only Orientierung:

$$
\\text{more holes}
\\neq
\\text{derived }J\\text{-sign}.
$$

Insbesondere bleibt zu prüfen, ob jede verwendete Umlaufs-, Flächen-, Trace- oder Hodge-artige Struktur wirklich aus der nicht-kausalen Vorstruktur entsteht oder durch Einbettung/Orientierung importiert wurde.

**Status**

Sinnvolle Substrat-Stressklasse, aber kein aktueller Fundament-Kandidat und keine Lösung des $J$-Vorzeichenproblems.

\---

§§X112§§32\. Ausgewiesene Artefaktlage dieser Fassung

Diese Fassung nennt nur noch Artefakte, die entweder angehängt wurden oder als Hugging-Face-Visualisierung ausdrücklich referenziert sind. Ältere Paketnamen, nicht angehängte Nachtests und hypothetische nächste Implementierungen werden nicht mehr als reproduzierbare Artefaktbasis dieser Datei geführt.

§§X113§§32.1 Hugging-Face-Visualisierung

§§X163§§

Die Visualisierung dient der Anschauung des ToC-/Approximanten-/UV-/Environment-Konzepts. Sie ist selbst nur eine Proxy- und Darstellungsebene; Tilt-, Winkel- oder Chartwerte daraus sind nicht als Schur-/DtN-Invarianten zu lesen.

§§X114§§32.2 Anhang §§X169§§

§§X164§§

Der Anhang enthält außerdem zugehörige CSV-, JSON-, PNG- und Markdown-Reports. Diese Artefakte bilden die ausgewiesene reproduzierbare Basis für die $\\alpha\_{\\mathrm{orth}}$-, Flow-Sign-, Cauchy-Shell-, Familien-Handoff-, triadischen Interface- und UV/Env-Cauchy-Shell-Befunde dieser Fassung.

§§X115§§32.3 Anhang §§X170§§

§§X165§§

Dieser Anhang dokumentiert den Baum-vs.-Sierpinski-Gasket-Kontrolltest: Baum als $b\_1=0$-Kontrollgruppe, Gasket als nichttrivialer $H\_1$-Stressfall, generatorischer $\\kappa$-Blindheitstest und $H\_1$-Dynamiktest.

\---

§§X116§§33\. Obstruktions-Orte nach Typ

§§X117§§33.1 Reziprozität

$$
\\Lambda=\\Lambda^T.
$$

Passive Schur-/DtN-Reduktion bleibt symmetrisch. Kein antisymmetrischer $J$-Generator.

§§X118§§33.2 Reelle Konjugationssymmetrie

$$
J\\mapsto -J.
$$

Reelle Strukturen wählen keine komplexe Orientierung.

§§X119§§33.3 $S\_b$-Äquivarianz

Geschwisterpermutationen halten kanonische Größen im trivialen Sektor. Keine Signum-Auswahl.

§§X120§§33.4 Radiale Einachsenstruktur (F1)

F1 liefert Ordnung:

$$
n\\to n+1.
$$

Aber nur entlang einer Achse. Nichtkommutativität braucht zwei unabhängige Achsen.

§§X121§§33.5 Degenerazien

Entartete Eigenräume dürfen nicht durch willkürliche numerische Basis geschnitten werden. Nur ganze Cluster sind relabeling-natürlich.

§§X122§§33.6 Keine partielle Spur auf Knoten

$$
\\mathbb R^N=\\mathbb R^S\\oplus\\mathbb R^E
$$

ist direkte Summe, kein Tensorprodukt.

§§X123§§33.7 Bit-Reversal

Adresszyklen können Drehsinn spiegeln:

$$
\\rho\\leftrightarrow\\rho^{-1}.
$$

§§X124§§33.8 Boundary reversal

UV/Env-Ko-Orientierung liefert:

$$
{J,-J}.
$$

§§X125§§33.9 Handoff reversal

$$
A\_{\\gamma^{-1}}=-A\_\\gamma.
$$

Ohne gerichtete Handoff-Sequenz kein absoluter Drehsinn.

§§X126§§33.10 OQS-Abhängigkeit von $i$

Lindblad/OQS kann Zeitrichtung liefern, setzt aber Hamilton-$i$ voraus.

§§X127§§33.11 Flachheit

Im flachen ToC-/DtN-Sektor fehlen:

$$
\\text{Connection},
\\qquad
\\text{Holonomy},
\\qquad
\\text{Curvature}.
$$

§§X128§§33.12 Kausalitätsimport

Eine primitive kausale Ordnung $\\leq$ ist kein neutraler Strukturträger. Sie würde bereits Zeit-/Kausalstruktur mitbringen und damit den späteren Rekonstruktionsschritt überspringen.

$$
\\boxed{
(E,\\leq,#)\\text{ is a target structure, not a foundation.}
}
$$

Der zulässige Test lautet daher nicht, ob ein kausales Substrat CNNA tragen kann, sondern ob CNNA aus einer nicht-kausalen Vorstruktur eine kausale Ordnung erzeugen kann.

\---

§§X129§§34\. Aktuelle Gesamtformel

$$
\\boxed{
\\text{All single-tree, single-approximant, passive Schur/DtN, and local triad tests end at }{J,-J}.
}
$$

$$
\\boxed{
\\text{Exact and cluster-safe handoff operators in the flat ToC/DtN sector commute.}
}
$$

$$
\\boxed{
\\text{Non-commutativity arises so far only through imposed order, non-canonical truncation, or asymmetric boundary conditions.}
}
$$

$$
\\boxed{
\\text{ToC nodes are provenance indices, not physical degrees of freedom.}
}
$$

$$
\\boxed{
\\text{The b-ary tree was chosen as the provenance side of SG/ST: }SG\\leftrightarrow b=3,\\quad ST\\leftrightarrow b=4.
}
$$

$$
\\boxed{
\\text{What is obstructed is not CNNA and not ToC in general, but rather the b-ary single tree as a global carrier of directed complex structure.}
}
$$

$$
\\boxed{
\\text{Complementary, handoff, and local algebra structures remain positively relevant for the AQFT connection.}
}
$$

$$
\\boxed{
\\text{UV/Env generate a genuine radial scale break, but no chirality.}
}
$$

$$
\\boxed{
\\omega\_\\partial\\Rightarrow{+J,-J},
\\qquad
\\omega\_{\\mathrm{lock}}\\text{ remains the open locking object.}
}
$$

$$
\\boxed{
\\text{Relative holonomy/frustration is not automatically absolute orientation.}
}
$$

$$
\\boxed{
\\text{The next genuine positive search space is not another flat single-ToC test, but rather curved-sector, multi-ToC frustration, or triadic handoff locking.}
}
$$

Der wichtigste nächste ToC-nahe Test vor Substratwechsel bleibt:

$$
\\boxed{
\\mathcal I\_{\\mathrm{ToCGrid}}=\\Gamma\_\\infty\\times T\_b^\\infty,
\\qquad
\\Omega\_{R,L}=W\_R\\times T\_{\\le L},
\\qquad
\\Lambda\_{R,L}.
}
$$

Ziel:

$$
\\text{Testing effective geometry based on subsystem membership},
$$

aber getrennt davon:

$$
\\text{Keep the J-/chirotopy-/non-commutativity gate open}.
$$

