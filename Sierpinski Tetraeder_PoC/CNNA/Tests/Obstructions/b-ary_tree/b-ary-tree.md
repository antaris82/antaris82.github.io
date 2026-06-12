# CNNA-ToC / J-Vorzeichen / Nichtkommutativität — vollständiges Test- und Obstruktionsinventar

Status: nach Chatstand, nicht als Lean-Theorem. Die meisten Befunde sind numerisch, konzeptionell oder aus den hier ausgewiesenen Diagnostikartefakten abgeleitet. Das zentrale Ergebnis ist inzwischen präziser als am Anfang:

Diese Fassung enthält zusätzlich das Substrat-Gate aus der überarbeiteten Paper-Fassung: Ereignisstrukturen werden nicht mehr als zulässiger Fundament-Kandidat behandelt, weil sie mit $\leq$ bereits eine kausale bzw. zeitartige Ordnung als primitives Datum enthalten würden. Sie bleiben nur Vergleichs- oder Zielstruktur.

> **Kernaussage.** Der flache, reellwertige, reziproke ToC-/Schur-/DtN-Sektor erzeugt keine ausgezeichnete J-Orientierung.

Er liefert mehrfach:

$$
\{+J,-J\},\qquad \{+\tau,-\tau\},\qquad \text{radiale Ordnung},\qquad \text{DtN-/Spektralstruktur}.
$$

Er liefert bisher nicht:

> **Kernaussage.** J statt -J.

Die einheitliche Obstruktion lautet jetzt nicht mehr nur „Symmetrie“, sondern genauer:

> **Kernaussage.** Eine abgeleitete Achse F1 genügt nicht. Nichtkommutativität braucht mindestens zwei nicht gemeinsam diagonalisierbare abgeleitete Operatorachsen; Chiralität braucht zusätzlich eine abgeleitete Orientierungs- bzw. Sign-Line-Auswahl.

---

# 0. Globaler Status der Testreihe

## 0.0 Warum der b-äre Baum als ToC-Referenzsubstrat gewählt wurde

Der b-äre Baum wurde nicht als beliebiger Weltbaum eingeführt. Die historische Motivation war die Zahmheit, Symmetrie und Skaleninvarianz von Sierpinski-Gasket (SG) und Sierpinski-Tetraeder/Tetrix (ST). Diese Objekte waren attraktiv, weil sie eine kontrollierte, selbstähnliche und hochsymmetrische Testklasse bilden. Gerade diese Zahmheit war methodisch wichtig: Wenn bereits der symmetrischste und kontrollierteste Kandidat die gewünschte Richtung von $J$ nicht erzwingt, dann liegt die Obstruktion nicht an numerischer Wildheit, sondern an der Struktur des flachen reell-reziproken Sektors.

Die zugehörige Provenienzseite von SG und ST ist ein b-ärer Adressbaum. Für das Sierpinski-Gasket ist der natürliche Adressbaum 3-är, für den Sierpinski-Tetraeder 4-är:

$$
SG:\quad b=3,\qquad ST:\quad b=4.
$$

Vor jeder geometrischen Einbettung, vor jeder Quotientrelation und vor jeder Orientierung liegt die reine Adress-/Provenienzstruktur

$$
A_b^{<\omega}.
$$

Der b-äre Baum ist daher die bewusst entkleidete Provenienzseite der SG/ST-Kandidaten. Er entfernt genau jene Strukturen, die nicht als Input zulässig sind: eingebettete Geometrie, zyklische Vertexordnung, Flächenorientierung, Hodge-Stern, komplexe Phase und gerichtete Zeit.

In dieser Datei bezeichnet der flache ToC-Referenzsektor daher nicht das gesamte mögliche CNNA-ToC-Konzept, sondern den minimalen b-ären Provenienzskelett-Sektor des Tree-of-Cliques/ToC. Clique-, Zell-, Quotient-, Gluing- oder lokale-Algebra-Anreicherungen sind spätere Schichten und dürfen nicht rückwirkend als Beweis für den flachen Einzelbaum gelesen werden.

Die Testfrage war damit absichtlich streng:

$$
\text{Kann die reine SG/ST-Provenienzseite bereits }J\text{ statt }-J\text{ erzwingen?}
$$

Der bisherige Befund lautet: nein, nicht im flachen, homogenen, reell-reziproken und relabeling-natürlichen Einzelbaumsektor. Das falsifiziert nicht CNNA und nicht das ToC-Konzept insgesamt. Es begrenzt die Rolle des b-ären Baums als einzelner globaler Träger komplexer gerichteter Strukturen.

## 0.1 Was A/ToC bisher positiv liefert

Der ToC-/DtN-Sektor liefert robuste Vorläufer:

$$
\text{radiale Provenienzordnung},
$$

$$
\text{UV/Env-Ko-Orientierung},
$$

$$
\text{Cauchy-Doppelung},
$$

$$
\{+J,-J\},
$$

$$
\{+\tau,-\tau\},
$$

$$
\text{reelle DtN-/Schur-Handoff-Matrizen}.
$$

Das ist nicht trivial. Es bedeutet:

> **Kernaussage.** Der ToC ist als lokale Provenienzfaser und flacher Referenzsektor wertvoll.

Dabei ist ein ToC-Knoten kein physikalischer Freiheitsgrad. Seine Rolle im flachen Referenzsektor ist zunächst die eines Provenienzindex. Der zulässige Lesepfad ist:

$$
\text{ToC-Knoten} \to \text{Provenienzindex} \to \text{Approximant} \to \text{Schur/DtN} \to \text{effektiver Handoff-Operator} \to \text{möglicher physikalischer Freiheitsgrad}.
$$

Nicht zulässig ist die Kurzidentifikation:

$$
\text{ToC-Knoten}=\text{physikalischer Freiheitsgrad}.
$$

## 0.2 Was A/ToC bisher nicht liefert

Nicht geliefert wird eine absolute Orientierung:

> **Kernaussage.** J ≠ derived uniquely from flat ToC data.

Auch nicht geliefert werden bisher:

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

## 0.3 Neuer Interpretationsstatus

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
\text{lokale Algebren} \Longleftrightarrow \text{Komplement-/Schnitt-/Handoff-Strukturen bleiben zentral.}
$$

Der Rollenwechsel ist daher selbst ein Finding:

> **Kernaussage.** Der b-äre Einzelbaum ist nicht Weltbaum, sondern lokale Provenienzfaser.

Ein endlicher Approximant ist entsprechend nicht automatisch ein Vielteilchensystem. Er ist zunächst ein effektiver lokaler Handoff-/Objektkandidat:

$$
\Omega(a,L) \Rightarrow \text{effektiver lokaler Handoff-/Objektkandidat}.
$$

Viele Objekte, Detektoren oder Vakuum-Gluing-Strukturen entstehen erst aus einer Familie lokaler Fasern und deren Verklebungen:

$$
\{T_i\}_{i\in I} \Rightarrow \text{Multi-ToC-/Gluing-Struktur}.
$$

## 0.4 Zusätzlicher Substrat-Gate: keine primitive Kausalität

Der nächste Substratkandidat darf keine der Strukturen enthalten, die CNNA erst rekonstruieren soll:

> **Kernaussage.** kein primitives i, · kein primitives J, · keine primitive Orientierung, · keine primitive Tensorstruktur, · keine primitive Kausalität.

Insbesondere sind Ereignisstrukturen mit einer gegebenen Relation $\leq$ nicht als Fundament zulässig, sofern $\leq$ kausal oder zeitartig gelesen wird. Eine solche Relation würde bereits eine Zeit-/Kausalordnung einführen. Zulässig ist nur die umgekehrte Richtung:

$$
\text{nicht-kausale CNNA-Vorstruktur} \longrightarrow \text{emergente Ereignisse} \longrightarrow \text{emergente kausale Ordnung}.
$$

Damit wird der Substratwechsel-Gate verschärft: Gesucht ist nicht einfach ein reichhaltigeres Substrat, sondern ein reichhaltigeres Substrat ohne importierte Kausalität.

---

## 0.5 Grunddefinitionen des flachen ToC-Sektors

Dieser Abschnitt fixiert die Minimalnotation, auf die alle folgenden Tests bezogen sind. Er ist keine zusätzliche physikalische Annahme, sondern eine Konventionsschicht für den flachen, homogenen, reell-reziproken ToC-/DtN-Referenzsektor. Dieser Referenzsektor ist die b-äre Provenienzseite der SG/ST-Motivation und nicht das vollständige CNNA-ToC-Konzept mit möglichen Clique-, Gluing-, Regulator- oder lokalen-Algebra-Anreicherungen.

### 0.5.1 Adressalphabet, Wortbaum und Konkatenation

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

### 0.5.2 Präfixordnung, Kantenrelation und Unit-edge-Graph

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
E_b^\infty = \{\{w,wi\}:w\in T_b^\infty,\ i\in A_b\}.
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

### 0.5.3 Relabeling-Gauge und Kanonizitätsbedingung

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

### 0.5.4 Endliche Approximanten als induzierte Teilgraphen

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
E_\Omega = \{\{x,y\}\in E_b^\infty:x,y\in\Omega(a,L)\}.
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

Er zerfällt in den UV-tail an den Blättern von $\Omega(a,L)$ und, falls $k>0$, in den Environment-Anteil auf der Parent-/Root-Seite. Für $k=0$ gilt die noOuterEnvironment-Lesart.

Der Environment-Port ist, falls $k>0$, der rootseitige Interface-Port am Approximantenroot $a$. Er ist kein zusätzlicher Bright-Knoten, sondern die Schnittstelle zur äußeren Komplementseite.

Auch der Approximant selbst hat zunächst eine Rollenbegrenzung: $\Omega(a,L)$ ist kein automatisch interpretiertes Vielteilchensystem. Im flachen ToC-Sektor ist er ein schnittrelativer lokaler Handoff-/Objektkandidat. Erst die aus ihm erzeugten Schur-/DtN-Daten und spätere Gluing-/Regimebildungen können physikalische Freiheitsgrade oder Vielobjektstruktur tragen.

### 0.5.5 Bright-Laplaceoperator und Komplement-Loads

Der Bright-Laplaceoperator $L_\Omega$ ist der Laplaceoperator des induzierten Bright-Graphen $G_\Omega$:

$$
(L_\Omega)_{xy} = \begin{cases} d_\Omega(x), & x=y,\\ -1, & x\sim y\text{ innerhalb von }\Omega,\\ 0, & \text{sonst}. \end{cases}
$$

Dabei ist

$$
d_\Omega(x)=|\{y\in\Omega:x\sim y\}|
$$

und zählt nur Nachbarn innerhalb von $\Omega$. Komplementzweige werden nicht in $\deg_\Omega$ mitgezählt. Ihre Wirkung wird ausschließlich über Schur-/DtN-/Load-Terme ergänzt. Dadurch wird eine Doppelzählung von Außenkanten vermieden.

Ein UV-cut oder Environment-cut ist bereits eine Dirichlet-artige Randsetzung. Die Schur-/DtN-Eliminierung wird daher nicht durch eine externe numerische Regularisierung stabilisiert, sondern durch den schnittrelativen Boundary-Status selbst. Die Regularisierung ist schnittintern:

$$
\text{UV-cut oder Environment-cut} \Rightarrow \text{Dirichlet-Boundary} \Rightarrow L_{II}^{-1}\text{ wohldefiniert},
$$

sofern der betrachtete Innenblock tatsächlich an die gesetzte Boundary gekoppelt ist. Externe Hilfssetzungen wie Ridge-Terme, Pseudoinversen oder künstliche Massenterme gehören nicht zum flachen derived-only ToC-/DtN-Kern.

Der effektive Operator hat die Form

$$
M_\Omega=L_\Omega+\Sigma_{\mathrm{Env}}+\Sigma_{\mathrm{UV}}.
$$

Im einfachsten load-basierten Proxy kann man schreiben

$$
\Sigma_{\mathrm{Env}} = \sigma_{\mathrm{Env}}\,P_{\mathrm{root}}, \qquad \Sigma_{\mathrm{UV}} = \sigma_{\mathrm{UV}}\,P_{\partial_{\mathrm{UV}}\Omega},
$$

wobei dies nur dann als derived gilt, wenn die Werte aus einer expliziten Schur-/DtN-Eliminierung der jeweiligen Komplementfamilien stammen. Frühe Konstanten- oder Ladder-Modelle für $\sigma_{\mathrm{Env}}$ bzw. $\alpha_{\mathrm{Env}}$ sind Diagnosemodelle, keine ontischen CNNA-Eingaben.

Die beiden Loads wirken an entgegengesetzten Seiten des Approximanten:

$$
\Sigma_{\mathrm{UV}}\text{ wirkt leafseitig an den feinsten/cut-Knoten}, \qquad \Sigma_{\mathrm{Env}}\text{ wirkt rootseitig am Parent-/Environment-Port}.
$$

Damit erzeugt der Schnitt einen echten inneren Skalenbruch des Approximanten. Dieser Skalenbruch ist jedoch zunächst radial bzw. longitudinal:

$$
\text{UV/Env-Skalenbruch}\neq\text{Chiralität}.
$$

Für Kanalquellen $f_{\mathrm{Env}}$ und $f_{\mathrm{UV}}$ sind die Antworten

$$
u_{\mathrm{Env}}=M_\Omega^{-1}f_{\mathrm{Env}}, \qquad u_{\mathrm{UV}}=M_\Omega^{-1}f_{\mathrm{UV}}.
$$

Standarddiagnostisch ist $f_{\mathrm{Env}}$ eine rootseitige Quelle am Environment-Port und $f_{\mathrm{UV}}$ eine symmetrische bzw. normierte Blattquelle auf $\partial_{\mathrm{UV}}\Omega$. Jede abweichende Normierung muss im jeweiligen Artefakt explizit dokumentiert werden.

Das Energie-Innenprodukt ist

$$
\langle x,y\rangle_M=x^TM_\Omega y.
$$

Die Orthogonalitätsdiagnose ist

$$
\rho_M = \frac{\langle u_{\mathrm{Env}},u_{\mathrm{UV}}\rangle_M} {\|u_{\mathrm{Env}}\|_M\|u_{\mathrm{UV}}\|_M}.
$$

Die Größen $\alpha_{\mathrm{UV}}$, $\alpha_{\mathrm{Env}}$, $C_k$ und $\Xi$ sind Diagnosegrößen, solange sie nicht aus den vollständigen Komplementfamilien abgeleitet sind. In den frühen Tests bedeutet $C_k$ eine schnitt- bzw. tiefenabhängige Normierungs-/Kapazitätsgröße des Approximanten; ihr genauer Wert ist artefakt- bzw. diagnostikabhängig und daher nicht als universale CNNA-Konstante zu lesen.

### 0.5.6 J-Problem, F1/F2 und Locking-Objekt

Eine komplexe Struktur auf einem reellen Handoff-Raum ist ein Endomorphismus $J$ mit

$$
J^2=-I.
$$

Das $J$-Vorzeichenproblem ist nicht die bloße Existenz eines solchen Blocks, sondern die derived-only-Auswahl von $J$ gegenüber $-J$. Eine reelle, symmetrische, relabeling-natürliche Struktur liefert daher höchstens

$$
\{+J,-J\},
$$

solange keine zusätzliche abgeleitete Orientierungs- oder Locking-Struktur vorliegt.

$F1$ bezeichnet die radiale Provenienz-/Tiefenachse

$$
|w|\mapsto |w|+1.
$$

Eine zweite Achse $F2$ ist kein Input, sondern ein offenes Zielobjekt: eine unabhängig abgeleitete transversale Struktur, die nicht durch volle $S_b$-Symmetrie trivialisiert wird.

$\omega_{\mathrm{lock}}$ bezeichnet die noch offene Handoff-Form, die eine $J$-Orientierung mit einer Fluss-/Zeit-/Handoff-Orientierung $\tau$ koppeln müsste. Sie ist nicht identisch mit einer bloßen Cauchy-Randform, solange diese nur

$$
\{+J,-J\}
$$

liefert. Die Cauchy-Shell kann also positiv sein, ohne das eigentliche Locking-Problem zu lösen:

$$
\omega_\partial\Rightarrow\{+J,-J\}, \qquad \omega_{\mathrm{lock}}:(J,\tau)\mapsto\text{stabiler orientierter Record}.
$$

---

# 1. Didaktische und Proxy-Tests

## 1.1 Hugging-Face-ToC-Concept-Explorer

**Artefaktbezug**

```text
Hugging-Face-Space: https://huggingface.co/spaces/antaris/b-ary_tree
app.py
```

`app.py` ist das Visualisierungsskript des Hugging-Face-Spaces. Es dient nur der Anschauung und nicht als Beweis- oder Primärdiagnostik.

**Ausgangslage**

Visualisierung eines (b)-ären ToC mit Parametern:

$$
b,\qquad L_{\max},\qquad \text{Approximant root},\qquad L.
$$

Dargestellte Stufen:

$$
\text{ToC} \to \text{proper subsystem} \to \text{UV-tail} \to \text{Environment} \to \text{Cauchy-/}J\text{-Kandidat} \to \text{Complex-plane overlay}.
$$

**Befund**

Didaktisch stark. Es trennt sichtbar:

$$
\text{Approximant}, \qquad \text{UV-tail}, \qquad \text{Environment}, \qquad \text{Interface}.
$$

**Obstruktions-Ort**

Visualisierung ist kein Beweis. Frühe Tilt-/Winkelwerte waren teilweise Chart-/Rendering-Proxies, nicht DtN-Invarianten.

**Status**

Didaktisch wertvoll, mathematisch sekundär.

---

## 1.2 Stage-6 Chart-Proxy / Tilt-Test

**Artefaktbezug**

Teil der Hugging-Face-Visualisierung `app.py`; nur Anschauungs- und Proxyebene.

**Ausgangslage**

Tiefe Einbettung von Approximanten, z. B.

$$
0.1,\qquad 0.1.1,\qquad 0.1.1.0,\ldots
$$

bei festen Parametern wie:

$$
b=3,\qquad L_{\max}=4.
$$

**Befund**

Visueller Tilt wurde mit tieferer Einbettung kleiner:

$$
|\mathrm{tilt}|\downarrow.
$$

**Interpretation**

Tiefer eingebettete Approximanten wirkten balancierter zwischen UV und Env.

**Obstruktions-Ort**

Kein echter Schur-/DtN-Wert:

$$
\text{Proxy} \neq \text{Invariante}.
$$

**Status**

Heuristische Motivation; später durch echte DtN-/Schur-Tests ersetzt.

---

# 2. Einzel-Approximant-Schur-/DtN-Tests

## 2.1 Projected-tail J-/Rotationstest

**Artefaktbezug**

Kein eigenständiger angehängter Artefakt in dieser Fassung; der Abschnitt bleibt als konsolidierter Befund aus der späteren $\alpha_{\mathrm{orth}}$- und DtN-Diagnostik.

**Ausgangslage**

Endlicher Approximant mit effektivem Operator:

$$
M=L_\Omega+\text{projected UV/Env loads}.
$$

Zwei Kanalantworten:

$$
u_{\mathrm{Env}},\qquad u_{\mathrm{UV}}.
$$

Messgröße:

$$
\rho_M = \frac{\langle u_{\mathrm{Env}},u_{\mathrm{UV}}\rangle_M} {\|u_{\mathrm{Env}}\|_M\,\|u_{\mathrm{UV}}\|_M}.
$$

**Befund**

Nahe Orthogonalität:

$$
|\rho_M|\ll 1,
$$

teilweise numerisch nahe $90^\circ$.

**Obstruktions-Ort**

Orthogonalität einer reellen 2-Ebene liefert höchstens:

$$
\{+J,-J\}.
$$

Die Ebene ist da; der Drehsinn nicht.

**Status**

Positiver Vorläufer einer prä-komplexen Ebene. Kein Vorzeichenbeweis.

---

## 2.2 Real finite-network Schur/DtN-Test

**Artefaktbezug**

Kein eigenständiger angehängter Artefakt in dieser Fassung; der Abschnitt bleibt als konsolidierter methodischer Befund.

**Ausgangslage**

Endlicher Baumgraph mit Laplace-Matrix:

$$
L_{\mathrm{graph}}.
$$

Rand (B), Innenknoten (I), Schur-Komplement:

$$
\Lambda_B = L_{BB}-L_{BI}L_{II}^{-1}L_{IB}.
$$

**Befund**

Für deterministische zentrierte Einzelmodi numerisch praktisch orthogonal, etwa:

$$
|\rho_M|\approx 10^{-18}.
$$

**Obstruktions-Ort**

Ein Einzelmodus kann orthogonal sein, während der volle Randantwortsraum noch Struktur trägt. Außerdem bleibt der DtN-Operator reell symmetrisch.

**Status**

Starker Hinweis auf echte Schur-/DtN-Orthogonalität in bestimmten Modi; kein $J$-Vorzeichen.

---

## 2.3 Dirichlet-/Cut-Regularisierungstest

**Artefaktbezug**

Kein eigenständiger angehängter Artefakt in dieser Fassung; der Abschnitt fixiert den methodischen Befund.

**Ausgangslage**

Frage:

$$
\text{Braucht man eine externe Regularisierung oder Pseudoinverse?}
$$

Genauer: Muss der Baum bzw. das Dirichlet-Netzwerk künstlich regularisiert werden, oder wirkt ein gesetzter UV- bzw. Environment-cut bereits selbst regularisierend?

**Befund**

Nein, das Dirichlet-Netzwerk muss nicht künstlich regularisiert werden. Ein echter UV-cut oder Environment-cut wirkt selbst bereits regularisierend, weil der entfernte Komplementanteil als Dirichlet-/Boundary-Seite behandelt wird. Dadurch wird der Innenblock

$$
L_{II}
$$

invertierbar, sofern der betrachtete Innenbereich tatsächlich an die gesetzte Boundary gekoppelt ist.

Die Regularisierung ist daher schnittintern:

$$
\text{UV-cut oder Environment-cut} \Rightarrow \text{Dirichlet-Boundary} \Rightarrow L_{II}^{-1}\text{ wohldefiniert}.
$$

Sie ist keine externe numerische Hilfssetzung:

$$
\text{kein Ridge},\qquad \text{keine Pseudoinverse},\qquad \text{kein künstlicher Massenterm}.
$$

**Obstruktions-Ort**

Der DtN-Operator bleibt cut-relativ:

$$
\Lambda_{\partial A}.
$$

Die schnittinterne Regularisierung liefert also einen wohldefinierten DtN-/Schur-Operator für den jeweiligen Cut, aber keinen cut-freien universalen DtN-Operator des ganzen unendlichen ToC.

**Status**

Wichtiges positives Ergebnis: UV- und Environment-cuts liefern die nötige Dirichlet-Regularisierung selbst. Keine Ridge-/Pseudoinversen-/Massenterm-Setzung nötig.

---

## 2.4 Harter UV/Env-Skalenbruch im Approximanten

**Artefaktbezug**

Konzeptionell aus den Schur-/DtN- und $\alpha_{\mathrm{orth}}$-Tests; in den angehängten Diagnostikartefakten über $M_\Omega$, $\Sigma_{\mathrm{UV}}$ und $\Sigma_{\mathrm{Env}}$ nachvollziehbar.

**Ausgangslage**

Ein proper subsystem besitzt zwei verschiedene Komplementseiten:

$$
\text{UV-tail an den feinsten/cut-Knoten}, \qquad \text{Environment am Root-/Parent-Port}.
$$

**Befund**

Die beiden Komplementprojektionen laden den Approximanten nicht gleichartig, sondern entgegengesetzt in der inneren Skalenrichtung:

$$
\text{UV-tail} \Rightarrow \text{Load an feinsten/cut-Knoten},
$$

$$
\text{Environment} \Rightarrow \text{Load am Root-/Parent-Port}.
$$

Also:

$$
\Sigma_{\mathrm{UV}}\text{ wirkt leafseitig}, \qquad \Sigma_{\mathrm{Env}}\text{ wirkt rootseitig}.
$$

Das ist ein echter harter Skalenbruch im Approximanten. Er ist nicht bloß Visualisierung oder Chart-Artefakt.

**Obstruktions-Ort**

Der Bruch ist radial bzw. longitudinal. Er unterscheidet innen/außen, fein/grob, UV/Environment, aber er erzeugt noch keine transversale Händigkeit:

$$
\text{Skalenbruch}\neq\text{Chiralität}.
$$

**Status**

Positives Finding für die Approximantenphysik und für $F1$. Kein $J$-Vorzeichenbeweis.

---

## 2.5 Passive Dirichlet-/Widerstandsnetzwerke erzeugen keine Phase

**Artefaktbezug**

Querschnittsbefund aus den realen Schur-/DtN-, Cauchy-Shell- und Motor-Analogie-Tests; kein eigenständiger zusätzlicher Artefaktbezug.

**Ausgangslage**

Der flache ToC-/DtN-Sektor ist reell, passiv und reziprok. Er verhält sich mathematisch wie ein Dirichlet-/Widerstandsnetzwerk mit Energieform, Diffusion und symmetrischer Randantwort.

**Befund**

Ein rein resistiver/passiver Sektor liefert Imbalance, Achsen, Loads, Dirichletenergie, Diffusion und DtN-Antworten:

$$
\text{passive resistance/load} \Rightarrow \text{imbalance/axis}.
$$

Er liefert aber keine eigenständige $90^\circ$-Phasenverschiebung und kein stabil gerichtetes Drehfeld:

$$
\text{passive resistance/load} \not\Rightarrow \text{rotating phase}.
$$

**Obstruktions-Ort**

Für Oszillation, Phase oder Hamilton-artige Rotation bräuchte man eine zweite Speicherstruktur, einen abgeleiteten skew-Sektor oder ein Handoff-Locking, das nicht bereits als komplexe Phase importiert wird.

**Status**

Technische Form der Motor-/Kondensator-Analogie: Reeller Widerstandssektor kann eine Achse und Pulsation liefern, aber nicht die fehlende Phase selbst.

---

# 3. alpha_orth- und Invarianten-Tests

## 3.1 Xi- / alpha_orth-Diagnostik

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/alpha_orth_invariant.py
```

**Ausgangslage**

Kontrollgröße:

$$
\Xi=(1+\lambda_{\mathrm{UV}})(1+\lambda_{\mathrm{Env}}),
$$

mit

$$
\lambda_{\mathrm{UV}} = \frac{b^k\alpha_{\mathrm{UV}}}{C_k}, \qquad \lambda_{\mathrm{Env}} = \frac{\alpha_{\mathrm{Env}}}{C_k}.
$$

Typische Orthogonalitätsdiagnose:

$$
|\rho|\sim \Xi^{-1/2}.
$$

**Befund**

Der UV-Term dominiert für wachsende Tiefe stark:

$$
|\rho|\sim b^{-k/2}.
$$

Also:

> **Kernaussage.** UV-Auflösung treibt Orthogonalität.

**Obstruktions-Ort**

$\alpha_{\mathrm{Env}}$ war in frühen Versionen modellabhängig:

```text
none
constant
power
exponential
ladder
```

Daher war der exakte Zahlenwert kein vollständig abgeleiteter physikalischer Wert.

**Status**

Gute Diagnosegröße. Kein Feinstrukturkonstanten-Claim. Kein $J$-Vorzeichen.

---

## 3.2 Environment-Sensitivitätsmodelle

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/alpha_orth_invariant.py
```

**Ausgangslage**

Vergleich verschiedener $\alpha_{\mathrm{Env}}$-Modelle.

**Befund**

Für große $k$ dominiert häufig der UV-Term so stark, dass die Environment-Modellwahl subdominant wird.

**Obstruktions-Ort**

In Regimen, in denen Environment nicht subdominant ist, braucht man eine echte Komplementfamilien-/DtN-Ableitung von $\alpha_{\mathrm{Env}}$.

**Status**

Guter methodischer Befund:

$$
\text{definierbar}\neq\text{erzwungen}.
$$

---

# 4. Parent–Child- und Handoff-Tests

## 4.1 Two-Approximant / Flow-Sign-Test

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/two_approximant_flow_sign.py
```

**Ausgangslage**

Parent–Child-Handoff:

$$
A_{\mathrm{parent}}\to A_{\mathrm{child}}.
$$

Ziel: prüfen, ob der Übergang ein $J$-Vorzeichen liefert.

**Befund**

Radiale Übergangssignaturen können entstehen.

**Obstruktions-Ort**

Radialität ist nicht Chiralität:

$$
\text{Parent}\to\text{Child}
$$

liefert Tieferichtung, aber keinen Drehsinn.

Außerdem kann Flow leicht durch Anregungsrichtung ein Vorzeichen einschmuggeln.

**Status**

Radiale Handoff-Struktur: ja. $J$-Vorzeichen: nein.

---

## 4.2 Schur-vor-Flow-Kriterium

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

> **Kernaussage.** Schur zuerst, Flow nur als Konsistenztest.

---

# 5. Zwei-Rand-/Shell-Chiralitätstests

## 5.1 V4 — Two-boundary shell chirality

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/two_boundary_shell_chirality.py
```

**Ausgangslage**

Parent–Child-Differenzschale, zwei Boundary-Ports, reale DtN-Matrix:

$$
\Lambda_\Delta.
$$

Cauchy-Paarung:

$$
\omega((q,p),(q',p'))=q^Tp'-p^Tq'.
$$

Auf einem DtN-Graphen gilt:

$$
p=\Lambda q.
$$

**Befund**

Für selbstadjungierten DtN-Graphen:

$$
\omega((q,\Lambda q),(r,\Lambda r)) = q^T\Lambda r-r^T\Lambda q = 0.
$$

**Obstruktions-Ort**

Ein einzelner passiver symmetrischer DtN-Graph ist Lagrangesch.

**Status**

Sauberes Negativergebnis. Zu eng für Familien-/Handoff-Tests, aber korrekt für Einzelgraph.

---

## 5.2 V5 — Family handoff chirality

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/family_handoff_chirality.py
```

**Ausgangslage**

Familie von DtN-Matrizen:

$$
{\Lambda_i}.
$$

Cross-Graph-Cauchy-Pairing:

$$
\omega_{ij}(q,r)=q^T\Lambda_jr-r^T\Lambda_iq.
$$

Zusätzlich Handoff-Square:

$$
A\to B_i\to C, \qquad A\to B_j\to C.
$$

**Befund**

Cross-Graph-Signale können auftreten:

$$
\omega_{ij}\neq 0.
$$

Aber:

```text
sibling_flip_detected = false
handoff_holonomy_detected = false
```

**Obstruktions-Ort**

Signal ist Familien-/Metrikdifferenz, nicht Chiralität. Keine Geschwister-Vorzeichenumkehr, keine echte Handoff-Holonomie.

**Status**

Wichtiger Test: „Nicht nur ein Graph“ wurde geprüft. Ergebnis bleibt achiral.

---

# 6. Triadische Tests

## 6.1 V6 — Triadic interface chirality

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/triadic_interface_chirality.py
```

**Ausgangslage**

Triade:

$$
\text{UV-channel}, \qquad \text{Environment-channel}, \qquad \text{Handoff/Regulator-channel}.
$$

Regulator-Kandidat:

$$
r_i=(\Lambda_{\mathrm{child},i}-\Lambda_{\mathrm{parent}})a.
$$

Triadische Fläche:

$$
\tau_i = \det(e_{\mathrm{UV}}-e_{\mathrm{Env}},\,r_i-e_{\mathrm{Env}}).
$$

**Befund**

Für kanonische Modi:

```text
tau_signs = 1,1,1
nonzero_tau_count = 3
sibling_flip_detected = false
```

**Obstruktions-Ort**

Die Triade ist radial bzw. sibling-invariant.

**Status**

Triadisches Signal ja. Chirale Geschwister-Asymmetrie nein.

---

## 6.2 Nichtkanonische positive Controls

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Dateien darin:
- cnna_alpha_orth_invariant_v7/triadic_interface_chirality.py
- cnna_alpha_orth_invariant_v7/family_handoff_chirality.py
```

**Ausgangslage**

Kontrollmodi:

```text
sibling_index
cyclic_order
```

**Befund**

Sie erzeugen erwartbar Vorzeichen-/Flip-Effekte.

**Obstruktions-Ort**

Sie brechen Symmetrie per Label oder externer Ordnung.

**Status**

Nur Detektorkontrolle. Kein CNNA-derived Beweis.

---

# 7. V7 — Oriented UV/Environment Cauchy shell

## 7.1 Gegengerichtete UV/Env-Randseiten

**Artefaktbezug**

```text
Anhang: cnna_alpha_orth_invariant_v7(1).zip
Datei darin: cnna_alpha_orth_invariant_v7/oriented_cauchy_shell_gate.py
```

**Ausgangslage**

UV-tail und Environment-tail werden als gegengerichtete Randseiten einer Shell gelesen.

Cauchy-Datenraum:

$$
(q_{\mathrm{Env}},q_{\mathrm{UV}},p_{\mathrm{Env}},p_{\mathrm{UV}}).
$$

Orientierte Randform:

$$
\omega_\partial=\omega_{\mathrm{Env}}-\omega_{\mathrm{UV}}.
$$

Metrik:

$$
g= \begin{pmatrix} k_{\mathrm{Env}} & 0 & 0 & 0\\ 0 & k_{\mathrm{UV}} & 0 & 0\\ 0 & 0 & k_{\mathrm{Env}}^{-1} & 0\\ 0 & 0 & 0 & k_{\mathrm{UV}}^{-1} \end{pmatrix}.
$$

Konstruktion:

$$
J=-g^{-1}\omega_\partial.
$$

**Befund**

Getestet:

```text
J_square_error = 0.0
metric_compat_error = 0.0
omega_compat_error = 0.0
swap_to_minus_J_error = 0.0
```

Also:

$$
J^2=-I, \qquad J^TgJ=g, \qquad J^T\omega J=\omega.
$$

**Obstruktions-Ort**

Die Ko-Orientierung wird gewählt. Mit der Gegenwahl entsteht ebenso konsistent:

$$
J\mapsto -J.
$$

Die Cauchy-Randform ist daher nicht identisch mit dem gesuchten Locking-Objekt. Sie liefert eine symplektisch-kompatible Cauchy-Struktur, aber noch keine Verriegelung von $J$ mit einer Fluss-/Zeit-/Handoff-Orientierung $\tau$:

$$
\omega_\partial\Rightarrow\{+J,-J\}, \qquad \omega_{\mathrm{lock}}:(J,\tau)\mapsto\text{stabiler orientierter Record}.
$$

**Status**

Sehr wichtiges Positivergebnis:

$$
\text{UV/Env-Ko-Orientierung}\Rightarrow \{+J,-J\}\text{-Cauchy-Struktur}.
$$

Cauchy-Shell positiv, aber Locking fehlt. Kein absolutes Vorzeichen.

---

# 8. Root-, Co-root- und Tiefenlesart-Tests

## 8.1 Root als äußerer Modellrand

**Ausgangslage**

Der ToC wächst nicht ontisch; er ist unendlich gegeben.

$$
\ell(\mathrm{root})=0, \qquad \ell\to\infty
$$

nach innen.

**Befund**

Tiefenordnung liefert relative Gegengerichtetheit:

$$
\text{Env-Seite}: \ell\downarrow, \qquad \text{UV-Seite}: \ell\uparrow.
$$

**Obstruktions-Ort**

Tiefenordnung ist polar, nicht chiral:

$$
\text{innen/außen}\neq\text{Drehsinn}.
$$

**Status**

Stützt V7 semantisch. Kein absolutes $J$.

---

## 8.2 Negative-root / Co-root-Hypothese

**Ausgangslage**

Hypothese:

$$
\text{formale Root ist Interface;} \qquad \text{dahinter liegt negative Wurzelfamilie}.
$$

**Befund**

Könnte Cauchy-Doppelung und $\alpha_{\mathrm{Env}}$-Ableitung unterstützen.

**Obstruktions-Ort**

Eine negative Wurzelfamilie bleibt bei reeller passiver Symmetrie nicht automatisch chiral.

**Status**

Möglicher Kandidat für Environment-Ableitung; kein Vorzeichenbeweis.

---

# 9. Geschwister-, S_b- und Adresssymmetrie-Tests

## 9.1 S_b-Sibling-Obstruktion

**Ausgangslage**

Im ungeordneten b-ären Baum sind Geschwister unter

$$
S_b
$$

austauschbar.

**Befund**

Kanonische Größen liegen in der trivialen $S_b$-Komponente.

**Obstruktions-Ort**

Die Signum-Darstellung wird nicht kanonisch ausgewählt:

$$
S_b\text{-Äquivarianz} \Rightarrow \text{keine kanonische sibling-chirality}.
$$

**Status**

Robuste Negativlinie.

---

## 9.2 Hamming-Gewichtsklassen

**Ausgangslage**

Blätter wie:

$$
000,001,010,011,100,101,110,111.
$$

Klassen:

$$
|x|_1=1, \qquad |x|_1=2.
$$

**Befund**

Adressintrinsische Relation quer zur Präfixstruktur.

**Obstruktions-Ort**

Hamming-Gewicht ist Betrag, keine Orientierung. Bit-Umkehr bleibt möglich.

**Status**

Strukturfund, aber achiral.

---

## 9.3 Zyklische Bitverschiebung

**Ausgangslage**

Auf etwa:

$$
\{001,010,100\}
$$

gibt es zyklische Verschiebung:

$$
001\to010\to100\to001.
$$

**Befund**

Adressintrinsische Schleife ohne geometrische Einbettung.

**Obstruktions-Ort**

Bit-Reversal konjugiert Links-Shift in Rechts-Shift:

$$
\mathrm{reverse}\circ\rho=\rho^{-1}\circ\mathrm{reverse}.
$$

Also:

$$
\text{Schleife ja, Drehsinn nein.}
$$

**Status**

Wichtiger Kandidat für Multi-ToC-/Frustrationsstrukturen. Kein lokales $J$-Vorzeichen.

---

# 10. SG/ST-, Chirotopie- und Sign-Line-Tests

## 10.1 SG/ST als IFS-/Quotient-Strukturen

**Ausgangslage**

Sierpinski-Gasket (SG) und Sierpinski-Tetrahedron/Tetrix (ST) wurden als ToC-nahe Quotient-/IFS-Strukturen betrachtet. Ihre Rolle in der Testgeschichte war nicht zufällig: SG und ST waren wegen Zahmheit, hoher Symmetrie, p.c.f.-Kontrollierbarkeit und Skaleninvarianz die ersten natürlichen Fraktal-Stressobjekte.

Der b-äre Baum ist dabei die Provenienz- bzw. Adressseite dieser Strukturen:

$$
SG:\quad A_3^{<\omega},\qquad ST:\quad A_4^{<\omega}.
$$

SG/ST selbst entstehen erst, wenn zur reinen Adressprovenienz zusätzliche IFS-/Quotient-/Randidentifikationen und meist eine geometrische Einbettung hinzukommen. Diese Zusatzrelationen dürfen im CNNA-derived-only-Test nicht unkontrolliert als Orientierung, Hodge-Struktur oder komplexe Phase zurückimportiert werden.

**Befund**

Sie bringen Schleifen und Kozyklen:

$$
H^1\neq0.
$$

Beispielhafte Größen:

$$
d_s(SG)=\frac{2\log 3}{\log 5}, \qquad d_s(ST)=\frac{2\log 4}{\log 6}.
$$

**Obstruktions-Ort**

SG/ST sind nicht der bare ToC. Sie sind IFS-/Adressquotienten. Ihre zusätzlichen Relationen sind nicht automatisch aus dem ToC abgeleitet.

**Status**

Nützlich als Vergleichs- und Strukturtest; kein direkter $J$-Durchbruch. Der b-äre Baum bleibt die bewusst entkleidete Provenienzseite von SG/ST, nicht deren geometrisch orientierte Einbettung.

---

## 10.2 Chirotopie / Sign-Line (S_b/A_b)

**Ausgangslage**

Chiralität auf Geschwistern liegt in der Signum-Information:

$$
S_b/A_b\simeq \mathbb Z_2.
$$

**Befund**

Wenn die lokale Isotropiegruppe $H$ nicht in $A_b$ liegt, gibt es keine kanonische nichtverschwindende Chirotopie.

Für den symmetrischen ToC:

$$
H=S_b.
$$

**Obstruktions-Ort**

$$
S_b\not\subset A_b.
$$

Daher ist eine Sign-Line nicht kanonisch ausgezeichnet.

**Status**

Sehr zentrale No-Go-Formulierung.

---

## 10.3 Z_b-Zyklizität ist nicht genug

**Ausgangslage**

Test, ob zyklische Ordnung $Z_b$ die fehlende Chirotopie ersetzt.

**Befund**

Nein. Bei $b=4$ kann ein 4-Zyklus als Labelpermutation ungerade sein; geometrische Orientierung und Permutationsparität fallen nicht automatisch zusammen.

**Obstruktions-Ort**

Zyklische Ordnung ist noch keine Sign-Line.

**Status**

Wichtige Korrektur gegen voreilige „Zyklus = Orientierung“-Schlüsse.

---
## 10.4 Angehangener SG/H₁-Kontrolltest

**Artefaktbezug**

```text
Anhang: files(1).zip
Dateien darin:
- F9_H1_test_zusammenfassung.md
- build_structures.py
- build_gasket.py
- generator_test.py
- h1_tests.py
```

**Ausgangslage**

Der b-äre Baum wurde als Kontrollgruppe gegen das Sierpinski-Gasket betrachtet: Der Baum hat keinen Zyklenraum, während das Gasket bereits auf Graphniveau viele Zyklen besitzt. Damit prüft der Test die Hypothese, ob das fehlende $i$ bzw. $J$ nicht im Baum, sondern in Schleifen bzw. $H_1$ liegen könnte.

**Befund**

Der Baum hat erwartungsgemäß $b_1=0$. Das Gasket besitzt nichttriviales $H_1$, aber die reine Graphenform reicht nicht aus, um ein kanonisches $J$ zu erzwingen. Die planare Zyklenorientierung hängt an einer gewählten Ebenenorientierung und kippt unter Spiegelung. Der reine Down-Kanten-Laplace annihiliert den Zyklenraum, weil Zyklen im Graphen harmonisch sind.

**Obstruktions-Ort**

Schleifen allein liefern noch keine Dynamik und keine kanonische komplexe Orientierung. Für eine nichttriviale Dynamik auf $H_1$ wären echte $2$-Zellen bzw. ein Kettenkomplex mit Up-Laplace erforderlich. Das wäre eine neue, separat zu prüfende Struktur und darf nicht aus der planaren Einbettung importiert werden.

**Status**

Der angehängte SG/H₁-Test stützt die Hauptlinie: mehr Topologie als im Baum ist hilfreich als Stressklasse, aber reine Graphenschleifen liefern noch kein derived-only $J$-Vorzeichen.

---

# 11. Hodge-, Dirac- und Dualkomplex-Tests

## 11.1 Cellular Dirac K = d - d*

**Ausgangslage**

Zellulärer Operator:

$$
K=d-d^*
$$

auf

$$
C^0\oplus C^1\oplus C^2.
$$

**Befund**

$K$ ist reell schief. Auf $\mathrm{im}K$ kann eine formale Polarstruktur einen J-artigen Anteil liefern.

**Obstruktions-Ort**

Der Operator mischt Grade. Auf reinem $C^1$-Raum ist der relevante Block nicht automatisch ein lokales $J$.

**Status**

Formale $J$-ähnliche Struktur möglich, aber nicht als lokaler Handoff-$J$ abgeleitet.

---

## 11.2 Hodge-Star / Dualkomplex

**Ausgangslage**

Test, ob duale Zellen oder Hodge-$\star$ die Orientierung liefern.

**Befund**

Ein echter Hodge-$\star$ braucht Orientierung bzw. Metrik-/Volumenstruktur.

Bei voller $S_b$-Symmetrie gibt es keinen kanonischen schiefen äquivarianten Operator.

Mit Chirotopie reduziert sich die Symmetrie und ein $J$-Block kann erscheinen.

**Obstruktions-Ort**

Richtung ist:

$$
\text{Chirotopie}\Rightarrow J\text{-Modus},
$$

nicht:

$$
J\text{-Modus}\Rightarrow\text{Chirotopie}.
$$

**Status**

Bestätigt, dass Orientierung nicht durch Hodge allein entsteht.

---

# 12. Recursive SG/ST- und Schur/DtN-Tests

## 12.1 Rekursive SG/ST-DtN-Matrizen

**Ausgangslage**

Boundary-DtN-Matrizen für rekursive SG/ST-Approximationen.

**Befund**

Boundary-DtN bleibt voll symmetrisch:

$$
\Lambda_n=a_n(bI-\mathbf 1\mathbf 1^T).
$$

Typisch:

$$
a_n(SG)=\left(\frac35\right)^n, \qquad a_n(ST)=\left(\frac23\right)^n.
$$

**Obstruktions-Ort**

Volle $S_b$-Invarianz bleibt erhalten. Keine Reduktion:

$$
S_b\to A_b.
$$

**Status**

SG/ST-Schur/DtN liefert Skalenstruktur, keine Chirotopie.

---

## 12.2 IFS-Erzeugungsprozess-Test

**Ausgangslage**

Test, ob der IFS-Wachstumsprozess selbst eine Ordnung erzeugt.

**Befund**

Ungeordnete Kontraktionen:

$$
\{\phi_i\}
$$

bleiben $S_b$-äquivariant.

**Obstruktions-Ort**

Eine geordnete/chirale IFS-Familie könnte Chirotopie tragen, aber nur, wenn die Ordnung selbst abgeleitet ist.

**Status**

IFS-Wachstum allein löst das Vorzeichenproblem nicht.

---

# 13. Mehrzellen-Holonomie

## 13.1 Permutations-Holonomie zwischen lokalen ToC-Fasern

**Ausgangslage**

Gluing-Kanten mit:

$$
\varphi_{\alpha\beta}\in S_b.
$$

Loop-Holonomie:

$$
h_\gamma = \varphi_{\alpha_{k-1}\alpha_k}\cdots\varphi_{\alpha_0\alpha_1}.
$$

**Befund**

Wenn der Zentralisator

$$
C_{S_b}(h_\gamma)
$$

in $A_b$ liegt, können lokale odd permutations ausgeschlossen werden.

Beispiel:

$$
b=3,\quad h=(012), \qquad C_{S_3}(h)=A_3.
$$

**Obstruktions-Ort**

Die Richtung

$$
h \text{ vs. } h^{-1}
$$

bleibt genau die chirale Wahl. Unorientierte Klasse:

$$
\{h,h^{-1}\}
$$

lokalisiert nur ein Paar.

**Status**

Starker Multi-ToC-Kandidat, aber ohne derived gerichtete Holonomie kein $J$-Vorzeichen.

---

# 14. F1-Holonomie und F1-only-No-Go

## 14.1 F1-only Port-Regeln

**Ausgangslage**

F1 ist der radiale Provenienz-/Auffüllpfeil. Test: Kann eine F1-only-Regel Ports nichttrivial permutieren?

**Befund**

Eine relabeling-natürliche F1-only-Portregel muss mit allen

$$
\sigma\in S_b
$$

kommutieren. Daher liegt sie im Zentrum:

$$
Z(S_b)=\{e\} \qquad (b\ge3).
$$

**Obstruktions-Ort**

F1 allein hat keine transversale Portordnung.

**Status**

Starker No-Go: Nichtlinearität in Tiefe hilft nicht, solange Relabeling-Natürlichkeit gilt.

---

## 14.2 Screw-Regel als Import

**Ausgangslage**

Regel wie:

$$
(n,i)\mapsto(n+1,\sigma(i)), \qquad \sigma=(012).
$$

**Befund**

Erzeugt scheinbar Drehung.

**Obstruktions-Ort**

Unter odd relabeling:

$$
\tau\sigma\tau^{-1}=\sigma^{-1}.
$$

Die Regel importiert eine Portordnung.

**Status**

Kontrollimport, kein CNNA-derived Mechanismus.

---

# 15. Value-based F1-Coupling

## 15.1 Tiefenabhängige Wertkopplung

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Abschnitt bleibt als konzeptioneller Befund.

**Ausgangslage**

Nicht Portpermutation, sondern wertbasierte Kopplung:

$$
w_{\alpha\beta}=f(d_\alpha,d_\beta,\ldots).
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

Zeigt, wie Nichtreziprozität entstehen könnte. Aber ohne derived Auswahl bleibt (\{+K,-K\}).

---

# 16. Block-RG und Schalenkopplung

## 16.1 Kollektive Schalenkopplung

**Ausgangslage**

Nicht Knoten-an-Knoten, sondern relabeling-natürliche Level-Schale an Level-Schale:

$$
S_k(A)\leftrightarrow S_k(B),
$$

mit Mean-Mode:

$$
u_{A,k} = \frac{1}{\sqrt{|S_k|}}\mathbf 1_{S_k(A)}.
$$

Kopplung:

$$
C_{AB} = \sum_k\gamma_k u_{A,k}u_{B,k}^T.
$$

**Befund**

Reziproke Schalenkopplung erzeugt Spektralstruktur und ggf. Zyklen.

**Obstruktions-Ort**

Die Kopplung bleibt symmetrisch:

$$
C_{AB}=C_{BA}^T.
$$

Daher überlebt A/B-Spiegelung.

**Status**

Struktur ja, Chiralität nein.

---

## 16.2 Vier-Fälle-Test: Adress-fixiert vs. Rollen-fixiert

**Ausgangslage**

Unterscheidung:

$$
\text{Adressort} \neq \text{Skalenrolle}.
$$

Vier Fälle:

| Fall | Skalenlesart | Verklebungsort                 |
| ---- | ------------ | ------------------------------ |
| A    | Wurzel grob  | Wurzel                         |
| B    | Wurzel fein  | Wurzel                         |
| C    | Wurzel grob  | grobes Ende = Wurzel           |
| D    | Wurzel fein  | grobes Ende = Level-$L$-Schale |

**Befund**

Fall D ist strukturell neu.

Gemeldeter Befund:

$$
\beta_1: 0\to 6560,
$$

$$
d_s: 1.385\to 3.647.
$$

**Obstruktions-Ort**

Trotz starker Strukturänderung überlebt A/B-Spiegelung in allen Fällen.

Grund:

$$
\text{Gate hängt an Reziprozität der transversalen Kopplung, nicht am Verklebungsort.}
$$

**Status**

Sehr wichtiger Befund: inverse Skalenlesart ist echter Strukturparameter, aber kein $J$-Mechanismus.

---

# 17. Inverser UV/Env-Cut

## 17.1 UV-cut unter umgekehrter Skalenlesart

**Ausgangslage**

Standard:

$$
\text{UV an Blättern}, \qquad \text{Env an Wurzel}.
$$

Inverse Lesart:

$$
\text{UV an Wurzel}, \qquad \text{Env an Blättern}.
$$

**Befund**

Als echter weiterer Test identifiziert; nicht vollständig als eigener finaler positiver Befund abgeschlossen.

**Obstruktions-Ort**

Würde Skalenrollen direkt in die Operatorstruktur einbringen. Aber solange die resultierenden Operatoren reell symmetrisch und relabeling-natürlich bleiben, ist Chiralität nicht zu erwarten.

**Status**

Offen bzw. als nächster präziser Test markiert, aber durch spätere DtN-/Flachheitsdiagnose teilweise eingeordnet.

---

# 18. DtN-Handoff-Operator-Tests

## 18.1 Zwei DtN-Matrizen auf gemeinsamem Handoff-Raum

**Ausgangslage**

Nach Korrektur: Handoff sieht keine ToC-Knoten mehr, sondern Operatoren:

$$
(H_\partial,\Lambda).
$$

Ziel:

$$
K=[\Lambda_A,\Lambda_B].
$$

**Befund**

Nur sinnvoll, wenn beide Operatoren auf demselben Handoff-Raum leben.

**Obstruktions-Ort**

Spektralordnung allein identifiziert keine Eigenräume. In jeweiliger Eigenbasis diagonalisiert, kommutieren beide trivial.

**Status**

Wichtige Kategoriekorrektur.

---

## 18.2 DtN-RG-Kommutator

**Ausgangslage**

Aufeinanderfolgende RG-/Schur-Stufen derselben Sequenz:

$$
\Lambda_n, \qquad \widetilde{\Lambda}_{n+1}.
$$

Kommutator:

$$
K_n=[\Lambda_n,\widetilde{\Lambda}_{n+1}].
$$

**Befund**

Gemeldet:

$$
K_n=0
$$

für kanonische RG-Projektion.

**Obstruktions-Ort**

Beide Operatoren liegen auf derselben radialen F1-Achse und teilen dieselbe symmetrieadaptierte Schalenbasis.

**Status**

Sehr wichtiger Mechanismus:

$$
\text{abgeleitete Reihenfolge durch F1} \Rightarrow \text{gleiche Achse} \Rightarrow \text{Kommutativität}.
$$

---

# 19. Überlagerte DtN-Matrixalgebra-Türme

## 19.1 Matrix-Tower-Idee

**Ausgangslage**

Vorschlag:

$$
M_2\to M_4\to M_8\to\cdots
$$

bzw. mehrere ToC-DtN-Matrizen auf wachsenden Handoff-Räumen.

**Befund**

Nichtkommutativität könnte entstehen, wenn mehrere symmetrische Operatoren auf demselben Raum keine gemeinsame Eigenbasis haben.

**Obstruktions-Ort**

Beispiele mit Spin-Ketten importieren Tensorproduktordnung und Nachbarschaft:

$$
A_{12},\qquad A_{23}.
$$

Diese Links-Rechts-Struktur ist nicht aus barem ToC abgeleitet.

**Status**

Als möglicher A→B-Algebraweg interessant, aber nur mit derived Einbettungen erlaubt.

---

## 19.2 Kinderpartition-/ToC-derived-Einbettungstest

**Ausgangslage**

Abgeleitete Einbettungen über Kinder-Teilbäume bzw. $S_b$-symmetrische Partitionen.

**Befund**

Kind-restringierte DtN-Operatoren kommutieren:

* disjunkte Supports → triviale Kommutatoren,
* volle DtN gegen blockdiagonalen Teil → kommutiert numerisch.

**Obstruktions-Ort**

Alle Zerlegungen respektieren dieselbe $S_b$-/Radialsymmetrie und teilen die symmetrieadaptierte Eigenbasis.

**Status**

Matrix-Tower-Route negativ im flachen abgeleiteten ToC-Sektor.

---

# 20. Connes-/Nichtkommutativitätsroute

## 20.1 Grundfrage: Woher kommt Nichtkommutativität bei Connes?

**Ausgangslage**

Connes ersetzt Raum durch Algebra:

$$
(\mathcal A,\mathcal H,D).
$$

Nichtkommutativität liegt in:

$$
ab\neq ba.
$$

**Befund**

Bei Connes ist die nichtkommutative Algebra typischerweise Eingabestruktur, nicht aus einem flachen ToC abgeleitet.

**Obstruktions-Ort für CNNA**

CNNA müsste erst eine Handoff-Algebra liefern:

$$
\mathcal A_{\mathrm{eff}} = \mathrm{Alg}\{\Lambda_i\}
$$

mit

$$
[\Lambda_i,\Lambda_j]\neq0.
$$

**Status**

Connes ist Ziel-/Vergleichsstruktur, nicht Generator.

---

## 20.2 Zwei Reduktionsregimes

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Befund bleibt als konsolidierter Diagnostikstand.

**Ausgangslage**

Vergleich:

$$
\Lambda_{\mathrm{UV}}
$$

gegen

$$
\Lambda_{\mathrm{Env}}
$$

auf demselben Leaf-Boundary-Raum.

**Befund**

Gemeldet:

$$
\|[\Lambda_{\mathrm{UV}},\Lambda_{\mathrm{Env}}]\|\sim 10^{-16}.
$$

**Obstruktions-Ort**

Root-Selbstenergie verschiebt Eigenwerte, aber dreht keine Eigenräume. Radial bleibt radial.

**Status**

Negativ für exakte derived Regime.

---

## 20.3 Spektral trunkierte Reduktion

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Befund bleibt als konsolidierter Diagnostikstand.

**Ausgangslage**

Vergleich:

$$
\Lambda_{\mathrm{full}}
$$

gegen spektral trunkierte Reduktion:

$$
\Lambda_{\mathrm{trunc}}.
$$

**Befund**

Bei beliebigem $m$:

$$
\|[\Lambda_{\mathrm{full}},\Lambda_{\mathrm{trunc}}]\|\approx 0.017
$$

für mittlere $m$-Werte; $K$ ist schief.

**Obstruktions-Ort**

Zunächst falsch interpretiert: $\pm i\lambda$-Paare wurden als „beide Chiralitäten“ gelesen. Korrektur:

$$
\pm i\lambda
$$

ist normales Spektrum eines reellen $J$-Blocks.

Der echte Vorzeichentest ist:

$$
K\text{ oder }-K\text{ ausgezeichnet?}
$$

**Status**

Nur scheinbar positiver Kandidat; musste degenerazien-sicher nachgetestet werden.

---

## 20.4 Degenerazien-sichere Cluster-Trunkierung

**Artefaktbezug**

Kein angehängter Artefaktbezug in dieser Fassung; der Abschnitt hält nur den methodischen Nachbefund fest.

**Ausgangslage**

Trunkierung nicht nach beliebigem $m$, sondern nur nach ganzen Eigenwert-Clustern:

$$
P_{\le \lambda} = \sum_{\mu\le\lambda}P_\mu.
$$

**Befund**

Bei allen kanonischen Cluster-Grenzen:

$$
|K|\approx 10^{-16}.
$$

Nichtkommutativität trat nur auf, wenn $m$ mitten durch degenerierte Eigenräume schnitt.

**Obstruktions-Ort**

Ein Schnitt durch entartete Eigenräume wählt eine nicht-kanonische `numpy`-Basis. Das ist kein ToC-derived Mechanismus.

Warnung: Nicht-kanonische Trunkierung mitten durch entartete Eigenräume ist ein Symmetriebruch durch numerische Basiswahl und darf nicht als derived Nichtkommutativität gezählt werden.

**Status**

Starker Negativbefund:

> **Kernaussage.** relabeling-natürliche exakte und cluster-sichere DtN-Reduktionen kommutieren.

---

# 21. Knoten-Elimination vs. partielle Spur

## 21.1 Falscher „Ausspuren“-Test

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
\text{Umgebung speist ein, System-Abfluss wird verworfen}.
$$

**Status**

Ungültig als OQS-/Partial-trace-Test. Höchstens Test einer asymmetrischen Randbedingung.

---

## 21.2 Korrekte Knotenreduktion

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

---

# 22. Flacher Sektor und Krümmung

## 22.1 Flacher reell-reziproker ToC-/DtN-Sektor

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
U_\gamma\neq I,
$$

und keine Krümmung:

$$
[\nabla_\mu,\nabla_\nu]\neq0.
$$

**Status**

Interpretationswechsel:

> **Kernaussage.** Die No-Gos betreffen den flachen ToC-/DtN-Sektor.

Nicht CNNA insgesamt.

---

## 22.2 Krümmung als möglicher späterer Ursprung von Nichtkommutativität

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
\text{Block-RG/DtN}\to\text{Connection}\to\text{Holonomie/Krümmung}.
$$

---

# 23. IDEAL-ToC-Faser-Gitter

## 23.1 Doppelt unendlicher IDEAL-Sektor

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
\text{flach, homogen, reziprok, intern ToC-skaleninvariant}.
$$

Transversale Isotropie nur diskret bzw. abhängig von $\Gamma_\infty$.

**Obstruktions-Ort**

Das Gitter bringt transversale Nachbarschaft als neues IDEAL-Vergleichsdatum mit. Sie ist nicht aus einem einzelnen ToC abgeleitet.

**Status**

Sehr sinnvoller letzter ToC-naher Test vor Substratwechsel.

---

## 23.2 Endlicher Doppelschnitt

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
\text{äußeres Gitter-Komplement},
$$

$$
\text{interner UV-tail},
$$

$$
\text{Rand/Ecken/Mischkomplemente}.
$$

**Obstruktions-Ort**

Subsystem-Sein erzeugt effektive Rand-/Spektral-/DtN-Geometrie, aber nicht automatisch Chirotopie.

**Status**

Positiver Geometrie-/DtN-Test, negativer $J$-Test im flachen reziproken Fall.

---

## 23.3 DtN auf dem ToC-Faser-Gitter

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

---

# 24. Holonomie-/Connection-Test im Faser-Gitter

## 24.1 Effektive Intertwiner zwischen lokalen Handoff-Räumen

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

---

# 25. Lorentz-/Zeitstruktur-Tests

## 25.1 Lorentz-Signatur

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

---

## 25.2 Reeller Zeitfluss-Vorläufer

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

---

# 26. Pillar C / OQS / Entropie

## 26.1 Lindblad-/OQS-Zeitpfeil

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

---

# 27. AQFT / Type-I / Type-III / Handoff-Struktur

## 27.1 A als Type-I-/Type-III-Vorläuferschicht

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

---

## 27.2 Triadischer Handoff (B|B'|C)

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

---

# 28. Multi-ToC / Detektor / Vielobjektstruktur

Dieser Abschnitt darf nicht als Rückfall in die Lesart „ToC-Knoten sind Teilchen“ verstanden werden. Viele Objekte entstehen nicht durch viele Knoten innerhalb eines einzelnen ToC, sondern durch viele lokale ToC-Fasern, deren Approximanten und Handoff-Daten relativ zueinander verklebt werden.

$$
\{T_i\}_{i\in I} \Rightarrow \text{Multi-ToC-/Gluing-Struktur}, \qquad T_i\text{-Knoten}\neq\text{Teilchen}.
$$

## 28.1 Mini-ToCs als Detektorelemente

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

---

## 28.2 Frustration / Spin-netz-artige Struktur

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

---

# 29. Motor-/Mehrphasen-Analogie

## 29.1 Zweiphasiger Dreiphasenmotor

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

---

## 29.2 Drei Phasen / Anschlussordnung

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

---

# 30. Cayley-Dickson / höhere Divisionsalgebren

## 30.1 CD-/Hurwitz-Kandidat

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

---

# 31. Substratwechsel-Kandidaten

## 31.1 ToC bleibt lokale Provenienzfaser

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

> **Kernaussage.** b-ärer Einzelbaum nicht Weltbaum, sondern lokale Provenienzfaser.

Die Komplementseite bleibt im Gegenteil strukturell notwendig, sobald lokale Handoff-Operatoren, lokale Algebren, relative Komplemente und spätere AQFT-Anschlussbedingungen ernst genommen werden.

---

## 31.2 Ereignisstrukturen als Vergleichsstruktur, kein Fundament

**Ausgangslage**

Ereignisstrukturen besitzen typischerweise zwei Relationen:

$$
\leq \qquad\text{und}\qquad \#.
$$

Dabei ist $\leq$ nicht neutral, sobald es als kausale oder zeitartige Ordnung gelesen wird. Die Relation $\#$ markiert Konflikt, Inkompatibilität oder Exklusion.

**Befund**

Als spätere Ziel- oder Vergleichsstruktur sind Ereignisstrukturen interessant. Sie könnten beschreiben, wie aus einer CNNA-derived Vorstruktur emergente Ereignisse, Konflikte und eine kausale Ordnung entstehen.

Die zulässige Richtung ist daher:

$$
\text{CNNA-derived nicht-kausale Vorstruktur} \longrightarrow \text{emergente Ereignisse} \longrightarrow (E,\leq,\#).
$$

**Obstruktions-Ort**

Als Fundament wären Ereignisstrukturen zu stark. Die Relation $\leq$ würde Kausalität bzw. Zeitordnung bereits als primitives Datum einführen. Damit würde genau das gesetzt, was CNNA erst rekonstruieren müsste.

Die unzulässige Richtung wäre:

$$
(E,\leq,\#) \longrightarrow \text{CNNA-Fundament}.
$$

Das wäre methodisch derselbe Importtyp wie:

$$
\text{komplexe Zahlen setzen},\qquad \text{Orientierung setzen},\qquad \text{Tensorprodukt setzen},\qquad \text{Hodge-Star setzen}.
$$

Nur wäre der importierte Inhalt hier:

> **Kernaussage.** Kausalität setzen.

**Status**

Ereignisstrukturen sind als nächster Fundament-Kandidat zurückzustufen. Sie bleiben Ziel-/Vergleichsstruktur, aber kein zulässiger Substratkern vor einer abgeleiteten Kausalitätsrekonstruktion.

> **Kernaussage.** Ereignisstrukturen: Vergleichsstruktur ja, Fundament nein.

## 31.3 Nicht-kausaler Substratwechsel-Gate

**Ausgangslage**

Der b-äre Einzelbaum ist als globaler Weltbaum für den $J$-Sektor unter den flach-reziproken Derived-only-Prämissen falsifiziert. Daraus folgt nicht, dass beliebig reichere relationale Substrate zulässig sind. Ein neues Substrat darf nicht einfach die fehlenden Zielstrukturen als primitive Relationen enthalten.

**Befund**

Ein zulässiger nächster Substratkandidat muss mindestens folgende Ausschlüsse erfüllen:

> **Kernaussage.** kein primitives i, · kein primitives J, · keine primitive Chirotopie, · keine primitive Orientierung, · keine primitive Tensorfaktorisierung, · keine primitive Kausalordnung.

Er darf eine nicht-kausale relationale, kombinatorische oder topologische Vorstruktur tragen, solange deren spätere kausale Lesart erst durch Handoff, Regimebildung, Spektralstruktur, Regulatoren oder Backreaction erzwungen wird.

**Obstruktions-Ort**

Jedes Substrat, das bereits eine gerichtete Zeit-, Kausal-, Orientierungs- oder Phasenstruktur enthält, umgeht den eigentlichen CNNA-Test. Dann wäre die fehlende zweite Achse nicht abgeleitet, sondern importiert.

**Status**

Der strengste derzeit zulässige Zwischenschritt bleibt daher das nicht-kausale IDEAL-ToC-Faser-Gitter als flacher Referenztest:

$$
\mathcal I_{\mathrm{ToCGrid}}=\Gamma_\infty\times T_b^\infty,\qquad \Omega_{R,L}=W_R\times T_{\le L}.
$$

Hier ist $\Gamma_\infty$ nur ein homogener relationaler Indexträger, nicht bereits Raumzeit und nicht bereits Kausalordnung. Jede metrische, räumliche, gerichtete oder orientierte Lesart von $\Gamma_\infty$ ist Vergleichs-/Teststruktur und kein ontischer Input.

---

## 31.4 Sierpinski-Teppich als nicht-p.c.f.-Stressklasse

**Ausgangslage**

Der Sierpinski-Teppich ist als nicht-p.c.f.-Stressklasse interessanter als SG/ST, wenn man mehrskalige Boundary-/Trace-Strukturen testen will. Der Mengerschwamm wird in dieser Fassung nicht weiterverfolgt.

**Befund**

Nicht-p.c.f.-Struktur bedeutet: wildere, mehrskalige Schnitt- und Randkontakte sind möglich. Das kann für Handoff-, Trace-, Gluing- und Frustrationstests nützlich sein:

$$
\text{nicht-p.c.f.} \Rightarrow \text{wildere, mehrskalige Boundary/Trace-Struktur}.
$$

**Obstruktions-Ort**

Mehr Löcher oder wildere Randstruktur liefern aber nicht automatisch eine derived-only Orientierung:

$$
\text{mehr Löcher} \neq \text{derived }J\text{-Vorzeichen}.
$$

Insbesondere bleibt zu prüfen, ob jede verwendete Umlaufs-, Flächen-, Trace- oder Hodge-artige Struktur wirklich aus der nicht-kausalen Vorstruktur entsteht oder durch Einbettung/Orientierung importiert wurde.

**Status**

Sinnvolle Substrat-Stressklasse, aber kein aktueller Fundament-Kandidat und keine Lösung des $J$-Vorzeichenproblems.

---

# 32. Ausgewiesene Artefaktlage dieser Fassung

Diese Fassung nennt nur noch Artefakte, die entweder angehängt wurden oder als Hugging-Face-Visualisierung ausdrücklich referenziert sind. Ältere Paketnamen, nicht angehängte Nachtests und hypothetische nächste Implementierungen werden nicht mehr als reproduzierbare Artefaktbasis dieser Datei geführt.

## 32.1 Hugging-Face-Visualisierung

```text
Hugging-Face-Space: https://huggingface.co/spaces/antaris/b-ary_tree
app.py
```

Die Visualisierung dient der Anschauung des ToC-/Approximanten-/UV-/Environment-Konzepts. Sie ist selbst nur eine Proxy- und Darstellungsebene; Tilt-, Winkel- oder Chartwerte daraus sind nicht als Schur-/DtN-Invarianten zu lesen.

## 32.2 Anhang `cnna_alpha_orth_invariant_v7(1).zip`

```text
cnna_alpha_orth_invariant_v7/alpha_orth_invariant.py
cnna_alpha_orth_invariant_v7/two_approximant_flow_sign.py
cnna_alpha_orth_invariant_v7/two_boundary_shell_chirality.py
cnna_alpha_orth_invariant_v7/family_handoff_chirality.py
cnna_alpha_orth_invariant_v7/triadic_interface_chirality.py
cnna_alpha_orth_invariant_v7/oriented_cauchy_shell_gate.py
```

Der Anhang enthält außerdem zugehörige CSV-, JSON-, PNG- und Markdown-Reports. Diese Artefakte bilden die ausgewiesene reproduzierbare Basis für die $\alpha_{\mathrm{orth}}$-, Flow-Sign-, Cauchy-Shell-, Familien-Handoff-, triadischen Interface- und UV/Env-Cauchy-Shell-Befunde dieser Fassung.

## 32.3 Anhang `files(1).zip`

```text
F9_H1_test_zusammenfassung.md
build_structures.py
build_gasket.py
generator_test.py
h1_tests.py
```

Dieser Anhang dokumentiert den Baum-vs.-Sierpinski-Gasket-Kontrolltest: Baum als $b_1=0$-Kontrollgruppe, Gasket als nichttrivialer $H_1$-Stressfall, generatorischer $\kappa$-Blindheitstest und $H_1$-Dynamiktest.

---

# 33. Obstruktions-Orte nach Typ

## 33.1 Reziprozität

$$
\Lambda=\Lambda^T.
$$

Passive Schur-/DtN-Reduktion bleibt symmetrisch. Kein antisymmetrischer $J$-Generator.

## 33.2 Reelle Konjugationssymmetrie

$$
J\mapsto -J.
$$

Reelle Strukturen wählen keine komplexe Orientierung.

## 33.3 S_b-Äquivarianz

Geschwisterpermutationen halten kanonische Größen im trivialen Sektor. Keine Signum-Auswahl.

## 33.4 Radiale Einachsenstruktur (F1)

F1 liefert Ordnung:

$$
n\to n+1.
$$

Aber nur entlang einer Achse. Nichtkommutativität braucht zwei unabhängige Achsen.

## 33.5 Degenerazien

Entartete Eigenräume dürfen nicht durch willkürliche numerische Basis geschnitten werden. Nur ganze Cluster sind relabeling-natürlich.

## 33.6 Keine partielle Spur auf Knoten

$$
\mathbb R^N=\mathbb R^S\oplus\mathbb R^E
$$

ist direkte Summe, kein Tensorprodukt.

## 33.7 Bit-Reversal

Adresszyklen können Drehsinn spiegeln:

$$
\rho\leftrightarrow\rho^{-1}.
$$

## 33.8 Boundary reversal

UV/Env-Ko-Orientierung liefert:

$$
{J,-J}.
$$

## 33.9 Handoff reversal

$$
A_{\gamma^{-1}}=-A_\gamma.
$$

Ohne gerichtete Handoff-Sequenz kein absoluter Drehsinn.

## 33.10 OQS-Abhängigkeit von i

Lindblad/OQS kann Zeitrichtung liefern, setzt aber Hamilton-$i$ voraus.

## 33.11 Flachheit

Im flachen ToC-/DtN-Sektor fehlen:

$$
\text{Connection}, \qquad \text{Holonomie}, \qquad \text{Krümmung}.
$$

## 33.12 Kausalitätsimport

Eine primitive kausale Ordnung $\leq$ ist kein neutraler Strukturträger. Sie würde bereits Zeit-/Kausalstruktur mitbringen und damit den späteren Rekonstruktionsschritt überspringen.

> **Kernaussage.** (E,≤,\#) ist Zielstruktur, nicht Fundament.

Der zulässige Test lautet daher nicht, ob ein kausales Substrat CNNA tragen kann, sondern ob CNNA aus einer nicht-kausalen Vorstruktur eine kausale Ordnung erzeugen kann.

---

# 34. Aktuelle Gesamtformel

> **Kernaussage.** Alle Einzelbaum-, Einzelapproximant-, passiven Schur-/DtN- und lokalen Triadentests enden bei {J,-J}.

> **Kernaussage.** Exakte und cluster-sichere Handoff-Operatoren im flachen ToC-/DtN-Sektor kommutieren.

> **Kernaussage.** Nichtkommutativität entsteht bisher nur durch gesetzte Ordnung, nicht-kanonische Trunkierung oder asymmetrische Randvorschrift.

> **Kernaussage.** ToC-Knoten sind Provenienzindizes, keine physikalischen Freiheitsgrade.

> **Kernaussage.** Der b-äre Baum wurde als Provenienzseite von SG/ST gewählt: SG↔ b=3, · ST↔ b=4.

> **Kernaussage.** Obstruiert ist nicht CNNA und nicht ToC allgemein, sondern der b-äre Einzelbaum als globaler Träger gerichteter komplexer Struktur.

> **Kernaussage.** Komplement-, Handoff- und lokale-Algebra-Strukturen bleiben für den AQFT-Anschluss positiv relevant.

> **Kernaussage.** UV/Env erzeugen einen echten radialen Skalenbruch, aber keine Chiralität.

> **Kernaussage.** \omega_\partial⇒{+J,-J}, · \omega_{lock} bleibt das offene Locking-Objekt.

> **Kernaussage.** Relative Holonomie/Frustration ist nicht automatisch absolute Orientierung.

> **Kernaussage.** Der nächste echte positive Suchraum ist nicht ein weiterer flacher Einzel-ToC-Test, sondern Curved-sector, Multi-ToC-Frustration oder triadisches Handoff-Locking.

Der wichtigste nächste ToC-nahe Test vor Substratwechsel bleibt:

$$
\mathcal I_{\mathrm{ToCGrid}}=\Gamma_\infty\times T_b^\infty, \qquad \Omega_{R,L}=W_R\times T_{\le L}, \qquad \Lambda_{R,L}.
$$

Ziel:

$$
\text{effektive Geometrie aus Subsystem-Sein testen},
$$

aber getrennt davon:

$$
\text{J-/Chirotopie-/Nichtkommutativitäts-Gate weiter offen halten}.
$$
