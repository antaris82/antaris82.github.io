# CNNA-ToC / (J)-Vorzeichen / Nichtkommutativität — vollständiges Test- und Obstruktionsinventar

Status: nach Chatstand, nicht als Lean-Theorem. Die meisten Befunde sind numerisch, konzeptionell oder aus Python-Diagnostik. Das zentrale Ergebnis ist inzwischen präziser als am Anfang:

$$
\boxed{
\text{Der flache, reellwertige, reziproke ToC-/Schur-/DtN-Sektor erzeugt keine ausgezeichnete }J\text{-Orientierung.}
}
$$

Er liefert mehrfach:

$$
{+J,-J},\qquad {+\tau,-\tau},\qquad \text{radiale Ordnung},\qquad \text{DtN-/Spektralstruktur}.
$$

Er liefert bisher nicht:

$$
\boxed{
J\text{ statt }-J.
}
$$

Die einheitliche Obstruktion lautet jetzt nicht mehr nur „Symmetrie“, sondern genauer:

$$
\boxed{
\text{Eine abgeleitete Achse }F1\text{ genügt nicht. Nichtkommutativität/Chiralität braucht eine zweite abgeleitete Achse.}
}
$$

---

# 0. Globaler Status der Testreihe

## 0.1 Was A/ToC bisher positiv liefert

Der ToC-/DtN-Sektor liefert robuste Vorläufer:

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

Das ist nicht trivial. Es bedeutet:

$$
\boxed{
\text{Der ToC ist als lokale Provenienzfaser und flacher Referenzsektor wertvoll.}
}
$$

## 0.2 Was A/ToC bisher nicht liefert

Nicht geliefert wird eine absolute Orientierung:

$$
\boxed{
J \neq \text{derived uniquely from flat ToC data}.
}
$$

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

$$
\boxed{
\text{flacher, homogener, reell-reziproker ToC-/DtN-Sektor.}
}
$$

Daraus folgt:

$$
\boxed{
\text{Der globale Einzel-ToC als ontischer Weltbaum ist für }J\text{ falsifiziert.}
}
$$

Aber nicht:

$$
\boxed{
\text{lokale ToC-Fasern, DtN-Geometrie oder CNNA als Gesamtprogramm sind falsifiziert.}
}
$$

---

# 1. Didaktische und Proxy-Tests

## 1.1 Gradio-ToC-Concept-Explorer

**Script / Datei**

```text
app.py
```

**Ausgangslage**

Visualisierung eines (b)-ären ToC mit Parametern:

$$
b,\qquad L_{\max},\qquad \text{Approximant root},\qquad L.
$$

Dargestellte Stufen:

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

**Befund**

Didaktisch stark. Es trennt sichtbar:

$$
\text{Approximant},
\qquad
\text{UV-tail},
\qquad
\text{Environment},
\qquad
\text{Interface}.
$$

**Obstruktions-Ort**

Visualisierung ist kein Beweis. Frühe Tilt-/Winkelwerte waren teilweise Chart-/Rendering-Proxies, nicht DtN-Invarianten.

**Status**

Didaktisch wertvoll, mathematisch sekundär.

---

## 1.2 Stage-6 Chart-Proxy / Tilt-Test

**Script / Datei**

Teil des interaktiven `app.py`.

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

## 2.1 Projected-tail (J)-/Rotationstest

**Script / Datei**

Im Chat implementiert; Funktionalität ging später in (\alpha_{\mathrm{orth}})- und DtN-Diagnostik ein.

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
\rho_M
======

\frac{\langle u_{\mathrm{Env}},u_{\mathrm{UV}}\rangle_M}
{|u_{\mathrm{Env}}|*M,|u*{\mathrm{UV}}|_M}.
$$

**Befund**

Nahe Orthogonalität:

$$
|\rho_M|\ll 1,
$$

teilweise numerisch nahe (90^\circ).

**Obstruktions-Ort**

Orthogonalität einer reellen 2-Ebene liefert höchstens:

$$
{+J,-J}.
$$

Die Ebene ist da; der Drehsinn nicht.

**Status**

Positiver Vorläufer einer prä-komplexen Ebene. Kein Vorzeichenbeweis.

---

## 2.2 Real finite-network Schur/DtN-Test

**Script / Datei**

Im Chat als endlicher Unit-edge-Graph-Test implementiert; kein eindeutig isolierter finaler Dateiname.

**Ausgangslage**

Endlicher Baumgraph mit Laplace-Matrix:

$$
L_{\mathrm{graph}}.
$$

Rand (B), Innenknoten (I), Schur-Komplement:

$$
\Lambda_B
=========

L_{BB}-L_{BI}L_{II}^{-1}L_{IB}.
$$

**Befund**

Für deterministische zentrierte Einzelmodi numerisch praktisch orthogonal, etwa:

$$
|\rho_M|\approx 10^{-18}.
$$

**Obstruktions-Ort**

Ein Einzelmodus kann orthogonal sein, während der volle Randantwortsraum noch Struktur trägt. Außerdem bleibt der DtN-Operator reell symmetrisch.

**Status**

Starker Hinweis auf echte Schur-/DtN-Orthogonalität in bestimmten Modi; kein (J)-Vorzeichen.

---

## 2.3 Dirichlet-/Cut-Regularisierungstest

**Script / Datei**

Teil der Schur-/DtN-Tests.

**Ausgangslage**

Frage:

$$
\text{Braucht man eine externe Regularisierung oder Pseudoinverse?}
$$

**Befund**

Nein, sofern ein echter Dirichlet-/Boundary-Cut gesetzt wird. Dann ist der Innenblock:

$$
L_{II}
$$

invertierbar.

**Obstruktions-Ort**

Der DtN-Operator ist cut-relativ:

$$
\Lambda_{\partial A}.
$$

Es gibt keinen cut-freien universalen DtN-Operator des ganzen unendlichen ToC.

**Status**

Wichtiges positives Ergebnis: keine Ridge-/Pseudoinversen-Setzung nötig.

---

# 3. $\alpha_{\mathrm{orth}}$- und Invarianten-Tests

## 3.1 $\Xi$- / $\alpha_{\mathrm{orth}}$-Diagnostik

**Script / Datei**

```text
alpha_orth_invariant.py
```

Pakete:

```text
cnna_alpha_orth_invariant_v2.zip
cnna_alpha_orth_invariant_v3.zip
cnna_alpha_orth_invariant_v4.zip
cnna_alpha_orth_invariant_v5.zip
cnna_alpha_orth_invariant_v6.zip
cnna_alpha_orth_invariant_v7.zip
```

**Ausgangslage**

Kontrollgröße:

$$
\Xi=(1+\lambda_{\mathrm{UV}})(1+\lambda_{\mathrm{Env}}),
$$

mit

$$
\lambda_{\mathrm{UV}}
=

\frac{b^k\alpha_{\mathrm{UV}}}{C_k},
\qquad
\lambda_{\mathrm{Env}}
=

\frac{\alpha_{\mathrm{Env}}}{C_k}.
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

$$
\boxed{
\text{UV-Auflösung treibt Orthogonalität.}
}
$$

**Obstruktions-Ort**

(\alpha_{\mathrm{Env}}) war in frühen Versionen modellabhängig:

```text
none
constant
power
exponential
ladder
```

Daher war der exakte Zahlenwert kein vollständig abgeleiteter physikalischer Wert.

**Status**

Gute Diagnosegröße. Kein Feinstrukturkonstanten-Claim. Kein (J)-Vorzeichen.

---

## 3.2 Environment-Sensitivitätsmodelle

**Script / Datei**

```text
alpha_orth_invariant.py
```

**Ausgangslage**

Vergleich verschiedener (\alpha_{\mathrm{Env}})-Modelle.

**Befund**

Für große (k) dominiert häufig der UV-Term so stark, dass die Environment-Modellwahl subdominant wird.

**Obstruktions-Ort**

In Regimen, in denen Environment nicht subdominant ist, braucht man eine echte Komplementfamilien-/DtN-Ableitung von (\alpha_{\mathrm{Env}}).

**Status**

Guter methodischer Befund:

$$
\text{definierbar}\neq\text{erzwungen}.
$$

---

# 4. Parent–Child- und Handoff-Tests

## 4.1 Two-Approximant / Flow-Sign-Test

**Script / Datei**

```text
two_approximant_flow_sign.py
```

**Ausgangslage**

Parent–Child-Handoff:

$$
A_{\mathrm{parent}}\to A_{\mathrm{child}}.
$$

Ziel: prüfen, ob der Übergang ein (J)-Vorzeichen liefert.

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

Radiale Handoff-Struktur: ja. (J)-Vorzeichen: nein.

---

## 4.2 Schur-vor-Flow-Kriterium

**Script / Datei**

Methodisch aus Parent–Child-Tests abgeleitet.

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
\boxed{
\text{Schur zuerst, Flow nur als Konsistenztest.}
}
$$

---

# 5. Zwei-Rand-/Shell-Chiralitätstests

## 5.1 V4 — Two-boundary shell chirality

**Script / Datei**

```text
two_boundary_shell_chirality.py
```

Paket:

```text
cnna_alpha_orth_invariant_v4.zip
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
\omega((q,\Lambda q),(r,\Lambda r))
===================================

# q^T\Lambda r-r^T\Lambda q

0.

$$

**Obstruktions-Ort**

Ein einzelner passiver symmetrischer DtN-Graph ist Lagrangesch.

**Status**

Sauberes Negativergebnis. Zu eng für Familien-/Handoff-Tests, aber korrekt für Einzelgraph.

---

## 5.2 V5 — Family handoff chirality

**Script / Datei**

```text
family_handoff_chirality.py
```

Paket:

```text
cnna_alpha_orth_invariant_v5.zip
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
A\to B_i\to C,
\qquad
A\to B_j\to C.
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

**Script / Datei**

```text
triadic_interface_chirality.py
```

Paket:

```text
cnna_alpha_orth_invariant_v6.zip
```

**Ausgangslage**

Triade:

$$
\text{UV-channel},
\qquad
\text{Environment-channel},
\qquad
\text{Handoff/Regulator-channel}.
$$

Regulator-Kandidat:

$$
r_i=(\Lambda_{\mathrm{child},i}-\Lambda_{\mathrm{parent}})a.
$$

Triadische Fläche:

$$
\tau_i
======

\det(e_{\mathrm{UV}}-e_{\mathrm{Env}},,r_i-e_{\mathrm{Env}}).
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

**Script / Datei**

```text
triadic_interface_chirality.py
family_handoff_chirality.py
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

**Script / Datei**

```text
oriented_cauchy_shell_gate.py
```

Paket:

```text
cnna_alpha_orth_invariant_v7.zip
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
g=\operatorname{diag}(k_{\mathrm{Env}},k_{\mathrm{UV}},1/k_{\mathrm{Env}},1/k_{\mathrm{UV}}).
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
J^2=-I,
\qquad
J^TgJ=g,
\qquad
J^T\omega J=\omega.
$$

**Obstruktions-Ort**

Die Coorientierung wird gewählt. Mit der Gegenwahl entsteht ebenso konsistent:

$$
J\mapsto -J.
$$

**Status**

Sehr wichtiges Positivergebnis:

$$
\text{UV/Env-Coorientierung}\Rightarrow {J,-J}\text{-Cauchy-Struktur}.
$$

Kein absolutes Vorzeichen.

---

# 8. Root-, Co-root- und Tiefenlesart-Tests

## 8.1 Root als äußerer Modellrand

**Ausgangslage**

Der ToC wächst nicht ontisch; er ist unendlich gegeben.

$$
\ell(\mathrm{root})=0,
\qquad
\ell\to\infty
$$

nach innen.

**Befund**

Tiefenordnung liefert relative Gegengerichtetheit:

$$
\text{Env-Seite}: \ell\downarrow,
\qquad
\text{UV-Seite}: \ell\uparrow.
$$

**Obstruktions-Ort**

Tiefenordnung ist polar, nicht chiral:

$$
\text{innen/außen}\neq\text{Drehsinn}.
$$

**Status**

Stützt V7 semantisch. Kein absolutes (J).

---

## 8.2 Negative-root / Co-root-Hypothese

**Ausgangslage**

Hypothese:

$$
\text{formale Root ist Interface;}
\qquad
\text{dahinter liegt negative Wurzelfamilie}.
$$

**Befund**

Könnte Cauchy-Dopplung und (\alpha_{\mathrm{Env}})-Ableitung unterstützen.

**Obstruktions-Ort**

Eine negative Wurzelfamilie bleibt bei reeller passiver Symmetrie nicht automatisch chiral.

**Status**

Möglicher Kandidat für Environment-Ableitung; kein Vorzeichenbeweis.

---

# 9. Geschwister-, $S_b$- und Adresssymmetrie-Tests

## 9.1 $S_b$-Sibling-Obstruktion

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
S_b\text{-Äquivarianz}
\Rightarrow
\text{keine kanonische sibling-chirality}.
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
|x|_1=1,
\qquad
|x|_1=2.
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
{001,010,100}
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

Wichtiger Kandidat für Multi-ToC-/Frustrationsstrukturen. Kein lokales (J)-Vorzeichen.

---

# 10. SG/ST-, Chirotopie- und Sign-Line-Tests

## 10.1 SG/ST als IFS-/Quotient-Strukturen

**Ausgangslage**

Sierpinski-Gasket (SG) und Sierpinski-Tetrahedron/Tetrix (ST) wurden als ToC-nahe Quotient-/IFS-Strukturen betrachtet.

**Befund**

Sie bringen Schleifen und Kozyklen:

$$
H^1\neq0.
$$

Beispielhafte Größen:

$$
d_s(SG)=\frac{2\log 3}{\log 5},
\qquad
d_s(ST)=\frac{2\log 4}{\log 6}.
$$

**Obstruktions-Ort**

SG/ST sind nicht der bare ToC. Sie sind IFS-/Adressquotienten. Ihre zusätzlichen Relationen sind nicht automatisch aus dem ToC abgeleitet.

**Status**

Nützlich als Vergleichs- und Strukturtest; kein direkter (J)-Durchbruch.

---

## 10.2 Chirotopie / Sign-Line (S_b/A_b)

**Ausgangslage**

Chiralität auf Geschwistern liegt in der Signum-Information:

$$
S_b/A_b\simeq \mathbb Z_2.
$$

**Befund**

Wenn die lokale Isotropiegruppe (H) nicht in (A_b) liegt, gibt es keine kanonische nichtverschwindende Chirotopie.

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

## 10.3 $Z_b$-Zyklizität ist nicht genug

**Ausgangslage**

Test, ob zyklische Ordnung $Z_b$ die fehlende Chirotopie ersetzt.

**Befund**

Nein. Bei (b=4) kann ein 4-Zyklus als Labelpermutation ungerade sein; geometrische Orientierung und Permutationsparität fallen nicht automatisch zusammen.

**Obstruktions-Ort**

Zyklische Ordnung ist noch keine Sign-Line.

**Status**

Wichtige Korrektur gegen voreilige „Zyklus = Orientierung“-Schlüsse.

---

# 11. Hodge-, Dirac- und Dualkomplex-Tests

## 11.1 Cellular Dirac $K=d-d^*$

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

(K) ist reell schief. Auf $\operatorname{im}K$ kann eine formale Polarstruktur einen J-artigen Anteil liefern.

**Obstruktions-Ort**

Der Operator mischt Grade. Auf reinem $C^1$-Raum ist der relevante Block nicht automatisch ein lokales (J).

**Status**

Formale (J)-ähnliche Struktur möglich, aber nicht als lokaler Handoff-(J) abgeleitet.

---

## 11.2 Hodge-Star / Dualkomplex

**Ausgangslage**

Test, ob duale Zellen oder Hodge-(\star) die Orientierung liefern.

**Befund**

Ein echter Hodge-$\star$ braucht Orientierung bzw. Metrik-/Volumenstruktur.

Bei voller $S_b$-Symmetrie gibt es keinen kanonischen schiefen äquivarianten Operator.

Mit Chirotopie reduziert sich die Symmetrie und ein (J)-Block kann erscheinen.

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
a_n(SG)=\left(\frac35\right)^n,
\qquad
a_n(ST)=\left(\frac23\right)^n.
$$

**Obstruktions-Ort**

Volle (S_b)-Invarianz bleibt erhalten. Keine Reduktion:

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
{\phi_i}
$$

bleiben (S_b)-äquivariant.

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
h_\gamma
========

\varphi_{\alpha_{k-1}\alpha_k}\cdots\varphi_{\alpha_0\alpha_1}.
$$

**Befund**

Wenn der Zentralisator

$$
C_{S_b}(h_\gamma)
$$

in (A_b) liegt, können lokale odd permutations ausgeschlossen werden.

Beispiel:

$$
b=3,\quad h=(012),
\qquad
C_{S_3}(h)=A_3.
$$

**Obstruktions-Ort**

Die Richtung

$$
h \text{ vs. } h^{-1}
$$

bleibt genau die chirale Wahl. Unorientierte Klasse:

$$
{h,h^{-1}}
$$

lokalisiert nur ein Paar.

**Status**

Starker Multi-ToC-Kandidat, aber ohne derived gerichtete Holonomie kein (J)-Vorzeichen.

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
Z(S_b)={e}
\qquad (b\ge3).
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
(n,i)\mapsto(n+1,\sigma(i)),
\qquad
\sigma=(012).
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

**Script / Datei**

Im Verlauf als `toc_paper_v10_f1_value_coupling` dokumentiert; Python-Test im Chat.

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

Zeigt, wie Nichtreziprozität entstehen könnte. Aber ohne derived Auswahl bleibt ({+K,-K}).

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
u_{A,k}
=======

\frac{1}{\sqrt{|S_k|}}\mathbf 1_{S_k(A)}.
$$

Kopplung:

$$
C_{AB}
======

\sum_k\gamma_k u_{A,k}u_{B,k}^T.
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
\text{Adressort}
\neq
\text{Skalenrolle}.
$$

Vier Fälle:

| Fall | Skalenlesart | Verklebungsort                 |
| ---- | ------------ | ------------------------------ |
| A    | Wurzel grob  | Wurzel                         |
| B    | Wurzel fein  | Wurzel                         |
| C    | Wurzel grob  | grobes Ende = Wurzel           |
| D    | Wurzel fein  | grobes Ende = Level-(L)-Schale |

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

Sehr wichtiger Befund: inverse Skalenlesart ist echter Strukturparameter, aber kein (J)-Mechanismus.

---

# 17. Inverser UV/Env-Cut

## 17.1 UV-cut unter umgekehrter Skalenlesart

**Ausgangslage**

Standard:

$$
\text{UV an Blättern},
\qquad
\text{Env an Wurzel}.
$$

Inverse Lesart:

$$
\text{UV an Wurzel},
\qquad
\text{Env an Blättern}.
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
K=$$\Lambda_A,\Lambda_B$$.
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
\Lambda_n,
\qquad
\widetilde{\Lambda}_{n+1}.
$$

Kommutator:

$$
K_n=$$\Lambda_n,\widetilde{\Lambda}_{n+1}$$.
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
\text{abgeleitete Reihenfolge durch F1}
\Rightarrow
\text{gleiche Achse}
\Rightarrow
\text{Kommutativität}.
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

Abgeleitete Einbettungen über Kinder-Teilbäume bzw. (S_b)-symmetrische Partitionen.

**Befund**

Kind-restringierte DtN-Operatoren kommutieren:

* disjunkte Supports → triviale Kommutatoren,
* volle DtN gegen blockdiagonalen Teil → kommutiert numerisch.

**Obstruktions-Ort**

Alle Zerlegungen respektieren dieselbe (S_b)-/Radialsymmetrie und teilen die symmetrieadaptierte Eigenbasis.

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
\mathcal A_{\mathrm{eff}}
=========================

\operatorname{Alg}{\Lambda_i},
$$

mit

$$
$$\Lambda_i,\Lambda_j$$\neq0.
$$

**Status**

Connes ist Ziel-/Vergleichsstruktur, nicht Generator.

---

## 20.2 Zwei Reduktionsregimes

**Script / Datei**

```text
two_reduction_regimes.py
```

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
|$$\Lambda_{\mathrm{UV}},\Lambda_{\mathrm{Env}}$$|\sim 10^{-16}.
$$

**Obstruktions-Ort**

Root-Selbstenergie verschiebt Eigenwerte, aber dreht keine Eigenräume. Radial bleibt radial.

**Status**

Negativ für exakte derived Regime.

---

## 20.3 Spektral trunkierte Reduktion

**Script / Datei**

```text
two_reduction_regimes.py
truncation_sign_test.py
```

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

Bei beliebigem (m):

$$
|$$\Lambda_{\mathrm{full}},\Lambda_{\mathrm{trunc}}$$|\approx 0.017
$$

für mittlere (m)-Werte; (K) ist schief.

**Obstruktions-Ort**

Zunächst falsch interpretiert: (\pm i\lambda)-Paare wurden als „beide Chiralitäten“ gelesen. Korrektur:

$$
\pm i\lambda
$$

ist normales Spektrum eines reellen (J)-Blocks.

Der echte Vorzeichentest ist:

$$
K\text{ oder }-K\text{ ausgezeichnet?}
$$

**Status**

Nur scheinbar positiver Kandidat; musste degenerazien-sicher nachgetestet werden.

---

## 20.4 Degenerazien-sichere Cluster-Trunkierung

**Script / Datei**

Nachtest zu `truncation_sign_test.py`; im Chat beschrieben.

**Ausgangslage**

Trunkierung nicht nach beliebigem (m), sondern nur nach ganzen Eigenwert-Clustern:

$$
P_{\le \lambda}
===============

\sum_{\mu\le\lambda}P_\mu.
$$

**Befund**

Bei allen kanonischen Cluster-Grenzen:

$$
|K|\approx 10^{-16}.
$$

Nichtkommutativität trat nur auf, wenn (m) mitten durch degenerierte Eigenräume schnitt.

**Obstruktions-Ort**

Ein Schnitt durch entartete Eigenräume wählt eine nicht-kanonische `numpy`-Basis. Das ist kein ToC-derived Mechanismus.

**Status**

Starker Negativbefund:

$$
\boxed{
\text{relabeling-natürliche exakte und cluster-sichere DtN-Reduktionen kommutieren.}
}
$$

---

# 21. Knoten-Elimination vs. partielle Spur

## 21.1 Falscher „Ausspuren“-Test

**Ausgangslage**

System/Umwelt-Knoten wurden getrennt:

$$
\mathbb R^N=\mathbb R^S\oplus\mathbb R^E.
$$

Dann wurde Diffusion (e^{-tL}) gerechnet und Umgebung mit festem Zustand behandelt.

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
L=
\begin{pmatrix}
L_{SS} & L_{SE}\
L_{ES} & L_{EE}
\end{pmatrix}.
$$

Korrekte Eliminierung:

$$
L_{\mathrm{eff}}
================

L_{SS}-L_{SE}L_{EE}^{-1}L_{ES}.
$$

**Befund**

Für reell symmetrisches (L):

$$
L_{\mathrm{eff}}^T=L_{\mathrm{eff}}.
$$

**Obstruktions-Ort**

Knoten-Elimination erzeugt keine OQS-Irreversibilität und keinen antisymmetrischen Hamilton-Teil.

**Status**

Zentrale Methodenkorrektur:

$$
\boxed{
\text{Auf Knoten wird eliminiert, nicht ausgespurt.}
}
$$

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
$$\nabla_\mu,\nabla_\nu$$\neq0.
$$

**Status**

Interpretationswechsel:

$$
\boxed{
\text{Die No-Gos betreffen den flachen ToC-/DtN-Sektor.}
}
$$

Nicht CNNA insgesamt.

---

## 22.2 Krümmung als möglicher späterer Ursprung von Nichtkommutativität

**Ausgangslage**

In Geometrie/Eichtheorie:

$$
$$\nabla_\mu,\nabla_\nu$$=R_{\mu\nu}
$$

bzw.

$$
$$D_\mu,D_\nu$$=F_{\mu\nu}.
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
\mathcal I_{\mathrm{ToCGrid}}
=============================

\Gamma_\infty\times T_b^\infty.
$$

Mit:

$$
x\in\Gamma_\infty,
\qquad
w\in T_b^\infty.
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

Transversale Isotropie nur diskret bzw. abhängig von (\Gamma_\infty).

**Obstruktions-Ort**

Das Gitter bringt transversale Nachbarschaft als neues IDEAL-Vergleichsdatum mit. Sie ist nicht aus einem einzelnen ToC abgeleitet.

**Status**

Sehr sinnvoller letzter ToC-naher Test vor Substratwechsel.

---

## 23.2 Endlicher Doppelschnitt

**Ausgangslage**

Berechenbarer Sektor:

$$
\Omega_{R,L}
============

W_R\times T_{\le L}.
$$

Mit:

$$
W_R\subset\Gamma_\infty,
\qquad
T_{\le L}\subset T_b^\infty.
$$

**Befund**

Subsystem-Sein bricht zwingend die IDEAL-Symmetrie:

$$
\operatorname{Aut}(\mathcal I_{\mathrm{ToCGrid}})
\to
\operatorname{Aut}(\Omega_{R,L}).
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

Positiver Geometrie-/DtN-Test, negativer (J)-Test im flachen reziproken Fall.

---

## 23.3 DtN auf dem ToC-Faser-Gitter

**Ausgangslage**

Operator auf (\Omega_{R,L}):

$$
L_{R,L}.
$$

Schur/DtN:

$$
\Lambda_{R,L}
=============

## L_{\partial\partial}

L_{\partial I}L_{II}^{-1}L_{I\partial}.
$$

**Befund**

Dies ist A→B-näher als rohe Knotenverklebung. B würde nicht ToC-Knoten sehen, sondern Handoff-Matrizen.

**Obstruktions-Ort**

Solange das Gitter homogen, reziprok und flach ist, entstehen zwar Spektrum und effektive Geometrie, aber keine ausgezeichnete Chirotopie.

**Status**

Wichtiger letzter Referenztest:

$$
\boxed{
\text{ToC-Faser-Gitter kann Geometrie testen, nicht }J\text{ erzwingen.}
}
$$

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
U_\gamma
========

U_{wx}U_{zw}U_{yz}U_{xy}.
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
\eta=\operatorname{diag}(-1,+1,\ldots,+1).
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

Reell-symmetrischer Generator (H), Flusspaar:

$$
{e^{+tH},e^{-tH}}.
$$

**Befund**

Liefert:

$$
{+\tau,-\tau}.
$$

**Obstruktions-Ort**

Für reell-symmetrisches (H) bleibt jede spektrale Funktion symmetrisch. Ein (J) ist antisymmetrisch:

$$
J\neq f(H).
$$

**Status**

Zeitpaar ja. Verriegelung mit (J) nein.

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
-i$$H,\rho$$.
$$

Also setzt OQS (i) bzw. (J) voraus.

**Status**

Pillar C kann (\tau) wählen, aber (J) nicht allein erzeugen.

---

# 27. AQFT / Type-I / Type-III / Handoff-Struktur

## 27.1 A als Type-I-/Type-III-Vorläuferschicht

**Ausgangslage**

Pillar A soll nicht direkt Type III beweisen, sondern Vorläufer liefern:

$$
\mathcal C_{d,k}
================

(Q_{d,k}\oplus P_{d,k},g_{d,k},\omega_{d,k},{J,-J}).
$$

Endlich:

$$
k<\infty
\Rightarrow
\text{Type-I-artige Vorläufer}.
$$

Unendlich:

$$
k\to\infty
\Rightarrow
\text{Type-III-fähige Komplementfamilien-Vorläufer}.
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
C\text{-Regulator}
\triangleright
H_{B|B'}(B,B')
\to
\text{stable record}.
$$

**Befund**

Bester Ort für:

$$
\omega_{\mathrm{lock}}.
$$

**Obstruktions-Ort**

Noch nicht formalisiert. Type-I/Type-III-Asymmetrie ist zunächst Algebra-/Dimensionsasymmetrie, nicht Orientierung.

**Status**

Weiterhin wichtigster offener (J)-Locking-Kandidat.

---

# 28. Multi-ToC / Detektor / Vielteilchenstruktur

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
\sigma_{ij},
\qquad
\Phi_\gamma=\prod_{(ij)\in\gamma}\sigma_{ij}.
$$

**Obstruktions-Ort**

Mechanismus für (\sigma_{ij}) ist noch nicht derived.

**Status**

Starker Kandidat für nächsten nichtlokalen Test.

---

## 28.2 Frustration / Spin-netz-artige Struktur

**Ausgangslage**

Viele lokale ToC-Fasern werden gekoppelt. Mögliches Zyklusprodukt:

$$
\Phi_\gamma=-1.
$$

**Befund**

Falls (\Phi_\gamma) invariant unter lokalen Gauge-Flips

$$
J_i\mapsto -J_i
$$

ist, entsteht echte globale Frustration.

**Obstruktions-Ort**

(\sigma_{ij}) darf nicht gesetzt werden.

**Status**

Wichtigster offener Multi-ToC-Testpfad.

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
{+J,-J}.
$$

**Obstruktions-Ort**

Ohne dritte Phasenordnung bzw. Anschlussordnung kein stabiler Drehsinn.

**Status**

Didaktisch stark.

---

## 29.2 Drei Phasen / Anschlussordnung

**Ausgangslage**

Balanciertes System:

$$
(1,a,a^2),
\qquad
a=e^{2\pi i/3}.
$$

Vertauschung:

$$
(1,a,a^2)
\leftrightarrow
(1,a^2,a).
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

Für das erste (J)-Vorzeichenproblem negativ. Höhere Algebra löst nicht die Herkunft der ersten komplexen Orientierung.

**Obstruktions-Ort**

Dimensionsverdopplung und Normmultiplikativität werden nicht aus Schnittdaten erzwungen.

Offene Objekte:

```text
positiveDefiniteNormSq
divisionFromNormSq
alternativeLaw
```

**Status**

Nicht aktueller Weg für (J)-Vorzeichen. Als spätere Zielstruktur nicht ausgeschlossen.

---

# 31. Substratwechsel-Kandidaten

## 31.1 ToC bleibt lokale Provenienzfaser

**Ausgangslage**

Der Einzel-ToC als globaler Weltbaum scheitert am (J)-Gate.

**Befund**

Als lokale Faser bleibt ToC wertvoll:

$$
\text{Provenienz}
\to
\text{Approximant}
\to
\text{Schur/DtN}
\to
\text{lokaler Handoff-Operator}.
$$

**Obstruktions-Ort**

Globale Ontologie als einzelner Baum ist zu arm für zweite Achse, Chirotopie, Krümmung.

**Status**

Kein Totalverwerfen des ToC; Rollenwechsel.

---

## 31.2 Ereignisstrukturen als Kandidat

**Ausgangslage**

Ereignisstrukturen besitzen zwei Relationen:

$$
\leq
\qquad\text{und}\qquad
#.
$$

**Befund**

Sie könnten zwei Achsen bzw. Kausalität und Konflikt tragen.

**Obstruktions-Ort**

Beide Relationen wären zunächst primitive Eingabedaten, solange sie nicht CNNA-derived sind.

**Status**

Starker Kandidat innerhalb (i)-freier Substratklassen, aber noch kein derived Ergebnis.

---

# 32. Wesentliche Scripts nach Chatstand

## 32.1 Sicher benannte ältere Scripts

```text
app.py
```

ToC-Concept-Explorer.

```text
alpha_orth_invariant.py
```

(\Xi)-, (\rho)-, (\alpha_{\mathrm{orth}})-Diagnostik.

```text
two_approximant_flow_sign.py
```

Parent–Child-/Flow-Sign-Diagnostik.

```text
two_boundary_shell_chirality.py
```

V4: Einzelgraph-Cauchy-Pairing.

```text
family_handoff_chirality.py
```

V5: Cross-Graph-Cauchy und Handoff-Square.

```text
triadic_interface_chirality.py
```

V6: UV/Env/Regulator-Triade.

```text
oriented_cauchy_shell_gate.py
```

V7: gegengerichtete UV/Env-Cauchy-Shell.

## 32.2 Neuere im Chat genannte Test-Scripts

```text
two_reduction_regimes.py
```

DtN-Regimes: bare UV-DtN vs. root-env-DtN; zusätzlich trunkierte Reduktion.

```text
truncation_sign_test.py
```

Vorzeichen-/Kommutator-Test für spektrale Trunkierung.

## 32.3 Als nächste Scripts sinnvoll

```text
cluster_safe_truncation_test.py
```

Degenerazien-sichere Cluster-Trunkierung explizit kapseln.

```text
toc_fiber_grid_dtn_test.py
```

IDEAL-ToC-Faser-Gitter, Doppelschnitt (\Omega_{R,L}), DtN-Spektraltests.

```text
fiber_grid_connection_holonomy_test.py
```

Derived Intertwiner (U_{xy}) und Loop-Holonomie im Faser-Gitter.

```text
multi_toc_frustration_gate.py
```

Gauge-invariante Zyklusprodukte:

$$
\Phi_\gamma=\prod\sigma_{ij}.
$$

```text
handoff_phase_sequence_gate.py
```

Handoff-Sequenz-/Mehrphasen-Gate.

---

# 33. Obstruktions-Orte nach Typ

## 33.1 Reziprozität

$$
\Lambda=\Lambda^T.
$$

Passive Schur-/DtN-Reduktion bleibt symmetrisch. Kein antisymmetrischer (J)-Generator.

## 33.2 Reelle Konjugationssymmetrie

$$
J\mapsto -J.
$$

Reelle Strukturen wählen keine komplexe Orientierung.

## 33.3 (S_b)-Äquivarianz

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

UV/Env-Coorientierung liefert:

$$
{J,-J}.
$$

## 33.9 Handoff reversal

$$
A_{\gamma^{-1}}=-A_\gamma.
$$

Ohne gerichtete Handoff-Sequenz kein absoluter Drehsinn.

## 33.10 OQS-Abhängigkeit von (i)

Lindblad/OQS kann Zeitrichtung liefern, setzt aber Hamilton-(i) voraus.

## 33.11 Flachheit

Im flachen ToC-/DtN-Sektor fehlen:

$$
\text{Connection},
\qquad
\text{Holonomie},
\qquad
\text{Krümmung}.
$$

---

# 34. Aktuelle Gesamtformel

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

Der wichtigste nächste ToC-nahe Test vor Substratwechsel bleibt:

$$
\boxed{
\mathcal I_{\mathrm{ToCGrid}}=\Gamma_\infty\times T_b^\infty,
\qquad
\Omega_{R,L}=W_R\times T_{\le L},
\qquad
\Lambda_{R,L}.
}
$$

Ziel:

$$
\text{effektive Geometrie aus Subsystem-Sein testen},
$$

aber getrennt davon:

$$
\text{(J)-/Chirotopie-/Nichtkommutativitäts-Gate weiter offen halten}.
$$
