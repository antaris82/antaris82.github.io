# RESULTS — `test_response_operator_refresh_rule.py`

## Fragestellung

Der Test vergleicht drei Semantiken für abgeschlossene lokale Geschwister-Tripel im wachsenden realen Komplementnetzwerk:

```text
record_only       = eingefrorener Birth-History-Record W_birth
live_only         = Current-State-Replay derselben lokalen Response-Regel
record_plus_live  = W_birth + positiver späterer Live-Inkrementkanal
```

Die Refresh-Regel wird nicht als freier Fit eingeführt. `live_only` rekonstruiert die lokale gerichtete Response aus bereits vorhandenen Größen:

```text
current child conductances g_i(t)
current ancestor environment
alpha_env
br_sibling
ancestor_env_decay
shell-normalized kernel K(d)=1/(3^(d-1)d^2)
```

Die Zwei-Ebenen-Variante nutzt parallele Kanäle:

```text
W_two = W_birth + max(W_live(t) - W_live(t_complete), 0)
```

Der abgeschnittene negative Anteil wird als Audit-Masse ausgegeben und nicht still als negativer Leitwert verwendet.

## Run

```bash
python3 test_response_operator_refresh_rule.py \
  --max-level 9 \
  --outdir response_operator_refresh_rule_out_L9
```

Finale Größe:

```text
nodes               = 29,524
completed parents   = 9,841
vertical pairs      = 9,840 pro Semantik
closure loops       = 16,400 pro Semantik
```

## Wichtigste numerische Befunde bei L=9

### Lokaler J-Sektor bleibt in allen drei Semantiken stabil

```text
record_only:       mean J² residual ≈ 1.20e-16
live_only:         mean J² residual ≈ 1.29e-16
record_plus_live:  mean J² residual ≈ 1.23e-16
```

Das ist ein sehr schwacher, aber wichtiger Sicherheitsbefund: die Refresh-Semantik zerstört den lokalen 2D-J-Kandidaten numerisch nicht. Er beweist aber weiterhin kein physikalisches `i`.

### Live reduziert level-zentrierte Loop-Residuen, kostet aber Achsen-Gluing

Über alle finalen Loop-Modi:

```text
record_only:       mean abs centered residual ≈ 0.12468°, p95 ≈ 0.29532°
live_only:         mean abs centered residual ≈ 0.07196°, p95 ≈ 0.17484°
record_plus_live:  mean abs centered residual ≈ 0.09318°, p95 ≈ 0.22147°
```

Damit ist `live_only` bei der residualen Krümmung am besten. Gegenüber `record_only` ist das eine Reduktion um ungefähr 42%. `record_plus_live` liegt dazwischen und reduziert die Residuen um ungefähr 25%.

### Vertikales Tower-Gluing spricht gegen reines Live-Only

```text
record_only:
  mean |dot|       ≈ 0.999996197
  mean angle       ≈ 0.12899°
  mean J mismatch  ≈ 3.18e-3

live_only:
  mean |dot|       ≈ 0.999739916
  mean angle       ≈ 1.17526°
  mean J mismatch  ≈ 2.90e-2

record_plus_live:
  mean |dot|       ≈ 0.999990660
  mean angle       ≈ 0.23509°
  mean J mismatch  ≈ 5.80e-3
```

Das reine Live-Update verbessert zwar die Residuen, verschlechtert aber die vertikale Achsenkohärenz deutlich. Die Zwei-Ebenen-Semantik ist hier der konstruktive Kompromiss: wesentlich näher am Record-Gluing, aber mit reduzierten Loop-Residuen.

### Old-interior aging wird sichtbar

`record_only` driftet definitionsgemäß nicht:

```text
record_only phase drift = 0
```

`live_only` und `record_plus_live` zeigen dagegen einen nichttrivialen Drift älterer Schichten:

```text
live_only distance=1:        drift ≈ 1.613°
live_only distance=2:        drift ≈ 1.902°
live_only distance=3:        drift ≈ 1.989°

record_plus_live distance=1: drift ≈ 1.655°
record_plus_live distance=2: drift ≈ 1.948°
record_plus_live distance=3: drift ≈ 2.022°
```

Das bestätigt die Semantikspaltung: der Birth-Record ist eingefroren, aber der Live-State altert weiter durch spätere Backreaction.

### Zwei-Layer-Audit

```text
mean live_delta_pos_sum ≈ 2.4218e-2
mean live_delta_neg_sum ≈ 3.3521e-4
negative/positive ratio ≈ 1.38%
```

Die nicht eingefügte negative Audit-Masse ist klein, aber nicht null. Das ist wichtig: `record_plus_live` ist kein exakter algebraischer Direct-Sum-Beweis, sondern ein positiver Parallelkanal-Test mit explizitem Audit der verworfenen negativen Differenz.

## Kritisches Urteil

Der Test spricht gegen eine naive reine Ersetzung des Records durch Live-State:

```text
live_only verbessert residual curvature,
aber verschlechtert tower gluing deutlich.
```

Der Test spricht vorläufig für eine Zwei-Ebenen-Semantik:

```text
record_plus_live hält das Record-Gluing fast stabil,
reduziert aber alte Residuen gegenüber record_only.
```

Die sauberste aktuelle Interpretation ist daher:

```text
Record layer bleibt als historischer Birth-Handoff nötig.
Live layer ist nötig, um spätere Backreaction/aging zu beschreiben.
Ein zusammenfallender Single-Layer-Operator ist wahrscheinlich zu grob.
```

## Nicht bewiesen

Weiterhin nicht gerechtfertigt:

```text
physikalisches i endgültig abgeleitet
physikalische Zeit abgeleitet
modularer Fluss bewiesen
Type III bewiesen
AQFT-Handoff bewiesen
```

## Nächster Testvorschlag

Der nächste harte Test sollte `record_plus_live` gegen alternative positive Zwei-Kanal-Regeln prüfen:

```text
A. positive incremental replay:
   W_birth + max(W_live(t)-W_live(t_complete),0)

B. exact signed diagnostic, aber getrennt als skew correction statt Leitwert:
   W_birth plus separat gespeicherter signed LiveDefect

C. true block-direct two-channel operator:
   W_total = W_record ⊕ W_live
   danach erst Projektion/Coarse-Graining prüfen
```

Variante C wäre methodisch am saubersten, weil sie Record und Live nicht vorschnell in eine einzige 3×3-Matrix presst.

---

# Adopted provisional model setting after this test

We adopt the two-layer response semantics as the current working model setting:

```text
Record layer = immutable birth/provenance handoff record
Live layer   = current conductance/response state under later backreaction
```

This is not treated as a mere numerical accident. The structural reason is that one single operator cannot simultaneously satisfy:

```text
record invariance under future extensions
live covariance under current backreaction state
nontrivial aging/backreaction
```

unless backreaction is trivialized, provenance is rewritten, or the live state is ignored. Therefore the current path keeps the roles separated.

See `TWO_LAYER_MODEL_SETTING.md` for the explicit single-layer obstruction and AQFT-net provenance note.
