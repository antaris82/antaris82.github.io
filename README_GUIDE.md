# README-Standard (für GitHub + deinen Indexer)

Ziel: **Maschinenfreundlich** für das Indizieren (du bekommst aus dem README strukturierte Infos) und **menschenfreundlich** für Leser.

## Regeln in Kürze
1. **Überschrift & Kurzfassung**: Klarer Titel, 1–2 Sätze TL;DR.
2. **Quick Facts**: Repo, Pfad, Status, Datum, Lizenz, Themen.
3. **Datei-Tabelle**: Name, Größe, Typ, *Rolle*, Kurzbeschreibung, Tags.
4. **Nutzung**: Schnellstart, Dependencies, Ordnerstruktur.
5. **Optional Mathe**: Nutze `$…$` (inline) und `$$…$$` (display). Mehrzeilig via `aligned`.
6. **Maschinenlesbarer Block**: Der `dirindex-json`-Kommentar am Ende mit Meta/Files (redundant zur Tabelle, aber robust parsbar).
7. **Cover-Bild**: `background.jpg|png` im selben Ordner für die Startseite.

## Automatische Erstellung
1. Fülle **`dirindex.meta.json`** aus (siehe Schema).
2. Lege optional **`files.meta.json`** an, um für einzelne Dateien Rollen/Beschreibungen/Tags vorzugeben.
3. Führe im Zielordner aus:
   ```bash
   python3 generate_readme.py
   ```
   Das Skript erzeugt **README.md** aus dem Template inkl. Dateiliste.

## Felder – Empfehlungen
- **status**: `draft` | `stable` | `archived`
- **role**: `data` | `code` | `doc` | `image` | `notebook` | `binary` | `other`
- **tags**: frei wählbar, klein halten (≤ 6)

Stand: 2025-08-18
