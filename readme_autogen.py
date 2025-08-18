#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatisierte README-Erstellung für mehrere Ordner.
- Arbeitet ordnerweise (nicht alles auf einmal).
- Unterstützt "nur geänderte Ordner" via Git.
- Optionales Entpacken von Archiven (.zip/.tar.* nativ; .rar via `rarfile` + unrar-Tool).
- Kann Änderungen committen/pushen (für CI).

Aufrufbeispiele:
  python readme_autogen.py --root . --changed-only --max-folders 20 --commit
  python readme_autogen.py --root ./docs --folders data/ projA/ --extract-archives
"""
from __future__ import annotations
import argparse, os, sys, subprocess, json, re, tarfile, zipfile, shutil
from pathlib import Path
from datetime import date

# ---------- Hilfsfunktionen aus Single-Ordner-Generator ----------
ROLE_GUESS = {
    ".py": "code",
    ".ipynb": "notebook",
    ".md": "doc",
    ".txt": "doc",
    ".csv": "data",
    ".tsv": "data",
    ".json": "data",
    ".parquet": "data",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".pdf": "doc"
}

def human_size(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"

def scan_files(folder: Path) -> list[dict]:
    out = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if p.name.startswith("."): 
            continue
        if p.is_dir():
            # nur direkte Dateien des Ordners
            continue
        if p.name.lower() in {"readme.md", "readme_template.md", "dirindex.meta.json", "files.meta.json"}:
            # tauchen trotzdem in der Tabelle nicht auf
            pass
        ext = p.suffix.lower()
        role = ROLE_GUESS.get(ext, "other")
        out.append({
            "name": p.name,
            "size": p.stat().st_size,
            "size_h": human_size(p.stat().st_size),
            "type": ext[1:] if ext else "file",
            "role": role
        })
    return out

def fill_table(files: list[dict], files_meta: dict) -> str:
    lines = ["| Name | Größe | Typ | Rolle | Kurzbeschreibung | Tags |",
             "|---|---:|---|---|---|---|"]
    for f in files:
        name = f["name"]
        meta = files_meta.get(name, {})
        role = meta.get("role", f["role"])
        desc = meta.get("desc", "")
        tags = ",".join(meta.get("tags", []))
        lines.append(f"| {name} | {f['size_h']} | {f['type']} | {role} | {desc} | {tags} |")
    return "\n".join(lines)

def render_readme(template: str, meta: dict, files: list[dict], files_meta: dict) -> str:
    # Tabelle
    table = fill_table(files, files_meta)
    # Platzhalter ersetzen
    today = date.today().isoformat()
    t = template
    repl = {
        "<TITEL DES ORDNERS/PROJEKTS>": meta.get("title", ""),
        "<owner>/<repo>": meta.get("repo", ""),
        "<ordner/pfad>": meta.get("path", ""),
        "<YYYY-MM-DD>": meta.get("updated", today),
        "<Lizenz>": meta.get("license", "MIT")
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    # Dateitabelle ersetzen
    t = re.sub(r"<!-- dirindex:files:start -->.*?<!-- dirindex:files:end -->",
               "<!-- dirindex:files:start -->\n" + table + "\n<!-- dirindex:files:end -->",
               t, flags=re.S)
    # JSON-Block aktualisieren
    t = re.sub(r"(<!-- dirindex-json\n)(.*?)(\ndirindex-json -->)",
               r"\1" + json.dumps({
                   "repo": meta.get("repo",""),
                   "path": meta.get("path",""),
                   "title": meta.get("title",""),
                   "updated": meta.get("updated", today),
                   "status": meta.get("status","draft"),
                   "tags": meta.get("tags",[]),
                   "links": meta.get("links",{}),
                   "files": [{"name": f["name"],
                              "role": (files_meta.get(f["name"],{}).get("role", f["role"])),
                              "desc": files_meta.get(f["name"],{}).get("desc",""),
                              "tags": files_meta.get(f["name"],{}).get("tags",[])}
                             for f in files]
               }, ensure_ascii=False, indent=2) + r"\3",
               t, flags=re.S)
    return t

# ---------- Archive extrahieren ----------
def extract_archives(folder: Path, allow_rar: bool=False) -> list[Path]:
    """
    Entpackt unterstützte Archive in ./extracted/<stem>/ im gleichen Ordner.
    .zip und .tar.*: Standardbibliothek
    .rar: optional via rarfile + unrar
    Gibt Liste der entstandenen Verzeichnisse zurück.
    """
    created = []
    ext_targets = folder / "extracted"
    ext_targets.mkdir(exist_ok=True)
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file(): 
            continue
        name = p.name.lower()
        stem = p.stem
        target = ext_targets / stem
        if target.exists():
            continue
        try:
            if name.endswith(".zip"):
                with zipfile.ZipFile(p, 'r') as z:
                    z.extractall(target)
                created.append(target)
            elif name.endswith((".tar.gz",".tgz",".tar.bz2",".tbz2",".tar.xz",".txz",".tar")):
                mode = "r:gz" if name.endswith((".tar.gz",".tgz")) else \
                       "r:bz2" if name.endswith((".tar.bz2",".tbz2")) else \
                       "r:xz" if name.endswith((".tar.xz",".txz")) else "r:"
                with tarfile.open(p, mode) as t:
                    t.extractall(target)
                created.append(target)
            elif allow_rar and name.endswith(".rar"):
                try:
                    import rarfile
                except Exception as e:
                    print(f"[warn] rarfile nicht installiert: {e}. Überspringe {p.name}.", file=sys.stderr)
                    continue
                try:
                    with rarfile.RarFile(p) as rf:
                        rf.extractall(target)
                    created.append(target)
                except Exception as e:
                    print(f"[warn] Konnte RAR nicht extrahieren ({p.name}): {e}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Fehler beim Extrahieren {p.name}: {e}", file=sys.stderr)
    return created

# ---------- Git-Helfer ----------
def run(cmd: list[str], cwd: Path|None=None, check: bool=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)

def git_changed_dirs(root: Path, base: str|None=None, head: str|None=None) -> list[str]:
    """Gibt Liste relativer Ordner zurück, in denen Dateien geändert wurden."""
    try:
        # Bestimme Default-Base, falls nicht gesetzt
        if base is None:
            # origin/HEAD -> refs/remotes/origin/main|master
            try:
                out = run(["git","symbolic-ref","refs/remotes/origin/HEAD"], cwd=root, check=True).stdout.strip()
                base = out.split("/")[-1]
            except Exception:
                base = "HEAD~1"
        if head is None:
            head = "HEAD"
        diff = run(["git","diff","--name-only", f"origin/{base}...{head}"], cwd=root, check=True).stdout
        paths = [Path(p) for p in diff.splitlines() if p.strip()]
        dirs = sorted({ (root / p).parent.relative_to(root).as_posix() for p in paths if (root/p).exists() })
        return [d if d.endswith("/") else (d + "/") if d != "." else "" for d in dirs]
    except Exception as e:
        print(f"[warn] git diff fehlgeschlagen: {e}. Fallback: keine Filterung.", file=sys.stderr)
        return []  # leere Liste heißt: Caller kann entscheiden (z.B. alle nehmen)

def git_commit_all(root: Path, message: str) -> bool:
    try:
        run(["git","config","user.name","dirindex-bot"], cwd=root)
        run(["git","config","user.email","github-actions[bot]@users.noreply.github.com"], cwd=root)
        run(["git","add","-A"], cwd=root)
        # commit nur, wenn Änderungen existieren
        status = run(["git","status","--porcelain"], cwd=root).stdout.strip()
        if not status:
            print("[info] Nichts zu committen.")
            return False
        run(["git","commit","-m", message], cwd=root)
        run(["git","push"], cwd=root)
        print("[info] Änderungen gepusht.")
        return True
    except Exception as e:
        print(f"[warn] Commit/Push fehlgeschlagen: {e}", file=sys.stderr)
        return False

# ---------- Hauptlogik ----------
def process_folder(root: Path, rel_folder: str, template_path: Path, extract: bool=False, allow_rar: bool=False) -> bool:
    """
    Erzeugt/aktualisiert README.md in rel_folder.
    Gibt True zurück, wenn README.md neu geschrieben/geändert wurde.
    """
    folder = (root / rel_folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"[skip] Ordner existiert nicht: {rel_folder}")
        return False

    # Optional: Archive entpacken
    if extract:
        extract_archives(folder, allow_rar=allow_rar)

    meta_path = folder / "dirindex.meta.json"
    files_meta_path = folder / "files.meta.json"
    readme_path = folder / "README.md"

    # Metadaten laden/erstellen
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # Stub erzeugen
        rel = folder.relative_to(root).as_posix()
        repo = os.getenv("GITHUB_REPOSITORY","<owner>/<repo>")
        meta = {
            "repo": repo,
            "path": rel if rel != "." else "/",
            "title": folder.name,
            "updated": date.today().isoformat(),
            "status": "draft",
            "tags": []
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[init] dirindex.meta.json angelegt in {rel}")

    files_meta = {}
    if files_meta_path.exists():
        try:
            files_meta = json.loads(files_meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] Konnte files.meta.json nicht lesen ({rel_folder}): {e}", file=sys.stderr)

    # Dateien scannen (nur direkt im Ordner)
    files = scan_files(folder)

    # Template laden
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Template nicht gefunden: {template_path}")

    # README rendern
    out = render_readme(template, meta, files, files_meta)

    old = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    if old == out:
        print(f"[ok] Keine Änderung: {rel_folder}")
        return False

    readme_path.write_text(out, encoding="utf-8")
    print(f"[write] README.md aktualisiert: {rel_folder}")
    return True

def main():
    ap = argparse.ArgumentParser(description="Automatisierte README-Generierung ordnerweise")
    ap.add_argument("--root", default=".", help="Repo-Root oder Start-Ordner")
    ap.add_argument("--folders", nargs="*", default=None, help="Konkrete Unterordner (relativ zu --root). Beispiel: data/ projA/")
    ap.add_argument("--changed-only", action="store_true", help="Nur Ordner mit geänderten Dateien (git diff)")
    ap.add_argument("--base", default=None, help="Basis-Branch/Ref für git diff (z. B. main)")
    ap.add_argument("--head", default=None, help="Head-Ref für git diff (default: HEAD)")
    ap.add_argument("--max-folders", type=int, default=None, help="Maximale Anzahl Ordner in diesem Lauf")
    ap.add_argument("--template", default="README_TEMPLATE.md", help="Pfad zum README-Template")
    ap.add_argument("--extract-archives", action="store_true", help="Archive (*.zip, *.tar.*, optional .rar) entpacken")
    ap.add_argument("--allow-rar", action="store_true", help="RAR-Entpacken erlauben (benötigt 'rarfile' + unrar)")
    ap.add_argument("--commit", action="store_true", help="Änderungen committen/pushen")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    template_path = (Path(args.template).resolve()
                     if Path(args.template).is_file()
                     else (root/args.template).resolve())
    if not root.exists():
        print("Root nicht gefunden.", file=sys.stderr); sys.exit(2)
    if not template_path.exists():
        print("Template nicht gefunden.", file=sys.stderr); sys.exit(3)

    # Kandidaten-Ordner bestimmen
    candidates: list[str] = []
    if args.folders:
        candidates = [f if f.endswith("/") else f + "/" for f in args.folders]
    else:
        # Alle direkten Unterordner + Root selbst
        for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if p.is_dir() and not p.name.startswith(".") and p.name != ".git" and p.name != ".github":
                candidates.append(p.relative_to(root).as_posix() + "/")
        # Root selbst verarbeiten (leer = Root)
        candidates = [""] + candidates

    # Changed-only filtern
    if args.changed-only:
        changed = git_changed_dirs(root, base=args.base, head=args.head)
        if changed:
            # Nur Schnittmenge
            candidates = [c for c in candidates if c in changed]
        else:
            # Wenn nichts ermittelt werden konnte, lieber alle nehmen (aber evtl. limitiert)
            pass

    # Limit
    if args.max-folders:
        candidates = candidates[: args.max-folders]

    print(f"[info] Ordner in diesem Lauf: {candidates}")

    # Ausführen
    wrote_any = False
    for rel in candidates:
        wrote = process_folder(root, rel, template_path,
                               extract=args.extract-archives, allow_rar=args.allow-rar)
        wrote_any = wrote_any or wrote

    # Committen (optional)
    if args.commit:
        git_commit_all(root, f"Auto: README aktualisiert ({len(candidates)} Ordner)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
