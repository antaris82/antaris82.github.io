#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, sys, subprocess, json, re, tarfile, zipfile, shutil
from pathlib import Path
from datetime import date

ROLE_GUESS = {
    ".py": "code", ".ipynb": "notebook", ".md": "doc", ".txt": "doc",
    ".csv": "data", ".tsv": "data", ".json": "data", ".parquet": "data",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".gif": "image", ".pdf": "doc"
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
        if p.is_dir(): continue
        ext = p.suffix.lower()
        role = ROLE_GUESS.get(ext, "other")
        out.append({"name": p.name, "size": p.stat().st_size, "size_h": human_size(p.stat().st_size),
                    "type": ext[1:] if ext else "file", "role": role})
    return out

def fill_table(files: list[dict], files_meta: dict) -> str:
    lines = ["| Name | Größe | Typ | Rolle | Kurzbeschreibung | Tags |",
             "|---|---:|---|---|---|---|"]
    for f in files:
        m = files_meta.get(f["name"], {})
        lines.append(f"| {f['name']} | {f['size_h']} | {f['type']} | {m.get('role', f['role'])} | {m.get('desc','')} | {','.join(m.get('tags',[]))} |")
    return "\n".join(lines)

def render_readme(template: str, meta: dict, files: list[dict], files_meta: dict) -> str:
    table = fill_table(files, files_meta)
    today = date.today().isoformat()
    t = template
    repl = {"<TITEL DES ORDNERS/PROJEKTS>": meta.get("title",""),
            "<owner>/<repo>": meta.get("repo",""),
            "<ordner/pfad>": meta.get("path",""),
            "<YYYY-MM-DD>": meta.get("updated", today),
            "<Lizenz>": meta.get("license","MIT")}
    for k,v in repl.items(): t = t.replace(k,v)
    t = re.sub(r"<!-- dirindex:files:start -->.*?<!-- dirindex:files:end -->",
               "<!-- dirindex:files:start -->\n"+table+"\n<!-- dirindex:files:end -->", t, flags=re.S)
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
               }, ensure_ascii=False, indent=2) + r"\3", t, flags=re.S)
    return t

def extract_archives(folder: Path, allow_rar: bool=False) -> None:
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file(): continue
        name = p.name.lower()
        target = folder / "extracted" / p.stem
        if target.exists(): continue
        try:
            if name.endswith(".zip"):
                with zipfile.ZipFile(p,'r') as z: z.extractall(target)
            elif name.endswith((".tar.gz",".tgz",".tar.bz2",".tbz2",".tar.xz",".txz",".tar")):
                mode = "r:gz" if name.endswith((".tar.gz",".tgz")) else \
                       "r:bz2" if name.endswith((".tar.bz2",".tbz2")) else \
                       "r:xz" if name.endswith((".tar.xz",".txz")) else "r:"
                import tarfile
                with tarfile.open(p, mode) as t: t.extractall(target)
            elif allow_rar and name.endswith(".rar"):
                try:
                    import rarfile
                    with rarfile.RarFile(p) as rf: rf.extractall(target)
                except Exception as e:
                    print(f"[warn] RAR nicht entpackt ({p.name}): {e}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Fehler beim Entpacken {p.name}: {e}", file=sys.stderr)

def run(cmd, cwd: Path|None=None, check=True):
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)

def git_changed_dirs(root: Path, base: str|None=None, head: str|None=None) -> list[str]:
    try:
        if base is None:
            try:
                out = run(["git","symbolic-ref","refs/remotes/origin/HEAD"], cwd=root, check=True).stdout.strip()
                base = out.split("/")[-1]
            except Exception:
                base = "HEAD~1"
        if head is None: head = "HEAD"
        diff = run(["git","diff","--name-only", f"origin/{base}...{head}"], cwd=root, check=True).stdout
        paths = [Path(p) for p in diff.splitlines() if p.strip()]
        dirs = sorted({ (root / p).parent.relative_to(root).as_posix() for p in paths if (root/p).exists() })
        return [d if d else ""] + [d+"/" if d and not d.endswith("/") else d for d in []]
    except Exception as e:
        print(f"[warn] git diff fehlgeschlagen: {e}", file=sys.stderr)
        return []

def git_commit_all(root: Path, message: str) -> bool:
    try:
        run(["git","config","user.name","dirindex-bot"], cwd=root)
        run(["git","config","user.email","github-actions[bot]@users.noreply.github.com"], cwd=root)
        run(["git","add","-A"], cwd=root)
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

def process_folder(root: Path, rel_folder: str, template_path: Path, extract: bool=False, allow_rar: bool=False) -> bool:
    folder = (root / rel_folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"[skip] Ordner existiert nicht: {rel_folder}")
        return False
    if extract: extract_archives(folder, allow_rar=allow_rar)
    meta_path = folder / "dirindex.meta.json"
    files_meta_path = folder / "files.meta.json"
    readme_path = folder / "README.md"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        rel = folder.relative_to(root).as_posix()
        repo = os.getenv("GITHUB_REPOSITORY","<owner>/<repo>")
        meta = {"repo": repo, "path": rel if rel != "." else "/", "title": folder.name,
                "updated": date.today().isoformat(), "status": "draft", "tags": []}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[init] dirindex.meta.json angelegt in {rel}")
    files_meta = {}
    if files_meta_path.exists():
        try: files_meta = json.loads(files_meta_path.read_text(encoding="utf-8"))
        except Exception as e: print(f"[warn] files.meta.json fehlerhaft: {e}", file=sys.stderr)
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Template nicht gefunden: {template_path}")
    files = scan_files(folder)
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
    ap.add_argument("--folders", nargs="*", default=None, help="Konkrete Unterordner (relativ zu --root)")
    ap.add_argument("--changed-only", dest="changed_only", action="store_true", help="Nur Ordner mit geänderten Dateien (git diff)")
    ap.add_argument("--max-folders", type=int, default=None, help="Maximale Anzahl Ordner in diesem Lauf")
    ap.add_argument("--template", default="README_TEMPLATE.md", help="Pfad zum README-Template")
    ap.add_argument("--extract-archives", dest="extract_archives", action="store_true", help="Archive entpacken")
    ap.add_argument("--allow-rar", dest="allow_rar", action="store_true", help="RAR-Entpacken erlauben")
    ap.add_argument("--commit", action="store_true", help="Änderungen committen/pushen")
    ap.add_argument("--base", default=None, help="Basis-Branch/Ref für git diff (z. B. main)")
    ap.add_argument("--head", default=None, help="Head-Ref für git diff (default: HEAD)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    template_path = (Path(args.template).resolve()
                     if Path(args.template).is_file()
                     else (root/args.template).resolve())
    if not root.exists(): print("Root nicht gefunden.", file=sys.stderr); return 2
    if not template_path.exists(): print("Template nicht gefunden.", file=sys.stderr); return 3

    candidates = []
    if args.folders:
        candidates = [f if f.endswith("/") else f + "/" for f in args.folders]
    else:
        for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if p.is_dir() and not p.name.startswith(".") and p.name not in {".git",".github"}:
                candidates.append(p.relative_to(root).as_posix() + "/")
        candidates = [""] + candidates

    if args.changed_only:
        changed = git_changed_dirs(root, base=args.base, head=args.head)
        if changed:
            candidates = [c for c in candidates if c in changed]
        else:
            pass

    # Bootstrap: wenn kein README im Root existiert, Root hinzufügen
    try:
        if not (root/"README.md").exists() and "" not in candidates:
            candidates = [""] + candidates
    except Exception:
        pass

    if args.max_folders:
        candidates = candidates[: args.max_folders]

    print(f"[info] Ordner in diesem Lauf: {candidates}")

    wrote_any = False
    for rel in candidates:
        wrote = process_folder(root, rel, template_path,
                               extract=args.extract_archives, allow_rar=args.allow_rar)
        wrote_any = wrote_any or wrote

    if args.commit:
        git_commit_all(root, f"Auto: README aktualisiert ({len(candidates)} Ordner)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
