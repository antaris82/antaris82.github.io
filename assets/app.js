// assets/app.js
import { OWNER, REPO, BRANCH } from "./config.js";
import { authHeaders, fetchManifest, fetchAllViaAPI, getManifestHeaders, rawUrl } from "./api.js";
import { renderMarkdownInto } from "./md.js";
import { renderGallery, renderFolder } from "./render.js";
import { getLang, initLangSelect } from "./i18n.js";

// State from URL
const QS = new URLSearchParams(location.search);
let cwd = decodeURIComponent(QS.get("path") || "");
if (cwd.startsWith("/")) cwd = cwd.slice(1);
if (cwd.endsWith("/"))  cwd = cwd.slice(0, -1);

const token = QS.get("token") || "";
const tokenInput = document.getElementById("tokenInput");
if (token) tokenInput.value = token;

// Lang
let LANG = getLang();
const langSelect = document.getElementById("langSelect");
initLangSelect(langSelect, LANG, (L)=>{ LANG=L; });

// UI refs
const statsEl = document.getElementById("stats");
const listingEl = document.getElementById("listing");
const readmeEl = document.getElementById("readme");
const galleryEl = document.getElementById("gallery");
const cwdLabelEl = document.getElementById("cwdLabel");
const currentPathEl = document.getElementById("currentPath");
const mdPathEl = document.getElementById("mdPath");
const openInNewEl = document.getElementById("openInNew");

cwdLabelEl.textContent = "/" + (cwd || "");
currentPathEl.textContent = "/" + (cwd || "");

// Buttons
document.getElementById("reloadBtn").addEventListener("click", () => location.reload());

function buildNavHref(path) {
  const base = `${location.pathname}?path=${encodeURIComponent(path)}`;
  const sp = new URLSearchParams(location.search);
  if (sp.get("lang")) return `${base}&lang=${encodeURIComponent(sp.get("lang"))}${sp.get("token")?`&token=${encodeURIComponent(sp.get("token"))}`:""}`;
  return `${base}${sp.get("token")?`&token=${encodeURIComponent(sp.get("token"))}`:""}`;
}

function langVariant(path, lang) {
  if (!/\.md$/i.test(path)) return [path];
  const dot=path.toLowerCase().lastIndexOf('.md');
  const base=path.slice(0,dot);
  return [`${base}.${lang}.md`, `${base}.md`];
}

async function openMarkdown(path) {
  if (mdPathEl) mdPathEl.textContent = "/" + path;
  const [first, fallback] = langVariant(path, LANG);
  const trials = [first, fallback];
  let txt=null, raw=null;
  for (const p of trials) {
    const url = rawUrl(encodeURIComponent(p).replace(/%2F/g,'/'));
    const r = await fetch(url, { headers: { ...authHeaders(token) }, cache: "no-store" });
    if (r.ok) { txt = await r.text(); raw = url; break; }
  }
  if (!txt) { readmeEl.textContent = "Datei nicht gefunden."; return; }
  // set 'open in new tab' link
  const base = `${location.pathname.replace(/[^/]+$/,'')}viewer.html?path=${encodeURIComponent(path)}`;
  const sp = new URLSearchParams(location.search);
  let href = base;
  if (sp.get("lang")) href += `&lang=${encodeURIComponent(sp.get("lang"))}`;
  if (sp.get("token")) href += `&token=${encodeURIComponent(sp.get("token"))}`;
  if (openInNewEl) openInNewEl.href = href;

  await renderMarkdownInto(readmeEl, txt);
}

// README loader (prefers README.<lang>.md)
async function loadReadme(manifest) {
  const prefix = cwd ? (cwd + "/") : "";
  const cand = [
    `${prefix}README.${LANG}.md`,
    `${prefix}Readme.${LANG}.md`,
    `${prefix}readme.${LANG}.md`,
    `${prefix}README.md`,
    `${prefix}Readme.md`,
    `${prefix}readme.md`,
    `${prefix}README`,
    `${prefix}README.txt`,
    `${prefix}readme.txt`
  ];
  let entry = null;
  for (const name of cand) {
    entry = manifest.find(x => x.path.toLowerCase() === name.toLowerCase());
    if (entry) break;
  }
  if (!entry) {
    readmeEl.textContent = "Kein README im aktuellen Ordner gefunden.";
    if (mdPathEl) mdPathEl.textContent = "";
    if (openInNewEl) openInNewEl.removeAttribute("href");
    return;
  }
  await openMarkdown(entry.path);
}

async function boot() {
  statsEl.textContent = "Lade Manifest…";

  let manifest = await fetchManifest();
  if (!manifest || !manifest.length) {
    statsEl.textContent = "Kein manifest.json, nutze GitHub API…";
    manifest = await fetchAllViaAPI(token);
  }

  if (!manifest || !manifest.length) {
    statsEl.textContent = "Keine Dateien gefunden.";
    return;
  }

  statsEl.textContent = "";
  try {
    const hdr = await getManifestHeaders();
    if (hdr?.lastModified) {
      document.getElementById("manifestUpdatedAt").textContent =
        new Date(hdr.lastModified).toLocaleString();
    } else {
      document.getElementById("manifestUpdatedAt").textContent = "–";
    }
  } catch {}

  const totalFiles = manifest.filter(x => x.type === "blob").length;
  document.getElementById("manifestTotalFiles").textContent = String(totalFiles);

  await loadReadme(manifest);
  renderFolder({ manifest, cwd, buildNavHref, listingEl, token, onOpenMd: openMarkdown });
  renderGallery({ manifest, cwd, token, galleryEl });
}

boot();
