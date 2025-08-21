// assets/app.js
import { OWNER, REPO, BRANCH } from "./config.js";
import { authHeaders, fetchManifest, fetchAllViaAPI, getManifestHeaders } from "./api.js";
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

// README loader (with language variants)
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
  let readmeEntry = null;
  for (const name of cand) {
    readmeEntry = manifest.find(x => x.path.toLowerCase() === name.toLowerCase());
    if (readmeEntry) break;
  }
  if (!readmeEntry) {
    readmeEl.textContent = "Kein README im aktuellen Ordner gefunden.";
    return;
  }
  try {
    const raw = `https://raw.githubusercontent.com/${OWNER}/${REPO}/${encodeURIComponent(BRANCH)}/${readmeEntry.path}`;
    const r = await fetch(raw, { headers: { ...authHeaders(token) }, cache: "no-store" });
    if (!r.ok) throw new Error("READMERaw error " + r.status);
    const txt = await r.text();
    await renderMarkdownInto(readmeEl, txt);
  } catch (e) {
    readmeEl.textContent = "README konnte nicht geladen werden.";
    console.error(e);
  }
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
  renderFolder({ manifest, cwd, buildNavHref, listingEl });
  renderGallery({ manifest, cwd, token, galleryEl });
}

boot();
