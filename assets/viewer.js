// assets/viewer.js
import { OWNER, REPO, BRANCH } from "./config.js";
import { authHeaders, rawUrl, ghBlobUrl } from "./api.js";
import { renderMarkdownInto } from "./md.js";

const QS=new URLSearchParams(location.search);
const filePath=QS.get("path")||"";
const lang=(QS.get("lang")||"de").toLowerCase();
const token=QS.get("token")||"";

// UI
const pathEl = document.getElementById("path");
const mdEl = document.getElementById("md");
const openRaw = document.getElementById("openRaw");
const openGh  = document.getElementById("openGitHub");
pathEl.textContent = "/" + filePath;

function langVariant(path, lang) {
  if (!/\.md$/i.test(path)) return [path];
  const dot=path.toLowerCase().lastIndexOf('.md');
  const base=path.slice(0,dot);
  return [`${base}.${lang}.md`, `${base}.md`];
}

async function load() {
  const [first, fallback] = langVariant(filePath, lang);
  const trials=[first, fallback];
  let txt=null, raw=null, gh=null;
  for (const p of trials) {
    const url = rawUrl(encodeURIComponent(p).replace(/%2F/g,'/'));
    const r = await fetch(url, {headers:{...authHeaders(token)}, cache:"no-store"});
    if (r.ok) { txt = await r.text(); raw=url; gh = ghBlobUrl(encodeURIComponent(p).replace(/%2F/g,'/')); break; }
  }
  if (!txt) { mdEl.textContent = "Datei nicht gefunden."; return; }

  openRaw.href = raw + (token?`?token=${encodeURIComponent(token)}`:"");
  openGh.href  = gh;

  await renderMarkdownInto(mdEl, txt);
}
load();
