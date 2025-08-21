import fs from "fs";
import path from "path";

const SOURCE_LANG = (process.env.SOURCE_LANG || "DE").toUpperCase();
const DEEPL_API_KEY = process.env.DEEPL_API_KEY || "";
let   DEEPL_API_URL = process.env.DEEPL_API_URL || "https://api-free.deepl.com/v2/translate";
const CHANGED_LIST = (process.env.CHANGED || "").split(/\r?\n/).map(s=>s.trim()).filter(Boolean);

const SKIP_DIRS = new Set([".git","node_modules",".github"]);
const isMd = f => f.toLowerCase().endsWith(".md");
function isReadme(f){
  const name = f.toLowerCase().split(/[\/\\]/).pop() || "";
  return name === "readme.md" || name === "readme.de.md" || name === "readme.en.md" || /^readme\.[a-z]{2}\.md$/.test(name);
}
function listAllMd(dir="."){
  const out=[]; for (const e of fs.readdirSync(dir,{withFileTypes:true})) {
    if (SKIP_DIRS.has(e.name)) continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...listAllMd(p));
    else if (isMd(p)) out.push(p);
  } return out;
}
function shouldTranslateFile(f){
  if (!isMd(f)) return false;
  if (/\.en\.md$/i.test(f)) return false;
  const parts = f.split(/[\/\\]/);
  if (parts.some(p => SKIP_DIRS.has(p))) return false;
  return true;
}
function dstFor(file) {
  if (/\.de\.md$/i.test(file)) return file.replace(/\.de\.md$/i, ".en.md");
  return file.replace(/\.md$/i, ".en.md");
}

// === Protection & placeholders ===
function protectSegments(src){
  const map=[]; let i=0;
  const take = (content)=>{ const id=`§§X${i++}§§`; map.push({id,content}); return id; };
  let out=src;

  // (H) Keep ATX heading prefixes (# ...), title remains translatable
  out = out.replace(/^(#{1,6})[ \t]+(.*)$/gm, (m, hashes, title) => {
    const id = take(hashes + " ");
    return `${id}${title}`;
  });
  // (H2) Keep Setext underline lines
  out = out.replace(/^(=+|-+)[ \t]*$/gm, m => take(m));

  // (B) Preserve hard line breaks "  \n"
  out = out.replace(/  \n/g, m => take(m));

  // (T) HTML tags
  out = out.replace(/<[^>]+>/g, m=>take(m));

  // (C) Code fences and inline code
  out = out.replace(/```[\s\S]*?```/g, m=>take(m));
  out = out.replace(/~~~[\s\S]*?~~~/g, m=>take(m));
  out = out.replace(/`[^`\n]+`/g, m=>take(m));

  // (M) Math
  out = out.replace(/\$\$[\s\S]*?\$\$/g, m=>take(m));
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, m=>take(m));
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, m=>take(m));

  // (L) Link targets keep URL intact
  out = out.replace(/\[([^\]]*)\]\(([^)]+)\)/g, (_,t,u)=>{
    const id=take("("+u+")");
    return `[${t}]${id}`;
  });

  // (E) Preserve emphasis markers while translating inner text
  // bold **...** and __...__
  out = out.replace(/\*\*([^\n*]+)\*\*/g, (_,inner)=> take("**") + inner + take("**"));
  out = out.replace(/__([^\n_]+)__/g,         (_,inner)=> take("__") + inner + take("__"));
  // italics *...* and _..._  (conservative: no nested/newlines)
  out = out.replace(/\*([^\n*]+)\*/g,         (_,inner)=> take("*")  + inner + take("*"));
  out = out.replace(/_([^\n_]+)_/g,           (_,inner)=> take("_")  + inner + take("_"));

  return {text: out, map};
}
function restoreSegments(txt,map){
  let out=txt;
  for (const {id,content} of map) out=out.split(id).join(content);
  return out;
}

// === Lossless chunking: keep exact newlines ===
function chunkSmart(s, max=3500){
  const chunks=[]; let i=0;
  while (i < s.length) {
    let end = Math.min(i+max, s.length);
    // prefer breaking at double newline within [i, end]
    let j = s.lastIndexOf("\n\n", end);
    if (j >= i && j !== -1 && (end-i) > 500) { end = j+2; }
    else {
      // else prefer single newline
      j = s.lastIndexOf("\n", end);
      if (j >= i && j !== -1 && (end-i) > 500) end = j+1;
    }
    if (end <= i) end = Math.min(i+max, s.length);
    chunks.push(s.slice(i, end));
    i = end;
  }
  return chunks.length ? chunks : [s];
}

// === DeepL helpers ===
const headers = () => ({
  "Authorization": `DeepL-Auth-Key ${DEEPL_API_KEY}`,
  "Content-Type": "application/json"
});
function baseUrlOf(endpoint){
  const m = endpoint.match(/^(https?:\/\/[^/]+)(?:\/.*)?$/i);
  return m ? m[1] : "https://api-free.deepl.com";
}
async function usage(endpoint){
  const url = baseUrlOf(endpoint) + "/v2/usage";
  const r = await fetch(url, { headers: headers(), method: "GET" });
  if (!r.ok) return { ok:false, status:r.status };
  const j = await r.json().catch(()=>({}));
  return { ok:true, status:r.status, count: j.character_count ?? 0, limit: j.character_limit ?? 0 };
}
async function deeplJSON(texts, targetLang, endpoint){
  const body = JSON.stringify({
    text: texts,
    source_lang: SOURCE_LANG,
    target_lang: targetLang.toUpperCase(),
    preserve_formatting: true
  });
  const r = await fetch(endpoint, { method: "POST", headers: headers(), body });
  if (!r.ok) { throw new Error(`DeepL HTTP ${r.status}: ${await r.text().catch(()=>r.statusText)}`); }
  const j = await r.json();
  return j.translations?.map(x=>x.text) || [];
}

(async()=>{
  const candidates = (CHANGED_LIST.length? CHANGED_LIST : listAllMd(".")).filter(shouldTranslateFile);
  if (!candidates.length) { console.log("[i18n] No Markdown files to translate."); return; }
  if (!DEEPL_API_KEY) { console.error("[i18n] DEEPL_API_KEY is missing."); process.exit(1); }

  const isFreeKey = DEEPL_API_KEY.endsWith(":fx");
  if (isFreeKey && /api\.deepl\.com/.test(DEEPL_API_URL)) DEEPL_API_URL = "https://api-free.deepl.com/v2/translate";
  if (!isFreeKey && /api-free\.deepl\.com/.test(DEEPL_API_URL)) DEEPL_API_URL = "https://api.deepl.com/v2/translate";

  // Build plan (README first)
  const plan = [];
  for (const f of candidates) {
    const raw = fs.readFileSync(f, "utf8");
    const { text: protectedText, map: segMap } = protectSegments(raw);
    const chunks = chunkSmart(protectedText, 3500);
    const charCost = chunks.reduce((a,t)=>a+(t?.length||0), 0);
    plan.push({ file:f, chunks, segMap, charCost, isReadme: isReadme(f), depth: f.split(/[\/\\]/).length });
  }
  plan.sort((a,b)=> (a.isReadme!==b.isReadme) ? (a.isReadme?-1:1) : (a.isReadme && b.isReadme ? a.depth-b.depth : a.file.localeCompare(b.file)));

  let u = await usage(DEEPL_API_URL);
  if (!u.ok && (u.status===401 || u.status===403)) {
    const alt = /api-free\.deepl\.com/.test(DEEPL_API_URL) ? "https://api.deepl.com/v2/translate" : "https://api-free.deepl.com/v2/translate";
    console.warn(`[i18n] Preflight ${u.status} on ${baseUrlOf(DEEPL_API_URL)} → trying ${baseUrlOf(alt)}.`);
    DEEPL_API_URL = alt;
    u = await usage(DEEPL_API_URL);
  }
  if (!u.ok) { console.error(`[i18n] Usage endpoint failed (HTTP ${u.status}).`); process.exit(1); }

  let left = Math.max(0, (u.limit||500000) - (u.count||0));
  for (const item of plan) {
    if (item.charCost > left) { console.warn(`[i18n] Skipping ${item.file} (${item.charCost} chars) → not enough quota (${left} left).`); continue; }
    const outChunks = [];
    for (const chunk of item.chunks) outChunks.push((await deeplJSON([chunk], "en", DEEPL_API_URL))[0] || chunk);
    let out = outChunks.join(""); // lossless rejoin (exact original newlines preserved in chunks)
    out = restoreSegments(out, item.segMap);

    const dstPath = (/\.de\.md$/i.test(item.file)) ? item.file.replace(/\.de\.md$/i, `.en.md`) : item.file.replace(/\.md$/i, `.en.md`);
    fs.mkdirSync(path.dirname(dstPath), {recursive:true});
    fs.writeFileSync(dstPath, out, "utf8");
    console.log("→", dstPath);
    left -= item.charCost;
  }
})();
