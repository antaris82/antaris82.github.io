// assets/render.js
import { rawUrl } from "./api.js";

/** Utility: human-readable bytes */
export function humanBytes(b) {
  const n = Number(b||0);
  if (!isFinite(n) || n<=0) return "—";
  const u = ["B","KB","MB","GB","TB"]; let i=0; let v=n;
  while (v>=1024 && i<u.length-1) { v/=1024; i++; }
  return (i? v.toFixed(1): String(v)) + " " + u[i];
}

const mediaRE = /\.(png|jpg|jpeg|gif|webp|svg|mp4|webm|ogg)$/i;
const backgroundRE = /^background\.(png|jpg|jpeg|gif|webp|svg)$/i;

/** Render the gallery for the current folder */
export function renderGallery({ manifest, cwd, token, galleryEl }) {
  galleryEl.innerHTML = "";
  const prefix = cwd ? (cwd + "/") : "";
  const direct = manifest
    .filter(x => x.path.startsWith(prefix))
    .map(x => ({ ...x, rest: x.path.slice(prefix.length) }))
    .filter(x => x.rest.length > 0 && !x.rest.includes("/"));

  const media = direct.filter(x => mediaRE.test(x.rest) && !backgroundRE.test(x.rest));
  for (const m of media) {
    const raw = rawUrl(m.path);
    const div = document.createElement("div");
    div.className = "thumb";
    if (/\.(mp4|webm|ogg)$/i.test(m.rest)) {
      const v = document.createElement("video");
      v.src = raw + (token?`?token=${encodeURIComponent(token)}`:"");
      v.controls = true;
      div.appendChild(v);
    } else {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = raw + (token?`?token=${encodeURIComponent(token)}`:"");
      img.alt = m.rest;
      div.appendChild(img);
    }
    galleryEl.appendChild(div);
  }

  const bg = direct.find(x => backgroundRE.test(x.rest));
  if (bg) {
    const bgUrl = rawUrl(bg.path) + (token?`?token=${encodeURIComponent(token)}`:"");
    document.body.style.backgroundImage = `url('${bgUrl}')`;
    document.body.classList.add("bg-on");
  } else {
    document.body.style.backgroundImage = "";
    document.body.classList.remove("bg-on");
  }
}

/** Render the folder listing */
export function renderFolder({ manifest, cwd, buildNavHref, listingEl }) {
  const prefix = cwd ? (cwd + "/") : "";
  const direct = manifest
    .filter(x => x.path.startsWith(prefix))
    .map(x => ({ ...x, rest: x.path.slice(prefix.length) }))
    .filter(x => x.rest.length > 0 && !x.rest.includes("/"));

  const dirsSet = new Set(
    manifest
      .filter(x => x.path.startsWith(prefix))
      .map(x => x.path.slice(prefix.length))
      .filter(x => x.includes("/"))
      .map(x => x.split("/")[0])
  );
  const dirs = Array.from(dirsSet).sort();

  listingEl.innerHTML = `
    <div class="head">Name</div>
    <div class="head">Typ</div>
    <div class="head">Größe</div>
  `;

  if (cwd) {
    const up = cwd.split("/").slice(0, -1).join("/");
    const row = document.createElement("div");
    row.className = "mono";
    row.style.gridColumn = "1 / span 3";
    row.innerHTML = `<a href="${buildNavHref(up)}">⬅︎ .. (nach oben)</a>`;
    listingEl.appendChild(row);
  }

  for (const d of dirs) {
    const rowName = document.createElement("div");
    rowName.className = "mono";
    rowName.innerHTML = `<a href="${buildNavHref(cwd ? (cwd + '/' + d) : d)}">📁 ${d}</a>`;
    const rowType = document.createElement("div");
    rowType.textContent = "dir";
    rowType.className = "muted mono";
    const rowSize = document.createElement("div");
    rowSize.textContent = "—";
    rowSize.className = "muted mono";
    listingEl.appendChild(rowName);
    listingEl.appendChild(rowType);
    listingEl.appendChild(rowSize);
  }

  for (const f of direct.filter(x => x.type === "blob")) {
    const rowName = document.createElement("div");
    rowName.className = "mono";
    const isMd = /\.md$/i.test(f.rest);
    let href;
    if (isMd) {
      const viewer = `${location.pathname.replace(/[^/]+$/,'')}viewer.html?path=${encodeURIComponent(cwd ? (cwd + "/" + f.rest) : f.rest)}`;
      href = viewer;
    } else {
      href = rawUrl(f.path);
    }
    // keep existing token/lang via search params
    const sp = new URLSearchParams(location.search);
    if (isMd) {
      // viewer handles token/lang; append both
      if (sp.get("lang")) href += `&lang=${encodeURIComponent(sp.get("lang"))}`;
      if (sp.get("token")) href += `&token=${encodeURIComponent(sp.get("token"))}`;
    } else {
      if (sp.get("token")) href += `?token=${encodeURIComponent(sp.get("token"))}`;
    }

    rowName.innerHTML = `${isMd ? "📰" : "📄"} <a href="${href}" target="_blank" rel="noopener">${f.rest}</a>`;

    const rowType = document.createElement("div");
    rowType.textContent = isMd ? "md (Viewer)" : "file";
    rowType.className = "muted mono";

    const rowSize = document.createElement("div");
    rowSize.textContent = humanBytes(f.size);
    rowSize.className = "muted mono";

    listingEl.appendChild(rowName);
    listingEl.appendChild(rowType);
    listingEl.appendChild(rowSize);
  }
}
