// assets/api.js
import { OWNER, REPO, BRANCH } from "./config.js";

/** Build auth headers from optional token */
export function authHeaders(token) {
  return token ? { "Authorization": "Bearer " + token } : {};
}

/** Try to fetch manifest.json from site root */
export async function fetchManifest() {
  try {
    const res = await fetch("/manifest.json", { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Fallback: GitHub Trees API (recursive tree) */
export async function fetchAllViaAPI(token) {
  try {
    const ref = encodeURIComponent(BRANCH);
    const api = `https://api.github.com/repos/${OWNER}/${REPO}/git/trees/${ref}?recursive=1`;
    const r = await fetch(api, { headers: { ...authHeaders(token) }, cache: "no-store" });
    if (!r.ok) throw new Error("Trees API " + r.status);
    const j = await r.json();
    const out = (j.tree || []).map(it => {
      if (it.type === "blob") return { path: it.path, size: it.size ?? 0, sha: it.sha, type: "blob", mode: it.mode || "100644" };
      if (it.type === "tree") return { path: it.path, size: 0, sha: it.sha, type: "tree", mode: it.mode || "040000" };
      return null;
    }).filter(Boolean);
    return out.sort((a,b)=> a.path.localeCompare(b.path));
  } catch (e) {
    console.warn(e);
    return [];
  }
}

/** Get headers for manifest.json (Last-Modified etc.) */
export async function getManifestHeaders() {
  try {
    const res = await fetch("/manifest.json", { method:"GET", cache:"no-store" });
    if (!res.ok) return null;
    return {
      lastModified: res.headers.get("last-modified") || null,
      contentLength: res.headers.get("content-length") || null
    };
  } catch { return null; }
}

/** Build raw URL for a repo path */
export function rawUrl(path) {
  return `https://raw.githubusercontent.com/${OWNER}/${REPO}/${encodeURIComponent(BRANCH)}/${path}`;
}

/** Build GitHub blob URL for a repo path */
export function ghBlobUrl(path) {
  return `https://github.com/${OWNER}/${REPO}/blob/${encodeURIComponent(BRANCH)}/${path}`;
}
