// assets/i18n.js
export const SUPPORTED_LANGS = ["de","en"];

export function getLang() {
  const qs = (new URLSearchParams(location.search)).get("lang");
  const stored = localStorage.getItem("lang");
  const nav = (navigator.language || "de").slice(0,2).toLowerCase();
  const c = (qs || stored || nav);
  return SUPPORTED_LANGS.includes(c) ? c : "de";
}

export function initLangSelect(selectEl, initialLang, onChange) {
  selectEl.value = initialLang;
  selectEl.addEventListener("change", (e) => {
    const LANG = e.target.value;
    localStorage.setItem("lang", LANG);
    const qs = new URLSearchParams(location.search);
    qs.set("lang", LANG);
    const curPath = qs.get("path");
    if (curPath) qs.set("path", curPath);
    const token = qs.get("token") || "";
    if (token) qs.set("token", token);
    onChange?.(LANG, qs);
    location.search = qs.toString();
  });
}
