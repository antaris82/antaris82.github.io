(() => {
  const STORE_KEY = "app.lang";
  const SUPPORTED_LANGS = ["de","en"];
  const I18n = {
    lang: null,
    dict: {},
    resolveLang() {
      const url = new URL(window.location.href);
      const queryLang = url.searchParams.get("lang");
      if (queryLang && SUPPORTED_LANGS.includes(queryLang)) return queryLang;
      const saved = localStorage.getItem(STORE_KEY);
      if (saved && SUPPORTED_LANGS.includes(saved)) return saved;
      const nav = (navigator.language || "de").slice(0,2).toLowerCase();
      return SUPPORTED_LANGS.includes(nav) ? nav : "de";
    },
    async load(lang) {
      const fallback = "de";
      try {
        const res = await fetch(`assets/locales/${lang}.json`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this.dict = await res.json();
        this.lang = lang;
      } catch (e) {
        if (lang !== fallback) return this.load(fallback);
      }
    },
    t(key, fallback="") {
      return key.split(".").reduce((acc,k)=>acc && acc[k], this.dict) || fallback || key;
    },
    apply() {
      document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        const text = this.t(key, el.textContent.trim());
        el.textContent = text;
      });
      document.querySelectorAll("[data-i18n-attr]").forEach(el => {
        const attrs = el.getAttribute("data-i18n-attr").split(",").map(s=>s.trim()).filter(Boolean);
        const baseKey = el.getAttribute("data-i18n") || "";
        for (const a of attrs) {
          const k = baseKey ? `${baseKey}.${a}` : el.getAttribute(`data-i18n-${a}`);
          if (!k) continue;
          const val = this.t(k, el.getAttribute(a) || "");
          el.setAttribute(a, val);
        }
      });
      document.documentElement.setAttribute("lang", this.lang || "de");
      const sel = document.getElementById("langSelect");
      if (sel && sel.value !== this.lang) sel.value = this.lang;
    },
    async set(lang) {
      if (!SUPPORTED_LANGS.includes(lang)) return;
      await this.load(lang);
      localStorage.setItem(STORE_KEY, lang);
      this.apply();
      const url = new URL(window.location.href);
      url.searchParams.set("lang", lang);
      window.history.replaceState({}, "", url);
    },
    async init(opts={}) {
      const defaultLang = (opts.defaultLang || "de");
      const lang = this.resolveLang() || defaultLang;
      await this.set(lang);
      const sel = document.getElementById("langSelect");
      if (sel) sel.addEventListener("change", e => this.set(e.target.value));
    }
  };
  window.I18n = I18n;
})();
