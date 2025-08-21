// assets/md.js
// Markdown + KaTeX helpers

export function extractMathSlots(src) {
  const slots = [];
  let out = src;
  out = out.replace(/```(math|latex|tex)\s*([\s\S]*?)```/gi, (_, __, body) => {
    const id = `§§MATHFENCE_${slots.length}§§`;
    slots.push({ id, tex: body.trim(), kind: 'fence' });
    return id;
  });
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => {
    const id = `§§MATHDSP_${slots.length}§§`;
    slots.push({ id, tex: body.trim(), kind: 'display' });
    return id;
  });
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, body) => {
    const id = `§§MATHINL_${slots.length}§§`;
    slots.push({ id, tex: body.trim(), kind: 'inline' });
    return id;
  });
  out = out.replace(/\$\$([\s\S]*?)\$\$/g, (_, body) => {
    const id = `§§MATHDOLL_${slots.length}§§`;
    slots.push({ id, tex: body.trim(), kind: 'display' });
    return id;
  });
  return { text: out, slots };
}

export function restoreMathSlots(html, slots) {
  for (const s of slots) {
    let repl;
    if (s.kind === 'fence' || s.kind === 'display') {
      repl = `<div class="math-fence">$$\n${s.tex}\n$$</div>`;
    } else {
      repl = `\\(${s.tex}\\)`;
    }
    html = html.split(s.id).join(repl);
  }
  return html;
}

export function renderKatex(el) {
  return new Promise((resolve) => {
    const opts = {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\[", right: "\\]", display: true},
        {left: "\\(", right: "\\)", display: false},
        {left: "$",  right: "$",  display: false}
      ],
      throwOnError: false,
      strict: "ignore"
    };
    let tries=0;
    (function tick(){
      if (window.renderMathInElement) {
        try { window.renderMathInElement(el, opts); } catch {}
        resolve();
      } else if (++tries>60) {
        resolve();
      } else {
        setTimeout(tick, 150);
      }
    })();
  });
}

export async function renderMarkdownInto(el, srcText) {
  const { text: placeholderMD, slots } = extractMathSlots(srcText);
  let html = window.marked.parse(placeholderMD);
  html = window.DOMPurify.sanitize(html);
  html = restoreMathSlots(html, slots);
  el.innerHTML = html;
  await renderKatex(el);
}
