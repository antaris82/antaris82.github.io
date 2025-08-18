# -*- coding: utf-8 -*-
"""
B1_make_two_tables.py
---------------------
Erzeugt Appendix-Tabelle B (Urgraph vs. Approximant-Subgraph) aus dem ST-Graph
und baut ein Beweisdokument mit zwei Appendices:
  - Appendix A: (wird als vorhandene Datei B1_ST_appendix_table.tex eingebunden)
  - Appendix B: B1_ST_urgraph_vs_subgraph_table.tex (wird hier erzeugt)

Voraussetzung:
- Eine Datei "st_2_partial trace on ST-Graph(1).py" mit
  * V0-Definition
  * build_graph_by_addresses(level:int)

Ausgaben:
- /mnt/data/B1_ST_urgraph_vs_subgraph_table.tex
- /mnt/data/B1_formal_ST_with_two_tables.tex
"""

from pathlib import Path
import numpy as np
import re, itertools, math
from textwrap import dedent

# ---------- Hilfsfunktionen ----------

def extract_V0_and_builder(src_path: Path):
    """Extrahiert V0 und die Funktion build_graph_by_addresses(level:int) aus der Nutzerdatei,
    ohne die übrigen Top-Level-Teile (Plots/Animationen) auszuführen.
    """
    src = src_path.read_text(encoding="utf-8", errors="ignore")
    mV0 = re.search(r"V0\s*=\s*np\.array\(\[[\s\S]+?\]\s*,\s*dtype=float\)\s*", src)
    if not mV0:
        raise RuntimeError("V0-Definition nicht gefunden.")
    start = re.search(r"^def\s+build_graph_by_addresses\s*\(level\s*:\s*int\)\s*:", src, flags=re.M)
    if not start:
        raise RuntimeError("Funktion build_graph_by_addresses(level:int) nicht gefunden.")
    rest = src[start.start():]
    m2 = re.search(r"\ndef\s+[A-Za-z_]\w*\s*\(", rest)
    func_block = rest if not m2 else rest[:m2.start()+1]
    ns = {"np": np, "itertools": itertools, "math": math}
    exec(mV0.group(0), ns, ns)
    exec(func_block, ns, ns)
    return ns["V0"], ns["build_graph_by_addresses"]

def build_aggregation(Vfine: np.ndarray, Vcoarse: np.ndarray):
    """Bildet Aggregationsmatrix C (4 x n) durch nearest-corner; R=C^T.
       Zeilennormierung bewirkt Gleichgewichtung innerhalb eines Clusters.
    """
    n = Vfine.shape[0]
    c = Vcoarse.shape[0]
    assign = np.argmin(((Vfine[:,None,:]-Vcoarse[None,:,:])**2).sum(axis=2), axis=1)
    C = np.zeros((c, n), dtype=float)
    for k in range(c):
        idxs = np.where(assign==k)[0]
        if len(idxs)==0: continue
        C[k, idxs] = 1.0/len(idxs)
    R = C.T
    return C, R

def observables_from_L(L: np.ndarray, beta: float):
    """E, S, P aus Gibbs(β) für Laplacian L via Spektraldarstellung."""
    evals, _ = np.linalg.eigh(L)
    w = np.exp(-beta*evals)
    Z = float(np.sum(w))
    p = w / Z
    E = float((p*evals).sum())
    S = float(-(p*np.log(p)).sum())
    P = float((p*p).sum())
    return E, S, P

# ---------- Hauptablauf ----------

def main():
    SRC = Path("/mnt/data/st_2_partial trace on ST-Graph(1).py")
    if not SRC.exists():
        raise FileNotFoundError(f"Benötigte Datei fehlt: {SRC}")
    V0, build_graph_by_addresses = extract_V0_and_builder(SRC)

    # ST-Urgraph Level 4 und Level 0
    V4, A4, L4 = build_graph_by_addresses(4)
    V0_points, A0, L0 = build_graph_by_addresses(0)

    # Aggregation/Lift
    C, R = build_aggregation(V4, V0_points)
    L_lift = R @ L0 @ C
    L_lift = 0.5*(L_lift + L_lift.T)  # numerisch symmetrieren

    # Subgraph wie im PoC
    rng = np.random.default_rng(42)
    N_SUB = 160
    n = L4.shape[0]
    idx_sub = np.sort(rng.choice(n, size=min(N_SUB, n), replace=False))
    L4_sub    = L4[np.ix_(idx_sub, idx_sub)].copy()
    Llift_sub = L_lift[np.ix_(idx_sub, idx_sub)].copy()

    beta = 3.0
    alphas = [0.00, 0.25, 0.50, 0.75, 1.00]

    # Urgraph (voll)
    E_U, S_U, P_U = observables_from_L(L4, beta)

    # Approximant-Subgraph je α
    rows = []
    for a in alphas:
        L_A_sub = (1.0-a)*L4_sub + a*Llift_sub
        E,S,P = observables_from_L(L_A_sub, beta)
        rows.append((a, E, S, P))

    # Tabelle B (Urgraph vs Subgraph)
    tex_lines = [r"\begin{table}[h]",
                 r"\centering",
                 r"\caption{Urgraph (voll) vs.\ Approximant-Subgraph (Level 4, Subgraph $N_{\rm sub}="+str(len(idx_sub))+r"$) bei $\beta=3$.}",
                 r"\begin{tabular}{l r r r}",
                 r"\toprule",
                 r"Modell & $E$ & $S$ & $P$ \\",
                 r"\midrule",
                 rf"Urgraph (n={n}) & {E_U:.6f} & {S_U:.6f} & {P_U:.6f} \\",
                 r"\midrule",
                 r"\multicolumn{4}{l}{\emph{Approximant-Subgraph} $L_A^{\rm sub}(\alpha)=(1-\alpha)L^{\rm sub}+\alpha L_{\rm lift}^{\rm sub}$} \\"]
    for a, E, S, P in rows:
        tex_lines.append(rf"$\alpha={a:.2f}$ & {E:.6f} & {S:.6f} & {P:.6f} \\")
    tex_lines += [r"\bottomrule",
                  r"\end{tabular}",
                  r"\vspace{0.25em}\small Hinweis: Werte sind \emph{nicht skaleninvariant} in $n$; der Subgraph hat geringere Dimension.",
                  r"\end{table}"]

    out_tbl  = Path("/mnt/data/B1_ST_urgraph_vs_subgraph_table.tex")
    out_tbl.write_text("\n".join(tex_lines), encoding="utf-8")

    # Beweis-Dokument mit zwei Appendices zusammenbauen
    base_path = Path("/mnt/data/B1_formal_extended.tex")
    if not base_path.exists():
        raise FileNotFoundError(f"Basis-Beweisdatei nicht gefunden: {base_path}")

    base = base_path.read_text(encoding="utf-8")
    combined = base.replace(r"\end{document}", r"""
\clearpage
\appendix
\section*{Appendix A: Numerische Konsistenzchecks (ST-Graph, Level 4)}
\input{B1_ST_appendix_table.tex}

\clearpage
\section*{Appendix B: Urgraph (voll) vs.\ Approximant-Subgraph}
\input{B1_ST_urgraph_vs_subgraph_table.tex}
\end{document}
""")
    out_doc = Path("/mnt/data/B1_formal_ST_with_two_tables.tex")
    out_doc.write_text(combined, encoding="utf-8")

    return str(out_tbl), str(out_doc)

if __name__ == "__main__":
    p_tbl, p_doc = main()
    print("Wrote:", p_tbl)
    print("Wrote:", p_doc)
