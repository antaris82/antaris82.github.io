test16 German
# A1 - Exclusive Metrics & Formal Derivation\n\n> **Path:** `Sierpinski Tetraeder_PoC/A1/` - **Owner:** antaris82
**Short description:** This folder contains the **formal derivation** of the metrics used in A1 (entropy, mutual information, intersection metrics) as well as the associated assets in the subfolder `files/`.
> **Math hint:** Inline \( … \), Display \[ … \].\n\n---\n\n## 🔗 Quick access
- 📄 **Formal derivation:** `A1_ST_exclusive_metrics_formal.pdf`
- 🗂 **Assets & data:** `./files/` - see [README in the files folder](./files/README.md)\n\n---\n\n## 1) Aim & context
A1 establishes the **exclusive observables and intersection metrics** on the ST-Ur graph.  
- Formal part: proofs and definitions in PDF (graph construction, correlation matrix, entropy and MI equations, exclusive rule, tables/plots).  
- Operational part: Assets in the subfolder `files/` (CSV, PNG, GIF, Python scripts).\n\n## 2) Axioms & core results
**Axioms (free-fermionic model).**
- ST graph (level L=4) via IFS construction.  
- Hamiltonian \(H=L\) (Graph-Laplacian).  
- Filling \(\nu=1/2\), Gaussian ground state, correlation matrix \(C=U_{occ}U_{occ}^\top\).  \n\n**Core results (from PDF)**
\[ S(\rho_A) = -\, \mathrm{tr}[C_A\log C_A + (I-C_A)\log(I-C_A)] \]
\[ MI(A:B) = S(A)+S(B)-S(A\cup B) \]
Exclusive rule: Prefix assignment (RED, YELLOW, GREEN), disjoint and total.  \n\n**Measured values (L=4)**
- Regional entropies: S(GREEN)=1.1933, S(YELLOW)=30.2059, S(RED)=1.8920.
- Pair metrics: MI(RED:YELLOW)=0.1649, cut=3, d_min=1.
- Layer scaling: S(4)=198.43, MI(4:Rest)=396.86.\n\n## 3) Methods / Formalism
- **Graph construction:** Prefix-IFS, edges as union of local tetrahedra.  
- **Correlation matrix:** from Hamilton eigenvectors.  
- **Exclusive rule:** Priority RED > YELLOW > GREEN.  
- **Metrics:** cut(A,B), \langle|C|\rangle_{cross}, d_min, S(A), MI(A:Rest), MI(A:B).\n\n## 4) File & folder overview
| path | type | short description |
|---|---|---|
| `./A1_ST_exclusive_metrics_formal.pdf` | PDF | Formal derivation and evaluation of entanglement and intersection metrics on the ST graph (L=4); contains definitions, propositions, tables, figures. |
| `./files/` | Folder | Contains assets (CSV, PNG, GIF, ST.py). See own README (A1/files). |\n\n---\n\n## 5) Acceptance criteria
- **K1:** PDF documents definitions, propositions and numerical values consistently.  
- **K2:** Subfolder `files/` contains assets (CSV, PNG, GIF) and is consistent with PDF.  
- **K3:** Exclusive assignment (RED/YELLOW/GREEN) disjoint and total (Lemma 3.2).  
- **K4:** Reproducibility: Numbers in PDF match CSVs in `files/` folder.\n\n## 6) Reproducibility
1. **Theory:** Consult PDF (derivations and tables).  
2. **Assets:** Use subfolder `files/` (CSV/PNG/GIF, see own README).  
3. **Pipeline:** Execute `ST.py`, compare results with PDF tables.  
4. **Validation:** Check that MI(A:A)=2S(A) applies; layer scaling consistent with CSV.\n\n## 7) Topic-related information
- A1 provides the foundation for subsequent folders (A2, A3, ...).  
- Link to literature: Peschel (2003), Eisert/Cramer/Plenio (2010).  \n\n## 8) Subfolder (structure)
```
A1/
├─ A1_ST_exclusive_metrics_formal.pdf
└─ files/
```§\n\n## 9) General notes
- Math delimiters: Inline \( … \), Display \[ … \].
- Subfolders bring their own README (here: files).  
- Keep paths relative.\n\n## 10) Open points / To-Do
- [ ] Automate consistency check between PDF and all CSVs.  
- [ ] Test layer scaling for higher L > 4.  
- [ ] Integration in ST-Graph PoC (A2+).\n\n## 11) Validation status
| Criterion | Status | Comment |
|---|---|---|
| K1 | 🟢 | PDF checked |
| K2 | 🟢 | files/ available, README consistent |
| K3 | 🟢 | Exclusive rule formally proven |
| K4 | 🟡 | CSV/PDF comparison still manual |\n\n## 12) References
- J. Eisert, M. Cramer, M. B. Plenio: *Colloquium: Area laws for the entanglement entropy*. Rev. Mod. Phys. 82 (2010) 277-306.
- I. Peschel: *Calculation of reduced density matrices from correlation functions*. J. Phys. A 36 (2003) L205.  
- I. Peschel, V. Eisler: *Reduced density matrices and entanglement entropy in free lattice models*. J. Phys. A 42 (2009) 504003.\n\n## 13) License
License\n\n Code (esp. in ./files/): MIT License.
    Non-code (e.g. PDFs, CSV/PNG): Creative Commons Attribution 4.0 International (CC BY 4.0).\n\n © 2025 antaris - Code: MIT; data/images/texts (incl. PDFs): CC BY 4.0.\n\n## 14) Changelog
- **v1.0 (2025-08-19):** First edition for `A1/` (Formal derivation + assets reference).
