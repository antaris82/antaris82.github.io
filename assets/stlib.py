# stlib.py
# Ein-Datei-Paket mit dynamischen Untermodule:
#   stlib.graph_st, stlib.graphs_regular, stlib.linops, stlib.heat,
#   stlib.quantum, stlib.gkls, stlib.dtn
#
# Ziel: kompatibel zum Demo-Skript; reine Python/NumPy/SciPy; Pyodide-tauglich.

from __future__ import annotations
import sys, types, math, itertools
from typing import List, Tuple, Dict, Iterable, Callable, Optional

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAVE_SCIPY = True
except Exception:
    sp = None  # type: ignore
    spla = None  # type: ignore
    HAVE_SCIPY = False


# --------------------------------------------------------------------------------------
# Hilfen
# --------------------------------------------------------------------------------------
def _to_csr(M) -> "sp.csr_matrix":
    if HAVE_SCIPY:
        if sp.issparse(M):
            return M.tocsr()
        return sp.csr_matrix(M)
    # Fallback: Minimaler CSR-Ersatz mit numpy (nur was wir brauchen)
    class _FakeCSR:
        def __init__(self, A):
            self.A = np.array(A)
            self.shape = self.A.shape
        def toarray(self): return self.A
        def diagonal(self): return np.diag(self.A)
        def dot(self, x): return self.A.dot(x)
    return _FakeCSR(np.array(M))


def _submatrix_csr(A: "sp.csr_matrix", rows: np.ndarray, cols: np.ndarray):
    if not HAVE_SCIPY:
        return _to_csr(A.toarray()[np.ix_(rows, cols)])
    A = A.tocsr()
    return A[rows][:, cols]


def _expm_multiply(A, v):
    """
    expm(A) @ v – bevorzugt sparse Variante, sonst Dense-Fallback.
    """
    if HAVE_SCIPY:
        try:
            return spla.expm_multiply(A, v)
        except Exception:
            pass
    # Fallback: dense expm
    try:
        from scipy.linalg import expm  # evtl. vorhanden
        return expm(A.toarray() if hasattr(A, "toarray") else A) @ v
    except Exception:
        # Notlösung: Serie (nur für sehr kleine Matrizen/Tests)
        X = np.eye(A.shape[0], dtype=complex) if np.iscomplexobj(A) else np.eye(A.shape[0])
        term = np.eye(A.shape[0], dtype=complex) if np.iscomplexobj(A) else np.eye(A.shape[0])
        for k in range(1, 24):
            term = term @ (A / k)
            X = X + term
        Av = X @ v
        return Av


def _expm_dense(A):
    if HAVE_SCIPY:
        try:
            from scipy.linalg import expm
            return expm(A.toarray() if hasattr(A, "toarray") else A)
        except Exception:
            pass
    # sehr einfache Serie (nur als Fallback)
    A = A.toarray() if hasattr(A, "toarray") else np.array(A)
    X = np.eye(A.shape[0], dtype=complex if np.iscomplexobj(A) else float)
    term = np.eye(A.shape[0], dtype=complex if np.iscomplexobj(A) else float)
    for k in range(1, 28):
        term = term @ (A / k)
        X = X + term
    return X


# ======================================================================================
# stlib.graph_st
# ======================================================================================
graph_st = types.ModuleType("stlib.graph_st")
def build_st_graph(level: int = 3) -> Tuple[List[str], List[Tuple[str, str]], List[str], Dict[str, Tuple[float,float,float]]]:
    """
    Erzeugt einen (vereinfachten) Sierpinski-Tetraeder-Graphen via IFS.
    - Knoten sind Adressen über {0,1,2,3} der Länge = level (Strings: '0123...').
    - Koordinaten entstehen rekursiv aus vier Eckpunkten (regulärer Tetraeder).
    - Kanten verbinden Punkte mit Abstand ~ Grundkantenlänge / 2**level (Toleranz 5%).
    Rückgabe: (nodes, edges, corners, coords)
    """
    assert level >= 1
    # regulärer Tetraeder (Seitenlänge ~ 1)
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.5, math.sqrt(3)/2, 0.0])
    v3 = np.array([0.5, math.sqrt(3)/6, math.sqrt(2/3)])
    V = [v0, v1, v2, v3]
    base_edge = np.linalg.norm(v0 - v1)

    # Alle Adressen der Länge 'level'
    addrs = [''.join(p) for p in itertools.product('0123', repeat=level)]

    def addr_to_coord(addr: str) -> np.ndarray:
        x = np.zeros(3)
        for k, ch in enumerate(addr, start=1):
            x = (x + V[int(ch)]) / 2.0
        return x

    coords = {a: addr_to_coord(a) for a in addrs}

    # Kanten: verbinde Nachbarn ~ Gitterabstand
    step = base_edge / (2 ** level)
    tol  = 0.055 * step
    nodes = addrs
    edges: List[Tuple[str, str]] = []
    arr = np.stack([coords[a] for a in nodes], axis=0)
    for i in range(len(nodes)):
        di = arr[i]
        # lokaler k-nächste Nachbarn-Check (begrenzen, um O(n^2) etwas zu reduzieren)
        # bei kleinen levels ok auch voll:
        for j in range(i+1, len(nodes)):
            dj = arr[j]
            d = np.linalg.norm(di - dj)
            if abs(d - step) <= tol or d < (step * 1.02):
                edges.append((nodes[i], nodes[j]))

    # Äußere Ecken ~ Adressen aus einer einzigen Ziffer
    corners = [str(k) * level for k in range(4)]
    return nodes, edges, corners, coords

def outer_corners(level: int = 3) -> List[str]:
    return [str(k) * level for k in range(4)]

graph_st.build_st_graph = build_st_graph
graph_st.outer_corners  = outer_corners


# ======================================================================================
# stlib.graphs_regular
# ======================================================================================
graphs_regular = types.ModuleType("stlib.graphs_regular")
def grid_graph_2d(nx: int, ny: int, periodic: bool = True) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
    2D-Gitter (nx × ny). Knoten = 0..N-1.
    """
    nodes = list(range(nx * ny))
    def idx(x,y): return x*ny + y
    edges = []
    for x in range(nx):
        for y in range(ny):
            u = idx(x,y)
            # rechts
            if x+1 < nx:
                edges.append((u, idx(x+1,y)))
            elif periodic:
                edges.append((u, idx(0,y)))
            # oben
            if y+1 < ny:
                edges.append((u, idx(x,y+1)))
            elif periodic:
                edges.append((u, idx(x,0)))
    return nodes, edges

def grid_graph_3d(nx: int, ny: int, nz: int, periodic: bool = True) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
    3D-Gitter (nx × ny × nz). Knoten = 0..N-1.
    """
    nodes = list(range(nx * ny * nz))
    def idx(x,y,z): return (x*ny + y)*nz + z
    edges = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                u = idx(x,y,z)
                # x-Richtung
                if x+1 < nx: edges.append((u, idx(x+1,y,z)))
                elif periodic: edges.append((u, idx(0,y,z)))
                # y-Richtung
                if y+1 < ny: edges.append((u, idx(x,y+1,z)))
                elif periodic: edges.append((u, idx(x,0,z)))
                # z-Richtung
                if z+1 < nz: edges.append((u, idx(x,y,z+1)))
                elif periodic: edges.append((u, idx(x,y,0)))
    return nodes, edges

graphs_regular.grid_graph_2d = grid_graph_2d
graphs_regular.grid_graph_3d = grid_graph_3d


# ======================================================================================
# stlib.linops
# ======================================================================================
linops = types.ModuleType("stlib.linops")
def laplacian_from_edges(nodes: List, edges: List[Tuple], weights: Optional[Iterable[float]] = None):
    """
    Ungewichteter/gewichteter Graph-Laplacian L = D - W.
    nodes: beliebige Hashables (z.B. Strings, ints)
    edges: Liste (u, v) mit u, v aus nodes
    weights: optional gleiche Länge wie edges
    Rückgabe: (L (csr), idx_map: node -> index)
    """
    n = len(nodes)
    idx = {u:i for i,u in enumerate(nodes)}
    if n == 0:
        return _to_csr(np.zeros((0,0))), idx

    rows, cols, data = [], [], []
    if weights is None:
        w_iter = itertools.repeat(1.0, len(edges))
    else:
        w_iter = weights

    deg = np.zeros(n, dtype=float)
    for (u,v), w in zip(edges, w_iter):
        if u not in idx or v not in idx: continue
        i, j = idx[u], idx[v]
        if i == j: 
            deg[i] += w
            continue
        rows += [i,j]; cols += [j,i]; data += [w,w]
        deg[i] += w; deg[j] += w

    if HAVE_SCIPY:
        W = sp.coo_matrix((data, (rows, cols)), shape=(n,n)).tocsr()
        D = sp.diags(deg)
        L = D - W
    else:
        W = np.zeros((n,n), float)
        for r,c,v in zip(rows,cols,data):
            W[r,c] += v
        L = np.diag(deg) - W
        L = _to_csr(L)

    return L, idx

def normalized_laplacian(L):
    """
    Normierter Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    """
    L = _to_csr(L)
    if HAVE_SCIPY:
        d = L.diagonal()
        with np.errstate(divide='ignore'):
            invsqrt = np.where(d > 0, 1.0/np.sqrt(d), 0.0)
        Dinv = sp.diags(invsqrt)
        return (Dinv @ L @ Dinv).tocsr()
    else:
        A = L.toarray()
        d = np.diag(A)
        invsqrt = np.where(d > 0, 1.0/np.sqrt(d), 0.0)
        Dinv = np.diag(invsqrt)
        return _to_csr(Dinv @ A @ Dinv)

def bfs_distances(nodes: List, edges: List[Tuple], seed_idx: int) -> np.ndarray:
    """
    BFS-Distanzen (ungewichtet) vom Seed-Index.
    """
    idx = {u:i for i,u in enumerate(nodes)}
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for (u,v) in edges:
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            if i != j:
                adj[i].append(j)
                adj[j].append(i)
    dist = np.full(n, -1, dtype=int)
    if not (0 <= seed_idx < n):
        return dist
    from collections import deque
    dq = deque([seed_idx])
    dist[seed_idx] = 0
    while dq:
        i = dq.popleft()
        for j in adj[i]:
            if dist[j] == -1:
                dist[j] = dist[i] + 1
                dq.append(j)
    return dist

linops.laplacian_from_edges = laplacian_from_edges
linops.normalized_laplacian  = normalized_laplacian
linops.bfs_distances         = bfs_distances


# ======================================================================================
# stlib.heat
# ======================================================================================
heat = types.ModuleType("stlib.heat")
def hutch_trace_expm(L, t: float, n_probes: int = 16, rng: Optional[np.random.Generator] = None):
    """
    Schätzt Tr(exp(-t L)) via Hutchinson.
    Rückgabe: (schätzer, standardfehler).
    """
    if rng is None:
        rng = np.random.default_rng()

    L = _to_csr(L)
    n = L.shape[0]
    if n == 0:
        return 0.0, 0.0

    # Hutchinson mit Rademacher (+/-1)
    ests = []
    for _ in range(n_probes):
        z = rng.integers(0, 2, size=n)*2 - 1
        y = _expm_multiply((-t)*L, z.astype(float))
        ests.append(float(z @ y))
    ests = np.array(ests, float)
    est = ests.mean()
    se  = ests.std(ddof=1) / math.sqrt(len(ests)) if len(ests) > 1 else 0.0
    return est, se

def spectral_dimension(ts: Iterable[float], pvals: Iterable[float]) -> np.ndarray:
    """
    d_s(t) = -2 d ln p / d ln t (diskrete Ableitung).
    """
    ts = np.asarray(list(ts), float)
    ps = np.asarray(list(pvals), float)
    ts = np.maximum(ts, 1e-15)
    ps = np.maximum(ps, 1e-300)
    g = np.gradient(np.log(ps), np.log(ts))
    return -2.0 * g

heat.hutch_trace_expm  = hutch_trace_expm
heat.spectral_dimension = spectral_dimension


# ======================================================================================
# stlib.quantum
# ======================================================================================
quantum = types.ModuleType("stlib.quantum")
def unitary_packet(H, psi0_idx: int, times: Iterable[float], normalize: bool = True) -> np.ndarray:
    """
    |psi(t)> = exp(-i H t) |psi0>
    Rückgabe: Array [len(times), N] der Wahrscheinlichkeiten |psi|^2.
    """
    H = _to_csr(H)
    N = H.shape[0]
    psi0 = np.zeros(N, dtype=complex); psi0[psi0_idx] = 1.0+0.0j
    probs = []
    for t in times:
        if HAVE_SCIPY:
            psi_t = spla.expm_multiply((-1j*t)*H, psi0)
        else:
            psi_t = _expm_multiply((-1j*t)*(H.toarray() if hasattr(H, "toarray") else H), psi0)
        p = np.abs(psi_t)**2
        if normalize:
            s = p.sum()
            if s > 0: p = p / s
        probs.append(p)
    return np.vstack(probs)

def lr_front_radius(prob_t: np.ndarray, dists: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """
    LR-Front-Radius als kleinstes r mit kumulativer Masse >= quantile.
    """
    dists = np.asarray(dists, int)
    rmax = int(dists[dists>=0].max()) if np.any(dists>=0) else 0
    out = []
    for p in prob_t:
        mass = np.zeros(rmax+1, float)
        for i,di in enumerate(dists):
            if di >= 0:
                mass[di] += p[i]
        c = np.cumsum(mass)
        r = int(np.searchsorted(c, quantile))
        out.append(r)
    return np.array(out, int)

quantum.unitary_packet  = unitary_packet
quantum.lr_front_radius = lr_front_radius


# ======================================================================================
# stlib.gkls
# ======================================================================================
gkls = types.ModuleType("stlib.gkls")
def gkls_dephasing_stepper(H, gamma: float = 0.5, dt: float = 0.02) -> Callable[[np.ndarray], np.ndarray]:
    """
    Einfache GKLS-Dephasierung in Ortsbasis:
      1) unitärer Schritt: ρ -> U ρ U†, U = exp(-i H dt)
      2) Dephasierung: Off-Diagonalen * exp(-gamma * dt)
    Für kleine N (<= ~ 300) ok.
    """
    if HAVE_SCIPY:
        U = _expm_dense((-1j*dt) * (H.toarray() if hasattr(H, "toarray") else H))
    else:
        U = _expm_dense((-1j*dt) * (H.toarray() if hasattr(H, "toarray") else H))

    damp = math.exp(-gamma * dt)

    def step(rho: np.ndarray) -> np.ndarray:
        rho = U @ rho @ U.conj().T
        if np.iscomplexobj(rho):
            d = np.diag(np.diag(rho))
            off = rho - d
            rho = d + damp * off
        else:
            d = np.diag(np.diag(rho))
            off = rho - d
            rho = d + damp * off
        return rho

    return step

gkls.gkls_dephasing_stepper = gkls_dephasing_stepper


# ======================================================================================
# stlib.dtn
# ======================================================================================
dtn = types.ModuleType("stlib.dtn")
def dirichlet_to_neumann(L, b_nodes: Iterable[int]) -> np.ndarray:
    """
    DtN mittels Schur-Komplement:
      Lambda = L_BB - L_BI * (L_II)^{-1} * L_IB
    """
    L = _to_csr(L)
    n = L.shape[0]
    B = np.array(list(b_nodes), dtype=int)
    I = np.array([i for i in range(n) if i not in set(B)], dtype=int)

    if len(B) == 0:
        return np.zeros((0,0), dtype=float)
    if len(I) == 0:
        # nur Rand -> DtN = L selbst
        return (L.toarray() if hasattr(L, "toarray") else np.array(L))

    L_BB = _submatrix_csr(L, B, B)
    L_BI = _submatrix_csr(L, B, I)
    L_IB = _submatrix_csr(L, I, B)
    L_II = _submatrix_csr(L, I, I)

    if HAVE_SCIPY:
        # löse L_II X = L_IB
        X = spla.spsolve(L_II.tocsc(), L_IB.toarray())
        S = L_BB.toarray() - L_BI.toarray() @ X
    else:
        A = L_II.toarray(); Bm = L_IB.toarray(); BI = L_BI.toarray()
        X = np.linalg.solve(A, Bm)
        S = L_BB.toarray() - BI @ X
    return np.array(S, float)

def perturb_edges_in_cell(nodes: List, edges: List[Tuple], cell_prefix: str, factor: float = 2.0) -> List[float]:
    """
    Erhöhe Kantengewichte für Kanten, deren beide Enden mit cell_prefix beginnen (String-Vergleich).
    Falls Knoten keine Strings sind, werden sie in Strings umgewandelt.
    """
    pref = str(cell_prefix)
    weights = []
    for u,v in edges:
        su = str(u); sv = str(v)
        if su.startswith(pref) and sv.startswith(pref):
            weights.append(float(factor))
        else:
            weights.append(1.0)
    return weights

dtn.dirichlet_to_neumann    = dirichlet_to_neumann
dtn.perturb_edges_in_cell   = perturb_edges_in_cell


# --------------------------------------------------------------------------------------
# Untermodule im sys.modules registrieren, damit `from stlib.X import ...` funktioniert.
# --------------------------------------------------------------------------------------
this = sys.modules.setdefault("stlib", types.ModuleType("stlib"))
# Exporte auf Top-Level optional (nicht notwendig für dein Skript)
this.graph_st       = graph_st
this.graphs_regular = graphs_regular
this.linops         = linops
this.heat           = heat
this.quantum        = quantum
this.gkls           = gkls
this.dtn            = dtn

# Registriere Namespaces für from-imports
sys.modules["stlib.graph_st"]       = graph_st
sys.modules["stlib.graphs_regular"] = graphs_regular
sys.modules["stlib.linops"]         = linops
sys.modules["stlib.heat"]           = heat
sys.modules["stlib.quantum"]        = quantum
sys.modules["stlib.gkls"]           = gkls
sys.modules["stlib.dtn"]            = dtn

__all__ = ["graph_st","graphs_regular","linops","heat","quantum","gkls","dtn"]
