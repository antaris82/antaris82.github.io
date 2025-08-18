# Create a new Jupyter-friendly script that ALSO computes


# cut-edges between regions, mean |C| over cross-blocks, and minimal graph distance,

# integrates these into CSVs and the annotated static PNG, and renders the GIF.

#

# We'll save the script into the current working directory and ALSO run the same logic here

# so you immediately get files in Jupyter Lab.


from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from itertools import product, combinations

import pandas as pd

import imageio, os, sys


SCRIPT = r'''

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from itertools import product, combinations

import pandas as pd

import imageio, os


# ------------------------ Core utilities ------------------------

def build_st_graph_with_cells(level: int):

    scale = 2**level

    corners = np.array([[0,0,0], [scale,0,0], [0,scale,0], [0,0,scale]], dtype=float)

    Vs = [tuple(v) for v in corners]

    maps = [lambda x,vi=vi: ((x[0]+vi[0])/2.0, (x[1]+vi[1])/2.0, (x[2]+vi[2])/2.0) for vi in Vs]

    words = list(product(range(4), repeat=level))

    vertices_set = set()

    def apply_prefix(word_prefix, x):

        for idx in word_prefix:

            x = maps[idx](x)

        return x

    for w in words:

        for k in range(level+1):

            wp = w[:k]

            imgs = [apply_prefix(wp, v) for v in Vs]

            for p in imgs:

                vertices_set.add(tuple(p))

    verts = np.array(sorted(list(vertices_set)), dtype=float)

    index = {tuple(v): i for i,v in enumerate(verts.tolist())}

    edges_set = set()

    for w in words:

        for k in range(level+1):

            wp = w[:k]

            imgs = [apply_prefix(wp, v) for v in Vs]

            inds = [index[tuple(p)] for p in imgs]

            for a,b in combinations(inds, 2):

                e = (a,b) if a<b else (b,a)

                edges_set.add(e)

    cell_words = [tuple(w) for w in words]

    cell_vertices_idx = []

    for w in words:

        imgs = [apply_prefix(w, v) for v in Vs]

        inds = np.array([index[tuple(p)] for p in imgs], dtype=int)

        cell_vertices_idx.append(inds)

    return verts, np.array(sorted(list(edges_set)), dtype=int), corners, cell_words, cell_vertices_idx


def adjacency_from_edges(n: int, edges: np.ndarray):

    A = np.zeros((n,n), dtype=float)

    for a,b in edges:

        A[a,b] = 1.0; A[b,a] = 1.0

    return A


def laplacian_from_adjacency(A: np.ndarray):

    return np.diag(A.sum(axis=1)) - A


def rotation_to_xy_base(corners: np.ndarray):

    base = corners[[0,1,2], :]

    v1 = base[1]-base[0]; v2 = base[2]-base[0]

    n = np.cross(v1, v2); n = n/np.linalg.norm(n)

    target = np.array([0,0,1.0])

    axis = np.cross(n, target); axlen = np.linalg.norm(axis)

    if axlen < 1e-14: return np.eye(3)

    axis = axis/axlen

    angle = np.arccos(np.clip(np.dot(n, target), -1.0, 1.0))

    K = np.array([[0, -axis[2], axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])

    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def correlation_matrix_groundstate(H: np.ndarray, filling=0.5, seed=99):

    from numpy.linalg import eigh

    rng = np.random.default_rng(seed)

    H = H + 1e-9*np.diag(rng.standard_normal(H.shape[0]))

    evals, evecs = eigh(H)

    N = H.shape[0]; M = int(round(filling*N))

    Uocc = evecs[:, :M]

    C = Uocc @ Uocc.T.conj()

    return C, evals


def von_neumann_entropy_from_C(CA: np.ndarray):

    evals = np.clip(np.linalg.eigvalsh(CA), 1e-12, 1-1e-12)

    return float(np.real(-np.sum(evals*np.log(evals) + (1-evals)*np.log(1-evals))))


def region_indices_from_prefixes(level:int, cell_words, cell_vertices_idx, prefixes):

    idxs = set()

    for w, inds in zip(cell_words, cell_vertices_idx):

        for pref in prefixes:

            if w[:len(pref)] == pref:

                for i in inds: idxs.add(int(i))

                break

    return np.array(sorted(list(idxs)), dtype=int)


def setup_axes_equal(ax, V):

    mins = V.min(axis=0); maxs = V.max(axis=0); spans = maxs - mins

    M = spans.max(); c = (maxs + mins)/2.0

    ax.set_box_aspect((1,1,1))

    ax.set_xlim(c[0]-M/2, c[0]+M/2)

    ax.set_ylim(c[1]-M/2, c[1]+M/2)

    ax.set_zlim(c[2]-M/2, c[2]+M/2)


def MI_disjoint(A_idx, B_idx, C):

    A = np.array(sorted(set(map(int, A_idx))), dtype=int)

    B = np.array(sorted(set(map(int, B_idx))), dtype=int)

    AB = np.array(sorted(set(A)|set(B)), dtype=int)

    if len(A)==0 or len(B)==0:

        return 0.0

    S_A = von_neumann_entropy_from_C(C[np.ix_(A,A)])

    S_B = von_neumann_entropy_from_C(C[np.ix_(B,B)])

    S_AB = von_neumann_entropy_from_C(C[np.ix_(AB,AB)])

    return S_A + S_B - S_AB


def cut_edges_between(A_idx, B_idx, edges):

    Aset = set(map(int, A_idx)); Bset = set(map(int, B_idx))

    cnt = 0

    for (u,v) in edges:

        if (u in Aset and v in Bset) or (u in Bset and v in Aset):

            cnt += 1

    return int(cnt)


def mean_abs_crossC(A_idx, B_idx, C):

    if len(A_idx)==0 or len(B_idx)==0: return 0.0

    block = C[np.ix_(A_idx, B_idx)]

    return float(np.mean(np.abs(block)))


def min_graph_distance(A_idx, B_idx, edges, N):

    from collections import deque

    Aset, Bset = set(map(int,A_idx)), set(map(int,B_idx))

    if len(Aset)==0 or len(Bset)==0: return None

    # adjacency list

    adj = [[] for _ in range(N)]

    for (u,v) in edges:

        adj[u].append(v); adj[v].append(u)

    # multi-source BFS from A until we hit any in B

    INF = 10**9

    dist = [INF]*N

    dq = deque()

    for a in Aset:

        dist[a] = 0

        dq.append(a)

    while dq:

        x = dq.popleft()

        if x in Bset: return dist[x]

        for y in adj[x]:

            if dist[y] == INF:

                dist[y] = dist[x] + 1

                dq.append(y)

    return None


# ------------------------ Main ------------------------

def main():

    BASE = Path.cwd()

    L = 4

    verts, edges, corners, cell_words, cell_vertices_idx = build_st_graph_with_cells(L)

    A = adjacency_from_edges(len(verts), edges)

    H = laplacian_from_adjacency(A)

    C, _ = correlation_matrix_groundstate(H, filling=0.5, seed=99)

    occ = np.real(np.diag(C))


    # Rotate to xy-plane base

    R = rotation_to_xy_base(corners)

    verts_rot = verts @ R.T

    corner_idx = [np.where((verts == v).all(axis=1))[0][0] for v in corners]


    # Regions (exclusive assignment: RED > YELLOW > GREEN)

    prefix_red = (0,1,2,3)

    prefix_yellow = (1,0)

    RED_base    = region_indices_from_prefixes(L, cell_words, cell_vertices_idx, [prefix_red])

    YELLOW_base = region_indices_from_prefixes(L, cell_words, cell_vertices_idx, [prefix_yellow])

    GREEN_base  = np.array(corner_idx, dtype=int)


    taken = set()

    def assign_exclusive(idx_list):

        out = []

        for i in idx_list:

            if int(i) not in taken:

                out.append(int(i)); taken.add(int(i))

        return np.array(sorted(out), dtype=int)


    RED_idx    = assign_exclusive(RED_base)

    YELLOW_idx = assign_exclusive(YELLOW_base)

    GREEN_idx  = assign_exclusive(GREEN_base)


    regions = {"RED": RED_idx, "YELLOW": YELLOW_idx, "GREEN": GREEN_idx}


    # Per-region stats

    def region_stats(name, idx, C):

        CA = C[np.ix_(idx, idx)] if len(idx)>0 else np.zeros((1,1))

        S = von_neumann_entropy_from_C(CA) if len(idx)>0 else 0.0

        MI_rest = 2.0*S

        if len(idx)>1:

            intra = C[np.ix_(idx, idx)]

            mean_intra_absC = float(np.mean(np.abs(intra - np.diag(np.diag(intra)))))

        else:

            mean_intra_absC = 0.0

        return {"region": name, "size": int(len(idx)), "S": S, "MI_with_rest": MI_rest,

                "mean_intra_absC": mean_intra_absC}


    df_regions = pd.DataFrame([region_stats(n, idx, C) for n, idx in regions.items()])

    df_regions.to_csv(BASE/"regions_observables_exclusive.csv", index=False)


    # Pairwise metrics

    pairs = [("RED","YELLOW"), ("RED","GREEN"), ("YELLOW","GREEN")]

    rows = []

    for a,b in pairs:

        Ai, Aj = regions[a], regions[b]

        row = {

            "A": a, "B": b,

            "overlap": 0,

            "cut_edges": cut_edges_between(Ai, Aj, edges),

            "mean_abs_crossC": mean_abs_crossC(Ai, Aj, C),

            "dmin": min_graph_distance(Ai, Aj, edges, len(verts)),

            "MI": MI_disjoint(Ai, Aj, C),

        }

        rows.append(row)

    df_pairs = pd.DataFrame(rows)

    df_pairs.to_csv(BASE/"pairs_observables_exclusive.csv", index=False)


    # Birth-level S & MI

    def birth_levels(level:int, verts: np.ndarray):

        scale = 2**level

        Vs = [np.array([0,0,0.0]), np.array([scale,0,0.0]), np.array([0,scale,0.0]), np.array([0,0,scale])]

        maps = [lambda x,vi=vi: ((x+vi)/2.0) for vi in Vs]

        words = list(product(range(4), repeat=level))

        first_seen = {}

        def apply_prefix(wp, x):

            for idx in wp: x = maps[idx](x)

            return tuple(x.tolist())

        for w in words:

            for k in range(level+1):

                wp = w[:k]

                img = [apply_prefix(wp, vi.copy()) for vi in Vs]

                for p in img:

                    if p not in first_seen: first_seen[p] = k

        idx_map = {tuple(v):i for i,v in enumerate(verts.tolist())}

        sets = [np.array([idx_map[tuple(v)] for v,l in first_seen.items() if l==ell], dtype=int)

                for ell in range(level+1)]

        return sets


    birth_sets = birth_levels(L, verts)

    rowsL = []

    for ell, idx in enumerate(birth_sets):

        if len(idx)==0:

            rowsL.append({"level": ell, "n": 0, "S": 0.0, "MI_with_rest": 0.0})

        else:

            CA = C[np.ix_(idx, idx)]

            S = von_neumann_entropy_from_C(CA)

            rowsL.append({"level": ell, "n": int(len(idx)), "S": S, "MI_with_rest": 2.0*S})

    df_levels = pd.DataFrame(rowsL)

    df_levels.to_csv(BASE/"levels_observables.csv", index=False)


    # ---------- Rendering ----------

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


    def setup_axes_equal(ax, V):

        mins = V.min(axis=0); maxs = V.max(axis=0); spans = maxs - mins

        M = spans.max(); c = (maxs + mins)/2.0

        ax.set_box_aspect((1,1,1))

        ax.set_xlim(c[0]-M/2, c[0]+M/2)

        ax.set_ylim(c[1]-M/2, c[1]+M/2)

        ax.set_zlim(c[2]-M/2, c[2]+M/2)


    def corners_of_prefix(prefix, L, verts):

        scale = 2**L

        base_corners = np.array([[0,0,0], [scale,0,0], [0,scale,0], [0,0,scale]], dtype=float)

        def apply_word(v, word):

            x = v.copy()

            for idx in word:

                x = (x + base_corners[idx]) / 2.0

            return tuple(x.tolist())

        pts = [apply_word(base_corners[i], prefix) for i in range(4)]

        idx_map = {tuple(v):i for i,v in enumerate(verts.tolist())}

        return [idx_map[tuple(p)] for p in pts]


    RED_corners    = corners_of_prefix(prefix_red, L, verts)

    YELLOW_corners = corners_of_prefix(prefix_yellow, L, verts)

    GREEN_corners  = list(corner_idx)


    def draw_edges(ax, idx_list, color, lw):

        for a,b in combinations(idx_list, 2):

            xs=[verts_rot[a,0], verts_rot[b,0]]

            ys=[verts_rot[a,1], verts_rot[b,1]]

            zs=[verts_rot[a,2], verts_rot[b,2]]

            ax.plot(xs, ys, zs, linewidth=lw, color=color)


    # Static 3D

    fig = plt.figure(figsize=(12.0,9.2))

    ax = fig.add_subplot(111, projection='3d')

    color_array = np.array(['#d3d3d3']*len(verts_rot), dtype=object)

    color_array[regions["RED"]] = 'red'

    color_array[regions["YELLOW"]] = 'gold'

    ax.scatter(verts_rot[:,0], verts_rot[:,1], verts_rot[:,2], s=5, c=color_array)

    # wireframe green

    for a,b in combinations(GREEN_corners, 2):

        xs=[verts_rot[a,0], verts_rot[b,0]]; ys=[verts_rot[a,1], verts_rot[b,1]]; zs=[verts_rot[a,2], verts_rot[b,2]]

        ax.plot(xs, ys, zs, color='green', linewidth=1.4)

    # red & yellow edges

    draw_edges(ax, RED_corners,    'red',  2.0)

    draw_edges(ax, YELLOW_corners, 'gold', 1.8)

    setup_axes_equal(ax, verts_rot)

    ax.view_init(elev=18, azim=-45)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


    # Annotation with pairs metrics

    def fmt_region_line(region):

        r = df_regions[df_regions["region"]==region].iloc[0]

        return f"{region}: |A|={r['size']}, S={r['S']:.4f}, MI(A:Rest)={r['MI_with_rest']:.4f}, ⟨|C|⟩_intra={r['mean_intra_absC']:.4f}"


    linesR = [fmt_region_line(n) for n in ["GREEN","YELLOW","RED"]]

    parts = []

    for _,row in df_pairs.iterrows():

        parts.append(f"I({row['A']}:{row['B']})={row['MI']:.4f} | cut={int(row['cut_edges'])} | "

                     f"⟨|C|⟩_cross={row['mean_abs_crossC']:.4f} | d_min={row['dmin']}")

    txt = " | ".join(linesR) + "\n" + " || ".join(parts)

    ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, va='top', fontsize=9,

              bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85))


    ax.set_title("Sierpinski-Tetraeder L=4 – EXKLUSIV + Schnitt-Metriken (GREEN L0, YELLOW L2, RED L4)", fontsize=11)

    (BASE/"static_colored_obs_exclusive.png").write_bytes(b"")  # touch marker for path stability

    fig.savefig(BASE/"static_colored_obs_exclusive.png", dpi=220, bbox_inches='tight', pad_inches=0.08)

    plt.close(fig)


    # Line chart

    figL = plt.figure(figsize=(8.2,4.8))

    plt.plot(df_levels["level"], df_levels["S"], marker='o', label="S(ℓ)")

    plt.plot(df_levels["level"], df_levels["MI_with_rest"], marker='s', label="MI(ℓ:Rest)")

    plt.xlabel("Konstruktionslevel ℓ"); plt.ylabel("Entropie / MI")

    plt.title("Verschränkungsentropie & MI pro ausgespurtem Layer (L=4)", fontsize=11)

    plt.xticks(df_levels["level"]); plt.legend()

    (BASE/"levels_S_MI.png").write_bytes(b"")

    figL.tight_layout()

    figL.savefig(BASE/"levels_S_MI.png", dpi=200, bbox_inches='tight', pad_inches=0.05)

    plt.close(figL)


    # Rotating GIF

    frames = []

    angles = np.linspace(0, 360, 30, endpoint=False)

    for az in angles:

        fig = plt.figure(figsize=(9.2,7.6))

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(verts_rot[:,0], verts_rot[:,1], verts_rot[:,2], s=5, c=color_array)

        for a,b in combinations(GREEN_corners, 2):

            xs=[verts_rot[a,0], verts_rot[b,0]]; ys=[verts_rot[a,1], verts_rot[b,1]]; zs=[verts_rot[a,2], verts_rot[b,2]]

            ax.plot(xs, ys, zs, color='green', linewidth=1.1)

        draw_edges(ax, RED_corners,    'red',  2.0)

        draw_edges(ax, YELLOW_corners, 'gold', 1.8)

        setup_axes_equal(ax, verts_rot)

        ax.view_init(elev=18, azim=az)

        ax.text2D(0.02, 0.98, "EXKLUSIV + Schnitt-Metriken", transform=ax.transAxes,

                  va='top', fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.75))

        ax.axis('off')

        tmp = BASE / f"__tmp_m_{int(az):03d}.png"

        fig.savefig(tmp, dpi=110, bbox_inches='tight', pad_inches=0.02)

        plt.close(fig)

        frames.append(imageio.v2.imread(tmp))

        os.remove(tmp)


    (BASE/"static_colored_obs_exclusive_rotate.gif").write_bytes(b"")

    imageio.mimsave(BASE/"static_colored_obs_exclusive_rotate.gif", frames, duration=0.10)


    print("Done. Files written in:", BASE.resolve())


if __name__ == "__main__":

    main()

'''


# Write the script into the current working directory of this kernel

script_path = Path("st_pipeline_metrics.py")

script_path.write_text(SCRIPT, encoding="utf-8")


# For convenience, also run the same logic here by importing the script as a module

# (so you immediately get outputs in Jupyter).

import importlib.util

spec = importlib.util.spec_from_file_location("st_pipeline_metrics", str(script_path.resolve()))

mod = importlib.util.module_from_spec(spec)

sys.modules["st_pipeline_metrics"] = mod

spec.loader.exec_module(mod)

# Run main()

mod.main()


# List generated files

out_files = [

    "static_colored_obs_exclusive.png",

    "levels_S_MI.png",

    "regions_observables_exclusive.csv",

    "pairs_observables_exclusive.csv",

    "levels_observables.csv",

    "static_colored_obs_exclusive_rotate.gif",

    "st_pipeline_metrics.py",

]

print("\nGenerated:")

for f in out_files:

    p = Path(f).resolve()

    print(" -", p)