"""
One-time generator of the spatially blocked splits (10 folds).
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from load_split import load_pos_to_xy
except ImportError:
    from data.load_split import load_pos_to_xy


# ============================================================
# Config
# ============================================================
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "saved_blocked_split"
CSV_PATH = OUT_DIR / "pos_to_xy.csv"

N_POS = 200
N_STATE = 4
SEED = 0
VAL_POS_NUM = 20

S_X = 0.40
S_Y_13_13 = 0.30
S_Y_HALF = S_Y_13_13 / 2.0

# Regions: edges lie halfway between grid columns / rows.
X_EDGES = [0.9, 1.9, 2.9, 3.9]
Y_EDGE = 1.125
N_X = len(X_EDGES) + 1
N_BLOCKS = 2 * N_X
BLOCK_SIZE = 20


# ============================================================
# Layout and uniform non-adjacent held-out selection
# ============================================================
def build_layout(coords):
    """Row index per position and row -> positions sorted by x."""
    rows = np.round(coords[:, 1] / S_Y_HALF).astype(np.int64)
    row_to_pos = {}
    for p in range(N_POS):
        row_to_pos.setdefault(int(rows[p]), []).append(p)
    for r in row_to_pos:
        row_to_pos[r].sort(key=lambda p: float(coords[p, 0]))
    return rows, row_to_pos


def build_neighbors(coords, rows):
    eps = 1e-6
    nb = {p: set() for p in range(N_POS)}
    for i in range(N_POS):
        for j in range(i + 1, N_POS):
            dx = abs(float(coords[i, 0] - coords[j, 0]))
            dr = abs(int(rows[i]) - int(rows[j]))
            same_row_adj = (rows[i] == rows[j]) and (eps < dx <= S_X + eps)
            adj_row_adj = (dr == 1) and (dx <= 0.5 * S_X + eps)
            if same_row_adj or adj_row_adj:
                nb[i].add(j)
                nb[j].add(i)
    return nb


def get_corners(coords, rows):
    top, bot = rows.min(), rows.max()
    top_p = np.where(rows == top)[0]
    bot_p = np.where(rows == bot)[0]
    return {
        int(top_p[np.argmin(coords[top_p, 0])]),
        int(top_p[np.argmax(coords[top_p, 0])]),
        int(bot_p[np.argmin(coords[bot_p, 0])]),
        int(bot_p[np.argmax(coords[bot_p, 0])]),
    }


def row_targets(row_to_pos, total, seed):
    rows = sorted(row_to_pos.keys())
    base, extra = divmod(total, len(rows))
    tgt = {r: base for r in rows}
    if extra:
        idx = np.floor(np.arange(extra) * len(rows) / extra).astype(int)
        offset = seed % max(1, len(rows) // extra)
        for i in idx:
            tgt[rows[(int(i) + offset) % len(rows)]] += 1
    return tgt


def select_heldout(row_to_pos, neighbors, excluded, total, seed, max_attempts=50000):
    rows = sorted(row_to_pos.keys())
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + 1009 * attempt)
        tgt = row_targets(row_to_pos, total, seed + attempt)
        selected = set()

        # shuffle rows first, then stable-sort by target desc.
        row_order = rows.copy()
        rng.shuffle(row_order)
        row_order = sorted(row_order, key=lambda r: tgt[r], reverse=True)

        ok = True
        for r in row_order:
            cands = [p for p in row_to_pos[r] if p not in excluded]
            rng.shuffle(cands)
            picked = 0
            for p in cands:
                if any(q in neighbors[p] for q in selected):
                    continue
                selected.add(p)
                picked += 1
                if picked == tgt[r]:
                    break
            if picked < tgt[r]:
                ok = False
                break
        if ok and len(selected) == total:
            return sorted(selected)
    raise RuntimeError(f"Failed to select {total} non-adjacent positions in {max_attempts} attempts.")


# ============================================================
# Regions
# ============================================================
def block_of(coords):
    ix = np.searchsorted(X_EDGES, coords[:, 0])
    iy = (coords[:, 1] > Y_EDGE).astype(int)
    return iy * N_X + ix


def block_rect(b, coords, pad=0.1):
    ix, iy = b % N_X, b // N_X
    xs = [coords[:, 0].min() - pad] + X_EDGES + [coords[:, 0].max() + pad]
    ys = [coords[:, 1].min() - pad, Y_EDGE, coords[:, 1].max() + pad]
    return xs[ix], ys[iy], xs[ix + 1] - xs[ix], ys[iy + 1] - ys[iy]


def nearest_train_dist(coords, test_pos, train_pos):
    d = np.linalg.norm(coords[test_pos][:, None, :] - coords[train_pos][None, :, :], axis=-1)
    return d.min(axis=1)


def write_fold(name, train_pos, val_pos, test_pos):
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    for split, positions in (("train", train_pos), ("val", val_pos), ("test", test_pos)):
        with open(d / f"{split}.txt", "w", newline="\n") as f:
            for p in positions:
                for s in range(N_STATE):
                    f.write(f"State_{s}_Pos_{p}.npz\n")


# ============================================================
# Plot
# ============================================================
def plot_folds(coords, folds, info, path):
    fig, axes = plt.subplots(2, N_X, figsize=(18, 6.2))
    style = {"train": ("#1f77b4", "o", "train"), "val": ("#2ca02c", "s", "val"), "test": ("#d62728", "^", "test")}
    for b, ax in enumerate(axes.ravel()):
        name = f"B{b + 1}"
        for split, (color, marker, label) in style.items():
            ps = folds[name][split]
            ax.scatter(coords[ps, 0], coords[ps, 1], c=color, marker=marker, s=35,
                       edgecolors="black", linewidths=0.4, label=f"{label} ({len(ps)})", zorder=3)
        x0, y0, w, h = block_rect(b, coords)
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="black", lw=1.0, ls="--", zorder=2))
        ax.set_title(f"{name}: nearest train {info[name]['dist_mean']:.2f} m "
                     f"(max {info[name]['dist_max']:.2f} m)", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(coords[:, 0].min() - 0.2, coords[:, 0].max() + 0.2)
        ax.set_ylim(coords[:, 1].max() + 0.2, coords[:, 1].min() - 0.2)     # row 0 on top
        ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.supxlabel("x (m)", fontsize=9)
    fig.supylabel("y (m)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    pos_to_xy = load_pos_to_xy(CSV_PATH)
    coords = np.array([pos_to_xy[p] for p in range(N_POS)], dtype=np.float64)
    rows, row_to_pos = build_layout(coords)
    neighbors = build_neighbors(coords, rows)
    corners = get_corners(coords, rows)

    blocks = block_of(coords)
    assert sorted(np.unique(blocks).tolist()) == list(range(N_BLOCKS))
    assert all(int((blocks == b).sum()) == BLOCK_SIZE for b in range(N_BLOCKS))

    folds, info = {}, {}
    for b in range(N_BLOCKS):
        name = f"B{b + 1}"
        test = sorted(np.where(blocks == b)[0].tolist())
        border = {q for p in test for q in neighbors[p]} - set(test)      # 1-hop border of the region
        val = select_heldout(row_to_pos, neighbors, excluded=corners | set(test) | border,
                             total=VAL_POS_NUM, seed=SEED)
        train = sorted(set(range(N_POS)) - set(test) - set(val))
        assert len(train) + len(val) + len(test) == N_POS
        assert not (set(val) & border) and not (set(val) & set(test))
        write_fold(name, train, val, test)
        folds[name] = {"train": train, "val": val, "test": test}

    for name, f in folds.items():
        d = nearest_train_dist(coords, f["test"], f["train"])
        info[name] = {"n_train_pos": len(f["train"]), "n_val_pos": len(f["val"]), "n_test_pos": len(f["test"]),
                      "dist_mean": float(d.mean()), "dist_max": float(d.max()), **f}
    (OUT_DIR / "folds.json").write_text(json.dumps(info, indent=1))
    plot_folds(coords, folds, info, OUT_DIR / "blocked_split_map.png")

    print(f"Wrote {OUT_DIR}")
    print(f"{'fold':5s} {'train':>5s} {'val':>4s} {'test':>4s}  nearest-train distance of the test positions")
    for name, d in info.items():
        print(f"{name:5s} {d['n_train_pos']:5d} {d['n_val_pos']:4d} {d['n_test_pos']:4d}  "
              f"mean {d['dist_mean']:.3f} m, max {d['dist_max']:.3f} m")


if __name__ == "__main__":
    main()
