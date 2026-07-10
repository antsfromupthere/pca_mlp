"""
One-time generator.

Reads the layout config, runs the uniform non-adjacent position split
(no corners, no adjacent held-out pairs), and writes four files:

    pos_to_xy.csv   pos_id,x,y         (200 rows, one per position, meters)
    train.txt       State_{s}_Pos_{p}.npz, one filename per line
    val.txt         same
    test.txt        same

Geometry (brick / staggered layout, 200 positions = 8 blocks of 25):
    - 13-row: 13 points spaced horizontally by S_X.
              x = 0, S_X, 2*S_X, ..., 12*S_X.
    - 12-row: 12 points, each horizontally at the midpoint between two
              13-row points.
              x = 0.5*S_X, 1.5*S_X, ..., 11.5*S_X.
    - Vertical:
              two consecutive 13-rows are S_Y_13_13 apart.
              the 12-row between them sits at the midpoint, i.e.
              S_Y_13_13 / 2 from each neighbouring 13-row.
    - So row index r (0..15) -> y = r * (S_Y_13_13 / 2).

Collection order within each 25-position block:
    row 0 (13 points): collected left to right   pos 0..12
    row 1 (12 points): collected right to left   pos 13..24
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Geometry config (meters)
# ============================================================

S_X = 0.40          # horizontal spacing between adjacent points in a 13-row
S_Y_13_13 = 0.30    # vertical distance between two consecutive 13-rows

S_Y_HALF = S_Y_13_13 / 2.0   # row-to-row vertical step (any row to next row)

N_POS = 200
N_STATE = 4
SEED = 0

HELDOUT_POS_NUM = 40
VAL_POS_NUM = 20
TEST_POS_NUM = 20

OUT_DIR = Path(".")
CSV_PATH = OUT_DIR / "pos_to_xy.csv"
TRAIN_TXT = OUT_DIR / "train.txt"
VAL_TXT = OUT_DIR / "val.txt"
TEST_TXT = OUT_DIR / "test.txt"
PLOT_PATH = OUT_DIR / "split_map.png"


# ============================================================
# Layout
# ============================================================

def pos_index_to_xy(p):
    """Map position index (0..199) to (x, y) in meters."""
    block = p // 25       # 0..7
    r = p % 25            # 0..24
    if r < 13:
        row = 2 * block               # 13-row
        x = r * S_X                   # 0, S_X, ..., 12*S_X
    else:
        row = 2 * block + 1           # 12-row
        j = r - 13                    # collected right-to-left
        x = (11.5 - j) * S_X          # 11.5*S_X, 10.5*S_X, ..., 0.5*S_X
    y = row * S_Y_HALF
    return x, y


def build_layout():
    """Return coords (N_POS, 2) in meters, row index per position, and row -> sorted positions."""
    coords = np.array([pos_index_to_xy(p) for p in range(N_POS)], dtype=np.float64)
    rows = np.round(coords[:, 1] / S_Y_HALF).astype(np.int64)
    row_to_pos = {}
    for p in range(N_POS):
        row_to_pos.setdefault(int(rows[p]), []).append(p)
    for r in row_to_pos:
        row_to_pos[r].sort(key=lambda p: float(coords[p, 0]))
    return coords, rows, row_to_pos


def build_neighbors(coords, rows):
    """
    Adjacency:
        - same row, |dx| ~ S_X (horizontally next point in that row)
        - neighbouring row (one above / one below), staggered:
            |dx| ~ 0.5 * S_X  (12-row point centered between two 13-row points)
            |dy| ~ S_Y_HALF
    """
    eps = 1e-6
    nb = {p: set() for p in range(N_POS)}
    for i in range(N_POS):
        for j in range(i + 1, N_POS):
            dx = abs(float(coords[i, 0] - coords[j, 0]))
            dr = abs(int(rows[i]) - int(rows[j]))
            same_row_adj = (rows[i] == rows[j]) and (eps < dx <= S_X + eps)
            adj_row_adj  = (dr == 1) and (dx <= 0.5 * S_X + eps)
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


# ============================================================
# Uniform non-adjacent held-out selection
# ============================================================

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

        # Match original: shuffle rows first, then stable-sort by target desc.
        # The shuffle consumes rng state and breaks ties between rows with equal target.
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


def split_val_test(heldout, coords, rows, val_num, seed):
    row_to_h = {}
    for p in heldout:
        row_to_h.setdefault(int(rows[p]), []).append(p)
    for r in row_to_h:
        row_to_h[r].sort(key=lambda p: float(coords[p, 0]))

    rs = sorted(row_to_h.keys())
    val_tgt = {r: len(row_to_h[r]) // 2 for r in rs}
    remaining = val_num - sum(val_tgt.values())
    if remaining > 0:
        cand = [r for r in rs if val_tgt[r] < len(row_to_h[r])]
        idx = np.floor(np.arange(remaining) * len(cand) / remaining).astype(int)
        offset = seed % max(1, len(cand))
        for i in idx:
            val_tgt[cand[(int(i) + offset) % len(cand)]] += 1

    rng = np.random.default_rng(seed)
    val_pos, test_pos = [], []
    for r in rs:
        ps = list(row_to_h[r])
        rng.shuffle(ps)
        val_pos.extend(ps[:val_tgt[r]])
        test_pos.extend(ps[val_tgt[r]:])
    return sorted(val_pos), sorted(test_pos)


# ============================================================
# Plot
# ============================================================

def plot_split(coords, train_pos, val_pos, test_pos, corners, neighbors,
               save_path=PLOT_PATH, show=False):
    """
    Scatter plot of all 200 positions, coloured by split (train / val / test),
    with the pos_id printed next to each point. Corners are circled in black,
    and any adjacent val/test pairs (which should never happen) are drawn in red.
    """
    train_pos = set(train_pos)
    val_pos = set(val_pos)
    test_pos = set(test_pos)
    heldout = val_pos | test_pos

    fig, ax = plt.subplots(figsize=(13, 7))

    # group plot for legend
    for label, positions, color, marker in [
        ("train", train_pos, "#1f77b4", "o"),
        ("val",   val_pos,   "#2ca02c", "s"),
        ("test",  test_pos,  "#d62728", "^"),
    ]:
        if not positions:
            continue
        ps = sorted(positions)
        ax.scatter(
            coords[ps, 0], coords[ps, 1],
            c=color, marker=marker, s=180,
            edgecolors="black", linewidths=0.6,
            label=f"{label} ({len(ps)})", zorder=3,
        )

    # corners: black ring
    if corners:
        cps = sorted(corners)
        ax.scatter(
            coords[cps, 0], coords[cps, 1],
            facecolors="none", edgecolors="black",
            s=380, linewidths=1.8, zorder=4,
            label=f"corner ({len(cps)})",
        )

    # adjacency violations among held-out (val ∪ test)
    bad_pairs = [
        (p, q) for p in heldout for q in neighbors[p]
        if q in heldout and p < q
    ]
    for p, q in bad_pairs:
        ax.plot(
            [coords[p, 0], coords[q, 0]],
            [coords[p, 1], coords[q, 1]],
            c="red", linewidth=2.5, alpha=0.9, zorder=5,
        )
    if bad_pairs:
        print(f"WARNING: {len(bad_pairs)} adjacent held-out pairs: {bad_pairs}")
    else:
        print("OK: no adjacent held-out pairs.")

    # pos_id labels
    for p in range(coords.shape[0]):
        ax.text(
            coords[p, 0], coords[p, 1] + 0.012,  # small offset above the point
            str(p),
            fontsize=7, ha="center", va="bottom", zorder=6,
        )

    ax.set_title(
        f"Position layout — train/val/test split "
        f"(S_X={S_X} m, S_Y_13_13={S_Y_13_13} m)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at top, matches collection order
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    coords, rows, row_to_pos = build_layout()
    neighbors = build_neighbors(coords, rows)
    corners = get_corners(coords, rows)

    print(f"Geometry: S_X = {S_X} m, S_Y_13_13 = {S_Y_13_13} m, row step = {S_Y_HALF} m")
    print(f"x range: {coords[:, 0].min():.3f} .. {coords[:, 0].max():.3f} m")
    print(f"y range: {coords[:, 1].min():.3f} .. {coords[:, 1].max():.3f} m")
    print(f"Corners excluded: {sorted(corners)}")

    heldout = select_heldout(
        row_to_pos, neighbors,
        excluded=corners,
        total=HELDOUT_POS_NUM,
        seed=SEED,
    )
    val_pos, test_pos = split_val_test(heldout, coords, rows, VAL_POS_NUM, SEED)
    train_pos = sorted(set(range(N_POS)) - set(heldout))

    print(f"train positions: {len(train_pos)}  val: {len(val_pos)}  test: {len(test_pos)}")

    # pos_to_xy.csv
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos_id", "x", "y"])
        for p in range(N_POS):
            w.writerow([p, float(coords[p, 0]), float(coords[p, 1])])
    print(f"Wrote {CSV_PATH}")

    # split txt files
    def write_split(path, positions):
        with open(path, "w") as f:
            for p in positions:
                for s in range(N_STATE):
                    f.write(f"State_{s}_Pos_{p}.npz\n")
        print(f"Wrote {path}  ({len(positions) * N_STATE} files)")

    write_split(TRAIN_TXT, train_pos)
    write_split(VAL_TXT, val_pos)
    write_split(TEST_TXT, test_pos)

    plot_split(coords, train_pos, val_pos, test_pos, corners, neighbors,
               save_path=PLOT_PATH, show=False)


if __name__ == "__main__":
    main()
