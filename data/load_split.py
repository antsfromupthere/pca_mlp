from pathlib import Path
import re
import csv
import numpy as np


_FNAME_RE = re.compile(r"State_(\d+)_Pos_(\d+)\.npz")

def load_pos_to_xy(csv_path):
    """Return dict {pos_id: (x, y)}."""
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["pos_id"])] = (float(row["x"]), float(row["y"]))
    return out


def load_split(txt_path, csv_path, data_measurement_dir, n_ACTION):
    """
    Read filenames from `txt_path`, load each npz, return (xy, states, powers).
    Filename format: State_{s}_Pos_{p}.npz
    """
    pos_to_xy = load_pos_to_xy(csv_path)
    xy, states, powers = [], [], []
    with open(txt_path) as f:
        names = [ln.strip() for ln in f if ln.strip()]

    for name in names:
        m = _FNAME_RE.match(name)
        if not m:
            raise ValueError(f"Bad filename in {txt_path}: {name}")
        s, p = int(m.group(1)), int(m.group(2))

        power = np.asarray(np.load(data_measurement_dir / name)["received_power"], dtype=np.float32).reshape(-1)
        if power.shape[0] != n_ACTION:
            raise ValueError(f"{name}: expected {n_ACTION} actions, got {power.shape[0]}")
        # linear to dB
        power = 10.0 * np.log10(np.maximum(power, 1e-30))

        xy.append(pos_to_xy[p])
        states.append(s)
        powers.append(power)

    return (
        np.asarray(xy, dtype=np.float32),
        np.asarray(states, dtype=np.int64),
        np.stack(powers, axis=0).astype(np.float32),
    )

