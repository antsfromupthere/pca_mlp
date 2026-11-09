from pathlib import Path
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

# PCA-MLP
from src.dataset import RFDataset, make_features
from src.utils import get_device
from src.model import PCADecoderMLP
from src.augment import add_coordinate_jitter
from src.pca import fit_pca_decoder_targets
from src.loss import pca_coeff_loss
from src.eval import evaluate_model
from data.load_split import load_split
from config.setup import load_config


# ============================================================
# Config
# ============================================================
cfg = load_config("./config/default.yaml")

N_STATE = cfg.system.n_state
N_BS = cfg.system.n_bs
N_UE = cfg.system.n_ue
N_RIS = cfg.system.n_ris
N_ACTION = N_BS * N_UE * N_RIS

BATCH_SIZE = cfg.learning.batch_size
EPOCHS = cfg.learning.epochs
LR = cfg.learning.lr
VAL_EVERY = cfg.learning.val_every

drop_out = cfg.learning.dropout
hidden_dim = cfg.learning.hidden_dim

K_LIST = cfg.evaluation.k_list

MIN_GAP_DB = cfg.pca.min_gap_db
RANK_R = cfg.pca.rank_r

ADD_COORD_JITTER = cfg.augmentation.add_coord_jitter
COORD_JITTER_SIGMA = cfg.augmentation.coord_jitter_sigma

device = get_device()

DATA_DIR = Path("data/raw_measurements")
SPLIT_DIR = Path("data/saved_blocked_split")
CSV_PATH = SPLIT_DIR / "pos_to_xy.csv"

FOLDS = [f"B{b}" for b in range(1, 11)]     # folds to run
SEEDS = [0, 1, 2, 3, 4]                     # training seeds


# Main
if __name__ == "__main__":
    print(f"device: {device}")

    power_drop_by_fold = {}

    for fold in FOLDS:
        TRAIN_TXT = SPLIT_DIR / fold / "train.txt"
        VAL_TXT = SPLIT_DIR / fold / "val.txt"
        TEST_TXT = SPLIT_DIR / fold / "test.txt"

        # Load split
        xy_train, s_train, y_train = load_split(
            txt_path=TRAIN_TXT, csv_path=CSV_PATH,
            data_measurement_dir=DATA_DIR, n_ACTION=N_ACTION,
        )
        xy_val, s_val, y_val = load_split(
            txt_path=VAL_TXT, csv_path=CSV_PATH,
            data_measurement_dir=DATA_DIR, n_ACTION=N_ACTION,
        )
        xy_test, s_test, y_test = load_split(
            txt_path=TEST_TXT, csv_path=CSV_PATH,
            data_measurement_dir=DATA_DIR, n_ACTION=N_ACTION,
        )

        x_train, feature_stats = make_features(
            features=xy_train, label=s_train, n_STATE=N_STATE, stats=None,
        )
        x_val, _ = make_features(
            features=xy_val, label=s_val, n_STATE=N_STATE, stats=feature_stats,
        )
        x_test, _ = make_features(
            features=xy_test, label=s_test, n_STATE=N_STATE, stats=feature_stats,
        )

        # PCA fitted on the training split of this fold
        pca_info = fit_pca_decoder_targets(
            label_train=y_train,
            label_val=y_val,
            label_test=y_test,
            rank_r=RANK_R,
            min_gap_db=MIN_GAP_DB,
        )
        r = pca_info["components"].shape[0]
        cum_explained_variance = float(pca_info["explained_variance_cum"][r - 1]) # Python’s 0-based indexing.
        print(f"\n----- {fold}: rank r={r}, cumulative explained variance={100 * cum_explained_variance:.2f}% -----")

        # Dataloaders
        train_ds = RFDataset(x_train, y_train, pca_info["train_coeff"])
        val_ds = RFDataset(x_val, y_val, pca_info["val_coeff"])
        test_ds = RFDataset(x_test, y_test, pca_info["test_coeff"])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        power_drop_by_seed = []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Model & optimizer
            model = PCADecoderMLP(
                input_dim=x_train.shape[1],
                mean_scaled_gap=pca_info["mean_scaled_gap"],
                components=pca_info["components"],
                hidden_dim=hidden_dim,
                dropout=drop_out,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # Select one checkpoint per k using validation regret@k only.
            best_val_metric_by_k = {k: float("inf") for k in K_LIST}
            best_model_state_by_k = {k: None for k in K_LIST}

            for epoch in range(1, EPOCHS + 1):
                model.train()

                for x_batch, power_batch, target_coeff_batch in train_loader:
                    x_batch = x_batch.to(device)
                    target_coeff_batch = target_coeff_batch.to(device)

                    if ADD_COORD_JITTER:
                        x_input = add_coordinate_jitter(x_batch, sigma=COORD_JITTER_SIGMA)
                    else:
                        x_input = x_batch

                    pred_coeff, _ = model(x_input)
                    loss = pca_coeff_loss(pred_coeff, target_coeff_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if epoch == 1 or epoch % VAL_EVERY == 0:
                    val_metrics = evaluate_model(
                        model=model, loader=val_loader, k_list=K_LIST, device=device,
                    )

                    for k in K_LIST:
                        monitor_value_k = val_metrics[f"regret@{k}"]
                        if monitor_value_k < best_val_metric_by_k[k]:
                            best_val_metric_by_k[k] = monitor_value_k
                            best_model_state_by_k[k] = copy.deepcopy(model.state_dict())

            # Evaluate every validation-selected checkpoint on the test set.
            final_test_by_selected_k = {}

            for selected_k in K_LIST:
                model.load_state_dict(best_model_state_by_k[selected_k])
                final_test_by_selected_k[selected_k] = evaluate_model(
                    model=model, loader=test_loader, k_list=K_LIST, device=device,
                )

            # Test regret@k of the checkpoint selected by validation regret@k.
            power_drop_db = [final_test_by_selected_k[k][f"regret@{k}"] for k in K_LIST]
            power_drop_by_seed.append(power_drop_db)

            print(f"{fold} | seed={seed} | Power drop (dB) @k={K_LIST} : "
                  f"{[round(v, 4) for v in power_drop_db]}", flush=True)

        # Average over the seeds of this fold.
        power_drop_by_fold[fold] = np.mean(power_drop_by_seed, axis=0)
        print(f"{fold} | average over {len(SEEDS)} seeds | Power drop (dB) @k={K_LIST} : "
              f"{[round(float(v), 4) for v in power_drop_by_fold[fold]]}", flush=True)

    # Average over the folds and seeds.
    power_drop_avg = np.mean(list(power_drop_by_fold.values()), axis=0)
    print(f"\n----- Average over {len(FOLDS)} folds and {len(SEEDS)} seeds -----")
    print(f"Power drop (dB) @k={K_LIST} : {[round(float(v), 4) for v in power_drop_avg]}")
