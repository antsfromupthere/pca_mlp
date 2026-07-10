from pathlib import Path
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

# PCA-MLP:
from src.dataset import RFDataset, make_features
from src.utils import get_device, best_regrets_from_testdataset
from src.model import PCADecoderMLP
from src.augment import add_coordinate_jitter
from src.pca import fit_pca_decoder_targets
from src.loss import pca_coeff_loss
from src.eval import evaluate_model, format_metrics
from data.load_split import load_split
from config.setup import load_config

# ============================================================
# Config
# ============================================================
cfg = load_config("./config/default.yaml")

# System Setup
N_STATE = cfg.system.n_state
N_BS = cfg.system.n_bs
N_UE = cfg.system.n_ue
N_RIS = cfg.system.n_ris
N_ACTION = N_BS * N_UE * N_RIS
SEED = cfg.system.seed

BATCH_SIZE = cfg.learning.batch_size
EPOCHS = cfg.learning.epochs
LR = cfg.learning.lr
WEIGHT_DECAY = cfg.learning.weight_decay
VAL_EVERY = cfg.learning.val_every

drop_out = cfg.learning.dropout
hidden_dim = cfg.learning.hidden_dim

K_LIST = cfg.evaluation.k_list
MONITOR_K = cfg.evaluation.monitor_k

MIN_GAP_DB = cfg.pca.min_gap_db
PCA_N_COMPONENTS = cfg.pca.n_components

ADD_COORD_JITTER = cfg.augmentation.add_coord_jitter
COORD_JITTER_SIGMA = cfg.augmentation.coord_jitter_sigma

device = get_device()

DATA_DIR = Path("data/raw_measurements")
SPLIT_DIR = Path("data/saved_split")

CSV_PATH = SPLIT_DIR / "pos_to_xy.csv"
TRAIN_TXT = SPLIT_DIR / "train.txt"
VAL_TXT = SPLIT_DIR / "val.txt"
TEST_TXT = SPLIT_DIR / "test.txt"


# Main
if __name__ == "__main__":

    torch.manual_seed(SEED)
    np.random.seed(SEED)

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

    # PCA
    pca_info = fit_pca_decoder_targets(
        label_train=y_train,
        label_val=y_val,
        label_test=y_test,
        n_components=PCA_N_COMPONENTS,
        min_gap_db=MIN_GAP_DB,
    )

    # Dataloaders
    train_ds = RFDataset(x_train, y_train, pca_info["train_coeff"])
    val_ds   = RFDataset(x_val,   y_val,   pca_info["val_coeff"])
    test_ds  = RFDataset(x_test,  y_test,  pca_info["test_coeff"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Model & optimizer
    model = PCADecoderMLP(
        input_dim=x_train.shape[1],
        mean_scaled_gap=pca_info["mean_scaled_gap"],
        components=pca_info["components"],
        hidden_dim=hidden_dim,
        dropout=drop_out,
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )


    # Training loop: keep, in memory, the best model per k (lowest val regret@k)
    best_val_metric_by_k = {k: float("inf") for k in K_LIST}
    best_model_state_by_k = {k: None for k in K_LIST}

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_num = 0

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

            bs = x_batch.shape[0]
            train_loss_sum += float(loss.item()) * bs
            train_num += bs

        train_loss = train_loss_sum / train_num

        if epoch == 1 or epoch % VAL_EVERY == 0:
            val_metrics = evaluate_model(
                model=model, loader=val_loader, k_list=K_LIST, device=device,
            )

            print(
                f"Epoch {epoch:04d} | "
                # f"lr={current_lr:.2e} | "
                f"train_loss={train_loss:.6f} | "
                + format_metrics(prefix="val", metrics=val_metrics, k_list=K_LIST)
            )

            for k in K_LIST:
                monitor_value_k = val_metrics[f"regret@{k}"]

                if monitor_value_k < best_val_metric_by_k[k]:
                    best_val_metric_by_k[k] = monitor_value_k
                    best_model_state_by_k[k] = copy.deepcopy(model.state_dict())

    final_test_by_selected_k = {}

    for selected_k in K_LIST:
        if best_model_state_by_k[selected_k] is None:
            print(f"No saved model for regret@{selected_k}")
            continue

        model.load_state_dict(best_model_state_by_k[selected_k])

        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            k_list=K_LIST,
            device=device,
        )

        final_test_by_selected_k[selected_k] = test_metrics

    # ---- final test ---- #
    best_power_drop, _ = best_regrets_from_testdataset(final_test_by_selected_k, K_LIST)
    power_drop_db = [round(best_power_drop[f"regret@{k}"], 4) for k in K_LIST]

    print(f"\n----- Final test (n_comp={PCA_N_COMPONENTS}) -----")
    print(f"Power drop (dB) @k={K_LIST} : {power_drop_db}")

