import numpy as np
import torch
from torch import nn


class PCADecoderMLP(nn.Module):
 
    def __init__(
        self,
        input_dim,
        mean_scaled_gap,
        components,
        hidden_dim=128,
        dropout=0.10,
    ):
        super().__init__()

        mean_scaled_gap = np.asarray(mean_scaled_gap, dtype=np.float32)
        components = np.asarray(components, dtype=np.float32)

        self.rank_r = components.shape[0]
        self.n_action = components.shape[1]

        # Fixed PCA decoder offset and basis (not trainable, follows .to(device)).
        self.register_buffer(
            "mean_scaled_gap",
            torch.tensor(mean_scaled_gap, dtype=torch.float32),
        )
        self.register_buffer(
            "components",
            torch.tensor(components, dtype=torch.float32),
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, self.rank_r),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        pred_coeff = self.net(x)
        pred_scaled_gap = (
            self.mean_scaled_gap.unsqueeze(0)
            + pred_coeff @ self.components
        )
        return pred_coeff, pred_scaled_gap