"""Shared Lightning base for all three forecasters.

Every model emits ``(batch, horizon, n_quantiles)`` and trains on the quantile
loss, so TCN / LSTM / TFT are directly comparable. Subclasses only implement
``forward(past, future)`` and build their own layers.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch

from smp.losses import quantile_loss


class BaseForecaster(pl.LightningModule):
    """Common training/validation/optim logic + quantile output contract."""

    def __init__(
        self,
        n_past_features: int,
        n_future_features: int,
        horizon: int,
        quantiles: list[float],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_past_features = n_past_features
        self.n_future_features = n_future_features
        self.horizon = horizon
        self.quantiles = list(quantiles)
        self.n_quantiles = len(self.quantiles)
        self.lr = lr
        self.weight_decay = weight_decay

    # Subclasses implement this and return (batch, horizon, n_quantiles).
    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        preds = self(batch["past"], batch["future"])
        loss = quantile_loss(preds, batch["target"], self.quantiles)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage == "val"),
            batch_size=batch["target"].shape[0],
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    @torch.no_grad()
    def predict_dataset(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        """Run the model over a dataloader, returning ``(preds, targets)``.

        Both arrays are in standardized target space; inverse-transform with the
        datamodule before computing metrics.
        """
        self.eval()
        preds, targets = [], []
        device = next(self.parameters()).device
        for batch in dataloader:
            p = self(batch["past"].to(device), batch["future"].to(device))
            preds.append(p.cpu().numpy())
            targets.append(batch["target"].numpy())
        if not preds:
            return (
                np.empty((0, self.horizon, self.n_quantiles), np.float32),
                np.empty((0, self.horizon), np.float32),
            )
        return np.concatenate(preds), np.concatenate(targets)


class QuantileHead(torch.nn.Module):
    """Map a feature vector to ``(horizon, n_quantiles)`` predictions."""

    def __init__(self, in_features: int, horizon: int, n_quantiles: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.proj = torch.nn.Linear(in_features, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        return out.view(-1, self.horizon, self.n_quantiles)
