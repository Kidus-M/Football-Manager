"""Player embeddings: a small autoencoder over per-90 output.

The encoder squeezes a player's eight per-90 numbers into a few dimensions
that capture *how* he contributes (shooter, creator, build-up player...).
Players close together in that space play alike, whatever their league or
listed position, which is what "players like X" searches need.

Trained on every qualifying player-season in the history; the trained model
and its scaler are saved under data/models/ and reused by later commands.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from recruit_intel import DATA_DIR
from recruit_intel.models.dataset import FEATURES, transform

MODEL_DIR = DATA_DIR / "models"
LATENT = 4
SEED = 7


class AutoEncoder(nn.Module):
    def __init__(self, n_in: int = len(FEATURES), latent: int = LATENT) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 16), nn.GELU(), nn.Linear(16, latent)
        )  # fmt: skip
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16), nn.GELU(), nn.Linear(16, n_in)
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class Embedder:
    model: AutoEncoder
    mean: np.ndarray
    std: np.ndarray

    def _scale(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.from_numpy((transform(df) - self.mean) / self.std)

    def embed(self, df: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            return self.model.encoder(self._scale(df)).numpy()

    def reconstruction_error(self, df: pd.DataFrame) -> float:
        x = self._scale(df)
        with torch.no_grad():
            return float(nn.functional.mse_loss(self.model(x), x))

    def save(self, directory: Path = MODEL_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "player_ae.pt")
        meta = {"features": FEATURES, "latent": LATENT,
                "mean": self.mean.tolist(), "std": self.std.tolist()}  # fmt: skip
        (directory / "player_ae.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> "Embedder":
        meta_path = directory / "player_ae.json"
        if not meta_path.exists():
            raise FileNotFoundError("no trained embedding model; run `recruit train`")
        meta = json.loads(meta_path.read_text())
        if meta["features"] != FEATURES:
            raise ValueError("embedding model was trained on other features; retrain it")
        model = AutoEncoder(len(FEATURES), meta["latent"])
        model.load_state_dict(torch.load(directory / "player_ae.pt", weights_only=True))
        model.eval()
        return cls(model, np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32))


def train(df: pd.DataFrame, epochs: int = 300, lr: float = 3e-3) -> tuple[Embedder, dict]:
    """Fit on 85% of rows; report reconstruction error on the held-out 15%."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x = transform(df)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    idx = rng.permutation(len(x))
    cut = int(len(x) * 0.85)
    train_df, test_df = df.iloc[idx[:cut]], df.iloc[idx[cut:]]

    model = AutoEncoder()
    emb = Embedder(model, mean.astype(np.float32), std.astype(np.float32))
    xt = emb._scale(train_df)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        for batch in torch.randperm(len(xt)).split(256):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[batch]), xt[batch])
            loss.backward()
            opt.step()
    model.eval()
    metrics = {
        "rows": len(df),
        "train_mse": emb.reconstruction_error(train_df),
        "test_mse": emb.reconstruction_error(test_df),
        # Baseline: predicting every feature as its mean has MSE 1.0 on scaled data.
        "baseline_mse": 1.0,
    }
    return emb, metrics


def nearest(
    vectors: np.ndarray, target: int, k: int = 10, mask: np.ndarray | None = None
) -> list[tuple[int, float]]:
    """Indices of the k rows closest to row `target` (Euclidean), excluding it.
    `mask` restricts which rows may be returned."""
    d = np.linalg.norm(vectors - vectors[target], axis=1)
    d[target] = np.inf
    if mask is not None:
        d[~mask] = np.inf
    order = np.argsort(d)[:k]
    return [(int(i), float(d[i])) for i in order if np.isfinite(d[i])]
