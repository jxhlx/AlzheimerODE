from __future__ import annotations

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONNECTOME_PATH = PROJECT_ROOT / "data" / "average_connectome.pt"


def load_average_connectome(
    connectome_path: str | Path = DEFAULT_CONNECTOME_PATH,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(connectome_path)
    if not path.exists():
        raise FileNotFoundError(f"missing average connectome data: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload["edge_index"].to(device), payload["edge_weight"].to(device)
