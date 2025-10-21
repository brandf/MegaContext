"""
Minimal training loop for the 32→1 GistNet compressor.

Usage:
    uv run python -m tools.train_gistnet --dataset data/sample_text/train.arrow
"""

from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Any

import pyarrow.ipc as pa_ipc
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Dataset

from megacontext.gistnet import GistNet, GistNetConfig


def load_train_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if path.suffix.lower() == ".json":
            return json.load(handle)
        raise ValueError(f"Unsupported config extension: {path.suffix}")


class GistArrowDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        with pa_ipc.open_file(shard_path.open("rb")) as reader:
            table = reader.read_all()
        self.tokens = torch.tensor(
            table.column("input_ids").to_pylist(), dtype=torch.long
        )
        self.attention_mask = torch.tensor(
            table.column("attention_mask").to_pylist(), dtype=torch.long
        )
        self.teacher_hidden = torch.tensor(
            table.column("teacher_hidden").to_pylist(), dtype=torch.float32
        )
        self.gist_target = torch.tensor(
            table.column("gist_target").to_pylist(), dtype=torch.float32
        )

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "attention_mask": self.attention_mask[index],
            "teacher_hidden": self.teacher_hidden[index],
            "gist_target": self.gist_target[index],
        }


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_step(
    model: GistNet,
    batch: dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    *,
    device: torch.device,
) -> float:
    model.train()
    embeddings = batch["teacher_hidden"].to(device)  # [B, block, hidden]
    inputs = embeddings.unsqueeze(1)  # [B, 1, block, hidden]
    targets = batch["gist_target"].to(device)  # [B, hidden]

    optimizer.zero_grad(set_to_none=True)
    preds = model(inputs)  # [B, 1, hidden]
    preds = preds[:, 0, :]
    loss = torch.nn.functional.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GistNet compressor.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to an Arrow shard produced by tools.prepare_dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config describing model and training hyperparameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gistnet"),
        help="Directory where checkpoints will be stored.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps from config.",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    dataset = GistArrowDataset(args.dataset)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset at {args.dataset} is empty.")

    device = torch.device(train_cfg.get("device", "cpu"))
    batch_size = int(train_cfg.get("batch_size", 4))
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 100))
    lr = float(train_cfg.get("lr", 1e-3))
    seed = int(train_cfg.get("seed", 0))
    torch.manual_seed(seed)

    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)
    model_config = GistNetConfig(**model_cfg)
    model = GistNet(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    data_iter = cycle(dataloader)
    losses: list[float] = []
    for step in range(1, max_steps + 1):
        batch = next(data_iter)
        loss = train_step(model, batch, optimizer, device=device)
        losses.append(loss)
        if step % max(1, max_steps // 10) == 0 or step == max_steps:
            avg_loss = sum(losses[-10:]) / min(len(losses), 10)
            print(f"step={step} loss={loss:.6f} avg_loss_10={avg_loss:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gistnet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.__dict__,
            "optimizer_state": optimizer.state_dict(),
            "training": {
                "steps": max_steps,
                "batch_size": batch_size,
                "lr": lr,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
