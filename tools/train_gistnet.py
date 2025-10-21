"""
Minimal training loop for the 32→1 GistNet compressor.

Usage:
    uv run python -m tools.train_gistnet --dataset data/sample_text/train.arrow
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

import pyarrow.ipc as pa_ipc
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

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
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars (useful for non-interactive logs).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires `wandb` package).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="megacontext-poc",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity/organization.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional explicit Weights & Biases run name.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Optional path to write training metrics (JSON).",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path to save a loss curve (requires matplotlib).",
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

    run_name = (
        args.wandb_run_name
        or train_cfg.get("run_name")
        or f"gistnet-{Path(args.dataset).stem}"
    )
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Weights & Biases logging requested but the `wandb` package is "
                "not installed. Install with `pip install wandb` or drop "
                "`--use-wandb`."
            ) from exc

        wandb_config = {
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
            "model": model_cfg,
            "training": {
                "batch_size": batch_size,
                "lr": lr,
                "max_steps": max_steps,
                "device": str(device),
                "seed": seed,
            },
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
        )

    data_iter = cycle(dataloader)
    losses: list[float] = []
    use_tqdm = (not args.no_tqdm) and tqdm is not None
    step_iter = range(1, max_steps + 1)
    progress = (
        tqdm(step_iter, desc="training", leave=False) if use_tqdm else step_iter  # type: ignore[arg-type]
    )
    print_interval = max(1, max_steps // 10)
    for step in progress:
        batch = next(data_iter)
        loss = train_step(model, batch, optimizer, device=device)
        losses.append(loss)
        if wandb_run is not None:
            wandb_run.log({"loss": loss, "step": step})

        if use_tqdm:
            assert tqdm is not None  # for mypy
            progress.set_postfix({"loss": f"{loss:.4f}"}, refresh=False)
        elif step == 1 or step % print_interval == 0 or step == max_steps:
            window = losses[-min(print_interval, len(losses)) :]
            avg_loss = sum(window) / len(window)
            print(f"step={step} loss={loss:.6f} avg_window={avg_loss:.6f}")

    final_loss = losses[-1]
    mean_loss = sum(losses) / len(losses)
    summary_line = (
        f"Training finished: steps={max_steps} final_loss={final_loss:.6f} "
        f"avg_loss={mean_loss:.6f}"
    )
    print(summary_line)
    if wandb_run is not None:
        wandb_run.log({"final_loss": final_loss, "avg_loss": mean_loss})

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

    if wandb_run is not None:  # pragma: no cover - optional dependency
        wandb_run.save(str(checkpoint_path))
        wandb_run.finish()

    if args.metrics_path is not None:
        metrics_payload = {
            "loss": losses,
            "final_loss": final_loss,
            "avg_loss": mean_loss,
        }
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        print(f"Wrote metrics to {args.metrics_path}")

    if args.save_plot is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            print(
                "matplotlib is not available; skipping loss plot.",
                file=sys.stderr,
            )
        else:
            args.save_plot.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(losses) + 1), losses, label="training loss")
            plt.xlabel("Step")
            plt.ylabel("MSE loss")
            plt.title(run_name)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            print(f"Saved loss plot to {args.save_plot}")


if __name__ == "__main__":
    main()
