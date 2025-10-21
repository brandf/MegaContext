"""
Minimal training loop for the 32→1 GistNet compressor.

Usage:
    uv run python -m tools.train_gistnet --dataset data/sample_text/train.arrow
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.types as pa_types
import torch
import yaml
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from megacontext.gistnet import GistNet, GistNetConfig
from megacontext.runtime import BaseModel
from megacontext.utils import in_notebook_env


def load_train_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if path.suffix.lower() == ".json":
            return json.load(handle)
        raise ValueError(f"Unsupported config extension: {path.suffix}")


class ContextArrowDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        self._mm_file = pa.memory_map(str(shard_path), "rb")
        self._reader = pa_ipc.open_file(self._mm_file)
        self._batches = [
            self._reader.get_batch(i) for i in range(self._reader.num_record_batches)
        ]
        teacher_field = self._reader.schema.field("teacher_context_hidden")
        value_type = teacher_field.type
        while pa_types.is_list(value_type):
            value_type = value_type.value_type
        if pa_types.is_float16(value_type):
            self.teacher_dtype = torch.float16
        elif hasattr(pa_types, "is_bfloat16") and pa_types.is_bfloat16(value_type):
            self.teacher_dtype = torch.bfloat16
        elif pa_types.is_float32(value_type):
            self.teacher_dtype = torch.float32
        else:
            raise ValueError(
                f"Unsupported teacher_context_hidden arrow type: {value_type!r}"
            )
        self._cumulative_rows: list[int] = []
        total = 0
        for batch in self._batches:
            total += batch.num_rows
            self._cumulative_rows.append(total)
        self._length = total
        if self._length == 0:
            self.context_tokens = 0
            self.horizon_tokens = 0
            self.hidden_size = 0
        else:
            first_batch = self._batches[0]
            context_example = first_batch.column("context_input_ids")[0].as_py()
            future_example = first_batch.column("future_input_ids")[0].as_py()
            hidden_example = first_batch.column("teacher_context_hidden")[0].as_py()
            self.context_tokens = len(context_example)
            self.horizon_tokens = len(future_example)
            self.hidden_size = len(hidden_example[0]) if hidden_example else 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        batch_idx = bisect.bisect_right(self._cumulative_rows, index)
        batch = self._batches[batch_idx]
        batch_start = 0 if batch_idx == 0 else self._cumulative_rows[batch_idx - 1]
        row = index - batch_start

        context_tokens = torch.tensor(
            batch.column("context_input_ids")[row].as_py(),
            dtype=torch.long,
        )
        future_tokens = torch.tensor(
            batch.column("future_input_ids")[row].as_py(),
            dtype=torch.long,
        )
        context_hidden = torch.tensor(
            batch.column("teacher_context_hidden")[row].as_py(),
            dtype=self.teacher_dtype,
        )
        future_hidden = torch.tensor(
            batch.column("teacher_future_hidden")[row].as_py(),
            dtype=self.teacher_dtype,
        )
        return {
            "context_tokens": context_tokens,
            "future_tokens": future_tokens,
            "context_hidden": context_hidden,
            "future_hidden": future_hidden,
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
    window_tokens: int,
    objective: str,
    base_model: BaseModel | None = None,
    embed_layer: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    model_dtype = next(model.parameters()).dtype

    context_tokens = batch["context_tokens"].to(device=device)
    future_tokens = batch["future_tokens"].to(device=device)
    context_hidden = batch["context_hidden"].to(device=device, dtype=model_dtype)
    context_len = context_tokens.shape[1]
    horizon_len = future_tokens.shape[1]
    window = min(window_tokens, context_len)
    if window % model.config.block_size != 0:
        raise ValueError(
            "window_tokens must be divisible by model block_size "
            f"({window} vs {model.config.block_size})"
        )

    context_start = context_len - window
    context_slice_ids = context_tokens[:, context_start:]
    context_slice_hidden = context_hidden[:, context_start:, :]
    batch_size, total_tokens, hidden = context_slice_hidden.shape
    block_size = model.config.block_size
    num_blocks = total_tokens // block_size
    reshaped = context_slice_hidden.view(batch_size, num_blocks, block_size, hidden)

    with torch.enable_grad():
        if objective == "pooling_mse":
            targets = reshaped.mean(dim=2).to(model_dtype)
            optimizer.zero_grad(set_to_none=True)
            preds = model(reshaped)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            return {"loss": float(loss.item())}

        if objective == "delta_nll":
            if base_model is None or embed_layer is None:
                raise ValueError(
                    "delta_nll objective requires a base model and embeddings"
                )

            combined_ids = torch.cat([context_slice_ids, future_tokens], dim=1)
            attn_ids = torch.ones_like(combined_ids, device=device)
            labels = torch.full_like(combined_ids, -100, device=device)
            labels[:, window:] = future_tokens
            with torch.no_grad():
                baseline_output = base_model.model(
                    input_ids=combined_ids,
                    attention_mask=attn_ids,
                    labels=labels,
                )
                baseline_loss = baseline_output.loss.detach()

            gists = model(reshaped).to(device=device)
            embed_dtype = embed_layer.weight.dtype
            gists = gists.to(embed_dtype)

            future_embeds = embed_layer(future_tokens).to(device=device)
            future_embeds = future_embeds.to(embed_dtype)
            inputs_embeds = torch.cat([gists, future_embeds], dim=1)

            attn_gist = torch.ones(
                batch_size,
                num_blocks + horizon_len,
                device=device,
                dtype=attn_ids.dtype,
            )
            labels_gist = torch.full(
                (batch_size, num_blocks + horizon_len),
                -100,
                dtype=torch.long,
                device=device,
            )
            labels_gist[:, num_blocks:] = future_tokens

            block_positions = (
                torch.arange(num_blocks, device=device, dtype=torch.long) * block_size
                + context_start
            )
            future_positions = (
                torch.arange(horizon_len, device=device, dtype=torch.long) + context_len
            )
            position_ids = torch.cat(
                [
                    block_positions.unsqueeze(0).expand(batch_size, -1),
                    future_positions.unsqueeze(0).expand(batch_size, -1),
                ],
                dim=1,
            )

            optimizer.zero_grad(set_to_none=True)
            gist_output = base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_gist,
                position_ids=position_ids,
                labels=labels_gist,
            )
            gist_loss = gist_output.loss
            training_loss = gist_loss - baseline_loss
            training_loss.backward()
            optimizer.step()

            return {
                "loss": float(training_loss.item()),
                "gist_loss": float(gist_loss.item()),
                "baseline_loss": float(baseline_loss.item()),
            }

    raise ValueError(f"Unknown training objective: {objective}")


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
        "--objective",
        type=str,
        choices=("pooling_mse", "delta_nll"),
        default=None,
        help="Override training objective (pooling_mse or delta_nll).",
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
    dataset = ContextArrowDataset(args.dataset)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset at {args.dataset} is empty.")
    device = torch.device(train_cfg.get("device", "cpu"))
    batch_size = int(train_cfg.get("batch_size", 4))
    default_window_tokens = int(train_cfg.get("window_tokens", dataset.context_tokens))
    default_max_steps = int(train_cfg.get("max_steps", 100))
    if args.max_steps is not None:
        default_max_steps = int(args.max_steps)
    base_lr = float(train_cfg.get("lr", 1e-3))
    seed = int(train_cfg.get("seed", 0))
    torch.manual_seed(seed)

    allowed_objectives = {"pooling_mse", "delta_nll"}
    phases_cfg = train_cfg.get("phases") if args.objective is None else None
    phases: list[dict[str, Any]] = []
    if phases_cfg:
        if not isinstance(phases_cfg, list):
            raise TypeError("training.phases must be a list of phase dictionaries")
        for idx, phase_cfg in enumerate(phases_cfg):
            if not isinstance(phase_cfg, dict):
                raise TypeError("Each training phase must be a mapping of settings")
            objective = phase_cfg.get("objective")
            if objective not in allowed_objectives:
                raise ValueError(
                    "training.phases[{idx}].objective must be one of "
                    f"{allowed_objectives}"
                )
            phase_steps = int(phase_cfg.get("max_steps", default_max_steps))
            if phase_steps <= 0:
                raise ValueError("phase max_steps must be > 0")
            phase_window = int(phase_cfg.get("window_tokens", default_window_tokens))
            if phase_window <= 0:
                raise ValueError("phase window_tokens must be > 0")
            phase_lr = float(phase_cfg.get("lr", base_lr))
            phases.append(
                {
                    "objective": objective,
                    "max_steps": phase_steps,
                    "window_tokens": phase_window,
                    "lr": phase_lr,
                    "name": phase_cfg.get("name", f"phase-{idx+1}"),
                }
            )
    else:
        objective = args.objective or train_cfg.get("objective")
        if objective is None:
            objective = "delta_nll" if train_cfg.get("base_model") else "pooling_mse"
        if objective not in allowed_objectives:
            raise ValueError(f"Unknown training objective: {objective}")
        phases.append(
            {
                "objective": objective,
                "max_steps": default_max_steps,
                "window_tokens": default_window_tokens,
                "lr": base_lr,
                "name": "phase-1",
            }
        )

    if dataset.hidden_size and "hidden_size" not in model_cfg:
        model_cfg["hidden_size"] = dataset.hidden_size
    if (
        dataset.hidden_size
        and "hidden_size" in model_cfg
        and model_cfg["hidden_size"] != dataset.hidden_size
    ):
        raise ValueError(
            "Model hidden_size does not match dataset teacher hidden size "
            f"({model_cfg['hidden_size']} vs {dataset.hidden_size})"
        )

    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)
    model_config = GistNetConfig(**model_cfg)
    if dataset.context_tokens and dataset.context_tokens % model_config.block_size != 0:
        raise ValueError(
            "context_tokens must be divisible by model block_size "
            f"({dataset.context_tokens} vs {model_config.block_size})"
        )
    for phase in phases:
        if phase["window_tokens"] % model_config.block_size != 0:
            raise ValueError(
                "window_tokens must be divisible by model block_size "
                f"for phase '{phase['name']}'"
            )
    model = GistNet(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=phases[0]["lr"])

    base_cfg = train_cfg.get("base_model") or {}
    requires_base_model = any(p["objective"] == "delta_nll" for p in phases)
    base_model: BaseModel | None = None
    embed_layer: torch.nn.Module | None = None
    if requires_base_model:
        base_model_name = base_cfg.get("name")
        if not base_model_name:
            raise ValueError(
                "training.base_model.name must be provided for delta_nll objective."
            )
        base_model = BaseModel.from_pretrained(
            base_model_name,
            torch_dtype=base_cfg.get("torch_dtype", "auto"),
            device=base_cfg.get("device"),
            trust_remote_code=base_cfg.get("trust_remote_code", False),
        )
        base_model.model.eval()
        base_model.model.requires_grad_(False)
        embed_layer = base_model.model.get_input_embeddings()
    elif base_cfg:
        print(
            "Warning: base_model config provided but no delta_nll phase is defined; "
            "the base model will be ignored."
        )

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
                "device": str(device),
                "seed": seed,
                "phases": phases,
                "base_model": base_cfg,
            },
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
        )

    data_iter = cycle(dataloader)
    all_losses: list[float] = []
    all_gist_losses: list[float] = []
    all_baseline_losses: list[float] = []
    delta_step_indices: list[int] = []
    delta_losses: list[float] = []
    phase_summaries: list[dict[str, Any]] = []
    global_step = 0
    use_tqdm = (not args.no_tqdm) and tqdm is not None

    def set_optimizer_lr(value: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = value

    for _phase_idx, phase in enumerate(phases, start=1):
        phase_name = phase["name"]
        objective = phase["objective"]
        phase_steps = phase["max_steps"]
        window_tokens = phase["window_tokens"]
        set_optimizer_lr(phase["lr"])

        local_losses: list[float] = []
        local_gist: list[float] = []
        local_baseline: list[float] = []

        if use_tqdm:
            assert tqdm is not None  # for mypy
            progress = tqdm(
                range(1, phase_steps + 1),
                desc=f"{phase_name} ({objective})",
                leave=True,
                position=0,
                dynamic_ncols=True,
                mininterval=0.2,
            )
        else:
            progress = range(1, phase_steps + 1)
        print_interval = max(1, phase_steps // 10)

        for local_step in progress:
            batch = next(data_iter)
            metrics = train_step(
                model,
                batch,
                optimizer,
                device=device,
                window_tokens=window_tokens,
                objective=objective,
                base_model=base_model,
                embed_layer=embed_layer,
            )
            loss = metrics["loss"]
            global_step += 1
            all_losses.append(loss)
            local_losses.append(loss)

            log_payload = {
                "loss": loss,
                "global_step": global_step,
                "phase": phase_name,
            }
            if objective == "delta_nll":
                gist_loss = metrics["gist_loss"]
                baseline_loss = metrics["baseline_loss"]
                all_gist_losses.append(gist_loss)
                all_baseline_losses.append(baseline_loss)
                delta_step_indices.append(global_step)
                delta_losses.append(loss)
                local_gist.append(gist_loss)
                local_baseline.append(baseline_loss)
                log_payload.update(
                    {
                        "delta_loss": loss,
                        "gist_loss": gist_loss,
                        "baseline_loss": baseline_loss,
                    }
                )

            if wandb_run is not None:
                wandb_run.log(log_payload)

            if use_tqdm:
                assert tqdm is not None  # for mypy
                if objective == "delta_nll":
                    progress.set_postfix(
                        {
                            "Δloss": f"{loss:.4f}",
                            "gist": f"{local_gist[-1]:.4f}",
                            "base": f"{local_baseline[-1]:.4f}",
                        },
                        refresh=False,
                    )
                else:
                    progress.set_postfix({"loss": f"{loss:.4f}"}, refresh=False)
            elif (
                local_step == 1
                or local_step % print_interval == 0
                or local_step == phase_steps
            ):
                window = local_losses[-min(print_interval, len(local_losses)) :]
                avg_loss = sum(window) / len(window)
                if objective == "delta_nll":
                    print(
                        f"{phase_name} step={local_step}/{phase_steps} "
                        f"Δloss={loss:.6f} gist={local_gist[-1]:.6f} "
                        f"baseline={local_baseline[-1]:.6f} avg_window={avg_loss:.6f}"
                    )
                else:
                    print(
                        f"{phase_name} step={local_step}/{phase_steps} "
                        f"loss={loss:.6f} avg_window={avg_loss:.6f}"
                    )

        if use_tqdm and tqdm is not None:
            progress.close()

        phase_summary: dict[str, Any] = {
            "name": phase_name,
            "objective": objective,
            "steps": phase_steps,
            "loss_final": local_losses[-1],
            "loss_avg": sum(local_losses) / len(local_losses),
            "lr": phase["lr"],
            "window_tokens": window_tokens,
        }
        if objective == "delta_nll":
            phase_summary.update(
                {
                    "gist_loss_final": local_gist[-1],
                    "gist_loss_avg": sum(local_gist) / len(local_gist),
                    "baseline_loss_final": local_baseline[-1],
                    "baseline_loss_avg": sum(local_baseline) / len(local_baseline),
                }
            )
        phase_summaries.append(phase_summary)
        summary_line = (
            f"Completed {phase_name}: objective={objective} steps={phase_steps} "
            f"loss_final={phase_summary['loss_final']:.6f} "
            f"loss_avg={phase_summary['loss_avg']:.6f}"
        )
        if objective == "delta_nll":
            summary_line += (
                f" Δloss_final={phase_summary['loss_final']:.6f}"
                f" gist_final={phase_summary['gist_loss_final']:.6f}"
                f" baseline_final={phase_summary['baseline_loss_final']:.6f}"
            )
        print(summary_line)

    final_loss = all_losses[-1]
    mean_loss = sum(all_losses) / len(all_losses)
    print(
        f"Training finished: total_steps={len(all_losses)} "
        f"loss_final={final_loss:.6f} loss_avg={mean_loss:.6f}"
    )
    if wandb_run is not None:
        final_payload = {
            "loss_final": final_loss,
            "loss_avg": mean_loss,
            "total_steps": len(all_losses),
        }
        if all_gist_losses:
            final_payload.update(
                {
                    "delta_loss_final": delta_losses[-1],
                    "delta_loss_avg": sum(delta_losses) / len(delta_losses),
                    "gist_loss_final": all_gist_losses[-1],
                    "baseline_loss_final": all_baseline_losses[-1],
                }
            )
        wandb_run.log(final_payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gistnet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.__dict__,
            "optimizer_state": optimizer.state_dict(),
            "training": {
                "total_steps": len(all_losses),
                "batch_size": batch_size,
                "phases": phases,
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
            "loss": all_losses,
            "final_loss": final_loss,
            "avg_loss": mean_loss,
            "phase_summaries": phase_summaries,
        }
        if all_gist_losses:
            metrics_payload["gist_loss"] = all_gist_losses
            metrics_payload["delta_step_indices"] = delta_step_indices
            metrics_payload["delta_loss"] = delta_losses
        if all_baseline_losses:
            metrics_payload["baseline_loss"] = all_baseline_losses
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        print(f"Wrote metrics to {args.metrics_path}")

    notebook_context = in_notebook_env()
    want_plot = notebook_context or args.save_plot is not None
    if want_plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            if args.save_plot is not None:
                print(
                    "matplotlib is not available; skipping loss plot.",
                    file=sys.stderr,
                )
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            steps = range(1, len(all_losses) + 1)
            if all_gist_losses and all_baseline_losses:
                ax.plot(steps, all_losses, label="Δ loss (gist - baseline)")
                ax.plot(delta_step_indices, all_gist_losses, label="gist loss")
                ax.plot(delta_step_indices, all_baseline_losses, label="baseline loss")
            else:
                ax.plot(steps, all_losses, label="Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(run_name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            fig.tight_layout()

            if args.save_plot is not None:
                args.save_plot.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.save_plot, dpi=150)
                print(f"Saved loss plot to {args.save_plot}")

            if notebook_context:
                try:  # pragma: no cover - requires IPython
                    from IPython.display import display  # type: ignore

                    display(fig)
                except ImportError:
                    plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
