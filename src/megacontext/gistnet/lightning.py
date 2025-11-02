"""
PyTorch Lightning utilities for training the GistNet compression model.

The legacy CLI trainer lived in ``tools/train_gistnet.py`` and mixed dataset IO,
optimizer management, logging, and plotting in one script. This module keeps the
model-focused pieces inside the `megacontext` package so notebooks can assemble
experiments with minimal boilerplate while reusing the tested dataset loader and
loss computation logic.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.types as pa_types
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback  # type: ignore
from lightning.pytorch.loggers import Logger  # type: ignore
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from megacontext.runtime import BaseModel

from .model import GistNet, GistNetConfig


def _ensure_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        msg = f"Dataset path does not exist: {resolved}"
        raise FileNotFoundError(msg)
    return resolved


class ContextArrowDataset(Dataset):
    """
    Memory-mapped Arrow dataset emitted by ``tools.prepare_dataset``.

    The dataset exposes the same fields that ``train_gistnet.py`` consumed:
    token ids, teacher hidden states for the retained context, and the future
    horizon targets. The tensors are materialised per row to keep Lightning's
    default collate function working without a custom collator.
    """

    def __init__(self, shard_path: Path | str) -> None:
        self.path = _ensure_path(shard_path)
        self._mm_file = pa.memory_map(str(self.path), "rb")
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


class GistNetDataModule(pl.LightningDataModule):
    """Lightweight LightningDataModule that wraps ContextArrowDataset."""

    def __init__(
        self,
        train_path: Path | str,
        *,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.train_path = _ensure_path(train_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._train_dataset: ContextArrowDataset | None = None

    @property
    def train_dataset(self) -> ContextArrowDataset:
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before requesting datasets.")
        return self._train_dataset

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = ContextArrowDataset(self.train_path)

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


@dataclass(slots=True)
class BaseModelSettings:
    """Hugging Face model metadata used when ``delta_nll`` phases are enabled."""

    name: str
    torch_dtype: str | torch.dtype | None = "auto"
    device: str | None = None
    trust_remote_code: bool = False
    load_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | BaseModelSettings | None,
    ) -> BaseModelSettings:
        if isinstance(data, cls):
            return data
        if data is None:
            raise ValueError(
                "BaseModelSettings.from_dict requires a mapping or instance"
            )
        payload = dict(data)

        allowed = {f.name for f in fields(cls)}
        base_payload: dict[str, Any] = {}
        extra_kwargs = payload.get("load_kwargs")

        model_name = payload.get("model_name")
        if model_name is not None:
            base_payload["name"] = model_name

        for key, value in payload.items():
            if key in allowed:
                base_payload[key] = value

        if "name" not in base_payload and "name" in payload:
            base_payload["name"] = payload["name"]

        instance = cls(**base_payload)
        if extra_kwargs:
            instance.load_kwargs.update(extra_kwargs)
        return instance


@dataclass(slots=True)
class GistNetTrainingPhase:
    """
    Piecewise-constant phase controlling the objective, window, and learning rate.
    """

    objective: str
    max_steps: int
    window_tokens: int
    lr: float
    name: str = "phase"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective"] = self.objective
        return payload

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | GistNetTrainingPhase,
    ) -> GistNetTrainingPhase:
        if isinstance(data, cls):
            return data
        return cls(**dict(data))


@dataclass(slots=True)
class GistNetTrainingConfig:
    """
    Container for experiment-level hyperparameters passed in from notebooks.
    """

    batch_size: int = 4
    seed: int = 0
    phases: Sequence[GistNetTrainingPhase] = field(default_factory=tuple)
    base_model: BaseModelSettings | None = None
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    weight_decay: float = 0.01
    accumulate_grad_batches: int = 1
    precision: str | int = "bf16-mixed"
    gradient_clip_val: float | None = None
    log_every_n_steps: int = 50

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | GistNetTrainingConfig,
    ) -> GistNetTrainingConfig:
        if isinstance(data, cls):
            return data

        payload = dict(data)
        phases_raw = payload.pop("phases", ())
        phases = tuple(
            GistNetTrainingPhase.from_dict(phase_raw) for phase_raw in phases_raw
        )

        base_model_raw = payload.pop("base_model", None)
        base_model = (
            BaseModelSettings.from_dict(base_model_raw)
            if base_model_raw is not None
            else None
        )

        return cls(phases=phases, base_model=base_model, **payload)


class GistNetLightningModule(pl.LightningModule):
    """
    LightningModule wrapping the GistNet compression model.

    The module reproduces the ``pooling_mse`` and ``delta_nll`` objectives from the
    legacy trainer while letting Lightning handle device placement, precision, and
    checkpointing.
    """

    valid_objectives = {"pooling_mse", "delta_nll"}

    def __init__(
        self,
        model_config: GistNetConfig,
        phases: Sequence[GistNetTrainingPhase],
        *,
        base_model_config: BaseModelSettings | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        if not phases:
            raise ValueError("At least one training phase must be provided.")
        for phase in phases:
            if phase.objective not in self.valid_objectives:
                msg = f"Unsupported objective '{phase.objective}' in phase {phase.name}"
                raise ValueError(msg)

        self.model = GistNet(model_config)
        self.phases = list(phases)
        self.weight_decay = weight_decay
        self.requires_base_model = any(
            phase.objective == "delta_nll" for phase in self.phases
        )
        if self.requires_base_model and base_model_config is None:
            raise ValueError(
                "delta_nll objective requires a BaseModelSettings configuration."
            )
        self.base_model_config = base_model_config
        self.base_model: BaseModel | None = None
        self.embed_layer: nn.Module | None = None

        self.phase_boundaries: list[int] = []
        cumulative = 0
        for phase in self.phases:
            if phase.max_steps <= 0:
                raise ValueError(f"{phase.name} max_steps must be > 0")
            if phase.window_tokens <= 0:
                raise ValueError(f"{phase.name} window_tokens must be > 0")
            cumulative += phase.max_steps
            self.phase_boundaries.append(cumulative)
        self.total_steps = cumulative
        self._model_dtype: torch.dtype | None = None
        self._current_phase_index = -1

        self.save_hyperparameters(
            {
                "model_config": model_config.__dict__,
                "phases": [phase.as_dict() for phase in self.phases],
                "weight_decay": weight_decay,
                "base_model": (
                    asdict(base_model_config) if base_model_config is not None else None
                ),
            }
        )

    @property
    def model_dtype(self) -> torch.dtype:
        if self._model_dtype is None:
            self._model_dtype = next(self.model.parameters()).dtype
        return self._model_dtype

    def setup(self, stage: str | None = None) -> None:
        if self.requires_base_model and self.base_model is None:
            cfg = self.base_model_config
            assert cfg is not None  # for mypy
            base_model = BaseModel.from_pretrained(
                cfg.name,
                torch_dtype=cfg.torch_dtype,
                device=cfg.device,
                trust_remote_code=cfg.trust_remote_code,
                **cfg.load_kwargs,
            )
            base_model.model.eval()
            base_model.model.requires_grad_(False)
            embed_layer = base_model.model.get_input_embeddings()
            if embed_layer is None:
                raise RuntimeError("Base model does not expose input embeddings.")
            self.base_model = base_model
            self.embed_layer = embed_layer

    # Lightning API -----------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        base_lr = self.phases[0].lr
        optimizer = AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=self.weight_decay,
        )

        if len(self.phases) == 1:
            return optimizer

        def lr_lambda(step: int) -> float:
            idx = bisect.bisect_right(self.phase_boundaries, step)
            idx = min(idx, len(self.phases) - 1)
            target_lr = self.phases[idx].lr
            return target_lr / base_lr

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        phase_index = self._phase_index_for_step(int(self.global_step))
        phase = self.phases[phase_index]
        loss, metrics = self._compute_phase_loss(batch, phase)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch["context_tokens"].shape[0],
        )
        if metrics:
            for key, value in metrics.items():
                self.log(
                    f"train/{key}",
                    value,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch["context_tokens"].shape[0],
                )
        if phase_index != self._current_phase_index:
            self._current_phase_index = phase_index
            self.log(
                "train/phase_index",
                float(phase_index),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        return loss

    # Internal helpers --------------------------------------------------------------

    def _phase_index_for_step(self, step: int) -> int:
        idx = bisect.bisect_right(self.phase_boundaries, step)
        return min(idx, len(self.phases) - 1)

    def _slice_context(
        self,
        context_tokens: Tensor,
        context_hidden: Tensor,
        window_tokens: int,
    ) -> tuple[Tensor, Tensor, int]:
        context_len = context_tokens.shape[1]
        window = min(window_tokens, context_len)
        block_size = self.model.config.block_size
        if window % block_size != 0:
            msg = f"window_tokens {window} must be divisible by block_size {block_size}"
            raise ValueError(msg)
        context_start = context_len - window
        context_slice_ids = context_tokens[:, context_start:]
        context_slice_hidden = context_hidden[:, context_start:, :]
        return context_slice_ids, context_slice_hidden, context_start

    def _reshape_blocks(
        self,
        hidden: Tensor,
    ) -> tuple[Tensor, int]:
        batch_size, total_tokens, hidden_size = hidden.shape
        block_size = self.model.config.block_size
        if total_tokens % block_size != 0:
            msg = (
                f"context window length {total_tokens} must align with block_size "
                f"{block_size}"
            )
            raise ValueError(msg)
        num_blocks = total_tokens // block_size
        reshaped = hidden.view(batch_size, num_blocks, block_size, hidden_size)
        return reshaped, num_blocks

    def _compute_phase_loss(
        self,
        batch: dict[str, Tensor],
        phase: GistNetTrainingPhase,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = self.device
        context_tokens = batch["context_tokens"].to(device=device)
        future_tokens = batch["future_tokens"].to(device=device)
        context_hidden = batch["context_hidden"].to(
            device=device, dtype=self.model_dtype
        )

        context_slice_ids, context_slice_hidden, context_start = self._slice_context(
            context_tokens,
            context_hidden,
            phase.window_tokens,
        )
        reshaped, num_blocks = self._reshape_blocks(context_slice_hidden)

        if phase.objective == "pooling_mse":
            targets = reshaped.mean(dim=2)
            preds = self.model(reshaped)
            loss = F.mse_loss(preds, targets)
            return loss, {}

        if phase.objective == "delta_nll":
            if self.base_model is None or self.embed_layer is None:
                raise RuntimeError("Base model must be initialised for delta_nll.")
            base_model = self.base_model

            batch_size, horizon_len = future_tokens.shape
            block_size = self.model.config.block_size
            context_len = context_tokens.shape[1]

            combined_ids = torch.cat([context_slice_ids, future_tokens], dim=1)
            attn_ids = torch.ones_like(combined_ids, device=device)
            labels = torch.full_like(combined_ids, -100, device=device)
            labels[:, context_slice_ids.shape[1] :] = future_tokens
            with torch.no_grad():
                baseline_output = base_model.model(
                    input_ids=combined_ids,
                    attention_mask=attn_ids,
                    labels=labels,
                )
                baseline_loss = baseline_output.loss.detach()

            gists = self.model(reshaped)
            embed_layer = self.embed_layer
            assert embed_layer is not None  # for mypy
            embed_dtype = embed_layer.weight.dtype
            gists = gists.to(dtype=embed_dtype)

            future_embeds = embed_layer(future_tokens).to(device=device)
            future_embeds = future_embeds.to(dtype=embed_dtype)
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

            gist_output = base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_gist,
                position_ids=position_ids,
                labels=labels_gist,
            )
            gist_loss = gist_output.loss
            delta_loss = gist_loss - baseline_loss
            metrics = {
                "delta_loss": delta_loss.detach(),
                "gist_loss": gist_loss.detach(),
                "baseline_loss": baseline_loss,
            }
            return delta_loss, metrics

        raise RuntimeError(f"Unhandled objective: {phase.objective}")


def build_gistnet_experiment(
    *,
    dataset_path: Path | str,
    model_config: GistNetConfig,
    training: GistNetTrainingConfig,
    trainer_kwargs: dict[str, Any] | None = None,
    logger: Logger | Sequence[Logger] | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> tuple[pl.Trainer, GistNetLightningModule, GistNetDataModule]:
    """
    Convenience helper that notebooks can call to assemble a full Lightning run.

    Returns the trainer, module, and data module so callers can tweak callbacks,
    loggers, or monitoring hooks before starting ``trainer.fit``.
    """

    phases: Sequence[GistNetTrainingPhase]
    if training.phases:
        phases = training.phases
    else:
        default_phase = GistNetTrainingPhase(
            objective="pooling_mse",
            max_steps=100,
            window_tokens=1024,
            lr=1e-3,
            name="phase-1",
        )
        phases = (default_phase,)

    pl.seed_everything(training.seed, workers=True)

    module = GistNetLightningModule(
        model_config=model_config,
        phases=phases,
        base_model_config=training.base_model,
        weight_decay=training.weight_decay,
    )
    data_module = GistNetDataModule(
        dataset_path,
        batch_size=training.batch_size,
        shuffle=training.shuffle,
        num_workers=training.num_workers,
        pin_memory=training.pin_memory,
    )

    default_trainer_kwargs: dict[str, Any] = {
        "max_steps": module.total_steps,
        "accumulate_grad_batches": training.accumulate_grad_batches,
        "precision": training.precision,
        "log_every_n_steps": training.log_every_n_steps,
    }
    if training.gradient_clip_val is not None:
        default_trainer_kwargs["gradient_clip_val"] = training.gradient_clip_val

    if trainer_kwargs:
        default_trainer_kwargs.update(trainer_kwargs)

    if logger is not None:
        default_trainer_kwargs["logger"] = logger
    else:
        default_trainer_kwargs.setdefault("logger", True)

    if callbacks:
        existing_callbacks = list(default_trainer_kwargs.get("callbacks", []))
        existing_callbacks.extend(callbacks)
        default_trainer_kwargs["callbacks"] = existing_callbacks

    trainer = pl.Trainer(**default_trainer_kwargs)
    return trainer, module, data_module


__all__ = [
    "BaseModelSettings",
    "ContextArrowDataset",
    "GistNetDataModule",
    "GistNetLightningModule",
    "GistNetTrainingConfig",
    "GistNetTrainingPhase",
    "build_gistnet_experiment",
]
