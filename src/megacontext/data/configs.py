"""Dataset configuration models shared between tooling and tests."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class SplitConfig(BaseModel):
    """Configuration for a single dataset split."""

    name: str = Field(description="Logical split name, e.g. train/validation/test.")
    source: str = Field(description="File or glob pointing to raw text files.")
    output_path: str = Field(description="Target .arrow file path.")
    max_files: int | None = Field(
        default=None, description="Optional cap on number of source files ingested."
    )
    max_tokens: int | None = Field(
        default=None, description="Optional cap on emitted tokens for quick smoke runs."
    )

    @field_validator("output_path")
    @classmethod
    def ensure_arrow_suffix(cls, value: str) -> str:
        if not value.endswith(".arrow"):
            msg = f"output_path must end with '.arrow', got {value!r}"
            raise ValueError(msg)
        return value


class DatasetConfig(BaseModel):
    """Top-level dataset config including teacher/cache settings."""

    dataset_name: str = Field(
        description="Identifier used for output metadata directories."
    )
    tokenizer: str = Field(description="Hugging Face tokenizer name or local path.")
    block_size: int = Field(
        default=32,
        description="Number of L0 tokens per block.",
    )
    context_tokens: int = Field(
        default=4096,
        description="Number of tokens retained as the MegaContext slice per example.",
    )
    context_stride: int | None = Field(
        default=None,
        description=(
            "Stride (in tokens) between successive context windows. "
            "Defaults to `context_tokens` when omitted."
        ),
    )
    horizon: int = Field(
        default=64,
        description="Total token horizon for teacher context windows.",
    )
    teacher_model: str | None = Field(
        default=None,
        description="Optional Hugging Face model name for caching teacher embeddings.",
    )
    teacher_batch_size: int = Field(
        default=4,
        description="Batch size used when computing teacher embeddings.",
    )
    teacher_dtype: str | None = Field(
        default="auto",
        description=(
            "Torch dtype for teacher model outputs. Use 'auto' to select float16 on "
            "older CUDA (e.g. T4) and bfloat16 on newer GPUs; fall back to float32 "
            "otherwise."
        ),
    )
    teacher_device: str = Field(
        default="auto",
        description=(
            "Device string for teacher model execution (e.g., cpu, cuda:0). "
            "Use 'auto' to pick a GPU when available, otherwise fall back to CPU."
        ),
    )
    teacher_trust_remote_code: bool = Field(
        default=False,
        description="Whether to enable HF trust_remote_code when loading the teacher",
    )
    splits: dict[str, SplitConfig]

    @field_validator("block_size")
    @classmethod
    def block_size_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("block_size must be > 0")
        return value

    @field_validator("context_tokens")
    @classmethod
    def context_multiple_of_block(cls, value: int, info) -> int:
        block_size = info.data.get("block_size", 32)
        if value <= 0:
            raise ValueError("context_tokens must be > 0")
        if value % block_size != 0:
            raise ValueError("context_tokens must be a multiple of block_size")
        return value

    @field_validator("context_stride")
    @classmethod
    def stride_multiple_of_block(cls, value: int | None, info) -> int | None:
        if value is None:
            return value
        block_size = info.data.get("block_size", 32)
        if value <= 0:
            raise ValueError("context_stride must be > 0 when provided")
        if value % block_size != 0:
            raise ValueError("context_stride must be a multiple of block_size")
        return value

    @field_validator("horizon")
    @classmethod
    def horizon_multiple_of_block(cls, value: int, info) -> int:
        block_size = info.data.get("block_size", 32)
        if value <= 0:
            raise ValueError("horizon must be > 0")
        if value % block_size != 0:
            raise ValueError("horizon must be a multiple of block_size")
        return value

    @field_validator("teacher_batch_size")
    @classmethod
    def teacher_batch_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("teacher_batch_size must be > 0")
        return value

    @field_validator("splits")
    @classmethod
    def ensure_named_split(
        cls,
        value: dict[str, SplitConfig],
    ) -> dict[str, SplitConfig]:
        if not value:
            raise ValueError("at least one split must be provided")
        for key, split in value.items():
            if split.name != key:
                msg = f"Split key {key!r} must match split.name {split.name!r}"
                raise ValueError(msg)
        return value

    def metadata_path(self) -> Path:
        return Path("data") / self.dataset_name / "metadata.yaml"
