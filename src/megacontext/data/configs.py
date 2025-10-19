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
    """Top-level dataset config (tokenizer, block size, and split definitions)."""

    dataset_name: str = Field(
        description="Identifier used for output metadata directories."
    )
    tokenizer: str = Field(description="Hugging Face tokenizer name or local path.")
    block_size: int = Field(
        default=32,
        description="Number of L0 tokens per block.",
    )
    splits: dict[str, SplitConfig]

    @field_validator("block_size")
    @classmethod
    def block_size_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("block_size must be > 0")
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
