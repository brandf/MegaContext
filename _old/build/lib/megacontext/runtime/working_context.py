"""
Working context management for the MegaContext runtime.

For Phase 1 we only maintain raw token spans, but the API mirrors the plan laid out in
``README.md`` so later phases can extend entries with gist-aware metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

Level = Literal["L0", "L1", "L2"]


@dataclass
class WorkingEntry:
    node_id: str
    level: Level
    cost: int
    span_width: int
    distance_to_cursor: int


class WorkingContext:
    """
    Minimal working-context container for the base LLM runtime.

    Later phases will extend this to mix gists and tokens; for the POC bootstrap we keep
    a single batch of token IDs while providing the interfaces LensNet will expect.
    """

    def __init__(self, token_ids: Tensor, attention_mask: Tensor) -> None:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be shaped [batch, seq]")
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids shape")
        if token_ids.shape[0] != 1:
            raise NotImplementedError("WorkingContext currently supports batch size 1.")
        self._token_ids = token_ids
        self._attention_mask = attention_mask
        self._entries: list[WorkingEntry] = self._build_entries()

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer,
        *,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> WorkingContext:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=bool(max_length),
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        token_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return cls(token_ids=token_ids, attention_mask=attention_mask)

    def entries(self) -> Iterable[WorkingEntry]:
        return tuple(self._entries)

    def tail_view(self, k: int = 256) -> Iterable[WorkingEntry]:
        return tuple(self._entries[-k:])

    def pack(self, embeddings: Tensor | None = None) -> dict[str, Tensor]:
        if embeddings is None:
            embeddings = self._zeros_like_embeddings()
        seq_len = embeddings.shape[-2]
        levels = torch.zeros(seq_len, dtype=torch.long)
        span_width = torch.ones(seq_len, dtype=torch.long)
        distance = torch.arange(seq_len, 0, -1, dtype=torch.long)
        return {
            "embeddings": embeddings,
            "levels": levels,
            "span_width": span_width,
            "distance_to_cursor": distance,
        }

    def to_tensors(self, embedding_layer: nn.Embedding | None = None) -> dict[str, Tensor]:
        if embedding_layer is not None:
            embeds = self.materialize_embeddings(embedding_layer)
            return {"inputs_embeds": embeds, "attention_mask": self._attention_mask}
        return {"input_ids": self._token_ids, "attention_mask": self._attention_mask}

    def materialize_embeddings(self, embedding_layer: nn.Embedding) -> Tensor:
        token_ids = self._token_ids.to(device=embedding_layer.weight.device)
        embeds = embedding_layer(token_ids)
        return embeds

    @property
    def token_ids(self) -> Tensor:
        return self._token_ids

    @property
    def attention_mask(self) -> Tensor:
        return self._attention_mask

    def _build_entries(self) -> list[WorkingEntry]:
        seq_len = self._token_ids.shape[-1]
        return [
            WorkingEntry(
                node_id=str(idx),
                level="L0",
                cost=1,
                span_width=1,
                distance_to_cursor=seq_len - idx,
            )
            for idx in range(seq_len)
        ]

    def _zeros_like_embeddings(self) -> Tensor:
        seq_len = self._token_ids.shape[-1]
        return torch.zeros(seq_len, 1)
