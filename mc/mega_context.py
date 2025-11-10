from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from .config import MegaContextConfig

if TYPE_CHECKING:
    from .gistnet import GistNetBase


class MegaContextTree:
    """Tensor-first MegaContext that stores LOD0 as tokens and higher LODs as embeddings."""

    def __init__(self, config: MegaContextConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.levels: Dict[int, torch.Tensor] = {}
        self.tokens: Optional[torch.Tensor] = None
        self.embedder: Optional[nn.Module] = None
        self._lod0_cache: Optional[torch.Tensor] = None
        self._cache_lod0: bool = True
        self.gistnet: Optional["GistNetBase"] = None
        self.batch_size: Optional[int] = None
        self.access_counters = {
            "child": defaultdict(int),
            "parent": defaultdict(int),
            "token_slice": 0,
        }

    @classmethod
    def from_tokens(
        cls,
        tokens: torch.Tensor,
        embedder: nn.Module,
        config: MegaContextConfig,
        gistnet: Optional["GistNetBase"] = None,
        precomputed_embeddings: Optional[torch.Tensor] = None,
    ) -> "MegaContextTree":
        tree = cls(config, tokens.device)
        tree.attach_gistnet(gistnet)
        tree.embedder = embedder
        tree.tokens = tokens.to(tree.device).long()
        tree.batch_size = tokens.shape[0]
        batch, seq_len = tree.tokens.shape
        if precomputed_embeddings is not None:
            lod0_embeddings = precomputed_embeddings.to(tree.device)
        else:
            lod0_embeddings = embedder(tree.tokens)
        tree._lod0_cache = lod0_embeddings
        tree._build_higher_levels(lod0_embeddings)
        return tree

    @classmethod
    def from_embeddings(
        cls,
        embeddings: torch.Tensor,
        config: MegaContextConfig,
        gistnet: Optional["GistNetBase"] = None,
    ) -> "MegaContextTree":
        tree = cls(config, embeddings.device)
        tree.attach_gistnet(gistnet)
        tree.batch_size = embeddings.shape[0]
        batch, seq_len, _ = embeddings.shape
        tree.levels[0] = embeddings.clone()
        tree._lod0_cache = embeddings.clone()
        tree._build_higher_levels(embeddings)
        return tree

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def attach_gistnet(self, gistnet: Optional["GistNetBase"]) -> None:
        self.gistnet = gistnet

    def _build_higher_levels(self, base_embeddings: torch.Tensor) -> None:
        batch, seq_len, _ = base_embeddings.shape
        level_embeddings = base_embeddings.to(self.device)
        for lod in range(1, self.config.max_lod + 1):
            prev_len = level_embeddings.shape[1]
            if prev_len == 0:
                break
            grouped, new_len, mask = self._reshape_for_pool(level_embeddings, self.config.block_size)
            if new_len == 0:
                break
            grouped = grouped.contiguous()
            mask = mask.contiguous()
            if self.gistnet is not None:
                masked = grouped * mask
                flat = masked.view(-1, self.config.block_size, self.config.embed_dim)
                pooled = self.gistnet(flat).view(batch, new_len, self.config.embed_dim)
            else:
                weighted = grouped * mask
                counts = mask.sum(dim=2).clamp_min(1.0)
                pooled = weighted.sum(dim=2) / counts
            self.levels[lod] = pooled
            level_embeddings = pooled

    def _reshape_for_pool(self, x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch, length, dim = x.shape
        new_len = math.ceil(length / block_size)
        if new_len == 0:
            empty = x.new_zeros(batch, 0, block_size, dim)
            mask = x.new_zeros(batch, 0, block_size, 1)
            return empty, 0, mask
        pad = new_len * block_size - length
        valid_mask = torch.ones(batch, length, 1, device=x.device, dtype=x.dtype)
        if pad > 0:
            pad_tensor = torch.zeros(batch, pad, dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)
            pad_mask = torch.zeros(batch, pad, 1, device=x.device, dtype=x.dtype)
            valid_mask = torch.cat([valid_mask, pad_mask], dim=1)
        reshaped = x.view(batch, new_len, block_size, dim)
        mask = valid_mask.view(batch, new_len, block_size, 1)
        return reshaped, new_len, mask

    def release_lod0_cache(self, disable_future_cache: bool = False) -> None:
        self._lod0_cache = None
        if disable_future_cache:
            self._cache_lod0 = False

    # ------------------------------------------------------------------ #
    # Incremental updates
    # ------------------------------------------------------------------ #
    def append(self, tokens: torch.Tensor) -> None:
        if self.tokens is None:
            raise ValueError("Tree not initialized with tokens")
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(1)
        elif tokens.dim() == 2:
            pass
        else:
            raise ValueError("append expects [B] or [B, 1] token tensor")
        tokens = tokens.to(self.device).long()
        self.tokens = torch.cat([self.tokens, tokens], dim=1)
        if self._lod0_cache is not None and self.embedder is not None:
            embedded = self.embedder(tokens)
            self._lod0_cache = torch.cat([self._lod0_cache, embedded], dim=1)
        self._refresh_after_append()

    def _refresh_after_append(self) -> None:
        total_tokens = self.tokens.shape[1] if self.tokens is not None else 0
        if total_tokens <= 0:
            return
        for lod in range(1, self.config.max_lod + 1):
            tokens_per_node = self.tokens_per_entry(lod)
            if tokens_per_node == 0:
                continue
            target_index = (total_tokens - 1) // tokens_per_node
            required_nodes = target_index + 1
            self._ensure_level_capacity(lod, required_nodes)
            self._recompute_node(lod, target_index)

    def _ensure_level_capacity(self, lod: int, required_nodes: int) -> None:
        if lod not in self.levels:
            batch = self._batch_size()
            empty = torch.zeros(batch, 0, self.config.embed_dim, device=self.device)
            self.levels[lod] = empty
        current = self.levels[lod]
        if current.shape[1] >= required_nodes:
            return
        batch, _, dim = current.shape
        extra = required_nodes - current.shape[1]
        pad_embeddings = torch.zeros(batch, extra, dim, device=self.device, dtype=current.dtype)
        self.levels[lod] = torch.cat([current, pad_embeddings], dim=1)

    def _recompute_node(self, lod: int, node_index: int) -> None:
        if lod == 0:
            return
        child_block = self._gather_child_block(lod, node_index)
        if self.gistnet is not None:
            gist = self.gistnet(child_block).unsqueeze(1)
        else:
            gist = child_block.mean(dim=1, keepdim=True)
        self.levels[lod][:, node_index : node_index + 1] = gist

    def _gather_child_block(self, lod: int, node_index: int) -> torch.Tensor:
        child_lod = lod - 1
        start = node_index * self.config.block_size
        end = start + self.config.block_size
        if child_lod == 0:
            block = self._embed_tokens_range(start, end)
        else:
            children = self.levels[child_lod]
            end = min(end, children.shape[1])
            block = children[:, start:end]
        if block.shape[1] < self.config.block_size:
            batch, _, dim = block.shape
            pad = torch.zeros(
                batch,
                self.config.block_size - block.shape[1],
                self.config.embed_dim,
                device=block.device,
                dtype=block.dtype,
            )
            block = torch.cat([block, pad], dim=1)
        return block

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_level(self, lod: int) -> torch.Tensor:
        if lod == 0:
            return self._get_lod0_embeddings()
        if lod not in self.levels:
            raise ValueError(f"LOD {lod} not available")
        return self.levels[lod]

    def _get_lod0_embeddings(self) -> torch.Tensor:
        if self._lod0_cache is not None:
            return self._lod0_cache
        if 0 in self.levels:
            return self.levels[0]
        if self.tokens is None or self.embedder is None:
            raise ValueError("Tree lacks tokens/embedder for LOD0")
        embeddings = self.embedder(self.tokens)
        if self._cache_lod0:
            self._lod0_cache = embeddings
        return embeddings

    def get_level_metadata(self, lod: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_level(lod), self._positions_for_lod(lod)

    def get_positions_for_lod(self, lod: int) -> torch.Tensor:
        return self._positions_for_lod(lod)

    def num_tokens(self) -> int:
        if self.tokens is not None:
            return int(self.tokens.shape[1])
        level0 = self.levels.get(0)
        return int(level0.shape[1]) if level0 is not None else 0

    def tokens_per_entry(self, lod: int) -> int:
        return int(self.config.block_size ** lod)

    def _node_index(self, lod: int, global_position: int) -> int:
        tokens = max(self.tokens_per_entry(lod), 1)
        return max(0, global_position // tokens)

    def get_node_embedding(self, lod: int, global_position: int) -> torch.Tensor:
        level = self.get_level(lod)
        idx = min(self._node_index(lod, global_position), level.shape[1] - 1)
        return level[:, idx : idx + 1]

    def get_children_embeddings(self, lod: int, global_position: int) -> torch.Tensor:
        if lod <= 0:
            raise ValueError("LOD 0 entries have no children")
        child_lod = lod - 1
        start = self._node_index(lod, global_position) * self.config.block_size
        end = start + self.config.block_size
        if child_lod == 0:
            children = self._embed_tokens_range(start, end)
        else:
            level = self.levels[child_lod]
            end = min(end, level.shape[1])
            children = level[:, start:end]
        self.access_counters["child"][lod] += 1
        return children

    def get_parent_embedding(self, lod: int, global_position: int) -> torch.Tensor:
        parent_lod = lod + 1
        if parent_lod not in self.levels:
            raise ValueError("Parent level not available")
        self.access_counters["parent"][parent_lod] += 1
        return self.get_node_embedding(parent_lod, global_position)

    def summary(self) -> Dict[int, Tuple[int, int]]:
        summary: Dict[int, Tuple[int, int]] = {}
        if self.tokens is not None:
            summary[0] = (self.tokens.shape[1], self.config.embed_dim)
        for lod, tensor in self.levels.items():
            summary[lod] = tensor.shape[1:]
        return summary

    def _embed_tokens_range(self, start: int, end: int) -> torch.Tensor:
        if self.tokens is None:
            raise ValueError("Tree lacks tokens/embedder for LOD0 operations")
        total = self.tokens.shape[1]
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        # Prefer cached embeddings if available
        if self._lod0_cache is not None:
            self.access_counters["token_slice"] += 1
            return self._lod0_cache[:, start:end]
        if self.embedder is None:
            raise ValueError("Tree lacks embedder for LOD0 operations")
        token_slice = self.tokens[:, start:end]
        self.access_counters["token_slice"] += 1
        return self.embedder(token_slice)

    def get_lod0_slice(self, start: int, end: int) -> torch.Tensor:
        start = max(0, start)
        if self._lod0_cache is not None:
            return self._lod0_cache[:, start:end]
        if self.tokens is not None and self.embedder is not None:
            return self._embed_tokens_range(start, end)
        return self.get_level(0)[:, start:end]

    def append_with_embeddings(self, tokens: torch.Tensor, embeddings: torch.Tensor) -> None:
        """Append tokens along with their precomputed LOD0 embeddings.

        Accepts embeddings shaped [B, D] or [B, T, D]. Updates the token buffer
        and the LOD0 cache without re-running the embedder.
        """
        if self.tokens is None:
            raise ValueError("Tree not initialized with tokens")
        # Normalize token shape to [B, T]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(1)
        elif tokens.dim() != 2:
            raise ValueError("append_with_embeddings expects tokens shaped [B] or [B, T]")
        tokens = tokens.to(self.device).long()
        self.tokens = torch.cat([self.tokens, tokens], dim=1)

        # Normalize embeddings to [B, T, D]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        elif embeddings.dim() != 3:
            raise ValueError("append_with_embeddings expects embeddings shaped [B, D] or [B, T, D]")
        embeddings = embeddings.to(self.device)
        if self._lod0_cache is None:
            # Initialize cache lazily if needed
            self._lod0_cache = embeddings
        else:
            self._lod0_cache = torch.cat([self._lod0_cache, embeddings], dim=1)
        self._refresh_after_append()

    def get_access_stats(self) -> Dict[str, Dict[int, int]]:
        return {
            "child": {int(k): int(v) for k, v in self.access_counters["child"].items()},
            "parent": {int(k): int(v) for k, v in self.access_counters["parent"].items()},
            "token_slice": int(self.access_counters["token_slice"]),
        }

    def _batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size
        if self.tokens is not None:
            self.batch_size = self.tokens.shape[0]
            return self.batch_size
        if 0 in self.levels:
            self.batch_size = self.levels[0].shape[0]
            return self.batch_size
        raise ValueError("Unable to infer batch size for MegaContextTree")

    def _positions_for_lod(self, lod: int) -> torch.Tensor:
        stride = self.tokens_per_entry(lod)
        if lod == 0:
            length = self.num_tokens()
        else:
            if lod not in self.levels:
                raise ValueError(f"LOD {lod} not available")
            length = self.levels[lod].shape[1]
        if length == 0:
            return torch.zeros(self._batch_size(), 0, dtype=torch.long, device=self.device)
        base_positions = torch.arange(length, device=self.device, dtype=torch.long) * stride
        return base_positions.unsqueeze(0).repeat(self._batch_size(), 1)


def build_mega_context(
    kind: str,
    tokens: torch.Tensor,
    embedder: nn.Module,
    config: MegaContextConfig,
    gistnet: Optional["GistNetBase"] = None,
    precomputed_embeddings: Optional[torch.Tensor] = None,
) -> MegaContextTree:
    key = kind.lower()
    try:
        target_device = next(embedder.parameters()).device
    except StopIteration:
        target_device = tokens.device
    tokens = tokens.to(target_device)
    if key == "ram":
        return MegaContextTree.from_tokens(
            tokens,
            embedder,
            config,
            gistnet=gistnet,
            precomputed_embeddings=precomputed_embeddings,
        )
    if key == "disk":
        raise NotImplementedError("Disk-backed MegaContextTree is not implemented yet")
    raise ValueError(f"Unknown MegaContext implementation: {kind}")
