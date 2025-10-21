"""
Lightweight tensor wrappers around long-form contexts used for gist training.

These helpers keep data contiguous and offer enumerators that surface all valid
working-context windows for a given token budget. They intentionally avoid any
Python object graphs beyond inexpensive slices so the training loop can batch
operations directly on tensors.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from torch import Tensor


@dataclass
class WorkingContext:
    """
    View over a contiguous window of tokens inside a ``MegaContext``.

    Instances are cheap wrappers that reference the parent context instead of
    copying data; callers should treat the exposed tensors as read-only.
    """

    parent: MegaContext
    start_token: int
    length: int

    @property
    def end_token(self) -> int:
        return self.start_token + self.length

    @property
    def token_slice(self) -> slice:
        return slice(self.start_token, self.end_token)

    @property
    def tokens(self) -> Tensor:
        return self.parent.tokens[self.token_slice]

    @property
    def hidden(self) -> Tensor:
        return self.parent.hidden[self.token_slice]

    @property
    def block_range(self) -> range:
        block_size = self.parent.block_size
        start_block = self.start_token // block_size
        end_block = (self.end_token + block_size - 1) // block_size
        return range(start_block, end_block)


@dataclass
class MegaContext:
    """
    Tensor-backed representation of a long-form context plus its teacher cache.

    Parameters
    ----------
    tokens:
        Token ids for the retained context (length ``context_tokens``).
    hidden:
        Teacher hidden states aligned with ``tokens`` (`context_tokens`, hidden_size).
    block_size:
        Number of L0 tokens per gist block.
    future_tokens / future_hidden:
        Continuation tokens and teacher hidden states over the prediction horizon.
    """

    tokens: Tensor
    hidden: Tensor
    block_size: int
    future_tokens: Tensor
    future_hidden: Tensor

    def num_tokens(self) -> int:
        return int(self.tokens.shape[0])

    def num_blocks(self) -> int:
        return self.num_tokens() // self.block_size

    @property
    def hidden_size(self) -> int:
        return int(self.hidden.shape[-1])

    def blocks(self) -> Tensor:
        """
        Reshape context hidden states into ``[num_blocks, block_size, hidden]``.

        Returns
        -------
        torch.Tensor
            View over the hidden states grouped by gist block.
        """

        total_tokens = self.num_tokens()
        if total_tokens % self.block_size != 0:
            msg = (
                f"Context length {total_tokens} is not divisible by block_size "
                f"{self.block_size}"
            )
            raise ValueError(msg)
        return self.hidden.view(self.num_blocks(), self.block_size, self.hidden_size)

    def gist_targets(self) -> Tensor:
        """Mean-hidden-state target for each block (used by the MSE baseline)."""

        return self.blocks().mean(dim=1)

    def iterate_working_windows(
        self,
        window_tokens: int,
        *,
        stride_tokens: int | None = None,
    ) -> Iterator[WorkingContext]:
        """
        Enumerate all contiguous working contexts of length ``window_tokens``.

        Parameters
        ----------
        window_tokens:
            Token budget for the working context.
        stride_tokens:
            Step size between generated windows. Defaults to ``window_tokens``.
        """

        if window_tokens <= 0:
            raise ValueError("window_tokens must be > 0")
        stride = stride_tokens or window_tokens
        if stride <= 0:
            raise ValueError("stride_tokens must be > 0")
        total_tokens = self.num_tokens()
        if total_tokens < window_tokens:
            return iter(())
        max_start = total_tokens - window_tokens + 1
        for start in range(0, max_start, stride):
            yield WorkingContext(self, start, window_tokens)
