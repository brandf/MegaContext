"""
Wrapper around Hugging Face causal language models used by the MegaContext runtime.

The wrapper keeps the tokenizer and model together, exposes a minimal forward API, and
centralises dtype/device handling so later phases can plug in additional adapters
without touching call site code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class ForwardOutput:
    """Tiny container so tests can easily assert on logits without importing HF."""

    logits: torch.Tensor


class BaseModel:
    """
    Convenience wrapper for loading and invoking frozen causal language models.

    Parameters are intentionally conservative: the wrapper defaults to bf16, keeps the
    model in eval mode, and leaves gradient tracking disabled.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._model = model.eval()
        self._tokenizer = tokenizer
        torch.set_grad_enabled(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        torch_dtype: str | torch.dtype | None = torch.bfloat16,
        device: str | None = None,
        use_fast_tokenizer: bool = True,
        trust_remote_code: bool = False,
        **kwargs: dict,
    ) -> BaseModel:
        """
        Load a pretrained causal LM and tokenizer from Hugging Face.

        Args:
            model_name: Hugging Face repository ID or local path.
            torch_dtype: Desired dtype (defaults to bf16). Use ``None`` to let
                Hugging Face decide automatically.
            device: Optional explicit device string (e.g. "cuda", "cuda:0", "cpu").
            use_fast_tokenizer: Toggle HF fast tokenizers.
            trust_remote_code: Forwarded to HF loaders for models requiring custom code.
            **kwargs: Forwarded to ``AutoModelForCausalLM.from_pretrained``.
        """

        resolved_dtype = _resolve_dtype(torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
        # Ensure the tokenizer always has a padding token for batched operations.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        if device is not None:
            model.to(device)

        return cls(model=model, tokenizer=tokenizer)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    def forward(self, **model_kwargs: dict) -> ForwardOutput:
        """
        Forward pass through the underlying model.

        The wrapper mirrors Hugging Face's signature flexibility. Callers can provide
        ``input_ids`` *or* ``inputs_embeds`` plus an ``attention_mask``. The output is
        wrapped in ``ForwardOutput`` for lightweight testing.
        """

        outputs = self._model(**model_kwargs)
        logits = outputs.logits
        return ForwardOutput(logits=logits)


def _resolve_dtype(torch_dtype: str | torch.dtype | None) -> torch.dtype | None:
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        if torch_dtype == "auto":
            return None
        if hasattr(torch, torch_dtype):
            return getattr(torch, torch_dtype)
    raise ValueError(f"Unsupported torch dtype: {torch_dtype!r}")
