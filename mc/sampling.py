from __future__ import annotations

import torch


def replay_sequence_with_controller(
    controller,
    prompt_tokens: list[int],
    generated_tokens: list[int],
) -> None:
    """
    Feed a prompt + generated continuation through the MC controller so the tree and
    working context reflect what the sampler produced. Does nothing if the controller
    is None or inputs are empty.
    """
    if controller is None or not prompt_tokens:
        return
    device = torch.device(controller.config.device)
    tokens = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    controller.begin_inference_session(tokens, rebuild=True)
    for token in generated_tokens:
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
        controller.inference_step(token_tensor)
