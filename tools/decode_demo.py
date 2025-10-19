"""Quick sanity check that the base LLM wrapper can decode text blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from megacontext.runtime import BaseModel
from megacontext.utils import maybe_init_wandb, setup_logging


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if path.suffix == ".json":
            return json.load(handle)
        raise ValueError(f"Unsupported config extension: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run a short decode demo with MegaContext's base model wrapper.")
    )
    parser.add_argument(
        "--config",
        default="configs/runs/base_llm.yaml",
        type=Path,
        help="Path to the run config YAML file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional override prompt for quick experiments.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = config.get("run_name", "base_llm_demo")
    logger = setup_logging(run_name)
    logger.info("Starting decode demo", extra={"config_path": str(args.config)})

    model_name = config["model_name"]
    torch_dtype = config.get("torch_dtype", "bfloat16")
    device = config.get("device")
    prompt = args.prompt or config.get("prompt", "MegaContext is ")
    max_new_tokens = int(config.get("max_new_tokens", 32))

    base_model = BaseModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=config.get("trust_remote_code", False),
    )

    tokenizer = base_model.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt")
    wandb_run = maybe_init_wandb(
        config={
            "model_name": model_name,
            "torch_dtype": str(torch_dtype),
            "max_new_tokens": max_new_tokens,
        },
        project=config.get("wandb_project", "megacontext-poc"),
        run_name=run_name,
    )

    start = perf_counter()
    outputs = base_model.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=config.get("do_sample", False),
        temperature=config.get("temperature", 0.7),
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    duration = perf_counter() - start

    generation_stats: dict[str, Any] = {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": len(outputs[0]) - int(inputs["input_ids"].shape[-1]),
        "duration_sec": duration,
    }

    logger.info("Decoded output", extra=generation_stats)
    print(decoded)

    if wandb_run is not None:
        wandb_run.log(generation_stats)
        wandb_run.finish()
        logger.info("Logged metrics to Weights & Biases", extra=generation_stats)

    logger.info("Decode demo finished")


if __name__ == "__main__":
    main()
