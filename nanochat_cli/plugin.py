from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class Plugin(Protocol):
    name: str
    description: str

    def applies(self, config: Dict[str, Any]) -> bool:
        ...

    def train_command(self, config: Dict[str, Any]) -> List[str]:
        ...

    def eval_command(self, checkpoint: str, prompt: str, mode: str, config: Dict[str, Any]) -> List[str]:
        ...

    def dataset_command(self, config: Dict[str, Any], shards: str) -> List[str]:
        ...


@dataclass
class VanillaPlugin:
    """Default plugin for vanilla nanochat workflows."""

    name: str = "vanilla"
    description: str = "Vanilla nanochat without MegaContext extensions."

    def applies(self, config: Dict[str, Any]) -> bool:
        # Default applies everywhere unless mc_enabled is hard-required by another plugin.
        return True

    def train_command(self, config: Dict[str, Any]) -> List[str]:
        # Directly invoke nanochat training modules instead of shell scripts.
        args = [
            "python",
            "-m",
            "scripts.base_train",
        ]
        device_batch_size = config.get("device_batch_size")
        if device_batch_size:
            args.extend(["--device_batch_size", str(device_batch_size)])
        depth = config.get("depth")
        if depth:
            args.extend(["--depth", str(depth)])
        max_seq = config.get("max_seq_len")
        if max_seq:
            args.extend(["--max_seq_len", str(max_seq)])
        block_size = config.get("block_size")
        if block_size:
            args.extend(["--block_size", str(block_size)])
        return args

    def eval_command(self, checkpoint: str, prompt: str, mode: str, config: Dict[str, Any]) -> List[str]:
        if mode == "chat":
            return ["python", "-m", "scripts.chat_cli", "--checkpoint", checkpoint, "-p", prompt]
        return ["python", "-m", "scripts.chat_eval", "--checkpoint", checkpoint, "--prompt", prompt]

    def dataset_command(self, config: Dict[str, Any], shards: str) -> List[str]:
        return ["python", "-m", "nanochat.dataset", "-n", shards]


class PluginRegistry:
    """Registry to resolve plugins based on config."""

    def __init__(self) -> None:
        self.plugins: List[Plugin] = [VanillaPlugin()]
        mc_plugin = self._load_mc_plugin()
        if mc_plugin:
            self.plugins.append(mc_plugin)

    def _load_mc_plugin(self) -> Optional[Plugin]:
        try:
            mod = importlib.import_module("nanochat_cli_mc")
            plugin = getattr(mod, "MegaContextPlugin", None)
            if plugin:
                return plugin()
        except Exception:
            return None
        return None

    def resolve(self, config: Dict[str, Any]) -> Plugin:
        for plugin in self.plugins:
            if plugin.applies(config):
                return plugin
        return self.plugins[0]
