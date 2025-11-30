from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from textual.containers import Horizontal, Vertical
from textual.binding import Binding
from textual.widgets import Button, Input, Log, TextArea

from ..orchestrator import RunOrchestrator
from ..plugin import PluginRegistry
from .config_view import ConfigSelected, ConfigChanged
from .setup_view import SetupPathsUpdated


class EvalView(Vertical):
    """Simple continuation eval."""

    can_focus = True

    BINDINGS = [
        Binding("ctrl+e", "run_eval", "Run eval"),
    ]

    def __init__(self, plugin_registry: PluginRegistry, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.plugin_registry = plugin_registry
        self.orchestrator: Optional[RunOrchestrator] = None
        self.prompt = TextArea(id="eval-prompt")
        self.run_btn = Button("Run", id="eval-run")
        self.output = Log(id="eval-output")
        self.base_dir = Path.home() / ".cache" / "nanochat"
        self.checkpoint_path: Optional[Path] = None

    def compose(self):
        yield self.prompt
        yield Horizontal(self.run_btn)
        yield self.output

    def set_checkpoint(self, path: Optional[Path]) -> None:
        self.checkpoint_path = path

    def set_base_dir(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "eval-run":
            await self._run_eval()

    async def action_run_eval(self) -> None:
        await self._run_eval()

    async def _run_eval(self) -> None:
        prompt = self.prompt.text.strip()
        if not prompt:
            self.output.write("Prompt required")
            return
        if not self.checkpoint_path:
            self.output.write("No checkpoint selected/compatible")
            return
        if self.orchestrator is None:
            self.output.write("No orchestrator available; select a config.")
            return
        cmd = self.orchestrator.build_eval_command(self.checkpoint_path, prompt, mode="continuation", config={})
        env = {"NANOCHAT_BASE_DIR": str(self.base_dir)}
        self.output.write(f"Running: {' '.join(cmd)}")
        queue, _ = await self.orchestrator.stream_process(cmd, env=env)
        while True:
            line = await queue.get()
            self.output.write(line)
            if line.startswith("[exit"):
                break

    # Cross-view messages
    def handle_app_message(self, message):
        if isinstance(message, (ConfigSelected, ConfigChanged)):
            base = Path(
                message.bundle.data.get("nanochat_base_dir")
                or os.environ.get("NANOCHAT_BASE_DIR")
                or Path.home() / ".cache" / "nanochat"
            ).expanduser()
            self.set_base_dir(base)
            self.orchestrator = RunOrchestrator(self.plugin_registry.resolve(message.bundle.data))
        elif isinstance(message, SetupPathsUpdated):
            self.set_base_dir(message.base_dir)
