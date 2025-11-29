from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Optional

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Log, Static

from ..orchestrator import RunOrchestrator
from ..config import ConfigBundle


class TrainStarted(Message):
    pass


class TrainView(Vertical):
    """Train orchestration view."""

    def __init__(self, orchestrator: RunOrchestrator, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.orchestrator = orchestrator
        self.log_widget = Log(id="train-log")
        self.run_name_input = Input(id="train-run-name", placeholder="Run name")
        self.active_task: Optional[asyncio.Task] = None
        self.active_proc: Optional[asyncio.subprocess.Process] = None
        self.config: Optional[ConfigBundle] = None
        self.base_dir = Path.home() / ".cache" / "nanochat"
        self.dataset_ready = False

    def compose(self):
        yield Horizontal(self.run_name_input, Button("Start", id="train-start"), id="train-actions")
        yield Static(id="train-stats")
        yield self.log_widget

    def set_config(self, bundle: ConfigBundle) -> None:
        self.config = bundle

    def set_base_dir(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def set_dataset_ready(self, ready: bool) -> None:
        self.dataset_ready = ready

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train-start":
            await self._start()

    async def _start(self) -> None:
        if not self.config:
            self.log_widget.write("No config loaded")
            return
        if not self.dataset_ready:
            self.log_widget.write("Dataset missing. Go to Dataset tab to prepare it.")
            return
        if self.active_task and not self.active_task.done():
            self.log_widget.write("Training already running")
            return
        run_name = self.run_name_input.value.strip() or self.config.data.get("wandb_run_name", "nanochat-cli-run")
        env = {"WANDB_RUN": run_name, "NANOCHAT_BASE_DIR": str(self.base_dir)}
        cmd = self.orchestrator.build_train_command(self.config)
        self.log_widget.write(f"Starting: {' '.join(cmd)}")

        async def _run() -> None:
            queue, proc = await self.orchestrator.stream_process(cmd, env=env)
            self.active_proc = proc
            while True:
                line = await queue.get()
                self.log_widget.write(line)
                if line.startswith("[exit"):
                    break
            self.active_proc = None

        self.active_task = asyncio.create_task(_run())
        self.post_message(TrainStarted())
