from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Optional

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Label, Static

from ..orchestrator import RunOrchestrator


class DatasetStatus(Message):
    def __init__(self, ready: bool) -> None:
        self.ready = ready
        super().__init__()
        self.message = "ready" if ready else "missing"


class DatasetDependencyMissing(Message):
    def __init__(self, dependency: str) -> None:
        self.dependency = dependency
        super().__init__()


class DatasetView(Vertical):
    """Detect/download/prep dataset."""

    def __init__(self, orchestrator: RunOrchestrator, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.orchestrator = orchestrator
        self.status = Static("")
        self.log_widget = Static("", id="dataset-log")
        self.base_dir = Path.home() / ".cache" / "nanochat"
        self.dataset_dir_override: Optional[Path] = None

    def compose(self):
        yield Label("Dataset")
        yield Horizontal(Button("Download/Prep", id="dataset-run"), id="dataset-actions")
        yield self.status
        yield self.log_widget

    def update_base_dir(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.check_status()

    def update_dataset_dir(self, dataset_dir: Path) -> None:
        self.dataset_dir_override = dataset_dir
        self.check_status()

    def check_status(self) -> bool:
        ready = self._is_ready()
        dataset_dir = self.dataset_dir_override or (self.base_dir / "base_data")
        msg = f"Dataset ready at {dataset_dir}" if ready else f"Dataset missing at {dataset_dir} (run download/prep)"
        self.status.update(msg)
        self.post_message(DatasetStatus(ready))
        return ready

    def _is_ready(self) -> bool:
        dataset_dir = self.dataset_dir_override or (self.base_dir / "base_data")
        marker = dataset_dir / "download_complete.txt"
        parquet = next(dataset_dir.glob("*.parquet"), None) if dataset_dir.exists() else None
        bin_file = next(dataset_dir.glob("*.bin"), None) if dataset_dir.exists() else None
        return dataset_dir.exists() and (parquet or bin_file or marker.exists())

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dataset-run":
            await self._run_dataset()

    async def _run_dataset(self) -> None:
        # Ensure pyarrow is present for dataset prep
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.post_message(DatasetDependencyMissing("pyarrow"))
            return
        cmd = ["python", "-m", "nanochat.dataset"]
        self.log_widget.update((self.log_widget.renderable or "") + "\n" + " ".join(cmd))
        env_base = self.dataset_dir_override.parent if self.dataset_dir_override else self.base_dir
        env = {"NANOCHAT_BASE_DIR": str(env_base)}
        queue, _ = await self.orchestrator.stream_process(cmd, env=env)
        while True:
            line = await queue.get()
            self.log_widget.update((self.log_widget.renderable or "") + "\n" + line)
            if line.startswith("[exit"):
                break
        self.check_status()

    def _python(self) -> str:
        import sys

        return sys.executable
