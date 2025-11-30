from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, Static
from textual.binding import Binding


class SetupCompleted(Message):
    pass


class SetupPathsUpdated(Message):
    def __init__(self, base_dir: Path, dataset_dir: Path, checkpoints_dir: Path) -> None:
        self.base_dir = base_dir
        self.dataset_dir = dataset_dir
        self.checkpoints_dir = checkpoints_dir
        super().__init__()


class SetupView(Vertical):
    """Self-managed environment setup (uv/torch/auth)."""

    can_focus = True

    BINDINGS = [
        Binding("ctrl+b", "check_setup", "Check setup"),
        Binding("ctrl+u", "run_setup", "Run setup"),
    ]

    def __init__(self, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.status = Static("")
        self.log_widget = Static("", id="setup-log")
        self.wandb_input = Input(id="setup-wandb", placeholder="WANDB_API_KEY")
        self.hf_input = Input(id="setup-hf", placeholder="HF_TOKEN")
        self.base_dir_input = Input(id="setup-base-dir", placeholder="Base dir (NANOCHAT_BASE_DIR)")
        self.dataset_dir_input = Input(id="setup-dataset-dir", placeholder="Dataset dir (optional)")
        self.checkpoints_dir_input = Input(id="setup-ckpt-dir", placeholder="Checkpoints dir (optional)")

    def compose(self):
        yield Label("Setup")
        yield Horizontal(self.wandb_input, self.hf_input, self.base_dir_input, self.dataset_dir_input, self.checkpoints_dir_input, Button("Run", id="setup-run"), id="setup-actions")
        yield self.status
        yield self.log_widget

    def check_status(self) -> None:
        checks = {
            "python": True,
            "uv": shutil.which("uv") is not None,
            "torch": self._has_torch(),
        }
        lines = [f"{k}: {'ok' if v else 'missing'}" for k, v in checks.items()]
        self.status.update("\n".join(lines))

    def _has_torch(self) -> bool:
        try:
            import torch  # type: ignore

            _ = torch.__version__
            return True
        except Exception:
            return False

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-run":
            await self._run_setup()

    def action_check_setup(self) -> None:
        self.check_status()

    async def action_run_setup(self) -> None:
        await self._run_setup()

    async def _run_setup(self) -> None:
        cmds = []
        env_lines = []
        if self.wandb_input.value.strip():
            env_lines.append(f"WANDB_API_KEY={self.wandb_input.value.strip()}")
        if self.hf_input.value.strip():
            env_lines.append(f"HF_TOKEN={self.hf_input.value.strip()}")
        if env_lines:
            env_path = Path.home() / ".nanochat-cli" / ".env"
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        if shutil.which("uv") is None:
            cmds.append([self._python(), "-m", "pip", "install", "uv"])
        if not self._has_torch():
            # Default to CPU wheel unless CUDA available
            if shutil.which("nvidia-smi"):
                cmds.append([self._python(), "-m", "pip", "install", "torch"])
            else:
                cmds.append([self._python(), "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
        # Optional dataset deps
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            cmds.append([self._python(), "-m", "pip", "install", "pyarrow"])
        for cmd in cmds:
            self.log_widget.update(self.log_widget.renderable + "\n" + " ".join(cmd) if self.log_widget.renderable else " ".join(cmd))
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            assert proc.stdout
            async for raw in proc.stdout:
                text = raw.decode("utf-8", errors="replace").rstrip()
                self.log_widget.update((self.log_widget.renderable or "") + "\n" + text)
            await proc.wait()
        self.check_status()
        base_dir = Path(self.base_dir_input.value.strip() or Path.home() / ".cache" / "nanochat").expanduser()
        dataset_dir = Path(self.dataset_dir_input.value.strip() or base_dir / "base_data").expanduser()
        ckpt_dir = Path(self.checkpoints_dir_input.value.strip() or base_dir).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.post_message(SetupPathsUpdated(base_dir, dataset_dir, ckpt_dir))
        self.post_message(SetupCompleted())

    def _python(self) -> str:
        import sys

        return sys.executable
