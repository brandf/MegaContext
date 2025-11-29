from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import ConfigBundle
from .plugin import Plugin


@dataclass
class RunningProcess:
    proc: subprocess.Popen
    command: List[str]
    working_dir: Path

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()

    def checkpoint_now(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGUSR1)
            except Exception:
                pass


class RunOrchestrator:
    """Builds commands and streams output for train/eval/chat."""

    def __init__(self, plugin: Plugin, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent
        self.plugin = plugin

    def build_train_command(self, config: ConfigBundle, checkpoint: Optional[Path] = None) -> List[str]:
        cmd = self.plugin.train_command(config.data)
        if checkpoint:
            cmd.extend(["--resume_from", str(checkpoint)])
        return cmd

    def build_eval_command(
        self, checkpoint: Path, prompt: str, mode: str = "continuation", config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        return self.plugin.eval_command(str(checkpoint), prompt, mode, config or {})

    async def stream_process(
        self, command: List[str], env: Optional[Dict[str, str]] = None
    ) -> tuple[asyncio.Queue[str], asyncio.subprocess.Process]:
        """Run a command and push stdout/stderr lines to a queue."""
        queue: asyncio.Queue[str] = asyncio.Queue()
        proc: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.repo_root,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def _reader() -> None:
            assert proc.stdout
            async for raw in proc.stdout:
                queue.put_nowait(raw.decode("utf-8", errors="replace").rstrip("\n"))
            await proc.wait()
            queue.put_nowait(f"[exit {proc.returncode}]")

        asyncio.create_task(_reader())
        return queue, proc
