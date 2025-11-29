from __future__ import annotations

import asyncio
import os
import shutil
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import yaml

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Log,
    Static,
    TabPane,
    TabbedContent,
    TextArea,
)

from .checkpoints import CheckpointRecord, CheckpointRegistry
from .config import ConfigBundle, ConfigManager, categorize_field, flatten_config, to_json
from .orchestrator import RunOrchestrator
from .plugin import PluginRegistry


class StatsModel:
    """Lightweight parser for train/eval log lines."""

    def __init__(self) -> None:
        self.values: Dict[str, Any] = {}
        self.step_regex = re.compile(r"step[:=]\s*(\d+)", re.IGNORECASE)
        self.loss_regex = re.compile(r"(?:loss|bpb)[:=]\s*([0-9.]+)")
        self.gpu_regex = re.compile(r"(?:mfu|gpu)\s*[:=]\s*([0-9.]+)")

    def ingest(self, line: str) -> Dict[str, Any]:
        if m := self.step_regex.search(line):
            self.values["step"] = int(m.group(1))
        if m := self.loss_regex.search(line):
            self.values["loss"] = float(m.group(1))
        if m := self.gpu_regex.search(line):
            self.values["gpu_util"] = float(m.group(1))
        return self.values


class NanochatApp(App):
    """Textual TUI for nanochat + MegaContext workflows."""

    CSS = """
    Screen {
        align: center middle;
    }
    #config-json {
        height: 16;
    }
    #log-view {
        height: 18;
    }
    #prompt-input {
        height: 4;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save_prefab", "Save prefab"),
        Binding("ctrl+r", "refresh_checkpoints", "Refresh checkpoints"),
        Binding("ctrl+t", "start_train", "Start train"),
        Binding("ctrl+e", "run_eval", "Run eval"),
    ]

    def __init__(self, config_manager: Optional[ConfigManager] = None) -> None:
        super().__init__()
        self.config_manager = config_manager or ConfigManager()
        self.checkpoint_registry = CheckpointRegistry()
        self.plugin_registry = PluginRegistry()
        self.orchestrator = RunOrchestrator(self.plugin_registry.resolve({}))
        prefabs = self.config_manager.list_prefabs()
        self.current_config: ConfigBundle = prefabs[0] if prefabs else ConfigBundle("empty", {})
        self.active_train_task: Optional[asyncio.Task] = None
        self.active_train_proc: Optional[asyncio.subprocess.Process] = None
        self.stats = StatsModel()
        self.checkpoints: List[CheckpointRecord] = []
        self.eval_output: List[str] = []
        self.train_log: List[str] = []
        self.setup_ready: bool = False

    # Compose -----------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            yield TabPane("Config", self._build_config_tab(), id="tab-config")
            yield TabPane("Checkpoints", self._build_checkpoints_tab(), id="tab-checkpoints")
            yield TabPane("Train", self._build_train_tab(), id="tab-train")
            yield TabPane("Eval/Chat", self._build_eval_tab(), id="tab-eval")
            yield TabPane("Dataset", self._build_dataset_tab(), id="tab-dataset")
            yield TabPane("Setup", self._build_setup_tab(), id="tab-setup")
        yield Footer()

    def _build_config_tab(self) -> Container:
        prefabs = ListView(
            *[ListItem(Label(bundle.name), id=bundle.name) for bundle in self.config_manager.list_prefabs()],
            id="prefab-list",
        )
        json_view = TextArea(id="config-json", language="json")
        overrides = TextArea(id="override-input")
        save_name = Input(placeholder="Save as name (optional)", id="save-name")
        controls = Horizontal(
            Button("Apply Override", id="apply-override"),
            Button("Save Prefab", id="save-prefab"),
            Button("Reload Prefabs", id="reload-prefabs"),
            save_name,
        )
        details = Vertical(Label("Prefab details"), json_view, Label("Overrides"), overrides, controls)
        return Horizontal(prefabs, details, id="config-tab")

    def _build_checkpoints_tab(self) -> Container:
        table = DataTable(id="checkpoint-table", zebra_stripes=True)
        table.add_columns("Stage", "Path", "Compatible", "Reasons")
        refresh_btn = Button("Refresh", id="btn-refresh-checkpoints")
        return Vertical(refresh_btn, table)

    def _build_train_tab(self) -> Container:
        run_name = Input(placeholder="Run name (WANDB + checkpoints)", id="run-name")
        start_btn = Button("Start/Resume", id="btn-start-train")
        stop_btn = Button("Stop", id="btn-stop-train")
        ckpt_btn = Button("Checkpoint Now", id="btn-ckpt-now")
        stats = Static(id="train-stats")
        log = Log(id="log-view", highlight=True)
        return Vertical(Horizontal(run_name, start_btn, stop_btn, ckpt_btn), stats, log)

    def _build_eval_tab(self) -> Container:
        prompt = TextArea(id="prompt-input")
        mode = Input(placeholder="Mode: chat|continuation (default continuation)", id="eval-mode")
        run_btn = Button("Run Eval/Chat", id="btn-run-eval")
        output = Log(id="eval-output", highlight=True)
        return Vertical(prompt, mode, run_btn, output)

    def _build_dataset_tab(self) -> Container:
        shard_input = Input(placeholder="Shard count (e.g., 200)", id="dataset-shards")
        run_btn = Button("Download/Prep Dataset", id="btn-run-dataset")
        log = Log(id="dataset-log", highlight=True)
        return Vertical(Horizontal(shard_input, run_btn), log)

    def _build_setup_tab(self) -> Container:
        status = Static(id="setup-status")
        check_btn = Button("Re-check", id="btn-check-setup")
        run_btn = Button("Run Setup", id="btn-run-setup")
        log = Log(id="setup-log", highlight=True)
        return Vertical(status, Horizontal(check_btn, run_btn), log)

    # Lifecycle ---------------------------------------------------------------
    async def on_mount(self) -> None:
        self._load_config_into_view(self.current_config)
        await self._refresh_checkpoints()
        self.set_interval(1.0, self._tick_stats)
        self._update_setup_status()

    # Actions -----------------------------------------------------------------
    async def action_save_prefab(self) -> None:
        save_field = self.query_one("#save-name", Input)
        name = save_field.value.strip() or self.current_config.name
        self.config_manager.save(self.current_config, as_name=name)
        await self._toast(f"Saved prefab {name}")

    async def action_refresh_checkpoints(self) -> None:
        await self._refresh_checkpoints()

    async def action_start_train(self) -> None:
        await self._start_train()

    async def action_run_eval(self) -> None:
        await self._run_eval()

    # Event handlers ----------------------------------------------------------
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-override":
            await self._apply_override()
        elif event.button.id == "save-prefab":
            await self.action_save_prefab()
        elif event.button.id == "reload-prefabs":
            prefabs = self.config_manager.list_prefabs()
            prefab_list = self.query_one("#prefab-list", ListView)
            prefab_list.clear()
            prefab_list.extend([ListItem(Label(bundle.name), id=bundle.name) for bundle in prefabs])
        elif event.button.id == "btn-refresh-checkpoints":
            await self._refresh_checkpoints()
        elif event.button.id == "btn-start-train":
            await self._start_train()
        elif event.button.id == "btn-stop-train":
            if self.active_train_proc and self.active_train_proc.returncode is None:
                self.active_train_proc.terminate()
            if self.active_train_task:
                self.active_train_task.cancel()
                self.active_train_task = None
                await self._toast("Train task cancelled")
        elif event.button.id == "btn-ckpt-now":
            await self._checkpoint_now()
        elif event.button.id == "btn-run-eval":
            await self._run_eval()
        elif event.button.id == "btn-run-dataset":
            await self._run_dataset()
        elif event.button.id == "btn-check-setup":
            self._update_setup_status()
        elif event.button.id == "btn-run-setup":
            await self._run_setup()

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "prefab-list":
            return
        if event.item and event.item.id:
            bundle = self.config_manager.load(event.item.id)
            self.current_config = bundle
            self.orchestrator = RunOrchestrator(self.plugin_registry.resolve(bundle.data))
            self._load_config_into_view(bundle)

    # Helpers -----------------------------------------------------------------
    def _load_config_into_view(self, bundle: ConfigBundle) -> None:
        json_view = self.query_one("#config-json", TextArea)
        json_view.load_text(to_json(bundle.data))
        stats = flatten_config(bundle.data)
        stats_lines = [f"{k}: {v}" for k, v in stats.items() if categorize_field(k) != "auth"]
        self.title = f"nanochat-cli — {bundle.name}"
        train_stats = self.query("#train-stats")
        if train_stats:
            train_stats.first().update("\n".join(stats_lines))

    async def _apply_override(self) -> None:
        overrides_area = self.query_one("#override-input", TextArea)
        raw = overrides_area.text.strip()
        if not raw:
            await self._toast("No overrides provided")
            return
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = json.loads(json.dumps(yaml.safe_load(raw)))
            except Exception as exc:
                await self._toast(f"Failed to parse overrides: {exc}")
                return
        self.current_config = self.config_manager.apply_overrides(self.current_config, parsed)
        self._load_config_into_view(self.current_config)
        await self._toast("Overrides applied")

    async def _refresh_checkpoints(self) -> None:
        self.checkpoints = self.checkpoint_registry.list_checkpoints(self.current_config.data)
        table = self.query_one("#checkpoint-table", DataTable)
        table.clear()
        for rec in self.checkpoints:
            table.add_row(
                rec.stage,
                str(rec.path),
                "yes" if rec.compatible else "no",
                "; ".join(rec.reasons) if rec.reasons else "",
            )
        await self._toast("Checkpoints refreshed")

    async def _start_train(self) -> None:
        if self.active_train_task and not self.active_train_task.done():
            await self._toast("Training already running")
            return
        run_name_input = self.query_one("#run-name", Input)
        run_name = run_name_input.value.strip() or self.current_config.data.get("wandb_run_name", "nanochat-cli-run")
        env = {
            "WANDB_RUN": run_name,
            "NANOCHAT_BASE_DIR": str(
                Path(
                    self.current_config.data.get("nanochat_base_dir")
                    or os.environ.get("NANOCHAT_BASE_DIR")
                    or Path.home() / ".cache" / "nanochat"
                ).expanduser()
            ),
        }
        cmd = self.orchestrator.build_train_command(self.current_config)
        log_widget = self.query_one("#log-view", Log)
        log_widget.write(f"Starting: {' '.join(cmd)}")

        async def _run() -> None:
            queue, proc = await self.orchestrator.stream_process(cmd, env=env)
            self.active_train_proc = proc
            while True:
                line = await queue.get()
                log_widget.write(line)
                self.stats.ingest(line)
                if line.startswith("[exit"):
                    break
            self.active_train_proc = None

        self.active_train_task = asyncio.create_task(_run())
        await self._toast(f"Started {run_name}")

    async def _checkpoint_now(self) -> None:
        await self._toast("Checkpoint signal not available for this process type")

    async def _run_eval(self) -> None:
        prompt_widget = self.query_one("#prompt-input", TextArea)
        mode_input = self.query_one("#eval-mode", Input)
        mode = mode_input.value.strip() or "continuation"
        prompt = prompt_widget.text.strip()
        output = self.query_one("#eval-output", Log)
        if not prompt:
            await self._toast("Prompt required")
            return
        ckpt = next((c for c in self.checkpoints if c.compatible), None)
        if not ckpt and self.checkpoints:
            ckpt = self.checkpoints[0]
            await self._toast("No compatible checkpoint; using first available (manual check advised)")
        if not ckpt:
            await self._toast("No compatible checkpoint found")
            return
        cmd = self.orchestrator.build_eval_command(ckpt.path, prompt, mode=mode, config=self.current_config.data)
        output.write(f"Running: {' '.join(cmd)}")

        async def _run() -> None:
            base_dir = Path(
                self.current_config.data.get("nanochat_base_dir")
                or os.environ.get("NANOCHAT_BASE_DIR")
                or Path.home() / ".cache" / "nanochat"
            ).expanduser()
            env = {"NANOCHAT_BASE_DIR": str(base_dir)}
            queue, _ = await self.orchestrator.stream_process(cmd, env=env)
            while True:
                line = await queue.get()
                output.write(line)
                if line.startswith("[exit"):
                    break

        asyncio.create_task(_run())

    async def _toast(self, message: str) -> None:
        self.sub_title = message

    def _tick_stats(self) -> None:
        if not self.stats.values:
            return
        lines = [f"{k}: {v}" for k, v in self.stats.values.items()]
        self.query_one("#train-stats", Static).update("\n".join(lines))

    async def _run_dataset(self) -> None:
        shards_input = self.query_one("#dataset-shards", Input)
        shard_val = shards_input.value.strip()
        shard_count = shard_val or str(self.current_config.data.get("dataset_shards", "200"))
        log = self.query_one("#dataset-log", Log)
        cmd = self.orchestrator.plugin.dataset_command(self.current_config.data, shard_count)
        base_dir = Path(
            self.current_config.data.get("nanochat_base_dir")
            or os.environ.get("NANOCHAT_BASE_DIR")
            or Path.home() / ".cache" / "nanochat"
        ).expanduser()
        env = {"NANOCHAT_BASE_DIR": str(base_dir)}
        log.write(f"Running: {' '.join(cmd)}")

        async def _run() -> None:
            queue, _ = await self.orchestrator.stream_process(cmd, env=env)
            while True:
                line = await queue.get()
                log.write(line)
                if line.startswith("[exit"):
                    break

        asyncio.create_task(_run())

    def _update_setup_status(self) -> None:
        status_widget = self.query_one("#setup-status", Static)
        checks = {
            "python": True,
            "uv": shutil.which("uv") is not None,
            "nanochat_import": self._can_import_nanochat(),
        }
        self.setup_ready = all(checks.values())
        lines = [f"{name}: {'ok' if ok else 'missing'}" for name, ok in checks.items()]
        status_widget.update("\n".join(lines))
        if not self.setup_ready:
            self.set_focus(status_widget)

    def _can_import_nanochat(self) -> bool:
        try:
            import importlib

            importlib.import_module("nanochat")
            return True
        except Exception:
            return False

    async def _run_setup(self) -> None:
        log = self.query_one("#setup-log", Log)
        script = "./mc_setup"
        if not Path(script).exists():
            await self._toast("mc_setup not found; skipping")
            return
        cmd = ["bash", script]
        log.write(f"Running: {' '.join(cmd)}")

        async def _run() -> None:
            queue, _ = await self.orchestrator.stream_process(cmd)
            while True:
                line = await queue.get()
                log.write(line)
                if line.startswith("[exit"):
                    break
            self._update_setup_status()

        asyncio.create_task(_run())
