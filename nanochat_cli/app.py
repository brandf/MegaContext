from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabPane, TabbedContent

from .config import ConfigBundle, ConfigManager
from .plugin import PluginRegistry
from .orchestrator import RunOrchestrator
from .checkpoints import CheckpointRegistry
from .views.config_view import ConfigView, ConfigChanged, ConfigSelected
from .views.checkpoints_view import CheckpointsView
from .views.dataset_view import DatasetView, DatasetStatus, DatasetDependencyMissing
from .views.eval_view import EvalView
from .views.setup_view import SetupView
from .views.train_view import TrainView


class NanochatApp(App):
    """Textual TUI for nanochat workflows."""

    CSS = ""
    BINDINGS = [
        Binding("ctrl+b", "check_setup", "Check setup", show=False),
        Binding("ctrl+u", "run_setup", "Run setup", show=False),
        Binding("ctrl+r", "refresh_checkpoints", "Refresh checkpoints", show=False),
        Binding("ctrl+s", "save_config", "Save config", show=False),
        Binding("ctrl+shift+s", "save_as_config", "Save config as", show=False),
        Binding("ctrl+l", "reload_config", "Reload config", show=False),
        Binding("ctrl+d", "run_dataset", "Run dataset prep", show=False),
        Binding("ctrl+t", "start_train", "Start train", show=False),
    ]

    def __init__(self, config_manager: Optional[ConfigManager] = None) -> None:
        super().__init__()
        self.config_manager = config_manager or ConfigManager()
        self.plugin_registry = PluginRegistry()
        self.checkpoint_registry = CheckpointRegistry()
        self.current_config: Optional[ConfigBundle] = None
        self.base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat")).expanduser()
        self.orchestrator = RunOrchestrator(self.plugin_registry.resolve({}))
        # views
        self.setup_view = SetupView()
        self.config_view = ConfigView(self.config_manager)
        self.dataset_view = DatasetView(self.orchestrator)
        self.checkpoints_view = CheckpointsView(self.checkpoint_registry)
        self.train_view = TrainView(self.orchestrator)
        self.eval_view = EvalView(self.orchestrator)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            yield TabPane("Setup", self.setup_view, id="tab-setup")
            yield TabPane("Config", self.config_view, id="tab-config")
            yield TabPane("Dataset", self.dataset_view, id="tab-dataset")
            yield TabPane("Checkpoints", self.checkpoints_view, id="tab-checkpoints")
            yield TabPane("Train", self.train_view, id="tab-train")
            yield TabPane("Eval", self.eval_view, id="tab-eval")
        yield Footer()

    async def on_mount(self) -> None:
        # load initial config
        configs = self.config_manager.list_prefabs()
        if configs:
            self._set_config(configs[0])
        self.setup_view.check_status()
        self.dataset_view.check_status()
        self.checkpoints_view.update_records(self.current_config.data if self.current_config else {})
        self._sync_binding_visibility()

    # Actions
    async def action_refresh_checkpoints(self) -> None:
        if self.query_one(TabbedContent).active == "tab-checkpoints":
            self.checkpoints_view.update_records(self.current_config.data if self.current_config else {})

    async def action_check_setup(self) -> None:
        if self.query_one(TabbedContent).active == "tab-setup":
            self.setup_view.check_status()

    async def action_run_setup(self) -> None:
        if self.query_one(TabbedContent).active == "tab-setup":
            await self.setup_view._run_setup()

    async def action_save_config(self) -> None:
        if self.query_one(TabbedContent).active == "tab-config":
            self.config_view.save_current()

    async def action_save_as_config(self) -> None:
        if self.query_one(TabbedContent).active == "tab-config":
            self.config_view.save_as_current()

    async def action_reload_config(self) -> None:
        if self.query_one(TabbedContent).active == "tab-config":
            self.config_view.reload_current()

    async def action_run_dataset(self) -> None:
        if self.query_one(TabbedContent).active == "tab-dataset":
            await self.dataset_view.on_button_pressed(type("evt", (), {"button": type("btn", (), {"id": "dataset-run"})})())  # reuse handler

    async def action_start_train(self) -> None:
        if self.query_one(TabbedContent).active == "tab-train":
            await self.train_view._start()

    # Message handlers
    async def on_config_selected(self, message: ConfigSelected) -> None:
        self._set_config(message.bundle)

    async def on_config_changed(self, message: ConfigChanged) -> None:
        # keep current config in sync
        self.current_config = message.bundle
        self._update_base_dir()
        self.train_view.set_config(message.bundle)
        self.eval_view.set_checkpoint(None)

    async def on_dataset_status(self, message: DatasetStatus) -> None:
        self.train_view.set_dataset_ready(message.ready)
        if not message.ready:
            # highlight dataset tab
            self.sub_title = "Dataset missing—prep before training"

    async def on_dataset_dependency_missing(self, message) -> None:
        self.sub_title = f"Missing dependency: {message.dependency}. Run setup."
        self.query_one(TabbedContent).active = "tab-setup"

    async def on_setup_paths_updated(self, message) -> None:
        self.base_dir = message.base_dir
        self.dataset_view.update_base_dir(message.base_dir)
        self.dataset_view.update_dataset_dir(message.dataset_dir)
        self.checkpoint_registry.set_base_dir(message.checkpoints_dir)
        self.train_view.set_base_dir(message.base_dir)
        self.eval_view.set_base_dir(message.base_dir)
        self.checkpoints_view.update_records(self.current_config.data if self.current_config else {})

    async def on_train_dataset_missing(self, message) -> None:
        # Switch to dataset tab when train attempted without dataset
        self.query_one(TabbedContent).active = "tab-dataset"
        self._sync_binding_visibility()

    # Helpers
    def _set_config(self, bundle: ConfigBundle) -> None:
        self.current_config = bundle
        self.title = f"nanochat-cli — {bundle.name}"
        # Update plugin/orchestrator
        self.orchestrator = RunOrchestrator(self.plugin_registry.resolve(bundle.data))
        # update dependent views
        self.train_view.set_config(bundle)
        self._update_base_dir()
        self.dataset_view.orchestrator = self.orchestrator
        self.checkpoints_view.update_records(bundle.data)
        self.eval_view.set_checkpoint(None)

    def _update_base_dir(self) -> None:
        if not self.current_config:
            return
        self.base_dir = Path(
            self.current_config.data.get("nanochat_base_dir")
            or os.environ.get("NANOCHAT_BASE_DIR")
            or Path.home() / ".cache" / "nanochat"
        ).expanduser()
        self.dataset_view.update_base_dir(self.base_dir)
        self.checkpoint_registry.set_base_dir(self.base_dir)
        self.train_view.set_base_dir(self.base_dir)
        self.eval_view.set_base_dir(self.base_dir)
        self.checkpoints_view.update_records(self.current_config.data if self.current_config else {})

    async def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._sync_binding_visibility()

    def _sync_binding_visibility(self) -> None:
        """Show only bindings relevant to the active tab."""
        active = self.query_one(TabbedContent).active
        show_for_tab = {
            "tab-setup": {"ctrl+b", "ctrl+u"},
            "tab-config": {"ctrl+s", "ctrl+shift+s", "ctrl+l"},
            "tab-dataset": {"ctrl+d"},
            "tab-checkpoints": {"ctrl+r"},
            "tab-train": {"ctrl+t"},
        }
        visible_keys = show_for_tab.get(active, set())
        binding_defs = {
            "ctrl+b": ("check_setup", "Check setup"),
            "ctrl+u": ("run_setup", "Run setup"),
            "ctrl+r": ("refresh_checkpoints", "Refresh checkpoints"),
            "ctrl+s": ("save_config", "Save config"),
            "ctrl+shift+s": ("save_as_config", "Save config as"),
            "ctrl+l": ("reload_config", "Reload config"),
            "ctrl+d": ("run_dataset", "Run dataset prep"),
            "ctrl+t": ("start_train", "Start train"),
        }
        for key, (action, desc) in binding_defs.items():
            # Rebind with updated visibility; bind() overwrites existing binding
            self.bind(key, action, description=desc, show=key in visible_keys)
