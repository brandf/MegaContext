from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Optional, Iterable, Callable, Awaitable

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, TabPane, TabbedContent, Button, Label
from textual.containers import Horizontal, Container
from datetime import datetime

from .config import ConfigBundle, ConfigManager
from .plugin import PluginRegistry
from .checkpoints import CheckpointRegistry
from .views.config_view import ConfigView, ConfigChanged, ConfigSelected
from .views.checkpoints_view import CheckpointsView
from .views.dataset_view import DatasetView
from .views.eval_view import EvalView
from .views.setup_view import SetupView
from .views.train_view import TrainView
from .messages import TabSwitchRequest
from .quit_screen import QuitConfirmScreen


class TopBar(Container):
    """Custom top bar with quit button, title, and clock."""

    def __init__(self) -> None:
        super().__init__(id="app-header")
        self.title_label = Label("", id="title-label")
        self.clock_label = Label("", id="clock-label")
        self.quit_btn = Button("◎", id="quit-btn", tooltip="Quit nanochat-cli")
        self.spacer_left = Label("", classes="header-spacer")
        self.spacer_right = Label("", classes="header-spacer")

    def compose(self) -> ComposeResult:
        yield Container(
            self.quit_btn,
            self.spacer_left,
            self.title_label,
            self.spacer_right,
            self.clock_label,
            id="app-header-row",
        )

    async def on_mount(self) -> None:
        self.title_label.update(self.app.title or "nanochat-cli")
        self.set_interval(1.0, self._tick_clock)

    def _tick_clock(self) -> None:
        self.clock_label.update(datetime.now().strftime("%H:%M:%S"))

    def update_title(self, title: str) -> None:
        self.title_label.update(title)

    async def on_mouse_enter(self, event: events.MouseEnter) -> None:
        if event.sender is self.quit_btn:
            self.quit_btn.label = "ⓧ"

    async def on_mouse_leave(self, event: events.MouseLeave) -> None:
        if event.sender is self.quit_btn:
            self.quit_btn.label = "◎"


class NanochatApp(App):
    """Textual TUI for nanochat workflows."""

    CSS_PATH = "styles/app.tcss"

    def __init__(self, config_manager: Optional[ConfigManager] = None) -> None:
        super().__init__()
        self.config_manager = config_manager or ConfigManager()
        self.plugin_registry = PluginRegistry()
        self.checkpoint_registry = CheckpointRegistry()
        self.current_config: Optional[ConfigBundle] = None
        # views (each handles its own bindings/actions)
        self.setup_view = SetupView()
        self.config_view = ConfigView(self.config_manager, plugin_registry=self.plugin_registry)
        self.dataset_view = DatasetView(plugin_registry=self.plugin_registry, checkpoint_registry=self.checkpoint_registry)
        self.checkpoints_view = CheckpointsView(self.checkpoint_registry)
        self.train_view = TrainView(plugin_registry=self.plugin_registry)
        self.eval_view = EvalView(plugin_registry=self.plugin_registry)
        self._views = [self.setup_view, self.config_view, self.dataset_view, self.checkpoints_view, self.train_view, self.eval_view]
        self._tab_map = {
            "tab-setup": self.setup_view,
            "tab-config": self.config_view,
            "tab-dataset": self.dataset_view,
            "tab-checkpoints": self.checkpoints_view,
            "tab-train": self.train_view,
            "tab-eval": self.eval_view,
        }

    def compose(self) -> ComposeResult:
        # Custom header with a quit affordance
        self.top_bar = TopBar()
        yield self.top_bar
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
        self.set_focus(self.setup_view)
        # ensure title label reflects current title
        self.top_bar.update_title(self.title)

    async def on_message(self, message) -> None:
        await super().on_message(message)
        # broadcast to sibling views for loose coupling
        for view in self._views:
            if view is message.sender:
                continue
            handler: Optional[Callable[[object], Awaitable[None]]] = getattr(view, "handle_app_message", None)
            if handler:
                result = handler(message)
                if hasattr(result, "__await__"):
                    await result
        # minimal app-level effects
        if isinstance(message, ConfigSelected):
            self._set_config(message.bundle)
        elif isinstance(message, ConfigChanged):
            self.current_config = message.bundle
            self.title = f"nanochat-cli — {message.bundle.name}"
            self.top_bar.update_title(self.title)
        elif isinstance(message, TabSwitchRequest):
            try:
                self.query_one(TabbedContent).active = message.tab_id
                target = self._tab_map.get(message.tab_id)
                if target:
                    self.set_focus(target)
            except Exception:
                pass

    async def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        target_id = None
        if hasattr(event, "tab") and event.tab is not None:
            target_id = event.tab.id
        elif hasattr(event, "pane") and event.pane is not None:
            target_id = event.pane.id
        elif hasattr(event, "tab_id"):
            target_id = event.tab_id
        target = self._tab_map.get(target_id)
        if target:
            self.set_focus(target)

    async def on_button_pressed(self, event) -> None:
        if getattr(event.button, "id", "") == "quit-btn":
            await self.push_screen(QuitConfirmScreen(), callback=self._handle_quit_result)

    async def _handle_quit_result(self, result: bool | None) -> None:
        if result:
            await self.action_quit()

    # Helpers
    def _set_config(self, bundle: ConfigBundle) -> None:
        self.current_config = bundle
        self.title = f"nanochat-cli — {bundle.name}"
        msg = ConfigSelected(bundle)
        for view in self._views:
            handler = getattr(view, "handle_app_message", None)
            if handler:
                result = handler(msg)
                if hasattr(result, "__await__"):
                    import asyncio
                    asyncio.create_task(result)
