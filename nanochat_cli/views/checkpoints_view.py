from __future__ import annotations

from typing import Dict, List
import os
from pathlib import Path

from textual.containers import Vertical
from textual.binding import Binding
from textual.message import Message
from textual.widgets import DataTable, Static

from ..checkpoints import CheckpointRecord, CheckpointRegistry
from .config_view import ConfigSelected, ConfigChanged
from .setup_view import SetupPathsUpdated


class CheckpointsRefreshed(Message):
    def __init__(self, records: List[CheckpointRecord]) -> None:
        self.records = records
        super().__init__()


class CheckpointsView(Vertical):
    """Checkpoint listing with compatibility info."""

    can_focus = True
    CSS_PATH = "styles/checkpoints.tcss"

    BINDINGS = [
        Binding("ctrl+r", "refresh_checkpoints", "Refresh checkpoints"),
    ]

    def __init__(self, registry: CheckpointRegistry, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.registry = registry
        self.records: List[CheckpointRecord] = []
        self.table = DataTable(zebra_stripes=True, id="checkpoint-table")
        self.status = Static("")
        self.last_config: Dict[str, object] = {}
        self.base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat")).expanduser()

    def compose(self):
        self.table.add_columns("Stage", "Path", "Compatible", "Reasons")
        yield self.table
        yield self.status

    def update_records(self, config: Dict[str, object]) -> None:
        self.last_config = config
        self.records = self.registry.list_checkpoints(config)
        self.table.clear()
        for rec in self.records:
            self.table.add_row(
                rec.stage,
                str(rec.path),
                "yes" if rec.compatible else "no",
                "; ".join(rec.reasons) if rec.reasons else "",
            )
        self.status.update(f"{len(self.records)} checkpoints")
        self.post_message(CheckpointsRefreshed(self.records))

    def action_refresh_checkpoints(self) -> None:
        self.update_records(self.last_config)

    # Cross-view messages
    def handle_app_message(self, message: Message):
        if isinstance(message, (ConfigSelected, ConfigChanged)):
            self.last_config = message.bundle.data
            base = Path(
                message.bundle.data.get("nanochat_base_dir")
                or os.environ.get("NANOCHAT_BASE_DIR")
                or Path.home() / ".cache" / "nanochat"
            ).expanduser()
            self.base_dir = base
            self.registry.set_base_dir(base)
            self.update_records(self.last_config)
        elif isinstance(message, SetupPathsUpdated):
            self.base_dir = message.checkpoints_dir
            self.registry.set_base_dir(message.checkpoints_dir)
            self.update_records(self.last_config)
