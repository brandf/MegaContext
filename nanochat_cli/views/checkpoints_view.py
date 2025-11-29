from __future__ import annotations

from typing import Dict, List

from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Static

from ..checkpoints import CheckpointRecord, CheckpointRegistry


class CheckpointsRefreshed(Message):
    def __init__(self, records: List[CheckpointRecord]) -> None:
        self.records = records
        super().__init__()


class CheckpointsView(Vertical):
    """Checkpoint listing with compatibility info."""

    def __init__(self, registry: CheckpointRegistry, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.registry = registry
        self.records: List[CheckpointRecord] = []
        self.table = DataTable(zebra_stripes=True, id="checkpoint-table")
        self.status = Static("")

    def compose(self):
        self.table.add_columns("Stage", "Path", "Compatible", "Reasons")
        yield self.table
        yield self.status

    def update_records(self, config: Dict[str, object]) -> None:
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
