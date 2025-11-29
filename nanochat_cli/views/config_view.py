from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label, Input, Select, Static

from ..config import ConfigBundle, ConfigManager, categorize_field, flatten_config


class ConfigSelected(Message):
    def __init__(self, bundle: ConfigBundle) -> None:
        self.bundle = bundle
        super().__init__()


class ConfigChanged(Message):
    def __init__(self, bundle: ConfigBundle, dirty: bool) -> None:
        self.bundle = bundle
        self.dirty = dirty
        super().__init__()


@dataclass
class FieldWidget:
    label: Label
    input: Input
    path: str
    category: str


class ConfigView(Vertical):
    """Categorized config editor with inline dirty tracking."""

    DEFAULT_CATEGORIES = ["setup", "core", "megacontext", "data", "telemetry", "auth", "visualization", "other"]

    def __init__(self, manager: ConfigManager, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self.manager = manager
        self.current: Optional[ConfigBundle] = None
        self.base_data: Dict[str, object] = {}
        self.fields: Dict[str, FieldWidget] = {}
        self.allow_discard = False
        self.status = Static("")
        self.initial_configs: List[ConfigBundle] = []

    def compose(self):
        configs = self.manager.list_prefabs()
        options = [(bundle.name, bundle.name) for bundle in configs]
        selector = Select(options=options, id="config-select", prompt="Select config")
        selector.styles.width = 30
        save_as = Input(placeholder="Save as", id="config-save-as")
        save_as.styles.width = 24
        bar = Horizontal(selector, Button("Save", id="config-save"), Button("Save As", id="config-save-as-btn"), Button("Reload", id="config-reload"), save_as, id="config-top")
        bar.styles.gap = 2
        yield bar
        # Category containers
        columns = []
        for cat in self.DEFAULT_CATEGORIES:
            col = Vertical(Label(cat.title(), classes="config-cat-header"), id=f"config-cat-{cat}")
            col.styles.width = 36
            col.styles.gap = 0
            columns.append(col)
        categories = Horizontal(*columns, id="config-categories")
        categories.styles.gap = 3
        yield VerticalScroll(categories, id="config-scroll")
        self.status.id = "config-status"
        yield self.status
        self.initial_configs = configs

    async def on_mount(self) -> None:
        if self.initial_configs:
            self._load_bundle(self.initial_configs[0])

    # Internal helpers
    def _load_bundle(self, bundle: ConfigBundle) -> None:
        self.current = bundle.clone(bundle.name)
        self.base_data = flatten_config(bundle.data)
        self._render_fields()
        self.post_message(ConfigSelected(self.current))

    def _render_fields(self) -> None:
        # Clear existing columns
        for cat in self.DEFAULT_CATEGORIES:
            column = self.query_one(f"#config-cat-{cat}", Vertical)
            column.remove_children()
            column.mount(Label(cat.title(), classes="config-cat-header"))
        self.fields.clear()
        flat = flatten_config(self.current.data if self.current else {})
        for path, value in flat.items():
            cat = categorize_field(path)
            if cat not in self.DEFAULT_CATEGORIES:
                cat = "other"
            column = self.query_one(f"#config-cat-{cat}", Vertical)
            label = Label(path)
            label.styles.width = 18
            safe_id = f"field-{path.replace('.', '-')}"
            input = Input(value=str(value), id=safe_id)
            input.styles.width = 24
            fw = FieldWidget(label=label, input=input, path=path, category=cat)
            self.fields[path] = fw
            row = Horizontal(label, input, classes="config-row")
            row.styles.gap = 1
            column.mount(row)
        self._update_dirty_labels()

    def _update_dirty_labels(self) -> None:
        for path, fw in self.fields.items():
            base_val = self.base_data.get(path)
            current_val = fw.input.value
            if str(base_val) != str(current_val):
                fw.label.update(f"* {path}")
            else:
                fw.label.update(path)
        dirty = self.is_dirty
        if dirty:
            self.status.update("Unsaved changes")
        else:
            self.status.update("")
        if self.current:
            self.post_message(ConfigChanged(self.current.clone(), dirty))

    @property
    def is_dirty(self) -> bool:
        for path, fw in self.fields.items():
            if str(self.base_data.get(path)) != str(fw.input.value):
                return True
        return False

    def _apply_inputs(self) -> None:
        if not self.current:
            return
        data = dict(self.current.data)
        for path, fw in self.fields.items():
            segments = path.split(".")
            cursor = data
            for seg in segments[:-1]:
                cursor = cursor.setdefault(seg, {})
            cursor[segments[-1]] = self._coerce_value(fw.input.value)
        self.current.data = data

    def _coerce_value(self, raw: str):
        # Simple coercion for ints/bools
        if raw.lower() in {"true", "false"}:
            return raw.lower() == "true"
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw

    # Event handlers
    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "config-save-as":
            return
        if event.input.id and event.input.id.startswith("field-"):
            self._apply_inputs()
            self._update_dirty_labels()
        elif event.input.id == "config-save-as":
            pass

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "config-select":
            return
        bundle = self.manager.load(event.value)
        self._load_bundle(bundle)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "config-save":
            self._apply_inputs()
            if self.current:
                self.manager.save(self.current)
                self.base_data = flatten_config(self.current.data)
                self._update_dirty_labels()
        elif event.button.id == "config-save-as-btn":
            self._apply_inputs()
            name = self.query_one("#config-save-as", Input).value.strip() or (self.current.name if self.current else "")
            if self.current and name:
                self.manager.save(self.current, as_name=name)
                self.current.name = name
                self.base_data = flatten_config(self.current.data)
                self._update_dirty_labels()
        elif event.button.id == "config-reload":
            if self.is_dirty and not self.allow_discard:
                self.status.update("Unsaved changes. Press reload again to discard.")
                self.allow_discard = True
                return
            self.allow_discard = False
            if self.current:
                bundle = self.manager.load(self.current.name)
                self._load_bundle(bundle)

    def save_current(self) -> None:
        self._apply_inputs()
        if self.current:
            self.manager.save(self.current)
            self.base_data = flatten_config(self.current.data)
            self._update_dirty_labels()

    def reload_current(self) -> None:
        if self.current:
            bundle = self.manager.load(self.current.name)
            self._load_bundle(bundle)
