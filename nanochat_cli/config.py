from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml

ConfigDict = Dict[str, Any]


@dataclasses.dataclass
class ConfigBundle:
    """In-memory representation of a config prefab + overrides."""

    name: str
    data: ConfigDict
    path: Optional[Path] = None
    description: str | None = None

    def clone(self, name: Optional[str] = None) -> "ConfigBundle":
        return ConfigBundle(name or self.name, copy.deepcopy(self.data), None, self.description)


class ConfigManager:
    """Loads/saves prefabs and applies overrides with categorization metadata."""

    BUILTIN_PREFIX = "builtin"

    def __init__(self, user_dir: Optional[Path] = None) -> None:
        self.user_dir = user_dir or Path.home() / ".nanochat-cli"
        self.prefab_dir = self.user_dir / "prefabs"
        self.builtin_dir = Path(__file__).resolve().parent / "prefabs"
        self.prefab_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_builtin_prefabs()

    # Public API
    def list_prefabs(self) -> List[ConfigBundle]:
        bundles: List[ConfigBundle] = []
        for path in sorted(self.prefab_dir.glob("*.yaml")):
            bundles.append(self._load_path(path))
        if not bundles:
            bundles.extend(self._load_builtin())
        return bundles

    def load(self, name: str) -> ConfigBundle:
        candidate = self.prefab_dir / f"{name}.yaml"
        if candidate.exists():
            return self._load_path(candidate)
        for bundle in self._load_builtin():
            if bundle.name == name:
                return bundle
        raise FileNotFoundError(f"No prefab named {name}")

    def save(self, bundle: ConfigBundle, as_name: Optional[str] = None) -> Path:
        name = as_name or bundle.name
        path = self.prefab_dir / f"{name}.yaml"
        payload = copy.deepcopy(bundle.data)
        payload.setdefault("name", name)
        if bundle.description:
            payload.setdefault("description", bundle.description)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        return path

    def apply_overrides(self, bundle: ConfigBundle, overrides: Mapping[str, Any]) -> ConfigBundle:
        merged = copy.deepcopy(bundle.data)
        self._deep_update(merged, overrides)
        new = ConfigBundle(bundle.name, merged, bundle.path, bundle.description)
        return new

    def diff(self, base: ConfigBundle, updated: ConfigBundle) -> Dict[str, Any]:
        return self._dict_diff(base.data, updated.data)

    # Internal helpers
    def _load_path(self, path: Path) -> ConfigBundle:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        name = data.get("name") or path.stem
        desc = data.get("description")
        return ConfigBundle(name=name, data=data, path=path, description=desc)

    def _load_builtin(self) -> List[ConfigBundle]:
        bundles: List[ConfigBundle] = []
        for path in sorted(self.builtin_dir.glob("*.yaml")):
            bundles.append(self._load_path(path))
        return bundles

    def _ensure_builtin_prefabs(self) -> None:
        for path in self.builtin_dir.glob("*.yaml"):
            target = self.prefab_dir / path.name
            if not target.exists():
                target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    def _deep_update(self, target: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _dict_diff(self, base: Mapping[str, Any], other: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
        diff: Dict[str, Any] = {}
        for key, base_val in base.items():
            path = f"{prefix}{key}"
            if key not in other:
                diff[path] = {"removed": base_val}
                continue
            other_val = other[key]
            if isinstance(base_val, Mapping) and isinstance(other_val, Mapping):
                nested = self._dict_diff(base_val, other_val, prefix=f"{path}.")
                diff.update(nested)
            elif base_val != other_val:
                diff[path] = {"from": base_val, "to": other_val}
        for key, other_val in other.items():
            path = f"{prefix}{key}"
            if key not in base:
                diff[path] = {"added": other_val}
        return diff


def categorize_field(key: str) -> str:
    """Rudimentary categorization to drive UI grouping."""
    lowered = key.lower()
    if lowered in {"wandb_api_key", "hf_token"}:
        return "auth"
    if lowered.startswith("mc") or "mc_" in lowered or lowered == "mc":
        return "megacontext"
    if lowered in {"gpu_profile", "device_batch_size", "max_seq_len", "depth"}:
        return "core"
    if "dataset" in lowered or "data" in lowered:
        return "data"
    if "vis" in lowered:
        return "visualization"
    if "telemetry" in lowered or "otel" in lowered:
        return "telemetry"
    return "other"


def flatten_config(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def to_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
