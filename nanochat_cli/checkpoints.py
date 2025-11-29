from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclasses.dataclass
class CheckpointRecord:
    path: Path
    stage: str
    metadata: Dict[str, Any]
    compatible: bool
    reasons: List[str]


class CheckpointRegistry:
    """Scans nanochat checkpoints and scores compatibility with a config."""

    def __init__(self) -> None:
        self.base_dir: Optional[Path] = None

    def set_base_dir(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def list_checkpoints(self, config: Dict[str, Any]) -> List[CheckpointRecord]:
        paths = self._find_checkpoint_paths(config)
        records: List[CheckpointRecord] = []
        for path, stage in paths:
            metadata = self._load_metadata(path)
            compatible, reasons = self._compatibility(config, metadata)
            records.append(
                CheckpointRecord(path=path, stage=stage, metadata=metadata, compatible=compatible, reasons=reasons)
            )
        def _mtime(rec: CheckpointRecord) -> float:
            try:
                return rec.path.stat().st_mtime
            except OSError:
                return 0

        return sorted(records, key=_mtime, reverse=True)

    def _find_checkpoint_paths(self, config: Dict[str, Any]) -> List[Tuple[Path, str]]:
        results: List[Tuple[Path, str]] = []
        base_dir = Path(
            config.get("nanochat_base_dir")
            or os.environ.get("NANOCHAT_BASE_DIR")
            or Path.home() / ".cache" / "nanochat"
        ).expanduser()
        if self.base_dir:
            base_dir = self.base_dir
        for stage in ("base", "mid", "chat"):
            stage_dir = base_dir / stage
            if not stage_dir.exists():
                continue
            for path in stage_dir.rglob("*"):
                if path.is_file() and path.suffix in {".pt", ".pth", ".bin", ".safetensors"}:
                    results.append((path, stage))
        return results

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        # Heuristic: look for metadata.json sibling or in parent.
        for candidate in [path.with_suffix(".metadata.json"), path.parent / "metadata.json"]:
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {}

    def _compatibility(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if not metadata:
            reasons.append("metadata missing; manual check recommended")
            return False, reasons
        mc_flag = bool(config.get("mc_enabled") or config.get("mc", {}).get("enabled"))
        if "mc_enabled" in metadata and metadata["mc_enabled"] != mc_flag:
            reasons.append("MC flag mismatch")
        if "block_size" in metadata and metadata["block_size"] != config.get("block_size"):
            reasons.append(f"block_size mismatch ({metadata['block_size']} vs {config.get('block_size')})")
        if "depth" in metadata and metadata["depth"] != config.get("depth"):
            reasons.append(f"depth mismatch ({metadata['depth']} vs {config.get('depth')})")
        if "vocab_size" in metadata and metadata["vocab_size"] != config.get("vocab_size"):
            reasons.append("vocab mismatch")
        return len(reasons) == 0, reasons
