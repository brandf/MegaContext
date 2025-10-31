"""
Lightning callbacks tailored for notebook usage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

from lightning.pytorch.callbacks import Callback  # type: ignore


class MetricsTracker(Callback):
    """
    Collects logged metrics during training so notebooks can render curves afterwards.
    """

    def __init__(self, metric_keys: Iterable[str] | None = None) -> None:
        super().__init__()
        self.metric_keys = tuple(metric_keys) if metric_keys else ()
        self.history: list[dict[str, float]] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore[override]  # pragma: no cover - Lightning runtime
        step = int(trainer.global_step)
        logged = trainer.logged_metrics
        record: dict[str, float] = {"step": step}
        if self.metric_keys:
            keys = self.metric_keys
        else:
            keys = tuple(logged.keys())
        for key in keys:
            if key in logged:
                try:
                    record[key] = float(logged[key].item())  # type: ignore[call-arg]
                except Exception:
                    record[key] = float(logged[key])
        if len(record) > 1:
            self.history.append(record)

    def plot(self, *, figsize: tuple[int, int] = (6, 4)) -> None:
        """Render tracked metrics using matplotlib."""

        if not self.history:
            print("No metrics captured yet.")
            return
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for plotting metrics. "
                "Install it with `pip install matplotlib`."
            )
        series = defaultdict(list)
        for entry in self.history:
            for key, value in entry.items():
                series[key].append(value)

        steps = series.pop("step")
        fig, ax = plt.subplots(figsize=figsize)
        for key, values in series.items():
            ax.plot(steps, values, label=key)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        plt.show()
