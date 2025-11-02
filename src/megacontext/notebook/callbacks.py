"""
Lightning callbacks tailored for notebook usage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from IPython.display import clear_output, display  # type: ignore

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

from lightning.pytorch.callbacks import Callback  # type: ignore


class MetricsTracker(Callback):
    """
    Collects logged metrics during training so notebooks can render curves afterwards.
    """

    def __init__(
        self,
        metric_keys: Iterable[str] | None = None,
        *,
        live_output=None,
        plot_every: int = 20,
    ) -> None:
        super().__init__()
        self.metric_keys = tuple(metric_keys) if metric_keys else ()
        self.history: list[dict[str, float]] = []
        self.live_output = live_output
        self.plot_every = max(1, int(plot_every))
        self._last_live_step: int = -1

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
            self._maybe_update_live_plot(step)

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
        fig = self._render_plot(figsize=figsize)
        if fig is None:
            return
        display(fig)
        plt.close(fig)

    # Internal helpers -----------------------------------------------------

    def _build_series(self) -> tuple[list[float], dict[str, list[float]]]:
        series = defaultdict(list)
        series = defaultdict(list)
        for entry in self.history:
            for key, value in entry.items():
                series[key].append(value)

        steps = series.pop("step", [])
        return steps, series

    def _render_plot(self, *, figsize: tuple[int, int]) -> plt.Figure | None:
        if plt is None:  # pragma: no cover - optional dependency
            return None
        steps, series = self._build_series()
        if not steps:
            return None
        fig, ax = plt.subplots(figsize=figsize)
        for key, values in series.items():
            ax.plot(steps, values, label=key)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2)
        if series:
            ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _maybe_update_live_plot(self, step: int) -> None:
        if (
            self.live_output is None
            or plt is None  # pragma: no cover - optional dependency
            or step == self._last_live_step
            or step % self.plot_every != 0
        ):
            return
        fig = self._render_plot(figsize=(6, 4))
        if fig is None:
            return
        with self.live_output:
            clear_output(wait=True)
            display(fig)
        plt.close(fig)
        self._last_live_step = step
