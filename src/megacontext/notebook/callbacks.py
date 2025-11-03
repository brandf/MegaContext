"""
Lightning callbacks tailored for notebook usage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

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
        metric_labels: Mapping[str, str] | None = None,
        reference_lines: Mapping[str, Sequence[float] | float] | None = None,
        y_label: str = "Value",
        warmup_steps: int = 5,
    ) -> None:
        super().__init__()
        self.metric_keys = tuple(metric_keys) if metric_keys else ()
        self.history: list[dict[str, float]] = []
        self.live_output = live_output
        self.plot_every = max(1, int(plot_every))
        self._last_live_step: int = -1
        self.metric_labels = dict(metric_labels or {})
        self.reference_lines = {
            key: (
                tuple(float(v) for v in value)
                if isinstance(value, Sequence) and not isinstance(value, (str | bytes))
                else (float(value),)
            )
            for key, value in (reference_lines or {}).items()
        }
        self.y_label = y_label
        self.warmup_steps = max(0, int(warmup_steps))
        self._post_warmup_ylim: float | None = None

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
        drawn_line_labels: set[str] = set()
        for key, values in series.items():
            label = self.metric_labels.get(key, key)
            if not values:
                continue
            if len(values) == len(steps):
                x_values = steps
            else:
                x_values = steps[-len(values) :]
            ax.plot(x_values, values, label=label)
            for value in self.reference_lines.get(key, ()):
                line_label = f"{label} target ({value:.3f})"
                show_label = line_label not in drawn_line_labels
                ax.axhline(
                    value,
                    color="#888888",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                    label=line_label if show_label else None,
                )
                drawn_line_labels.add(line_label)
        all_values: list[float] = []
        for values in series.values():
            all_values.extend(values)

        if self._post_warmup_ylim is None and len(steps) > self.warmup_steps:
            idx = min(self.warmup_steps, len(steps) - 1)
            warmup_values = [
                values[idx] for values in series.values() if len(values) > idx
            ]
            if warmup_values:
                ref_vals = [
                    val for vals in self.reference_lines.values() for val in vals
                ]
                candidate = (
                    max(warmup_values + ref_vals) if ref_vals else max(warmup_values)
                )
                self._post_warmup_ylim = float(candidate)

        ax.set_xlabel("Global Step")
        ax.set_ylabel(self.y_label)
        ax.grid(True, alpha=0.2)
        if self._post_warmup_ylim is not None:
            top = self._post_warmup_ylim
            if top == 0.0:
                top = 0.05
            margin = abs(top) * 0.05
            min_value = min(all_values) if all_values else 0.0
            ax.set_ylim(
                bottom=min_value - margin if all_values else None,
                top=top + margin,
            )
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
