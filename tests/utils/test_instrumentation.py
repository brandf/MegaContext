import sys
from pathlib import Path
from types import SimpleNamespace

from megacontext.utils import WANDB_ENV_FLAG, maybe_init_wandb, setup_logging


def test_setup_logging_creates_log(tmp_path: Path, monkeypatch) -> None:
    logger = setup_logging("test-run", log_dir=tmp_path)
    logger.info("hello world")
    log_files = list(tmp_path.glob("test-run-*.log"))
    assert log_files, "expected log file to be created"


def test_maybe_init_wandb_disabled(monkeypatch) -> None:
    monkeypatch.delenv(WANDB_ENV_FLAG, raising=False)
    run = maybe_init_wandb(config={"foo": 1}, project="demo")
    assert run is None


def test_maybe_init_wandb_enabled(monkeypatch) -> None:
    calls = {}

    def fake_init(**kwargs):
        calls["kwargs"] = kwargs
        return SimpleNamespace(finish=lambda: None, log=lambda *args, **kwargs: None)

    monkeypatch.setenv(WANDB_ENV_FLAG, "1")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))

    run = maybe_init_wandb(config={"foo": 1}, project="demo", run_name="test")
    assert run is not None
    assert calls["kwargs"]["project"] == "demo"
    assert calls["kwargs"]["config"]["foo"] == 1
