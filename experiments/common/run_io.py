"""Run directory I/O and environment capture."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"


def git_commit(repo: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return None


def collect_environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "commit": git_commit(),
        "upstream": "https://github.com/pathwaycom/bdh",
        "fork": "https://github.com/mpiekarski78/bdh",
    }
    if torch.cuda.is_available():
        env["cudnn_version"] = torch.backends.cudnn.version()
    if extra:
        env.update(extra)
    return env


def next_run_dir(prefix: str | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    n = 1
    while True:
        name = f"{day}_{n:03d}"
        if prefix:
            name = f"{day}_{prefix}_{n:03d}"
        path = runs_dir / name
        if not path.exists():
            path.mkdir(parents=True)
            (path / "state").mkdir()
            (path / "checkpoints").mkdir()
            (path / "plots").mkdir()
            return path
        n += 1


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def init_run(
    config: dict[str, Any],
    *,
    prefix: str | None = None,
    env_extra: dict[str, Any] | None = None,
) -> Path:
    run_dir = next_run_dir(prefix=prefix)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", collect_environment(env_extra))
    return run_dir


def write_summary(run_dir: Path, text: str) -> None:
    (run_dir / "summary.md").write_text(text if text.endswith("\n") else text + "\n")
