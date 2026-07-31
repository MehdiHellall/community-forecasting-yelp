"""Project path helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def resolve_repo_path(path: str | Path | None, default: Path) -> Path:
    """Resolve a CLI path relative to the current working directory."""
    if path is None:
        return default
    return Path(path).expanduser().resolve()
