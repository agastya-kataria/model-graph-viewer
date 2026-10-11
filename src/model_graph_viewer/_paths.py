"""Output preflight and atomic writes shared by orchestration and the CLI."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory


def _expand(path: str | Path) -> Path:
    try:
        return Path(path).expanduser()
    except RuntimeError as exc:
        raise ValueError(f"Cannot expand path '{path}': {exc}") from exc


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve path '{path}': {exc}") from exc


def _same_file(first: Path, second: Path) -> bool:
    if _resolve(first) == _resolve(second):
        return True
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"Cannot compare paths '{first}' and '{second}': {exc}") from exc


def validate_output_paths(
    model: str | Path,
    output: str | Path,
    stats: str | Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Reject outputs that alias inputs, including symlinks and hard links.

    OpenVINO may resolve a symlinked XML relative to its supplied or resolved
    filename. Protect both associated weight paths rather than guessing which
    one the reader uses. No files or directories are created by this check.
    """
    model_path, output_path = _expand(model), _expand(output)
    stats_path = _expand(stats) if stats is not None else None
    protected = [model_path]
    if model_path.suffix.lower() == ".xml":
        protected.extend([model_path.with_suffix(".bin"), _resolve(model_path).with_suffix(".bin")])
    destinations = [("SVG output", output_path)]
    if stats_path is not None:
        destinations.append(("Statistics output", stats_path))
    for description, destination in destinations:
        for source in protected:
            if _same_file(destination, source):
                raise ValueError(
                    f"{description} '{destination}' must not overwrite the model "
                    f"or its weights ('{source}'). Choose a different output path."
                )
    if stats_path is not None and _same_file(stats_path, output_path):
        raise ValueError("Statistics output must be different from the SVG output path.")
    return model_path, output_path, stats_path


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file only after writing succeeds; always remove staging files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-graph-stats-", dir=path.parent) as directory:
        staged = Path(directory) / "statistics.json"
        staged.write_text(text, encoding="utf-8")
        os.replace(staged, path)
