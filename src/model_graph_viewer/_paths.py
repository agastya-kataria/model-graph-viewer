"""Output preflight and atomic writes shared by orchestration and the CLI."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from .metadata import metadata_path_for


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
    metadata: str | Path | None = None,
) -> tuple[Path, Path, Path | None, Path]:
    """Reject outputs that alias inputs, including symlinks and hard links.

    OpenVINO may resolve a symlinked XML relative to its supplied or resolved
    filename. Protect both associated weight paths rather than guessing which
    one the reader uses. No files or directories are created by this check.
    """
    model_path, output_path = _expand(model), _expand(output)
    stats_path = _expand(stats) if stats is not None else None
    metadata_path = metadata_path_for(output_path, metadata)
    protected = [model_path]
    if model_path.suffix.lower() == ".xml":
        protected.extend([model_path.with_suffix(".bin"), _resolve(model_path).with_suffix(".bin")])
    destinations = [("SVG output", output_path), ("Metadata output", metadata_path)]
    if stats_path is not None:
        destinations.append(("Statistics output", stats_path))
    for description, destination in destinations:
        for source in protected:
            if _same_file(destination, source):
                raise ValueError(
                    f"{description} '{destination}' must not overwrite the model "
                    f"or its weights ('{source}'). Choose a different output path."
                )
    for index, (description, destination) in enumerate(destinations):
        for other_description, other in destinations[index + 1 :]:
            if _same_file(destination, other):
                raise ValueError(
                    f"{description} and {other_description.lower()} must be different."
                )
    return model_path, output_path, stats_path, metadata_path


def publish_files(files: Sequence[tuple[Path, Path]]) -> None:
    """Publish staged files together, restoring previous destinations on failure.

    Each staged file lives on its destination filesystem. Backups are prepared
    before any replacement; rollback also restores destinations that were absent.
    This provides failure recovery, rather than cross-file atomic visibility.
    """
    with ExitStack() as stack:
        backups: dict[Path, Path | None] = {}
        for _, destination in files:
            if destination.exists() or destination.is_symlink():
                directory = stack.enter_context(
                    TemporaryDirectory(prefix=".model-graph-backup-", dir=destination.parent)
                )
                backup = Path(directory) / "previous"
                shutil.copy2(destination, backup, follow_symlinks=False)
                backups[destination] = backup
            else:
                backups[destination] = None
        replaced = []
        try:
            for staged, destination in files:
                os.replace(staged, destination)
                replaced.append(destination)
        except OSError:
            for destination in reversed(replaced):
                backup = backups[destination]
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            raise


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file only after writing succeeds; always remove staging files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-graph-stats-", dir=path.parent) as directory:
        staged = Path(directory) / "statistics.json"
        staged.write_text(text, encoding="utf-8")
        os.replace(staged, path)
