#!/usr/bin/env python3
"""Fetch the pinned OpenVINO model, resuming partial downloads and verifying SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verified(path: Path, expected: dict) -> bool:
    """Check both the byte count and digest without reading weights into memory."""
    if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
        return False
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest == expected["sha256"]


def download(directory: Path) -> None:
    source = json.loads((ROOT / "examples/model_sources.json").read_text())["tinyllama"]
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for HTTPS downloads; install it and try again.")
    directory.mkdir(parents=True, exist_ok=True)
    for entry in source["files"]:
        destination = directory / entry["name"]
        if verified(destination, entry):
            print(f"Verified existing {destination}", flush=True)
            continue
        if destination.exists():
            raise RuntimeError(
                f"Checksum mismatch: {destination}. Move or remove it before downloading again."
            )
        partial = destination.with_suffix(destination.suffix + ".part")
        url = (
            f"https://huggingface.co/{source['repository']}/resolve/"
            f"{source['revision']}/{entry['name']}"
        )
        if not verified(partial, entry):
            print(f"Downloading {entry['name']} ({entry['size_bytes']:,} bytes)", flush=True)
            subprocess.run(
                [
                    curl,
                    "--fail",
                    "--location",
                    "--show-error",
                    "--silent",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "30",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    url,
                ],
                check=True,
            )
        if not verified(partial, entry):
            raise RuntimeError(
                f"Size or SHA-256 mismatch: {partial}. Remove this partial file and retry."
            )
        partial.replace(destination)
        print(f"Verified {destination}: SHA-256 {entry['sha256']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "models/tinyllama")
    args = parser.parse_args()
    try:
        download(args.directory)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
