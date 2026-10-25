#!/usr/bin/env python3
"""Verify a wheel in clean Python 3.11–3.13 environments and record the evidence.

Run with a Python containing requirements.lock, for example:
    .venv/bin/python scripts/verify_package.py

The uv command and Graphviz's dot executable must be on PATH. uv may download
missing Python interpreters and dependencies. Temporary environments and copied
test fixtures are removed after the run; docs/verification.json records results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHONS = ("3.11", "3.12", "3.13")

PROBE = """
import importlib.metadata as metadata
import json
import platform
import sys
import model_graph_viewer
print(json.dumps({
    'python': platform.python_version(),
    'executable': sys.executable,
    'prefix': sys.prefix,
    'package_file': model_graph_viewer.__file__,
    'dependencies': dict(sorted(
        (item.metadata['Name'], item.version) for item in metadata.distributions()
    )),
}))
"""

SMOKE_CHECK = """
import importlib.util
import json
import sys
from pathlib import Path
from model_graph_viewer import (
    HideConstantConverts, HideConstants, HighlightNodes, ReferenceHighFanout,
    apply_passes, load_model,
)
from model_graph_viewer.metadata import snapshot_graph
checker_path, model, svg, stats_path = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('artifact_checker', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
graph = load_model(model)
original = snapshot_graph(graph)
apply_passes(graph, [
    HideConstantConverts(), HideConstants(), ReferenceHighFanout(threshold=1),
    HighlightNodes(['projection']),
])
artifact = checker.verify_svg(graph, svg, original, label_detail='compact')
stats = json.loads(stats_path.read_text(encoding='utf-8'))
assert (stats['before_nodes'], stats['before_edges']) == (9, 11), stats
assert (stats['after_nodes'], stats['after_edges']) == (7, 8), stats
assert (stats['rendered_edges'], stats['referenced_edges']) == (4, 4), stats
assert artifact['nodes'] == stats['after_nodes'], artifact
assert artifact['logical_edges'] == stats['after_edges'], artifact
assert artifact['rendered_edges'] == stats['rendered_edges'], artifact
assert artifact['referenced_edges'] == stats['referenced_edges'], artifact
assert Path(stats['output_path']) == svg, stats
assert Path(stats['metadata_path']) == svg.with_suffix('.json'), stats
view = json.loads(svg.with_suffix('.json').read_text(encoding='utf-8'))['view']
highlighted = [
    node for node in view['nodes'].values() if node['visual']['border_color'] == 'red'
]
assert len(highlighted) == 2 and all(node['name'] == 'projection' for node in highlighted)
print(json.dumps({'statistics': stats, 'artifact': artifact, 'highlighted_nodes': 2}))
"""


class VerificationFailure(RuntimeError):
    """A recorded command or evidence check did not succeed."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory: str, pattern: str) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted((ROOT / directory).rglob(pattern))
        if path.is_file()
    }


def input_manifest() -> dict:
    return {
        "source_sha256": hashes("src", "*.py"),
        "test_sha256": hashes("tests", "*.py"),
        "script_sha256": hashes("scripts", "*.py"),
        "fixture_sha256": hashes("examples/models", "*"),
        "requirements_sha256": sha256(ROOT / "requirements.lock"),
        "pyproject_sha256": sha256(ROOT / "pyproject.toml"),
    }


def run_command(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    records: list[dict],
    label: str,
    *,
    timeout: float = 600,
) -> dict:
    """Keep complete command output and emit a short progress line."""
    print(f"{label} ...", flush=True)
    started = perf_counter()
    record = {"label": label, "command": command, "cwd": str(cwd)}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        record.update(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except OSError as exc:
        record.update(exit_code=None, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        record.update(
            exit_code=None,
            stdout=_decode(exc.stdout),
            stderr=_decode(exc.stderr),
            timeout_seconds=timeout,
        )
    record["elapsed_seconds"] = perf_counter() - started
    records.append(record)
    if record["exit_code"] != 0:
        raise VerificationFailure(f"{label} failed; see the recorded command output")
    return record


def _decode(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def copied_inputs(destination: Path) -> dict:
    """Copy only test tooling and the two small IR fixtures, never source imports."""
    destination.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")
    for directory in ("tests", "scripts", "examples/models"):
        shutil.copytree(ROOT / directory, destination / directory, ignore=ignore)
    for filename in ("pyproject.toml", "requirements.lock"):
        shutil.copy2(ROOT / filename, destination / filename)
    if (destination / "src").exists():
        raise VerificationFailure("Copied tests must not have an importable source package")
    files = [path for path in destination.rglob("*") if path.is_file()]
    return {
        "path": str(destination),
        "contains_package_source": False,
        "files": {path.relative_to(destination).as_posix(): sha256(path) for path in sorted(files)},
    }


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    return {
        field: sum(int(suite.get(attribute, "0")) for suite in suites)
        for field, attribute in (
            ("total", "tests"),
            ("failed", "failures"),
            ("errors", "errors"),
            ("skipped", "skipped"),
        )
    }


def verify_environment(
    requested: str,
    temporary: Path,
    clean_source: Path,
    wheel: Path,
    uv: str,
    environment: dict[str, str],
    evidence: dict,
) -> None:
    directory = temporary / requested
    directory.mkdir()
    venv = directory / "environment"
    python = venv / "bin" / "python"
    records = evidence["commands"]

    def run(command, label):
        return run_command(command, directory, environment, records, f"Python {requested}: {label}")

    run([uv, "venv", "--python", requested, str(venv)], "create clean environment")
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(clean_source / "requirements.lock"),
        ],
        "install locked dependencies",
    )
    run([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)], "install wheel")
    probe = json.loads(run([str(python), "-c", PROBE], "inspect installed package")["stdout"])
    evidence.update(probe)
    package = Path(probe["package_file"]).resolve()
    installed = (
        package.is_relative_to(venv.resolve())
        and "site-packages" in package.parts
        and not package.is_relative_to(ROOT)
        and not package.is_relative_to(clean_source)
    )
    evidence["imported_installed_wheel"] = installed
    if not installed or not probe["python"].startswith(requested + "."):
        raise VerificationFailure(f"Python {requested}: installed package provenance is incorrect")
    junit = directory / "tests.xml"
    run(
        [str(python), "-m", "pytest", str(clean_source / "tests"), "-q", f"--junitxml={junit}"],
        "run full tests from outside source",
    )
    evidence["tests"] = junit_counts(junit)
    if (
        evidence["tests"]["total"] == 0
        or evidence["tests"]["failed"]
        or evidence["tests"]["errors"]
    ):
        raise VerificationFailure(
            f"Python {requested}: test results are incomplete or unsuccessful"
        )
    evidence["cli_smoke"] = []
    for name, entrypoint in (
        ("module", [str(python), "-m", "model_graph_viewer"]),
        ("console", [str(venv / "bin" / "model-graph-viewer")]),
    ):
        help_output = run([*entrypoint, "--help"], f"{name} CLI help")["stdout"]
        for option in ("--metadata", "--label-detail", "--fanout-threshold"):
            if option not in help_output:
                raise VerificationFailure(f"Python {requested}: {name} help is missing {option}")
        svg = directory / f"branching-{name}.svg"
        stats = directory / f"branching-{name}.stats.json"
        model = clean_source / "examples/models/branching.xml"
        run(
            [
                *entrypoint,
                "--model",
                str(model),
                "--output",
                str(svg),
                "--passes",
                "hide-constant-converts,hide-constants,reference-high-fanout,highlight",
                "--highlight",
                "projection",
                "--fanout-threshold",
                "1",
                "--stats",
                str(stats),
            ],
            f"{name} CLI transform and render",
        )
        checked = run(
            [
                str(python),
                "-c",
                SMOKE_CHECK,
                str(clean_source / "scripts/benchmark.py"),
                str(model),
                str(svg),
                str(stats),
            ],
            f"validate {name} SVG, metadata, and statistics",
        )
        evidence["cli_smoke"].append(
            {
                "entrypoint": name,
                **json.loads(checked["stdout"]),
                "artifact_sha256": {
                    path.name: sha256(path) for path in (svg, svg.with_suffix(".json"), stats)
                },
            }
        )
    evidence["success"] = True


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".verification-", delete=False
    ) as staged:
        temporary = Path(staged.name)
        try:
            json.dump(report, staged, indent=2, ensure_ascii=True, allow_nan=False)
            staged.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/verification.json")
    parser.add_argument(
        "--python",
        action="append",
        choices=DEFAULT_PYTHONS,
        help="Python version to verify; repeat to override the default 3.11, 3.12, 3.13 matrix",
    )
    args = parser.parse_args()
    versions = list(dict.fromkeys(args.python or DEFAULT_PYTHONS))
    environment = os.environ.copy()
    removed = [
        key
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        if environment.pop(key, None) is not None
    ]
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = perf_counter()
    original = input_manifest()
    report = {
        "schema_version": 2,
        "generator": "scripts/verify_package.py",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "method": (
            "Build a wheel; install locked dependencies and the wheel into clean uv environments; "
            "run copied tests and both CLI entrypoints from outside all source trees."
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "runner_python": sys.version,
        "requested_pythons": versions,
        "environment": {
            "removed_variables": removed,
            "PYTHONPATH_present": False,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        **original,
        "commands": [],
        "environments": [],
        "failures": [],
    }
    try:
        uv = shutil.which("uv")
        dot = shutil.which("dot")
        if uv is None or dot is None:
            raise VerificationFailure("Prerequisites missing: uv and Graphviz dot must be on PATH")
        for command, label in (
            ([uv, "--version"], "uv version"),
            ([dot, "-V"], "Graphviz version"),
        ):
            run_command(command, ROOT, environment, report["commands"], label)
        run_command(
            [sys.executable, "-m", "ruff", "check", "."],
            ROOT,
            environment,
            report["commands"],
            "Ruff lint",
        )
        run_command(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "src",
                "tests",
                "scripts",
                "starter_code.py",
            ],
            ROOT,
            environment,
            report["commands"],
            "Ruff format check",
        )
        with tempfile.TemporaryDirectory(prefix="model-graph-verification-") as directory:
            temporary = Path(directory)
            clean_source = temporary / "test-fixtures"
            report["test_inputs"] = copied_inputs(clean_source)
            build = run_command(
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(temporary / "wheel")],
                ROOT,
                environment,
                report["commands"],
                "Build wheel",
            )
            report["build"] = build
            wheels = list((temporary / "wheel").glob("*.whl"))
            if len(wheels) != 1:
                raise VerificationFailure("Build must produce exactly one wheel")
            report["wheel_name"], report["wheel_sha256"] = wheels[0].name, sha256(wheels[0])
            for version in versions:
                evidence = {"requested_python": version, "commands": [], "success": False}
                report["environments"].append(evidence)
                try:
                    verify_environment(
                        version, temporary, clean_source, wheels[0], uv, environment, evidence
                    )
                except (VerificationFailure, OSError, ValueError, KeyError) as exc:
                    report["failures"].append({"python": version, "error": str(exc)})
            copied_unchanged = all(
                sha256(clean_source / filename) == digest
                for filename, digest in report["test_inputs"]["files"].items()
            )
            report["copied_test_inputs_unchanged"] = copied_unchanged
            if not copied_unchanged:
                report["failures"].append(
                    {"error": "Copied test inputs changed during verification"}
                )
        report["temporary_environments_cleaned"] = True
    except (VerificationFailure, OSError, ValueError) as exc:
        report["failures"].append({"error": str(exc)})
    report["inputs_unchanged_during_verification"] = input_manifest() == original
    if not report["inputs_unchanged_during_verification"]:
        report["failures"].append(
            {
                "error": (
                    "Source, tests, scripts, fixtures, or configuration changed during verification"
                )
            }
        )
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = perf_counter() - started
    report["success"] = not report["failures"] and len(report["environments"]) == len(versions)
    write_report(args.output, report)
    print(f"{'Passed' if report['success'] else 'Failed'}: {args.output}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
