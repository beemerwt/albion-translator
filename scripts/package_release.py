#!/usr/bin/env python3
"""Assemble Linux release directories with binary and prepared models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models-cache"
DEFAULT_DIST_DIR = REPO_ROOT / "dist"
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "albion-translator"


def main() -> int:
    args = parse_args()
    release_dir = args.dist_dir.resolve() / args.target
    binary = args.binary.resolve()
    models_dir = args.models_dir.resolve()

    verify_binary(binary)
    verify_models(models_dir)
    verify_release_destination(release_dir, args.force)

    if args.dry_run:
        print(f"would create release directory {release_dir}")
        print(f"would copy {binary} -> {release_dir / binary.name}")
        print(f"would copy {models_dir} -> {release_dir / 'models'}")
        if args.target == "linux-cuda":
            print(f"would write CUDA shared-library packaging TODO in {release_dir}")
        return 0

    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    shutil.copy2(binary, release_dir / binary.name)
    shutil.copytree(models_dir, release_dir / "models", ignore=ignore_non_release_model_files)

    if args.target == "linux-cuda":
        write_cuda_todo(release_dir)

    print(f"assembled {args.target} release at {release_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a Linux release folder with the Rust binary and prepared translation models."
    )
    parser.add_argument("--target", required=True, choices=["linux-cpu", "linux-cuda"])
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help=f"Rust binary to package, default: {DEFAULT_BINARY}",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Prepared model directory, default: {DEFAULT_MODELS_DIR}",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help=f"Release output root, default: {DEFAULT_DIST_DIR}",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing target release directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying files.")
    return parser.parse_args()


def verify_binary(path: Path) -> None:
    if not path.is_file():
        fail(f"Rust binary does not exist: {path}")


def verify_models(path: Path) -> None:
    if not path.is_dir():
        fail(f"prepared model directory does not exist: {path}")

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        fail(f"bundled model manifest is missing: {manifest_path}")

    manifest = read_json(manifest_path)
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        fail(f"{manifest_path} must contain a non-empty 'models' array")

    converted_dirs = [candidate for candidate in path.iterdir() if candidate.is_dir() and looks_like_ct2_model(candidate)]
    if not converted_dirs:
        fail(f"no converted CTranslate2 model directories were found in {path}")

    missing = []
    for model in models:
        if not isinstance(model, dict):
            fail(f"{manifest_path} contains a non-object model entry")
        model_path = model.get("path")
        if not isinstance(model_path, str) or not model_path:
            fail(f"model entry {model.get('id', '<missing-id>')} is missing a string 'path'")
        if not looks_like_ct2_model(path / model_path):
            missing.append(model_path)

    if missing:
        fail("manifest references missing or invalid converted model directories: " + ", ".join(missing))


def verify_release_destination(path: Path, force: bool) -> None:
    if path.exists() and not force:
        fail(f"release directory already exists: {path}. Re-run with --force to overwrite it.")


def looks_like_ct2_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and any(path.glob("model*.bin"))


def ignore_non_release_model_files(_directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache"}
    return {name for name in names if name in ignored or name.endswith(".tmp")}


def write_cuda_todo(release_dir: Path) -> None:
    todo = release_dir / "CUDA_SHARED_LIBS_TODO.txt"
    todo.write_text(
        "TODO: copy or document the exact CTranslate2/CUDA shared libraries required by this build.\n"
        "Keep CUDA artifacts separate from linux-cpu releases and avoid vendoring large CUDA runtimes in git.\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        fail(f"failed to parse {path}: {error}")
    except OSError as error:
        fail(f"failed to read {path}: {error}")


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


if __name__ == "__main__":
    sys.exit(main())
