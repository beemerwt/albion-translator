#!/usr/bin/env python3
"""Prepare CTranslate2 translation models for release bundling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "models" / "manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models-cache"
DEFAULT_MODEL = {
    "id": "opus-mt-es-en-ct2",
    "version": "2026-06-07",
    "source": "es",
    "target": "en",
    "hf_model": "Helsinki-NLP/opus-mt-es-en",
    "path": "opus-mt-es-en-int8",
    "quantization": "int8",
    "model_type": "marian",
    "copy_files": [
        "source.spm",
        "target.spm",
        "tokenizer_config.json",
    ],
    "tokenizer": {
        "source": "source.spm",
        "target": "target.spm",
        "tokenizer_json": None,
    },
    "archive": None,
}


def main() -> int:
    args = parse_args()
    manifest = load_or_create_manifest(DEFAULT_MANIFEST, args.dry_run)
    selected = select_models(manifest, args.model, args.all)
    output_dir = args.output_dir.resolve()

    bundled_models: list[dict[str, Any]] = []
    for model in selected:
        quantization = args.quantization or model.get("quantization") or "int8"
        prepared_name = prepared_model_dir_name(model, quantization)
        prepared_dir = output_dir / prepared_name

        if looks_like_ct2_model(prepared_dir) and not args.force:
            print(f"skip {model_label(model)}; {prepared_dir} already looks prepared")
        else:
            prepare_model(model, quantization, prepared_dir, args.force, args.dry_run)

        bundled_models.append(bundled_manifest_model(model, prepared_name, quantization, prepared_dir, args.dry_run))

    write_bundled_manifest(manifest, bundled_models, output_dir, args.dry_run)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face translation models into CTranslate2 release assets."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", help="Model id or language pair, for example es-en.")
    selection.add_argument("--all", action="store_true", help="Prepare every model in the manifest.")
    parser.add_argument("--quantization", default="int8", help="CTranslate2 quantization, default: int8.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Prepared model root, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if output already looks valid.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files or running conversion.")
    return parser.parse_args()


def load_or_create_manifest(path: Path, dry_run: bool) -> dict[str, Any]:
    if path.is_file():
        return read_json(path)

    manifest = {"version": 1, "models": [DEFAULT_MODEL]}
    if dry_run:
        print(f"would create minimal model manifest at {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)
        print(f"created minimal model manifest at {path}")
    return manifest


def download_hf_snapshot(hf_model: str, dry_run: bool) -> Path | None:
    if dry_run:
        print(f"would download Hugging Face snapshot for {hf_model}")
        return None

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        fail(
            "missing Python package huggingface_hub. "
            "Install with: pip install huggingface_hub"
        )

    return Path(snapshot_download(repo_id=hf_model))


def existing_copy_files(snapshot_dir: Path | None, copy_files: list[str]) -> list[str]:
    if snapshot_dir is None:
        return copy_files

    existing = []
    missing = []

    for filename in copy_files:
        if (snapshot_dir / filename).is_file():
            existing.append(filename)
        else:
            missing.append(filename)

    if missing:
        print("warning: skipping missing Hugging Face files: " + ", ".join(missing))

    return existing


def select_models(manifest: dict[str, Any], model_selector: str | None, include_all: bool) -> list[dict[str, Any]]:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        fail("model manifest must contain a non-empty 'models' array")

    if include_all:
        return [normalize_model(model) for model in models]

    assert model_selector is not None
    selector = model_selector.lower()
    for model in models:
        normalized = normalize_model(model)
        pair = f"{normalized['source']}-{normalized['target']}".lower()
        arrow_pair = f"{normalized['source']}->{normalized['target']}".lower()
        if selector in {normalized["id"].lower(), pair, arrow_pair}:
            return [normalized]

    fail(f"no model matched {model_selector!r}; available: {', '.join(model_names(models))}")


def normalize_model(model: Any) -> dict[str, Any]:
    if not isinstance(model, dict):
        fail("each manifest model entry must be an object")

    normalized = dict(model)
    for key in ["id", "source", "target"]:
        if not normalized.get(key):
            fail(f"manifest model entry is missing required field {key!r}")

    normalized.setdefault("hf_model", "Helsinki-NLP/opus-mt-es-en")
    normalized.setdefault("quantization", "int8")
    normalized.setdefault("model_type", "marian")
    normalized.setdefault("copy_files", copy_files_from_tokenizer(normalized.get("tokenizer")))
    normalized.setdefault("path", prepared_model_dir_name(normalized, normalized["quantization"]))
    normalized.setdefault("archive", None)
    return normalized


def copy_files_from_tokenizer(tokenizer: Any) -> list[str]:
    if not isinstance(tokenizer, dict):
        return []
    files = []
    for value in tokenizer.values():
        if isinstance(value, str) and value:
            files.append(value)
    return files


def model_names(models: list[Any]) -> list[str]:
    names = []
    for model in models:
        if isinstance(model, dict):
            source = model.get("source", "?")
            target = model.get("target", "?")
            names.append(f"{model.get('id', '<missing-id>')} ({source}-{target})")
    return names


def require_tools() -> None:
    missing_packages = [
        package
        for package in ["ctranslate2", "transformers", "sentencepiece", "huggingface_hub"]
        if importlib.util.find_spec(package) is None
    ]
    if missing_packages:
        fail(
            "missing Python packages: "
            + ", ".join(missing_packages)
            + ". Install them with: pip install ctranslate2 transformers sentencepiece huggingface_hub"
        )

    if shutil.which("ct2-transformers-converter") is None:
        fail("missing ct2-transformers-converter CLI; install the ctranslate2 Python package")


def verify_sentencepiece_tokenizer(path: Path) -> None:
    paired_candidates = [
        ("source.spm", "target.spm"),
        ("src.spm.model", "tgt.spm.model"),
    ]

    single_candidates = [
        "spm.model",
        "sentencepiece.model",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    ]

    if any((path / src).is_file() and (path / tgt).is_file() for src, tgt in paired_candidates):
        return

    if any((path / name).is_file() for name in single_candidates):
        return

    fail(
        f"{path} does not contain a recognized SentencePiece tokenizer file; "
        "expected source/target pair or one shared tokenizer model"
    )


def build_converter_command(
    model_path: str,
    prepared_dir: Path,
    quantization: str,
    copy_files: list[str],
) -> list[str]:
    command = [
        "ct2-transformers-converter",
        "--model",
        model_path,
        "--output_dir",
        str(prepared_dir),
        "--quantization",
        quantization,
    ]

    if copy_files:
        command.extend(["--copy_files", *copy_files])

    return command


def prepare_preconverted_ct2_model(
    model: dict[str, Any],
    prepared_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    hf_model = model.get("hf_model")
    if not hf_model:
        fail(f"model {model['id']} is missing 'hf_model'")

    copy_files = [str(path) for path in model.get("copy_files", []) if path]

    if not copy_files:
        fail(f"preconverted model {model['id']} must declare copy_files")

    if prepared_dir.exists() and force:
        if dry_run:
            print(f"would remove existing prepared model directory {prepared_dir}")
        else:
            shutil.rmtree(prepared_dir)

    if dry_run:
        print(f"would download preconverted CTranslate2 model {hf_model}")
        for file in copy_files:
            print(f"would copy {file} -> {prepared_dir / file}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        fail(
            "missing Python package huggingface_hub. "
            "Install with: pip install huggingface_hub"
        )

    snapshot_dir = Path(snapshot_download(repo_id=str(hf_model)))

    prepared_dir.parent.mkdir(parents=True, exist_ok=True)

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    prepared_dir.mkdir(parents=True)

    missing = []
    for relative in copy_files:
        source = snapshot_dir / relative
        destination = prepared_dir / relative

        if not source.is_file():
            missing.append(relative)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    if missing:
        fail(
            f"preconverted model {model['id']} is missing expected files: "
            + ", ".join(missing)
        )

    verify_ct2_model(prepared_dir)
    verify_sentencepiece_tokenizer(prepared_dir)


def prepare_model(
    model: dict[str, Any],
    quantization: str,
    prepared_dir: Path,
    force: bool,
    dry_run: bool
) -> None:
    hf_model = model.get("hf_model")
    if not hf_model:
        fail(f"model {model['id']} is missing 'hf_model'")

    if prepared_dir.exists() and force:
        if dry_run:
            print(f"would remove existing prepared model directory {prepared_dir}")
        else:
            shutil.rmtree(prepared_dir)

    raw_copy_files = [str(path) for path in model.get("copy_files", []) if path]

    if dry_run:
        command = build_converter_command(
            model_path=str(hf_model),
            prepared_dir=prepared_dir,
            quantization=quantization,
            copy_files=raw_copy_files,
        )
        print("+ " + " ".join(command))
        return

    require_tools()
    snapshot_dir = download_hf_snapshot(str(hf_model), dry_run)
    copy_files = existing_copy_files(snapshot_dir, raw_copy_files)

    if model.get("preconverted") is True or model.get("format") == "ctranslate2":
        prepare_preconverted_ct2_model(model, prepared_dir, force, dry_run)
        return

    command = build_converter_command(
        model_path=str(snapshot_dir),
        prepared_dir=prepared_dir,
        quantization=quantization,
        copy_files=copy_files,
    )

    prepared_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"converting {model_label(model)} -> {prepared_dir}")

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        fail("ct2-transformers-converter was not found in PATH")
    except subprocess.CalledProcessError as error:
        fail(f"model conversion failed for {model_label(model)} with exit code {error.returncode}")

    verify_ct2_model(prepared_dir)


def bundled_manifest_model(
    model: dict[str, Any], prepared_name: str, quantization: str, prepared_dir: Path, dry_run: bool
) -> dict[str, Any]:
    bundled = dict(model)
    bundled["path"] = prepared_name
    bundled["quantization"] = quantization
    bundled["prepared_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not dry_run and prepared_dir.exists():
        bundled["files"] = checksum_files(prepared_dir)
    return bundled


def write_bundled_manifest(
    source_manifest: dict[str, Any], bundled_models: list[dict[str, Any]], output_dir: Path, dry_run: bool
) -> None:
    manifest = {
        "version": source_manifest.get("version", 1),
        "models": bundled_models,
    }
    path = output_dir / "manifest.json"
    if dry_run:
        print(f"would write bundled manifest to {path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    print(f"wrote bundled manifest to {path}")


def prepared_model_dir_name(model: dict[str, Any], quantization: str | None) -> str:
    manifest_path = model.get("path")
    if isinstance(manifest_path, str) and manifest_path:
        return manifest_path

    base = str(model.get("hf_model") or model.get("id") or f"{model['source']}-{model['target']}")
    slug = base.split("/")[-1].lower().replace("_", "-")

    if slug.endswith("-ct2"):
        slug = slug[:-4]

    if model.get("preconverted") is True or model.get("format") == "ctranslate2":
        return slug

    return f"{slug}-{quantization or model.get('quantization') or 'int8'}"


def looks_like_ct2_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and any(path.glob("model*.bin"))


def verify_ct2_model(path: Path) -> None:
    if not looks_like_ct2_model(path):
        fail(f"{path} does not look like a converted CTranslate2 model; expected config.json and model*.bin")


def checksum_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    return files


def model_label(model: dict[str, Any]) -> str:
    return f"{model['id']} ({model['source']}->{model['target']})"


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        fail(f"failed to parse {path}: {error}")
    except OSError as error:
        fail(f"failed to read {path}: {error}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


if __name__ == "__main__":
    sys.exit(main())
