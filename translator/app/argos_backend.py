from __future__ import annotations

from dataclasses import dataclass
import os
import json
from pathlib import Path
import re
import shutil
import urllib.request
from typing import Iterable

ARGOS_HOME = Path(__file__).resolve().parents[1] / ".argos"
os.environ.setdefault("XDG_DATA_HOME", str(ARGOS_HOME / "data"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ARGOS_HOME / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(ARGOS_HOME / "cache"))
os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")

import argostranslate.package
import argostranslate.translate
from minisbd import models as minisbd_models


minisbd_models.cache_dir = str(ARGOS_HOME / "data" / "argos-translate" / "minisbd")


SUPPORTED_SOURCES = ("es", "pt", "zh", "vi")
DEFAULT_TARGET = "en"
REQUIRED_PAIRS = tuple((source, DEFAULT_TARGET) for source in SUPPORTED_SOURCES)
SPANISH_TO_ENGLISH_PACKAGE = "translate-es_en-1_0"
SPANISH_TO_ENGLISH_PACKAGE_URL = (
    "https://data.argosopentech.com/argospm/v1/"
    f"{SPANISH_TO_ENGLISH_PACKAGE}.argosmodel"
)


class ArgosBackendError(RuntimeError):
    pass


class MissingLanguagePairError(ArgosBackendError):
    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        super().__init__(
            f"Argos package {source}->{target} is not installed or not available."
        )


@dataclass(frozen=True)
class PackageInstallSummary:
    installed: list[str]
    skipped: list[str]
    unavailable: list[str]
    failed: list[str]


@dataclass(frozen=True)
class PackageOverride:
    package_name: str
    url: str
    installed_names: tuple[str, ...]


PACKAGE_OVERRIDES = {
    ("es", DEFAULT_TARGET): PackageOverride(
        package_name=SPANISH_TO_ENGLISH_PACKAGE,
        url=SPANISH_TO_ENGLISH_PACKAGE_URL,
        installed_names=(SPANISH_TO_ENGLISH_PACKAGE, "es_en"),
    ),
}


def installed_pairs() -> list[str]:
    pairs = set()
    packages_dir = _packages_dir()
    if not packages_dir.is_dir():
        return []

    for metadata_path in packages_dir.rglob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(metadata, dict):
            continue

        source = metadata.get("from_code") or metadata.get("from")
        target = metadata.get("to_code") or metadata.get("to")
        if source and target:
            pairs.add(f"{normalize_language_code(source)}->{normalize_language_code(target)}")

    for path in packages_dir.iterdir():
        match = re.search(r"translate-([a-z]{2,3})_([a-z]{2,3})(?:-|$)", path.name.lower())
        if match:
            pairs.add(
                f"{normalize_language_code(match.group(1))}->{normalize_language_code(match.group(2))}"
            )

    return sorted(set(pairs))


def translate_text(text: str, source: str, target: str = DEFAULT_TARGET) -> str:
    source = normalize_language_code(source)
    target = normalize_language_code(target)

    if source == target:
        return text

    source_language = _installed_language(source)
    target_language = _installed_language(target)
    if source_language is None or target_language is None:
        raise MissingLanguagePairError(source, target)

    try:
        translation = source_language.get_translation(target_language)
    except Exception as error:
        raise MissingLanguagePairError(source, target) from error

    if translation is None:
        raise MissingLanguagePairError(source, target)

    try:
        return translation.translate(text)
    except Exception as error:
        raise ArgosBackendError(str(error)) from error


def install_required_packages(
    pairs: Iterable[tuple[str, str]] = REQUIRED_PAIRS,
) -> PackageInstallSummary:
    installed = []
    skipped = []
    unavailable = []
    failed = []
    available_packages = None

    for source, target in pairs:
        source = normalize_language_code(source)
        target = normalize_language_code(target)
        pair = f"{source}->{target}"
        override = PACKAGE_OVERRIDES.get((source, target))
        if override is not None:
            try:
                _remove_conflicting_pair_packages(
                    source, target, keep=set(override.installed_names)
                )
                if _is_override_installed(override):
                    _cache_sentence_boundary_model(source)
                    skipped.append(pair)
                    continue

                package_path = _download_package_override(override)
                argostranslate.package.install_from_path(str(package_path))
                _cache_sentence_boundary_model(source)
                installed.append(pair)
            except Exception as error:
                failed.append(f"{pair}: {error}")
            continue

        if is_pair_installed(source, target):
            try:
                _cache_sentence_boundary_model(source)
            except Exception as error:
                failed.append(f"{pair}: {error}")
                continue
            skipped.append(pair)
            continue

        if available_packages is None:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()

        package = next(
            (
                candidate
                for candidate in available_packages
                if candidate.from_code == source and candidate.to_code == target
            ),
            None,
        )
        if package is None:
            unavailable.append(pair)
            continue

        try:
            package_path = Path(package.download())
            argostranslate.package.install_from_path(str(package_path))
            _cache_sentence_boundary_model(source)
            installed.append(pair)
        except Exception as error:
            failed.append(f"{pair}: {error}")

    return PackageInstallSummary(
        installed=installed,
        skipped=skipped,
        unavailable=unavailable,
        failed=failed,
    )


def is_pair_installed(source: str, target: str) -> bool:
    return f"{normalize_language_code(source)}->{normalize_language_code(target)}" in installed_pairs()


def _download_package_override(override: PackageOverride) -> Path:
    download_dir = ARGOS_HOME / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    package_path = download_dir / f"{override.package_name}.argosmodel"
    urllib.request.urlretrieve(override.url, package_path)
    return package_path


def _is_override_installed(override: PackageOverride) -> bool:
    return any(_package_dir(package_name).is_dir() for package_name in override.installed_names)


def _remove_conflicting_pair_packages(source: str, target: str, keep: set[str]) -> None:
    packages_dir = _packages_dir()
    if not packages_dir.is_dir():
        return

    for path in packages_dir.iterdir():
        if path.name in keep:
            continue
        if _package_matches_pair(path, source, target):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _package_matches_pair(path: Path, source: str, target: str) -> bool:
    metadata_path = path / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}

        if isinstance(metadata, dict):
            metadata_source = metadata.get("from_code") or metadata.get("from")
            metadata_target = metadata.get("to_code") or metadata.get("to")
            if metadata_source and metadata_target:
                return (
                    normalize_language_code(metadata_source) == source
                    and normalize_language_code(metadata_target) == target
                )

    match = re.search(r"translate-([a-z]{2,3})_([a-z]{2,3})(?:-|$)", path.name.lower())
    if match is None:
        return False

    return (
        normalize_language_code(match.group(1)) == source
        and normalize_language_code(match.group(2)) == target
    )


def _cache_sentence_boundary_model(source: str) -> None:
    language = {
        "zh": "zh-hans",
    }.get(source, source)
    minisbd_models.get_model_file(language)


def normalize_language_code(language: str) -> str:
    normalized = language.strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "en": "en",
        "eng": "en",
        "english": "en",
        "es": "es",
        "spa": "es",
        "spanish": "es",
        "pt": "pt",
        "por": "pt",
        "portuguese": "pt",
        "zh": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-tw": "zh",
        "zh-hant": "zh",
        "cmn": "zh",
        "chinese": "zh",
        "vi": "vi",
        "vie": "vi",
        "vietnamese": "vi",
    }
    return aliases.get(normalized, normalized)


def _installed_language(code: str):
    for language in argostranslate.translate.get_installed_languages():
        if language.code == code:
            return language
    return None


def _packages_dir() -> Path:
    return ARGOS_HOME / "data" / "argos-translate" / "packages"


def _package_dir(package_name: str) -> Path:
    return _packages_dir() / package_name
