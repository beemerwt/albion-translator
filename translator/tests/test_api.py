from __future__ import annotations

import pytest
from fastapi import HTTPException

from app import argos_backend
from app.main import DetectRequest, TranslateRequest, detect, health, translate


def test_health_shape() -> None:
    response = health()

    body = response.model_dump()
    assert body["ok"] is True
    assert body["backend"] == "argos"
    assert isinstance(body["installed_pairs"], list)


def test_detect_schema_for_spanish() -> None:
    response = detect(DetectRequest(text="hola mundo gracias"))

    body = response.model_dump()
    assert set(body) == {"language", "confidence"}
    assert body["language"] in {"es", "pt", "zh", "vi", "en", "unknown"}


def test_normalizes_language_codes() -> None:
    assert argos_backend.normalize_language_code("Spanish") == "es"
    assert argos_backend.normalize_language_code("zh-CN") == "zh"
    assert argos_backend.normalize_language_code("Vietnamese") == "vi"


def test_spanish_package_override_removes_newer_conflicting_package(
    tmp_path, monkeypatch
) -> None:
    packages_dir = tmp_path / "data" / "argos-translate" / "packages"
    broken_package_dir = packages_dir / "translate-es_en-1_9"
    desired_package_dir = packages_dir / "translate-es_en-1_0"
    broken_package_dir.mkdir(parents=True)
    desired_package_dir.mkdir()
    (broken_package_dir / "metadata.json").write_text(
        '{"from_code":"es","to_code":"en"}'
    )
    (desired_package_dir / "metadata.json").write_text(
        '{"from_code":"es","to_code":"en"}'
    )
    monkeypatch.setattr(argos_backend, "ARGOS_HOME", tmp_path)

    argos_backend._remove_conflicting_pair_packages(
        "es", "en", keep={"translate-es_en-1_0"}
    )

    assert not broken_package_dir.exists()
    assert desired_package_dir.is_dir()


def test_unsupported_explicit_language_error() -> None:
    with pytest.raises(HTTPException) as raised:
        translate(TranslateRequest(text="bonjour", source="fr", target="en"))

    assert raised.value.status_code == 422
    detail = raised.value.detail
    assert detail["error"] == "unsupported_language"
    assert detail["source"] == "fr"


def test_missing_pair_error_when_package_not_installed(monkeypatch) -> None:
    def raise_missing_pair(text: str, source: str, target: str) -> str:
        raise argos_backend.MissingLanguagePairError(source, target)

    monkeypatch.setattr("app.main.translate_text", raise_missing_pair)

    with pytest.raises(HTTPException) as raised:
        translate(TranslateRequest(text="hola mundo", source="es", target="en"))

    assert raised.value.status_code == 503
    detail = raised.value.detail
    assert detail["error"] == "missing_language_pair"
    assert detail["source"] == "es"
    assert detail["target"] == "en"
