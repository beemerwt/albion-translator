from __future__ import annotations

from contextlib import asynccontextmanager
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from langdetect import DetectorFactory, LangDetectException, detect_langs
from pydantic import BaseModel, Field

from .argos_backend import (
    ArgosBackendError,
    DEFAULT_TARGET,
    MissingLanguagePairError,
    install_required_packages,
    installed_pairs,
    normalize_language_code,
    translate_text,
)


DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"es", "pt", "zh", "vi", "en"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if os.environ.get("ALBION_TRANSLATOR_AUTO_INSTALL") == "1":
        summary = install_required_packages()
        print(f"Argos auto-install summary: {summary}")
    yield


app = FastAPI(title="Albion Translator Sidecar", lifespan=lifespan)


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    installed_pairs: list[str]


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    language: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TranslateRequest(BaseModel):
    text: str
    source: str = "auto"
    target: str = "en"


class TranslateResponse(BaseModel):
    source: str
    target: str
    translated_text: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, backend="argos", installed_pairs=installed_pairs())


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    return _detect_language(request.text)


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    source = _resolve_source_language(request.text, request.source)
    target = normalize_language_code(request.target or DEFAULT_TARGET)

    if source not in SUPPORTED_LANGUAGES:
        raise _unsupported_language(source, target)
    if target != DEFAULT_TARGET:
        raise _unsupported_language(source, target)

    try:
        translated_text = translate_text(request.text, source, target)
    except MissingLanguagePairError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "missing_language_pair",
                "source": error.source,
                "target": error.target,
                "message": str(error),
            },
        ) from error
    except ArgosBackendError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "translation_backend_error",
                "source": source,
                "target": target,
                "message": str(error),
            },
        ) from error

    return TranslateResponse(source=source, target=target, translated_text=translated_text)


def _detect_language(text: str) -> DetectResponse:
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return DetectResponse(language="zh", confidence=None)

    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return DetectResponse(language="unknown", confidence=None)

    if not candidates:
        return DetectResponse(language="unknown", confidence=None)

    best = candidates[0]
    language = normalize_language_code(best.lang)
    if language not in SUPPORTED_LANGUAGES:
        language = "unknown"

    # langdetect probabilities are especially unreliable for short chat text.
    confidence = best.prob if len(text.strip()) >= 20 else None
    return DetectResponse(language=language, confidence=confidence)


def _resolve_source_language(text: str, source: str) -> str:
    source = normalize_language_code(source)
    if source == "auto":
        detected = _detect_language(text)
        if detected.language == "unknown":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unsupported_language",
                    "source": "unknown",
                    "target": DEFAULT_TARGET,
                    "message": "Could not detect a supported source language.",
                },
            )
        return detected.language
    return source


def _unsupported_language(source: str, target: str) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "error": "unsupported_language",
            "source": source,
            "target": target,
            "message": f"Unsupported translation direction {source}->{target}.",
        },
    )
