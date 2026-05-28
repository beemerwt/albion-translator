from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import FastAPI


app = FastAPI(title="Albion Translator Sidecar")


class HealthResponse(BaseModel):
    ok: bool


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    language: str
    confidence: float = Field(ge=0.0, le=1.0)


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
    return HealthResponse(ok=True)


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    return _detect_language(request.text)


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    detected = _detect_language(request.text)
    source = detected.language if request.source == "auto" else request.source

    # TODO: Replace this placeholder with an offline translation backend.
    # The stub is deliberately obvious so callers do not mistake it for a real
    # translation.
    return TranslateResponse(
        source=source,
        target=request.target,
        translated_text=f"[stub translation {source}->{request.target}] {request.text}",
    )


def _detect_language(text: str) -> DetectResponse:
    lowered = text.lower()

    # TODO: Replace these simple heuristics with a real offline language
    # detector. These guesses are intentionally low confidence.
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return DetectResponse(language="zh", confidence=0.35)

    vietnamese_markers = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    if any(char in vietnamese_markers for char in lowered):
        return DetectResponse(language="vi", confidence=0.3)

    portuguese_markers = ("ção", "ões", " você ", " não ", " olá ")
    if any(marker in f" {lowered} " for marker in portuguese_markers):
        return DetectResponse(language="pt", confidence=0.25)

    spanish_markers = ("¿", "¡", " el ", " la ", " que ", " hola ", " gracias ")
    if any(marker in f" {lowered} " for marker in spanish_markers):
        return DetectResponse(language="es", confidence=0.25)

    return DetectResponse(language="unknown", confidence=0.0)
