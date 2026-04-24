"""FastAPI server for Miniforge WebUI.

Provides an OpenAI-compatible chat completions API and a polished
single-page chat interface, plus Prometheus metrics for Grafana.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Optional prometheus_client
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    HAS_PROMETHEUS = True
    REQUEST_COUNT = Counter(
        "miniforge_requests_total", "Total requests", ["method", "endpoint", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "miniforge_request_duration_seconds", "Request latency", ["method", "endpoint"]
    )
    GENERATION_TOKENS = Counter(
        "miniforge_generation_tokens_total", "Total generated tokens", ["model"]
    )
    ACTIVE_CONTEXT = Gauge("miniforge_context_tokens", "Current context tokens", ["model"])
except Exception:
    HAS_PROMETHEUS = False
    REQUEST_COUNT = None  # type: ignore
    REQUEST_LATENCY = None  # type: ignore
    GENERATION_TOKENS = None  # type: ignore
    ACTIVE_CONTEXT = None  # type: ignore


class ServerState:
    """Shared server state including model and metrics."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_info: dict[str, Any] = {}
        self.config: Any | None = None
        self.load_error: str | None = None
        self._lock = asyncio.Lock()
        self._generations = 0
        self._total_tokens = 0
        self._start_time = time.time()

    async def load(self) -> None:
        """Load model on startup."""
        from miniforge import Miniforge
        from miniforge.utils.config import M7Config

        model_id = os.environ.get("MINIFORGE_MODEL", "MiniMaxAI/MiniMax-M2.7")
        quantization = os.environ.get("MINIFORGE_QUANTIZATION")
        backend = os.environ.get("MINIFORGE_BACKEND", "llama_cpp")
        download_dir = os.environ.get("MINIFORGE_DOWNLOAD_DIR")

        config = M7Config.from_env()

        if quantization:
            config.quantization = quantization
        if backend:
            config.backend = backend
        if download_dir:
            config.download_dir = download_dir
        if model_id:
            config.model_id = model_id

        self.config = config
        self.load_error = None

        logger.info("Loading model %s with backend %s...", model_id, backend)
        self.model = await Miniforge.from_pretrained(
            model_id,
            config=config,
            backend=backend,
            download_dir=download_dir,
        )
        self.model_info = {
            "id": model_id,
            "object": "model",
            "owned_by": "miniforge",
            "backend": getattr(self.model, "backend_name", config.backend),
            "quantization": config.quantization,
            "offline": config.offline,
        }
        logger.info("Model loaded successfully.")

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        stream: bool = False,
    ) -> Any:
        """Run chat completion."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        response = await self.model.chat_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )
        return response

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        import psutil

        mem = psutil.virtual_memory()
        config_summary = self.config.summary() if self.config is not None else None
        healthy = self.model is not None
        return {
            "status": "healthy" if healthy else "degraded",
            "load_error": self.load_error,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "generations": self._generations,
            "total_tokens": self._total_tokens,
            "memory": {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent,
            },
            "config": config_summary,
            "model": self.model_info,
        }


_state = ServerState()


def _static_dir() -> Path:
    """Return the directory containing static files."""
    return Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    try:
        await _state.load()
    except Exception as exc:
        _state.load_error = str(exc)
        logger.error("Failed to load model: %s", exc)
        # Continue without model; health endpoint will report degraded
    yield
    if _state.model is not None:
        try:
            await _state.model.cleanup()
        except Exception as exc:
            logger.warning("Cleanup error: %s", exc)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Miniforge WebUI",
        description="OpenWebUI-like interface for Miniforge inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    static_dir = _static_dir()
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Any) -> Any:
        """Record request metrics."""
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        if HAS_PROMETHEUS and REQUEST_LATENCY is not None and REQUEST_COUNT is not None:
            method = request.method
            endpoint = request.url.path
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status=response.status_code
            ).inc()
        return response

    @app.get("/", response_model=None)
    async def index() -> FileResponse | dict[str, str]:
        """Serve the main UI page."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Miniforge WebUI — place static/index.html for the UI"}

    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint."""
        healthy = _state.model is not None
        status_code = 200 if healthy else 503
        config_summary = _state.config.summary() if _state.config is not None else None
        return JSONResponse(
            content={
                "status": "healthy" if healthy else "degraded",
                "model_loaded": healthy,
                "load_error": _state.load_error,
                "config": config_summary,
                "timestamp": time.time(),
            },
            status_code=status_code,
        )

    @app.get("/api/runtime")
    async def api_runtime() -> dict[str, Any]:
        """Resolved runtime config and server readiness for the frontend."""
        return {
            "status": "healthy" if _state.model is not None else "degraded",
            "load_error": _state.load_error,
            "model": _state.model_info,
            "config": _state.config.summary() if _state.config is not None else None,
        }

    @app.get("/api/stats")
    async def api_stats() -> dict[str, Any]:
        """Server statistics for the frontend."""
        return _state.get_stats()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """OpenAI-compatible models list."""
        data: list[dict[str, Any]] = []
        if _state.model_info:
            data.append(_state.model_info)
        try:
            from miniforge.models.registry import get_registry

            cache_dir = (
                Path(_state.config.cache_dir) if _state.config and _state.config.cache_dir else None
            )
            for hosted in get_registry(cache_dir).list_hosted_models():
                if any(model.get("id") == hosted.id for model in data):
                    continue
                data.append(
                    {
                        "id": hosted.id,
                        "object": "model",
                        "owned_by": "miniforge-local",
                        "backend": hosted.backend,
                        "quantization": hosted.quantization,
                        "path": hosted.path,
                    }
                )
        except Exception as exc:
            logger.debug("Could not list hosted models: %s", exc)
        return {
            "object": "list",
            "data": data,
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
    ) -> StreamingResponse | JSONResponse | dict[str, Any]:
        """OpenAI-compatible chat completions with streaming support."""
        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 0.95)
        stream = body.get("stream", False)
        model_id = body.get("model", "miniforge")

        if _state.model is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Model not loaded"},
            )

        if stream:
            return StreamingResponse(
                _stream_chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    model_id=model_id,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming
        response_text = await _state.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )

        # Rough token count
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
        completion_tokens = len(response_text.split()) if isinstance(response_text, str) else 0
        total_tokens = prompt_tokens + completion_tokens

        _state._generations += 1
        _state._total_tokens += completion_tokens
        if HAS_PROMETHEUS and GENERATION_TOKENS is not None:
            GENERATION_TOKENS.labels(model=model_id).inc(completion_tokens)
        if HAS_PROMETHEUS and ACTIVE_CONTEXT is not None:
            ACTIVE_CONTEXT.labels(model=model_id).set(total_tokens)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        """Prometheus metrics endpoint."""
        if not HAS_PROMETHEUS:
            return PlainTextResponse("prometheus-client not installed", status_code=501)
        return PlainTextResponse(
            generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


async def _stream_chat(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    model_id: str,
) -> AsyncIterator[str]:
    """Stream chat completion in SSE format."""
    response = await _state.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    n_tokens = 0

    # Send role first
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    if hasattr(response, "__aiter__"):
        async for token in response:
            n_tokens += 1
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    else:
        # Fallback if response is string
        text = response if isinstance(response, str) else str(response)
        for word in text.split():
            n_tokens += 1
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    _state._generations += 1
    _state._total_tokens += n_tokens
    if HAS_PROMETHEUS and GENERATION_TOKENS is not None:
        GENERATION_TOKENS.labels(model=model_id).inc(n_tokens)

    done_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model: str | None = None,
    quantization: str | None = None,
    backend: str | None = None,
    download_dir: str | None = None,
    preset: str | None = None,
    model_dirs: list[str] | None = None,
    offline: bool | None = None,
) -> None:
    """Run the WebUI server (synchronous entry point)."""
    import uvicorn

    if model:
        os.environ["MINIFORGE_MODEL"] = model
    if quantization:
        os.environ["MINIFORGE_QUANTIZATION"] = quantization
    if download_dir:
        os.environ["MINIFORGE_DOWNLOAD_DIR"] = download_dir
    if preset:
        os.environ["MINIFORGE_PRESET"] = preset
    if backend:
        os.environ["MINIFORGE_BACKEND"] = backend
    if model_dirs:
        os.environ["MINIFORGE_MODEL_DIRS"] = ";".join(model_dirs)
    if offline:
        os.environ["MINIFORGE_OFFLINE"] = "1"

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
