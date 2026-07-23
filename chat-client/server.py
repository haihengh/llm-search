"""Chat client proxy server for LLM Search.

Serves a static chat UI and reverse-proxies API calls to the
llm-search middleware over the Docker internal network.

Routes:
    GET  /              → Chat UI (index.html)
    GET  /static/*      → Static assets (CSS, JS)
    ALL  /v1/*          → Proxied to middleware
    GET  /health        → Proxied to middleware
    GET  /stats         → Proxied to middleware
    GET  /api/health    → Chat client's own health check
"""

import os

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

MIDDLEWARE_URL = os.getenv("MIDDLEWARE_URL", "http://llm-search:8000")

app = FastAPI(title="LLM Search Chat")


# ── Static files ─────────────────────────────────────────────────
# Must be mounted before the catch-all proxy route.

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    """Serve the chat UI."""
    return FileResponse("static/index.html", media_type="text/html")


@app.get("/sw.js")
async def service_worker():
    """Serve the service worker from root so its scope covers the whole app."""
    return FileResponse("static/sw.js", media_type="application/javascript")


@app.get("/api/health")
async def chat_health():
    """Chat client health check — also probes the middleware."""
    import httpx

    middleware_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MIDDLEWARE_URL}/health")
            middleware_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if middleware_ok else "degraded",
        "middleware_url": MIDDLEWARE_URL,
        "middleware_ok": middleware_ok,
    }


# ── API proxy ────────────────────────────────────────────────────

# Hop-by-hop headers that should NOT be forwarded
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding",
    "upgrade",
}

# Response headers to strip (let the proxy server set its own)
_STRIP_RESP = _HOP_BY_HOP | {"content-encoding", "server", "date"}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(request: Request, path: str):
    """Proxy requests to the llm-search middleware."""
    import httpx

    # Only proxy known API paths
    if not (
        path.startswith("v1/")
        or path in ("health", "stats")
        or path == "models"  # some clients query /models not /v1/models
    ):
        return Response(status_code=404, content="Not found")

    url = f"{MIDDLEWARE_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    # Forward headers (strip hop-by-hop)
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP | {"host", "content-length"}
    }

    body = await request.body()

    # IMPORTANT: Do NOT use `async with` for the httpx client — the
    # StreamingResponse body is consumed *after* this view returns, so
    # the client must stay alive until the stream is exhausted.
    client = httpx.AsyncClient(timeout=300.0)

    req = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
    )
    upstream = await client.send(req, stream=True)

    content_type = upstream.headers.get("content-type", "")

    # Build response headers
    resp_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _STRIP_RESP | {"content-length"}
    }

    # Cache-control for SSE
    if "text/event-stream" in content_type:
        resp_headers.setdefault("cache-control", "no-cache")
        resp_headers.setdefault("connection", "keep-alive")
        resp_headers.setdefault("x-accel-buffering", "no")

    async def body_stream():
        """Yield bytes from upstream, then close the client."""
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(
        content=body_stream(),
        status_code=upstream.status_code,
        media_type=content_type or "application/json",
        headers=resp_headers,
    )
