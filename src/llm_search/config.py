"""Configuration from environment variables with sensible defaults.

Two layers:
  - Settings: read-once from env vars at startup (immutable)
  - RuntimeConfig: mutable copy that can be updated via the API at runtime
"""

import threading
from typing import Any, Callable, Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Search provider ---
    search_provider: Literal["searxng", "brave", "serpapi"] = "searxng"
    searxng_url: str = "http://searxng:8080"
    search_api_key: str = ""

    # --- LM Studio ---
    lm_studio_url: str = "http://host.docker.internal:1234/v1"
    lm_studio_timeout: float = 120.0

    # --- Middleware server ---
    middleware_host: str = "0.0.0.0"
    middleware_port: int = 8000

    # --- Limits ---
    max_tool_loop_iterations: int = 10
    search_cache_ttl_seconds: int = 300
    rate_limit_per_minute: int = 30
    max_search_results: int = 5
    max_client_tools: int = 12

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Singleton (immutable — init-time defaults from env)
settings = Settings()


# ── Runtime Mutable Config ──────────────────────────────────────

class RuntimeConfig:
    """Mutable configuration that can be changed at runtime via the API.

    Initialised from ``Settings`` at startup.  Callers that need
    live-updatable values should read from this object rather than
    from the ``settings`` singleton directly.

    Thread-safe for concurrent reads/writes.
    """

    # Subset of keys exposed for runtime editing via the web UI.
    # (Infra-level keys like host/port are intentionally excluded.)
    _EDITABLE_KEYS = frozenset({
        "lm_studio_url",
        "search_provider",
        "searxng_url",
        "search_api_key",
        "max_tool_loop_iterations",
        "max_client_tools",
        "lm_studio_timeout",
        "max_search_results",
    })

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._change_hooks: list[Callable[[str, Any, Any], None]] = []

        # Seed from immutable settings
        self.lm_studio_url: str = settings.lm_studio_url
        self.search_provider: str = settings.search_provider
        self.searxng_url: str = settings.searxng_url
        self.search_api_key: str = settings.search_api_key
        self.max_tool_loop_iterations: int = settings.max_tool_loop_iterations
        self.max_client_tools: int = settings.max_client_tools
        self.lm_studio_timeout: float = settings.lm_studio_timeout
        self.max_search_results: int = settings.max_search_results

    # ── Read helpers ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return all editable fields as a dict (for the API)."""
        return {k: getattr(self, k) for k in self._EDITABLE_KEYS}

    # ── Update ───────────────────────────────────────────────────

    def update(self, **kwargs: Any) -> dict[str, Any]:
        """Update one or more editable fields at runtime.

        Returns a dict of {key: new_value} for the fields that actually
        changed.  Unknown or non-editable keys are silently ignored.
        """
        changed: dict[str, Any] = {}
        with self._lock:
            for key, new_val in kwargs.items():
                if key not in self._EDITABLE_KEYS:
                    continue
                old_val = getattr(self, key, None)
                if old_val == new_val:
                    continue
                # Coerce types to match existing field type
                if old_val is not None and new_val is not None:
                    try:
                        new_val = type(old_val)(new_val)
                    except (ValueError, TypeError):
                        continue
                setattr(self, key, new_val)
                changed[key] = new_val
                # Fire hooks outside the loop so hooks see the updated
                # state of all keys that changed in this batch.
            for key, new_val in changed.items():
                old_val = getattr(settings, key, None)
                for hook in self._change_hooks:
                    try:
                        hook(key, old_val, new_val)
                    except Exception:
                        pass
        return changed

    def on_change(self, hook: Callable[[str, Any, Any], None]) -> None:
        """Register a callback fired after any config key changes.

        Signature: ``hook(key: str, old_value: Any, new_value: Any)``
        """
        self._change_hooks.append(hook)


# Singleton
runtime_config = RuntimeConfig()
