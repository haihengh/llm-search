"""Configuration from environment variables with sensible defaults."""

from typing import Literal

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

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Singleton
settings = Settings()
