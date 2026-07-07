"""Fetch and extract readable text from web pages.

Security: validates URLs, rejects internal/private IPs, enforces timeouts.
Uses only stdlib (no new dependencies).
"""

import ipaddress
import logging
import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlparse

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# ── URL Validation ─────────────────────────────────────────────

# Private/reserved IP ranges to block
_BLOCKED_NETWORKS = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
    ipaddress.IPv6Network("fe80::/10"),
]


def _is_private_ip(hostname: str) -> bool:
    """Check if a hostname resolves to a private/reserved IP."""
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return False  # Not an IP literal — DNS resolution handled below
    return any(addr in net for net in _BLOCKED_NETWORKS)


def validate_url(url: str) -> Optional[str]:
    """Validate a URL is safe to fetch.

    Returns an error message string if invalid, or None if OK.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return f"Invalid URL: {url}"

    if parsed.scheme not in ("http", "https"):
        return f"Unsupported protocol: {parsed.scheme}. Only http/https allowed."

    hostname = parsed.hostname
    if not hostname:
        return f"URL has no hostname: {url}"

    # Block private IP literals
    if _is_private_ip(hostname):
        return f"URL resolves to a private/internal address: {hostname}"

    # Block common internal hostnames
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return f"URL points to localhost: {hostname}"

    # Block hostnames that look internal (no dots, e.g. "internal-server")
    if "." not in hostname and hostname != "localhost":
        return f"Hostname looks internal (no domain): {hostname}"

    return None


# ── HTML to Text ────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style/head content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in ("script", "style", "head"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str):
        if tag in ("script", "style", "head") and self._skip_depth > 0:
            self._skip_depth -= 1
        # Add newlines after block elements for readability
        if tag in ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
                    "br", "article", "section", "header", "footer"):
            self._text.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._text.append(text)

    def get_text(self) -> str:
        """Return extracted text, cleaned up."""
        raw = " ".join(self._text)
        # Collapse whitespace
        raw = re.sub(r"\s+", " ", raw)
        # Collapse repeated newlines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def extract_text_from_html(html: str) -> str:
    """Extract readable text from an HTML string.

    Strips script, style, and head content. Returns clean plain text.
    """
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        logger.debug("HTML parse error — returning partial text")
    return parser.get_text()


# ── Main Fetch API ──────────────────────────────────────────────

async def fetch_page_text(
    url: str,
    *,
    timeout: float = 15.0,
    max_chars: int = 8000,
) -> str:
    """Fetch a URL and return its readable text content.

    Args:
        url: The URL to fetch (http/https only).
        timeout: Request timeout in seconds.
        max_chars: Maximum characters to return (truncated with note).

    Returns:
        Extracted text, or an error message string.
    """
    # Validate URL
    error = validate_url(url)
    if error:
        return error

    # Fetch
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "llm-search/0.1 (self-hosted LLM search agent)",
                    "Accept": "text/html,text/plain,*/*",
                },
                follow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Unsupported content type: {content_type}. Only text/html and text/plain are supported."

            text = extract_text_from_html(response.text)

    except httpx.ConnectError:
        return f"Could not connect to {url}"
    except httpx.TimeoutException:
        return f"Request to {url} timed out after {timeout}s"
    except httpx.HTTPStatusError as exc:
        return f"HTTP {exc.response.status_code} when fetching {url}"
    except Exception as exc:
        logger.warning("Unexpected error fetching %s: %s", url, exc)
        return f"Error fetching {url}: {exc}"

    # Truncate if needed
    if len(text) > max_chars:
        text = text[:max_chars] + (
            f"\n\n[Truncated at {max_chars} characters. "
            f"Original was {len(text)} chars after extraction.]"
        )

    if not text.strip():
        return f"No readable text found at {url}"

    return text
