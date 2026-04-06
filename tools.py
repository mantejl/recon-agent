"""
HTTP recon helpers (Step 3–4): headers and same-domain link extraction.

Facts only in tool outputs; severity/judgment is for the agent or report step later.
"""

from __future__ import annotations

import json
import re
import ssl
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import truststore
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

# Canonical header names as servers return them
_SECURITY_HEADERS: dict[str, str] = {
    "x-frame-options": "X-Frame-Options",
    "content-security-policy": "Content-Security-Policy",
    "strict-transport-security": "Strict-Transport-Security",
    "x-content-type-options": "X-Content-Type-Options",
}

_DEFAULT_TIMEOUT = 15.0

def _sanitize_tool_url(raw: object) -> str:
    """Pull a clean http(s) URL from messy ReAct tool input (dict, newlines, backticks)."""
    if isinstance(raw, dict):
        raw = (
            raw.get("url")
            or raw.get("input")
            or next((str(v) for v in raw.values() if v not in (None, "")), "")
        )
    s = str(raw)
    m = re.search(r"https?://[^\s`\'\"<>]+", s, re.IGNORECASE)
    if m:
        return m.group(0).rstrip(".,);]`\"'")
    return re.sub(r"\s+", "", s).strip()


def _http_get(url: str) -> httpx.Response:
    url = _sanitize_tool_url(url)
    with httpx.Client(
        verify=_SSL_CONTEXT,
        timeout=_DEFAULT_TIMEOUT,
        follow_redirects=True,
    ) as client:
        return client.get(url)


def fetch_headers(url: str) -> dict[str, Any]:
    """
    GET ``url``, inspect response headers, and classify key security headers.

    Returns a dict suitable for JSON / logging:
      - url, status_code
      - headers_present: {canonical_name: value}
      - findings: list of {header, status, severity} for each missing header
    """
    resp = _http_get(url)
    lowered = {k.lower(): v for k, v in resp.headers.items()}

    present: dict[str, str] = {}
    findings: list[dict[str, str]] = []
    for key_lower, canonical in _SECURITY_HEADERS.items():
        if key_lower in lowered:
            present[canonical] = lowered[key_lower]
        else:
            findings.append(
                {
                    "header": canonical,
                    "status": "missing",
                    "severity": "medium",
                }
            )

    return {
        "url": str(resp.url),
        "status_code": resp.status_code,
        "headers_present": present,
        "findings": findings,
    }


def fetch_headers_summary(url: str) -> str:
    """Compact string for printing or feeding an LLM later."""
    return json.dumps(fetch_headers(url), indent=2)


def extract_links(url: str) -> dict[str, Any]:
    """
    GET ``url``, parse HTML, return unique same-host links only.

    Facts only:
      - page_url, status_code
      - links: sorted list of absolute URLs (no fragments), http/https only
      - count

    Same-domain rule: link's ``hostname`` must equal the final response URL's
    ``hostname`` (after redirects), case-insensitive.
    """
    resp = _http_get(url)
    if resp.status_code != 200 or not resp.text:
        return {
            "check": "extract_links",
            "page_url": str(resp.url),
            "status_code": resp.status_code,
            "links": [],
            "count": 0,
            "note": "non-200 or empty body; no links parsed",
        }

    base = urlparse(str(resp.url))
    if not base.hostname:
        return {
            "check": "extract_links",
            "page_url": str(resp.url),
            "status_code": resp.status_code,
            "links": [],
            "count": 0,
            "note": "could not determine base hostname",
        }

    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    for tag in soup.find_all("a", href=True):
        raw_href = (tag.get("href") or "").strip()
        if not raw_href or raw_href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute, _frag = urldefrag(urljoin(str(resp.url), raw_href))
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue
        if parsed.hostname.lower() != base.hostname.lower():
            continue
        seen.add(absolute)

    links = sorted(seen)
    return {
        "check": "extract_links",
        "page_url": str(resp.url),
        "status_code": resp.status_code,
        "links": links,
        "count": len(links),
    }


def extract_links_summary(url: str) -> str:
    return json.dumps(extract_links(url), indent=2)

# LangChain tools (ReAct agent calls these by name) 
#  tool → summary → core logic

@tool
def fetch_headers_for_audit(url: str) -> str:
    """Fetch HTTP response headers for a URL.

    Reports status code, which security headers are present (X-Frame-Options,
    Content-Security-Policy, Strict-Transport-Security, X-Content-Type-Options),
    and which are missing. Use this first when auditing a single page.

    Args:
        url: Full http(s) URL to GET (e.g. https://example.com).

    Returns:
        JSON string with url, status_code, headers_present, findings.
    """
    return fetch_headers_summary(_sanitize_tool_url(url))


@tool
def extract_same_host_links(url: str) -> str:
    """List unique same-host links found in the HTML of a page.

    Resolves relative URLs, drops mailto/tel/javascript anchors, keeps only
    http(s) links whose hostname matches the final page URL after redirects.
    Use after header checks to see what paths the site links to.

    Args:
        url: Full http(s) URL whose HTML will be parsed.

    Returns:
        JSON string with page_url, status_code, links, count.
    """
    return extract_links_summary(_sanitize_tool_url(url))


RECON_TOOLS = [fetch_headers_for_audit, extract_same_host_links]


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] == "links":
        target = argv[1] if len(argv) > 1 else "https://example.com"
        print(extract_links_summary(target))
    else:
        target = argv[0] if argv else "https://example.com"
        print(fetch_headers_summary(target))
