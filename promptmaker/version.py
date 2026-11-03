"""Version checks against the GitHub repository (standard library only).

    from promptmaker.version import check_for_update
    info = check_for_update()          # {"current": "0.3.0", "latest": "0.3.0", "update_available": False, ...}

Releases are git tags named ``v<version>`` with a GitHub Release each; see docs/GUIDE.md (9).
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

REPO = "wowjjang83-sketch/PromptMaker"
API = "https://api.github.com"


def current_version() -> str:
    from . import __version__

    return __version__


def parse_version(text: str) -> Tuple[int, ...]:
    """``"v0.3.1"`` -> ``(0, 3, 1)``. Anything after the numbers (``-rc1``) is ignored."""
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", text or "")
    if not m:
        raise ValueError(f"버전 형식이 아닙니다: {text!r}")
    return tuple(int(p) for p in m.group(1).split("."))


def _get(url: str, timeout: float, token: Optional[str]) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "PromptMaker-version-check"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def latest_release(repo: str = REPO, timeout: float = 5.0, token: Optional[str] = None) -> Dict[str, Any]:
    """Newest published release (falls back to the highest ``v*`` tag when there is none)."""
    try:
        r = _get(f"{API}/repos/{repo}/releases/latest", timeout, token)
        tag = r["tag_name"]
        return {"tag": tag, "version": ".".join(map(str, parse_version(tag))), "url": r.get("html_url", ""),
                "published_at": r.get("published_at", ""), "notes": r.get("body") or ""}
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    tags = [t["name"] for t in _get(f"{API}/repos/{repo}/tags?per_page=100", timeout, token)
            if re.match(r"^v\d", t["name"])]
    if not tags:
        raise LookupError(f"{repo} 에 릴리스나 v* 태그가 없습니다.")
    tag = max(tags, key=parse_version)
    return {"tag": tag, "version": ".".join(map(str, parse_version(tag))),
            "url": f"https://github.com/{repo}/releases/tag/{tag}", "published_at": "", "notes": ""}


def check_for_update(repo: str = REPO, timeout: float = 5.0, token: Optional[str] = None) -> Dict[str, Any]:
    """Compare the installed version with the latest GitHub release."""
    current = current_version()
    latest = latest_release(repo, timeout, token)
    return {
        "current": current,
        "latest": latest["version"],
        "update_available": parse_version(latest["version"]) > parse_version(current),
        "url": latest["url"],
        "install": f"pip install --upgrade git+https://github.com/{repo}.git@{latest['tag']}",
    }
