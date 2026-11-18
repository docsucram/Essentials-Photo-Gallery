"""
Application version metadata and lightweight GitHub Releases update checker.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

APP_VERSION = "0.14.0"
APP_NAME = "Essentials Photo Gallery"
APP_MUTEX_NAME = "EssentialsPhotoGallery_Mutex"
GITHUB_RELEASES_API = "https://api.github.com/repos/docsucram/Essentials-Photo-Gallery/releases/latest"
GITHUB_RELEASES_URL = "https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest"


def parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse a semantic version string like 'v0.13.1' or '0.13.0' into a comparable int tuple."""
    cleaned = (version_str or "").strip().lstrip("vV")
    nums = [int(p) for p in re.findall(r"\d+", cleaned)]
    return tuple(nums) if nums else (0,)


def is_newer_version(latest: str, current: str = APP_VERSION) -> bool:
    """Return True if *latest* is strictly newer than *current*."""
    t_latest = parse_version_tuple(latest)
    t_curr = parse_version_tuple(current)
    max_len = max(len(t_latest), len(t_curr))
    t_latest += (0,) * (max_len - len(t_latest))
    t_curr += (0,) * (max_len - len(t_curr))
    return t_latest > t_curr


def fetch_latest_release(timeout: float = 8.0) -> dict[str, Any]:
    """
    Query GitHub Releases API for the latest release metadata.
    Returns dict with keys: tag, version, name, html_url, body, installer_url, installer_name.
    Raises Exception on network or parse error.
    """
    req = urllib.request.Request(
        GITHUB_RELEASES_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"EssentialsPhotoGallery/{APP_VERSION}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    tag = str(data.get("tag_name") or "").strip()
    installer_url = ""
    installer_name = ""
    for asset in data.get("assets") or []:
        name = str(asset.get("name") or "")
        dl_url = str(asset.get("browser_download_url") or "")
        if name.lower().endswith(".exe") and dl_url:
            installer_url = dl_url
            installer_name = name
            break

    return {
        "tag": tag,
        "version": tag.lstrip("vV") or APP_VERSION,
        "name": str(data.get("name") or tag),
        "html_url": str(data.get("html_url") or GITHUB_RELEASES_URL),
        "body": str(data.get("body") or ""),
        "installer_url": installer_url,
        "installer_name": installer_name,
    }


def download_installer_exe(
    url: str,
    filename: str = "EssentialsPhotoGallery_Setup_x64.exe",
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    timeout: float = 20.0,
) -> Path | None:
    """Download an installer .exe to %TEMP% atomically and return its Path, or None if cancelled/failed."""
    safe_name = Path(filename).name or "EssentialsPhotoGallery_Setup_x64.exe"
    dest_path = Path(tempfile.gettempdir()) / safe_name
    tmp_path = dest_path.with_suffix(".exe.part")
    req = urllib.request.Request(url, headers={"User-Agent": f"EssentialsPhotoGallery/{APP_VERSION}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp_path, "wb") as fh:
                while True:
                    if cancel_cb is not None and cancel_cb():
                        raise InterruptedError("Cancelled")
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress_cb is not None:
                        progress_cb(done, total)
        os.replace(tmp_path, dest_path)
        return dest_path
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return None
