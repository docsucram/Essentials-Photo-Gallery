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

APP_VERSION = "0.19.0"
APP_NAME = "Essentials Photo Gallery"
APP_MUTEX_NAME = "EssentialsPhotoGallery_Mutex"
GITHUB_RELEASES_API = "https://api.github.com/repos/docsucram/Essentials-Photo-Gallery/releases/latest"
GITHUB_RELEASES_URL = "https://github.com/docsucram/Essentials-Photo-Gallery/releases/latest"

AUTO_UPDATE_CHECK_KEY = "updates/auto_check_on_startup"
AUTO_UPDATE_FREQ_KEY = "updates/check_frequency"
AUTO_UPDATE_LAST_CHECK_KEY = "updates/last_check_ts"
AUTO_UPDATE_SKIPPED_VER_KEY = "updates/skipped_version"

DEFAULT_UPDATE_FREQUENCY = "weekly"
UPDATE_FREQUENCY_SECONDS: dict[str, int] = {
    "daily": 86400,
    "weekly": 604800,
    "monthly": 2592000,
}

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_DETAILS_BLOCK_RE = re.compile(
    r"<details\b[^>]*>\s*(?:<summary\b[^>]*>(.*?)</summary>)?(.*?)</details>",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_release_notes_markdown(
    raw_md: str, *, include_technical: bool = False
) -> tuple[str, bool]:
    """
    Prepares GitHub / RELEASES.md Markdown cleanly for ``QTextBrowser.setMarkdown()``:
      - Strips remote ``![...](...)`` image embeds and HTML ``<img>`` tags (avoids broken image placeholders)
      - Extracts or hides ``<details><summary>...</summary>...</details>`` technical changelog sections
      - Returns ``(cleaned_markdown, has_technical_section)``.
    """
    text = (raw_md or "").replace("\r\n", "\n").strip()
    if not text:
        return ("", False)

    # Remove image links / standalone image tags
    text = _MD_IMAGE_RE.sub("", text)
    text = _HTML_IMG_RE.sub("", text)

    has_tech = bool(_DETAILS_BLOCK_RE.search(text))

    def _replace_details(match: re.Match[str]) -> str:
        if not include_technical:
            return ""
        summary = re.sub(r"<[^>]+>", "", (match.group(1) or "Full Technical Changelog")).strip()
        body = (match.group(2) or "").strip()
        if not body:
            return ""
        return f"\n\n### {summary}\n\n{body}\n"

    text = _DETAILS_BLOCK_RE.sub(_replace_details, text)
    # Strip any leftover raw <br> or horizontal rule clutter
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (text, has_tech)


def load_local_release_notes(
    version_str: str = APP_VERSION,
    releases_path: Path | str | None = None,
) -> str:
    """
    Reads bundled ``RELEASES.md`` (offline fallback) and extracts the release section matching
    ``v{version_str}``, or the top release section if no exact tag match is found.
    Returns empty string cleanly if ``RELEASES.md`` is missing.
    """
    if releases_path is None:
        try:
            from src.utils.paths import get_resource_path
            cand = get_resource_path("RELEASES.md")
            if not cand.is_file():
                cand = Path(__file__).resolve().parents[1] / "RELEASES.md"
            p = cand
        except Exception:
            p = Path(__file__).resolve().parents[1] / "RELEASES.md"
    else:
        p = Path(releases_path)

    if not p.is_file():
        return ""
    try:
        content = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    except OSError:
        return ""

    sections = re.split(r"(?m)^(?=##\s+Essentials Photo Gallery v)", content)
    release_secs = [s.strip() for s in sections if s.lstrip().startswith("## Essentials Photo Gallery v")]
    if not release_secs:
        return ""

    target = (version_str or "").strip().lstrip("vV")
    chosen = release_secs[0]
    if target:
        for sec in release_secs:
            first_line = sec.splitlines()[0]
            if f"v{target}" in first_line:
                chosen = sec
                break

    # Drop the top `## Essentials Photo Gallery v...` header line & trailing `---`
    lines = chosen.splitlines()
    body_lines = lines[1:] if len(lines) > 1 else lines
    body = "\n".join(body_lines).strip()
    body = re.sub(r"\n+---\s*$", "", body).strip()
    return body


def get_update_check_frequency() -> str:
    """Return the configured startup update check cadence ('daily', 'weekly', or 'monthly')."""
    try:
        from PySide6.QtCore import QSettings

        val = str(
            QSettings("PhotoGallery64", "PhotoGallery64").value(
                AUTO_UPDATE_FREQ_KEY, DEFAULT_UPDATE_FREQUENCY
            )
            or DEFAULT_UPDATE_FREQUENCY
        ).strip().lower()
        return val if val in UPDATE_FREQUENCY_SECONDS else DEFAULT_UPDATE_FREQUENCY
    except Exception:
        return DEFAULT_UPDATE_FREQUENCY


def get_update_cooldown_seconds(freq: str | None = None) -> int:
    """Return cooldown in seconds for the configured or specified update frequency (default weekly)."""
    key = (freq if freq is not None else get_update_check_frequency()).strip().lower()
    return UPDATE_FREQUENCY_SECONDS.get(key, UPDATE_FREQUENCY_SECONDS[DEFAULT_UPDATE_FREQUENCY])


def should_run_auto_update_check(
    now_ts: int | None = None,
    cooldown_seconds: int | None = None,
) -> bool:
    """
    Returns True if background startup update check should run:
      - Not inside pytest (``PYTEST_CURRENT_TEST`` unset)
      - User setting ``updates/auto_check_on_startup`` is enabled (default True)
      - At least ``cooldown_seconds`` (or configured daily/weekly/monthly cadence) has elapsed since ``updates/last_check_ts``
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    try:
        from PySide6.QtCore import QSettings

        settings = QSettings("PhotoGallery64", "PhotoGallery64")
        enabled = settings.value(AUTO_UPDATE_CHECK_KEY, True, type=bool)
        if not enabled:
            return False
        effective_cooldown = (
            get_update_cooldown_seconds() if cooldown_seconds is None else max(0, int(cooldown_seconds))
        )
        if now_ts is None:
            import time
            now_ts = int(time.time())
        last_ts = int(settings.value(AUTO_UPDATE_LAST_CHECK_KEY, 0) or 0)
        return (now_ts - last_ts) >= effective_cooldown
    except Exception:
        return False


def get_skipped_update_version() -> str:
    try:
        from PySide6.QtCore import QSettings
        return str(QSettings("PhotoGallery64", "PhotoGallery64").value(AUTO_UPDATE_SKIPPED_VER_KEY, "") or "").strip()
    except Exception:
        return ""


def set_skipped_update_version(version_str: str | None) -> None:
    try:
        from PySide6.QtCore import QSettings
        s = QSettings("PhotoGallery64", "PhotoGallery64")
        if version_str and str(version_str).strip():
            s.setValue(AUTO_UPDATE_SKIPPED_VER_KEY, str(version_str).strip().lstrip("vV"))
        else:
            s.remove(AUTO_UPDATE_SKIPPED_VER_KEY)
    except Exception:
        pass


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
