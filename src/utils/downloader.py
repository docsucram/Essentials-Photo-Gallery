"""
Shared resumable, integrity-verified HTTP file downloader for optional AI/CV add-ons.
Supports HTTP Range resume via `.part` files, Content-Length truncation checks,
automatic SHA-256 validation (explicit `expected_sha256` or HuggingFace `X-Linked-Etag`),
HTML error-page rejection, and atomic `os.replace` commit.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CONTENT_RANGE_TOTAL_RE = re.compile(r"bytes\s+\d+-\d+/(\d+)", re.IGNORECASE)


def compute_file_sha256(path: Path | str, chunk_size: int = 262144) -> str:
    """Computes lowercase hex SHA-256 digest of a file without loading it into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest().lower()


def extract_header_sha256(headers: object) -> str | None:
    """
    Extracts a 64-hex SHA-256 digest from HuggingFace LFS `X-Linked-Etag`
    when the server supplies a SHA-256 hash in that header.
    """
    if headers is None or not hasattr(headers, "get"):
        return None
    for key in ("X-Linked-Etag", "x-linked-etag"):
        raw = headers.get(key)
        if not raw:
            continue
        cleaned = str(raw).strip()
        if cleaned.startswith("W/"):
            cleaned = cleaned[2:].strip()
        cleaned = cleaned.strip('"').strip("'").strip()
        if _HEX64_RE.match(cleaned):
            return cleaned.lower()
    return None


def _unlink_quiet(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def download_file_resumable(
    url: str,
    dest_path: Path | str,
    *,
    expected_sha256: str | None = None,
    min_bytes: int = 1024,
    timeout: float = 30.0,
    chunk_size: int = 262144,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    reject_html: bool = True,
    user_agent: str | None = None,
) -> Path:
    """
    Download ``url`` to ``dest_path`` with HTTP Range resume (``.part``), Content-Length
    truncation checking, SHA-256 verification (explicit or HuggingFace ``X-Linked-Etag``),
    and atomic ``os.replace``.

    On network timeout or cancellation, ``.part`` is preserved so the next call resumes.
    On SHA-256 mismatch, HTML error page, or ``< min_bytes``, ``.part`` is deleted and
    ``ValueError`` is raised.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_name(f"{dest.name}.part")

    if user_agent is None:
        try:
            from src.version import APP_VERSION
            user_agent = f"EssentialsPhotoGallery/{APP_VERSION}"
        except Exception:
            user_agent = "EssentialsPhotoGallery/1.0"

    for attempt in range(2):
        existing_bytes = part_path.stat().st_size if part_path.is_file() else 0
        headers = {"User-Agent": user_agent}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as http_err:
            # 416 Range Not Satisfiable -> reset stale .part and retry once
            if http_err.code == 416 and existing_bytes > 0 and attempt == 0:
                _unlink_quiet(part_path)
                continue
            raise

        with resp:
            status_code = getattr(resp, "status", 200)
            hdrs = getattr(resp, "headers", None)
            content_len = 0
            if hdrs is not None and hasattr(hdrs, "get"):
                try:
                    content_len = int(hdrs.get("Content-Length") or 0)
                except (TypeError, ValueError):
                    content_len = 0

            append_mode = (status_code == 206 and existing_bytes > 0)
            expected_total = 0
            if append_mode:
                cr = str(hdrs.get("Content-Range") or "") if (hdrs is not None and hasattr(hdrs, "get")) else ""
                m = _CONTENT_RANGE_TOTAL_RE.search(cr)
                if m:
                    expected_total = int(m.group(1))
                elif content_len > 0:
                    expected_total = existing_bytes + content_len
            else:
                expected_total = content_len

            header_sha = extract_header_sha256(hdrs)
            done = existing_bytes if append_mode else 0
            open_mode = "ab" if append_mode else "wb"

            with open(part_path, open_mode) as fh:
                while True:
                    if cancel_cb is not None and cancel_cb():
                        raise InterruptedError("Download cancelled")
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress_cb is not None:
                        progress_cb(done, expected_total)

        actual_size = part_path.stat().st_size if part_path.is_file() else 0
        if expected_total > 0 and actual_size != expected_total:
            raise ValueError(
                f"Incomplete download for {dest.name} ({actual_size} of {expected_total} bytes)"
            )
        if actual_size < min_bytes:
            _unlink_quiet(part_path)
            raise ValueError(
                f"Downloaded file {dest.name} is unexpectedly small ({actual_size} bytes)"
            )
        if reject_html:
            with open(part_path, "rb") as chk:
                head = chk.read(64).lstrip()
                if head.startswith(b"<") and (b"<html" in head.lower() or b"<!doctype" in head.lower()):
                    _unlink_quiet(part_path)
                    raise ValueError(
                        f"Downloaded {dest.name} appears to be an HTML page rather than a binary model"
                    )

        target_sha = (expected_sha256.strip().lower() if expected_sha256 else None) or header_sha
        if target_sha:
            actual_sha = compute_file_sha256(part_path)
            if actual_sha != target_sha:
                _unlink_quiet(part_path)
                raise ValueError(
                    f"SHA-256 mismatch for {dest.name} (expected {target_sha[:12]}..., got {actual_sha[:12]}...)"
                )

        os.replace(part_path, dest)
        if progress_cb is not None and expected_total > 0:
            progress_cb(expected_total, expected_total)
        return dest

    raise RuntimeError(f"Failed to download {dest.name} from {url}")
